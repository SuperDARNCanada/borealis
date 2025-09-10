"""
spectrum
~~~~~~~~
This process handles realtime spectrum measurement.

:copyright: 2025 SuperDARN Canada
:author: Remington Rohel
"""

import datetime as dt
import math
import mmap
import os
import sys
import time
from typing import Optional

import numpy as np
import plotille
import posix_ipc as ipc
from pydantic import ConfigDict, model_validator, NonNegativeInt
from pydantic.dataclasses import dataclass

try:
    import cupy as xp

    mempool = xp.get_default_memory_pool()
except ImportError:
    cupy_available = False
    import numpy as xp
else:
    cupy_available = True


sys.path.append(os.environ["BOREALISPATH"])

from experiment_prototype.experiment_utils.decimation_scheme import (
    create_default_cfs_scheme,
)
from utils.message_formats import (
    DriverRxMetadata,
    SpectrumStart,
    parse_msg,
)
from utils.signals import DSP
from utils.options import Options
import utils.socket_operations as so


@dataclass(config=ConfigDict(validate_assignment=True, arbitrary_types_allowed=True))
class SpectrumParams:
    ringbuffer: Optional[xp.ndarray] = None
    init_time: float = 0.0
    last_sqn_start_time: float = 0.0
    start_idx: NonNegativeInt = 0
    rxrate: float = 0.0
    end_idx: NonNegativeInt = 0

    @model_validator(mode="after")
    def wrap_idx(self):
        """
        Ensure the indices are between 0 and the size of the buffer.
        """
        if self.ringbuffer is not None:
            ringbuffer_size = self.ringbuffer.shape[1]
            if self.start_idx > ringbuffer_size:
                self.start_idx -= ringbuffer_size
            if self.end_idx > ringbuffer_size:
                self.end_idx -= ringbuffer_size
        return self


def compute_spectrum(processor: DSP, samples: xp.ndarray, num_bins: int):
    """
    Compute the spectrum within the band.
    """
    processor.apply_filters(samples)
    processor.move_filter_results()
    num_intervals = int((processor.antennas_iq_samples.shape[-1]) / num_bins)
    data = processor.antennas_iq_samples[..., : num_intervals * num_bins]
    data_chunks = np.reshape(data, data.shape[:-1] + (num_intervals, num_bins))

    fft_data = np.fft.fftshift(np.fft.fft(data_chunks, axis=-1), axes=-1)
    processor.clear_results()

    return fft_data


def extract_samples(params: SpectrumParams):
    """
    Grabs the applicable samples from the ringbuffer. Will adjust `params.start_idx` to
    limit the number of extracted samples to a reasonable amount (0.1 seconds worth).
    """
    ringbuffer_size = params.ringbuffer.shape[1]
    num_samples = params.end_idx - params.start_idx
    max_num_samps = int(0.1 * params.rxrate)
    if num_samples <= 0:
        num_samples += ringbuffer_size
    if num_samples > max_num_samps:
        num_samples = max_num_samps
        start_idx = params.end_idx - num_samples
        if start_idx < 0:
            start_idx += ringbuffer_size
        params.start_idx = start_idx

    if params.start_idx > params.end_idx:
        piece1 = params.ringbuffer[:, params.start_idx :]
        piece2 = params.ringbuffer[:, : params.end_idx]
        tmp1 = xp.array(piece1)
        tmp2 = xp.array(piece2)
        sequence_samples = xp.concatenate((tmp1, tmp2), axis=1)
    else:
        sequence_samples = xp.array(
            params.ringbuffer[:, params.start_idx : params.end_idx]
        )

    return sequence_samples


def plot_spectrum(freqs, spectrum):
    """
    Plots a frequency spectrum using unicode characters.

    :param    freqs: Frequencies being plotted, in Hz.
    :type     freqs: np.ndarray
    :param spectrum: Power for each frequency, in linear units.
    :param spectrum: np.ndarray
    """
    print(
        plotille.plot(
            freqs / 1e3,
            spectrum,
            X_label="kHz",
            Y_label="dB",
            width=56,  # TODO(Remington): specify in config.ini
            height=22,  # TODO(Remington): specify in config.ini
        )
    )


def spectral_analysis(params: SpectrumParams, processor: DSP, num_bins: int):
    """
    Extract samples, compute spectrum, and plot in the terminal.
    """
    log.verbose("Conducting spectral analysis")
    # Conduct spectral analysis
    sequence_samples = extract_samples(params)
    fft_data = compute_spectrum(processor, sequence_samples, num_bins)
    cfs_data = 20 * np.log10(np.sum(np.abs(np.average(fft_data, axis=2)), axis=1))
    cfs_data = np.nan_to_num(cfs_data, nan=-60)
    fs = processor.rx_rate / np.prod(processor.dm_rates)  # Sampling frequency in Hz
    cfs_freqs = np.fft.fftshift(np.fft.fftfreq(num_bins, d=1 / fs))

    plot_spectrum(cfs_freqs, cfs_data[0])

    return fft_data, cfs_freqs


def main():
    options = Options()
    params = SpectrumParams()

    sockets = so.create_sockets(
        options.router_address,
        options.spectrum_to_driver_identity,
        options.spectrum_to_driverrx_identity,
    )
    spectrum_to_driver = sockets[0]
    spectrum_to_driverrx = sockets[1]

    total_antennas = len(options.rx_main_antennas) + len(options.rx_intf_antennas)

    # TODO: Get DecimationScheme from radar_control
    ############ Set up the filtering/downsampling strategy #########################
    dec_scheme = create_default_cfs_scheme()
    taps_per_stage = []
    dm_rates = dec_scheme.dm_rates
    dm_scheme_taps = []
    extra_samples = 0
    for stage in dec_scheme.stages:
        dm_rates.append(stage.dm_rate)
        dm_scheme_taps.append(np.array(stage.filter_taps, dtype=np.complex64))
        taps_per_stage.append(len(stage.filter_taps))
    log.verbose(
        "CFS decimation and filter taps",
        decimation_rates=dm_rates,
        filter_taps_per_stage=taps_per_stage,
    )
    dm_rates = np.array(dm_rates, dtype=np.uint32)
    for dm, taps in zip(reversed(dm_rates), reversed(dm_scheme_taps)):
        extra_samples = (extra_samples * dm) + len(taps) // 2
    total_dm_rate = np.prod(dm_rates)

    ########################## Get start message from driver ########################
    driver_started_msg = so.recv_bytes(
        spectrum_to_driverrx, options.driverrx_to_spectrum_identity, log
    )
    log.verbose("Driver RX message", message=driver_started_msg.decode("utf-8"))
    msg = parse_msg(driver_started_msg.decode("utf-8"), SpectrumStart)
    params.start_idx = msg.idx
    params.rxrate = msg.rxrate
    params.end_idx = params.start_idx + int(0.08 * params.rxrate)
    start_time = dt.datetime.now(dt.timezone.utc)

    ############################ Set up ringbuffer ##################################
    # usrp_driver creates the shared memory, so we must wait for the previous message before attaching to it
    shm = ipc.SharedMemory(options.ringbuffer_name)
    mapped_mem = mmap.mmap(shm.fd, shm.size)
    params.ringbuffer = np.frombuffer(mapped_mem, dtype=np.complex64).reshape(
        total_antennas, -1
    )

    if cupy_available:
        xp.cuda.runtime.hostRegister(
            params.ringbuffer.ctypes.data, params.ringbuffer.size, 0
        )
        xp.cuda.runtime.setDevice(0)

    ########################### Set up the DSP object ###############################
    cfs_processor = DSP(
        dec_scheme.rxrate,
        dm_scheme_taps,
        [-1.5e6],  # TODO: get this from somewhere
        dec_scheme.dm_rates,
        use_shared_mem=False,
    )
    n = 512  # TODO: get this from somewhere
    first_rx_sample_off = 1050  # TODO: Get from radar_control

    ################# Sleep until enough samples have been connected ################
    end_time = dt.datetime.now(dt.timezone.utc)
    to_sleep = dt.timedelta(milliseconds=80) - (end_time - start_time)
    if to_sleep > dt.timedelta(0):
        time.sleep(to_sleep.total_seconds())

    #################################################################################
    # Run an initial spectral analysis before Tx starts
    cfs_data, cfs_freqs = spectral_analysis(params, cfs_processor, n)

    while True:
        msg = so.recv_bytes(
            spectrum_to_driver, options.driver_to_spectrum_identity, log
        )
        rx_metadata = parse_msg(msg.decode("utf-8"), DriverRxMetadata)
        log.verbose("rx_metadata", metadata=rx_metadata)

        sample_time_diff = (
            rx_metadata.sequence_start_time - rx_metadata.initialization_time
        )
        # Time delay between start of ringbuffer and start of this sequence

        sample_in_time = (sample_time_diff * cfs_processor.rx_rate) - extra_samples
        # Move the time delay into units of samples, and adjust to avoid pulse leaking in

        sqn_start_sample = int(math.fmod(sample_in_time, params.ringbuffer.shape[1]))
        samples_needed = int(
            math.ceil(
                float(rx_metadata.num_rx_samps + 2 * extra_samples)
                / float(total_dm_rate)
            )
            * total_dm_rate
        )
        sqn_end_sample = sqn_start_sample + samples_needed + 2 * first_rx_sample_off
        params.end_idx = sqn_start_sample

        cfs_data, cfs_freqs = spectral_analysis(params, cfs_processor, n)

        params.start_idx = sqn_end_sample


if __name__ == "__main__":
    from utils import log_config

    log = log_config.log()
    log.info("SPECTRUM BOOTED")
    if not cupy_available:
        log.warning("cupy not installed")
    try:
        main()
        log.info("SPECTRUM EXITED")
    except Exception as main_exception:
        log.critical("SPECTRUM CRASHED", error=main_exception)
        log.exception("SPECTRUM CRASHED", exception=main_exception)
