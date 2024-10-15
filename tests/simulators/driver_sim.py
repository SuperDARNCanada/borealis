"""
Simulator to mock the usrp_driver module and create arbitrary data.
"""

import datetime as dt
import math
import mmap
import os
import posix_ipc as ipc
import sys

import numpy as np

sys.path.append(os.environ["BOREALISPATH"])
if __debug__:
    from build.debug.src.utils.protobuf.driverpacket_pb2 import DriverPacket
    from build.debug.src.utils.protobuf.rxsamplesmetadata_pb2 import RxSamplesMetadata
else:
    from build.release.src.utils.protobuf.driverpacket_pb2 import DriverPacket
    from build.release.src.utils.protobuf.rxsamplesmetadata_pb2 import RxSamplesMetadata

from src.utils import socket_operations as so
from src.utils.options import Options


def driver_thread():
    options = Options()
    identities = (
        options.driver_to_radctrl_identity,
        options.driver_to_dsp_identity,
        options.driver_to_brian_identity,
    )
    radctrl_socket, dsp_socket, brian_socket = so.create_sockets(
        options.router_address,
        *identities,
    )

    expected_sqn_num = 0
    sqn_num = 0

    # set up the ringbuffer
    num_antennas = len(options.rx_main_antennas) + len(options.rx_intf_antennas)
    buffer_size_per_antenna_samps = int(round(200.0e6 / 8))
    ringbuffer_size = num_antennas * buffer_size_per_antenna_samps
    shm = ipc.SharedMemory(
        options.ringbuffer_name, flags=ipc.O_CREX, size=ringbuffer_size
    )
    mapped_mem = mmap.mmap(shm.fd, shm.size)
    ringbuffer = np.frombuffer(mapped_mem, dtype=np.complex64).reshape(num_antennas, -1)

    # initialize ringbuffer as "noise"
    rng = np.random.default_rng(0)
    noise_pwr = -50.0 + 3.0 * rng.standard_normal(
        size=ringbuffer.shape, dtype=np.float32
    )  # -50.0 +/- 3.0 dB
    noise_phase = (
        2 * np.pi * rng.random(size=ringbuffer.shape, dtype=np.float32)
    )  # [0, 2pi)
    ringbuffer[:] = np.power(10, noise_pwr / 10.0) * np.exp(1j * noise_phase)

    initialization_time = dt.datetime.utcnow()

    # On startup, radar_control sends some preliminary data to help the driver set up
    radctrl_msg = so.recv_bytes(radctrl_socket, options.radctrl_to_driver_identity, log)
    driver_packet = DriverPacket()
    driver_packet.ParseFromString(radctrl_msg)
    rx_rate = driver_packet.rxrate
    # tx_rate = driver_packet.txrate
    # tx_ctr_freq = driver_packet.txcenterfreq
    # rx_ctr_freq = driver_packet.rxcenterfreq

    # let radar_control know that driver is good to start
    so.send_string(radctrl_socket, options.radctrl_to_driver_identity, "DRIVER_READY")

    while True:
        # Get parameters from radar_control
        more_pulses = True
        pulse_starts = []
        pulse_durations = []
        pulse_samples = []
        tx_samples = []
        align_sqns = False
        num_rx_samps = 0

        while more_pulses:
            radctrl_msg = so.recv_bytes(
                radctrl_socket, options.radctrl_to_driver_identity, log
            )
            driver_packet = DriverPacket()
            driver_packet.ParseFromString(radctrl_msg)
            sqn_num = driver_packet.sequence_num
            if sqn_num != expected_sqn_num:
                raise ValueError(
                    f"Sequence number received {sqn_num} did not match expected {expected_sqn_num}"
                )
            rx_rate = driver_packet.rxrate
            # tx_rate = driver_packet.txrate
            # tx_ctr_freq = driver_packet.txcenterfreq
            # rx_ctr_freq = driver_packet.txcenterfreq
            pulse_durations.append(driver_packet.seqtime)
            pulse_starts.append(driver_packet.timetosendsamples)
            burst_start = driver_packet.SOB
            burst_end = driver_packet.EOB
            align_sqns = driver_packet.align_sequences
            if burst_start:
                num_rx_samps = driver_packet.numberofreceivesamples
                for chan in driver_packet.channel_samples:
                    tx_samples.append(
                        np.array(chan.real, dtype=float)
                        + 1j * np.array(chan.imag, dtype=float)
                    )
                tx_samples = np.array(tx_samples, dtype=np.complex64)
                # mock_samples[:, ]
            pulse_samples.append(tx_samples)

            if burst_end:
                more_pulses = False

        # Generate some mock data, of the correct shape and size based on the parameters
        # generate background noise
        shape = (num_antennas, num_rx_samps)
        noise_pwr = -50.0 + 3.0 * rng.standard_normal(
            size=shape, dtype=np.float32
        )  # -50.0 +/- 3.0 dB
        noise_phase = 2 * np.pi * rng.random(size=shape, dtype=np.float32)  # [0, 2pi)
        mock_samples = np.power(10, noise_pwr / 10.0) * np.exp(1j * noise_phase)

        # todo: put the pulses into the mock data
        starting_pulses = dt.datetime.utcnow()
        sqn_start_time = starting_pulses + dt.timedelta(milliseconds=5)
        if align_sqns:
            next_tenth = int(np.ceil(sqn_start_time.microsecond * 1e-5))
            if next_tenth == 10:
                sqn_start_time.replace(second=sqn_start_time.second + 1, microsecond=0)
            else:
                sqn_start_time.replace(microsecond=int(next_tenth * 100_000))

        # place mock data in ringbuffer
        time_since_start = sqn_start_time.timestamp() - initialization_time.timestamp()
        start_idx = int(math.fmod(time_since_start * rx_rate, ringbuffer.shape[1]))
        end_idx = start_idx + num_rx_samps
        if end_idx >= ringbuffer.shape[1]:
            ringbuffer[:, start_idx:] = mock_samples[
                :, : ringbuffer.shape[1] - start_idx
            ]
            ringbuffer[:, : end_idx - ringbuffer.shape[1]] = mock_samples[
                :, ringbuffer.shape[1] - start_idx :
            ]
        else:
            ringbuffer[:, start_idx:end_idx] = mock_samples

        # create metadata message for other modules
        rx_metadata = RxSamplesMetadata()
        rx_metadata.rx_rate = rx_rate
        rx_metadata.initialization_time = initialization_time.timestamp()
        rx_metadata.sequence_start_time = sqn_start_time.timestamp()
        rx_metadata.ringbuffer_size = buffer_size_per_antenna_samps
        rx_metadata.numberofreceivesamples = num_rx_samps
        rx_metadata.sequence_num = sqn_num
        rx_metadata.sequence_time = (
            dt.datetime.utcnow() - starting_pulses
        ).total_seconds()
        rx_metadata.agc_status_bank_h = np.int16(0)
        rx_metadata.agc_status_bank_l = np.int16(0)
        rx_metadata.lp_status_bank_h = np.int16(0)
        rx_metadata.lp_status_bank_l = np.int16(0)
        rx_metadata.gps_locked = True
        rx_metadata.gps_to_system_time_diff = 2.0e-8
        metadata_msg = rx_metadata.SerializeToString()

        # Wait for request for metadata from rx_signal_processing
        so.recv_string(dsp_socket, options.dsp_to_driver_identity, log)

        # Send the metadata
        so.send_bytes(dsp_socket, options.dsp_to_driver_identity, metadata_msg)

        # Wait for request for metadata from brian
        so.recv_string(brian_socket, options.brian_to_driver_identity, log)

        # Send the metadata
        so.send_bytes(brian_socket, options.brian_to_driver_identity, metadata_msg)

        expected_sqn_num += 1


if __name__ == "__main__":
    from src.utils import log_config

    log = log_config.log()
    log.info("DRIVER_SIM BOOTED")
    try:
        driver_thread()
        log.info("DRIVER_SIM EXITED")
    except Exception as main_exception:
        log.critical("DRIVER_SIM CRASHED", error=main_exception)
        log.exception("DRIVER_SIM CRASHED", exception=main_exception)
