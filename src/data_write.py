#!/usr/bin/python3

"""
data_write package
~~~~~~~~~~~~~~~~~~
This package contains utilities to parse protobuf packets containing antennas_iq data, bfiq
data, rawacf data, etc. and write that data to HDF5 or DMAP files.

:copyright: 2017 SuperDARN Canada
"""

# built-in
import argparse as ap
import copy
import datetime
import faulthandler
from multiprocessing import shared_memory
import os
import pickle
import subprocess as sp
import sys
import threading
import time

# third-party
import numpy as np
import zmq
from scipy.constants import speed_of_light

# local
from utils.data_aggregator import Aggregator
from utils import socket_operations as so
from utils.message_formats import AveperiodMetadataMessage
from utils.options import Options
from utils.file_formats import SliceData
from utils import writers


class DataWrite:
    """
    This class contains the functions used to write out processed data to files.

    :param  data_write_options: The options parsed from config file
    :type   data_write_options: Options
    :param  rawacf_format:      The format for rawacf files. Either "hdf5" or "dmap".
    :type   rawacf_format:      str
    """

    def __init__(self, data_write_options: Options, rawacf_format: str):
        # Used for getting info from config.
        self.options = data_write_options

        # String format used for output files names that have slice data.
        self.two_hr_format = "{dt}.{site}.{sliceid}.{{ext}}"

        # Special name and format for rawrf. Contains no slice info.
        self.raw_rf_two_hr_format = "{dt}.{site}.rawrf"
        self.raw_rf_two_hr_name = None

        # Special name and format for tx data. Contains no slice info
        self.tx_data_two_hr_format = "{dt}.{site}.txdata"
        self.tx_data_two_hr_name = None

        # A dict to hold filenames for all available slices in the experiment as they are received.
        self.slice_filenames = {}

        # The git hash used to identify what version of Borealis is running.
        self.git_hash = sp.check_output("git describe --always".split()).strip()

        # The next two-hour boundary for files.
        self.next_boundary = None

        # Default this to true so we know if we are running for the first time.
        self.first_time = True

        # Timestamp of the first sequence in a file
        self.timestamp = None

        # Directory where output files are written
        self.dataset_directory = None

        # Socket for sending rawacf data to realtime
        self.realtime_socket = so.create_sockets(
            self.options.router_address, self.options.dw_to_rt_identity
        )

        # Format of file that rawacf data should be written to
        self.rawacf_format = rawacf_format

    @staticmethod
    def two_hr_ceiling(dt):
        """
        Finds the next 2hr boundary starting from midnight

        :param  dt: A datetime to find the next 2hr boundary.
        :type   dt: DateTime

        :returns:   2hr aligned datetime
        :rtype:     DateTime
        """

        midnight_today = dt.replace(hour=0, minute=0, second=0, microsecond=0)
        boundary_time = midnight_today + datetime.timedelta(hours=2)
        while dt > boundary_time:
            boundary_time += datetime.timedelta(hours=2)

        return boundary_time

    def _write_file(self, aveperiod_data, two_hr_file_with_type, data_type):
        """
        Writes the final data out to the location based on the type of file extension required

        :param  aveperiod_data:         Collection of data from sequences
        :type   aveperiod_data:         SliceData
        :param  two_hr_file_with_type:  Name of the two-hour file with data type added
        :type   two_hr_file_with_type:  str
        :param  data_type:              Data type, e.g. 'antennas_iq', 'bfiq', etc.
        :type   data_type:              str
        """

        os.makedirs(self.dataset_directory, exist_ok=True)
        full_two_hr_file = f"{self.dataset_directory}/{two_hr_file_with_type}.hdf5.site"

        try:
            if data_type == "rawacf" and self.rawacf_format == "dmap":
                writer = writers.DMAPWriter
            else:
                writer = writers.HDF5Writer
            writer.write_record(
                full_two_hr_file, aveperiod_data, self.timestamp, data_type
            )
        except Exception as e:
            if "No space left on device" in str(e):
                log.critical("no space left on device", error=e)
                log.exception("no space left on device", exception=e)
                sys.exit(-1)
            else:
                log.critical("error when saving to file", error=e)
                log.exception("error when saving to file", exception=e)
                sys.exit(-1)

    def output_data(
        self,
        write_bfiq,
        write_antenna_iq,
        write_raw_rf,
        write_tx,
        aveperiod_meta,
        data_parsing,
        write_rawacf=True,
    ):
        """
        Parse through samples and write to file.

        A file will be created using the file extension for each requested data product.

        :param  write_bfiq:         Should beamformed IQ be written to file?
        :type   write_bfiq:         bool
        :param  write_antenna_iq:   Should pre-beamformed IQ be written to file?
        :type   write_antenna_iq:   bool
        :param  write_raw_rf:       Should raw rf samples be written to file?
        :type   write_raw_rf:       bool
        :param  write_tx:           Should the generated tx samples and metadata be written to file?
        :type   write_tx:           bool
        :param  aveperiod_meta:     Metadata from radar control about averaging period
        :type   aveperiod_meta:     AveperiodMetadataMessage
        :param  data_parsing:       All parsed and concatenated data from averaging period
        :type   data_parsing:       Aggregator
        :param  write_rawacf:       Should rawacfs be written to file? Defaults to True.
        :type   write_rawacf:       bool, optional
        """

        start = time.perf_counter()

        # Format the name and location for the dataset
        time_now = datetime.datetime.utcfromtimestamp(data_parsing.timestamps[0])
        today_string = time_now.strftime("%Y%m%d")
        self.timestamp = time_now.strftime("%Y%m%d.%H%M.%S.%f")
        self.dataset_directory = f"{self.options.data_directory}/{today_string}"

        if self.first_time:
            self.raw_rf_two_hr_name = self.raw_rf_two_hr_format.format(
                dt=time_now.strftime("%Y%m%d.%H%M.%S"), site=self.options.site_id
            )
            self.tx_data_two_hr_name = self.tx_data_two_hr_format.format(
                dt=time_now.strftime("%Y%m%d.%H%M.%S"), site=self.options.site_id
            )
            self.next_boundary = self.two_hr_ceiling(time_now)
            self.first_time = False

        for slice_id in data_parsing.slice_ids:
            if slice_id not in self.slice_filenames:
                two_hr_str = self.two_hr_format.format(
                    dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                    sliceid=slice_id,
                    site=self.options.site_id,
                )
                self.slice_filenames[slice_id] = two_hr_str

        if time_now > self.next_boundary:
            self.raw_rf_two_hr_name = self.raw_rf_two_hr_format.format(
                dt=time_now.strftime("%Y%m%d.%H%M.%S"), site=self.options.site_id
            )
            self.tx_data_two_hr_name = self.tx_data_two_hr_format.format(
                dt=time_now.strftime("%Y%m%d.%H%M.%S"), site=self.options.site_id
            )
            for slice_id in self.slice_filenames.keys():
                two_hr_str = self.two_hr_format.format(
                    dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                    sliceid=slice_id,
                    site=self.options.site_id,
                )
                self.slice_filenames[slice_id] = two_hr_str

            self.next_boundary = self.two_hr_ceiling(time_now)

        all_slice_data = {}
        for sqn_meta in aveperiod_meta.sequences:
            for rx_channel in sqn_meta.rx_channels:
                # time to first range and back. convert to meters, div by c then convert to us
                rtt = (rx_channel.first_range * 2 * 1.0e3 / speed_of_light) * 1.0e6

                encodings = []
                for encoding in rx_channel.sequence_encodings:
                    encoding = np.array(encoding, dtype=np.float32)
                    encodings.append(encoding)
                encodings = np.array(encodings, dtype=np.float32)

                lags = [
                    [lag.pulse_position[0], lag.pulse_position[1]]
                    for lag in rx_channel.ltabs
                ]

                parameters = SliceData()
                parameters.agc_status_word = np.uint32(data_parsing.agc_status_word)
                parameters.averaging_method = rx_channel.averaging_method
                parameters.beam_azms = [beam.beam_azimuth for beam in rx_channel.beams]
                parameters.beam_nums = [
                    np.uint32(beam.beam_num) for beam in rx_channel.beams
                ]
                parameters.blanked_samples = np.array(sqn_meta.blanks, dtype=np.uint32)
                parameters.borealis_git_hash = self.git_hash.decode("utf-8")

                if np.uint32(rx_channel.slice_id) in aveperiod_meta.cfs_slice_ids:
                    parameters.cfs_freqs = np.array(aveperiod_meta.cfs_freqs)
                    parameters.cfs_noise = np.array(aveperiod_meta.cfs_noise)
                    parameters.cfs_range = np.array(
                        aveperiod_meta.cfs_range[np.uint32(rx_channel.slice_id)]
                    )
                    parameters.cfs_masks = np.array(
                        aveperiod_meta.cfs_masks[np.uint32(rx_channel.slice_id)]
                    )
                else:
                    parameters.cfs_freqs = np.array([])
                    parameters.cfs_noise = np.array([])
                    parameters.cfs_range = np.array([])
                    parameters.cfs_masks = np.array([])

                parameters.data_normalization_factor = (
                    aveperiod_meta.data_normalization_factor
                )
                parameters.experiment_comment = aveperiod_meta.experiment_comment
                parameters.experiment_id = np.int16(aveperiod_meta.experiment_id)
                parameters.experiment_name = aveperiod_meta.experiment_name
                parameters.first_range = np.float32(rx_channel.first_range)
                parameters.first_range_rtt = np.float32(rtt)
                parameters.freq = np.uint32(rx_channel.rx_freq)
                parameters.gps_locked = data_parsing.gps_locked
                parameters.gps_to_system_time_diff = (
                    data_parsing.gps_to_system_time_diff
                )
                parameters.int_time = np.float32(aveperiod_meta.aveperiod_time)
                parameters.intf_antenna_count = np.uint32(
                    len(rx_channel.rx_intf_antennas)
                )
                parameters.lags = np.array(lags, dtype=np.uint32)
                parameters.lag_numbers = parameters.lags[:, 1] - parameters.lags[:, 0]
                parameters.lp_status_word = np.uint32(data_parsing.lp_status_word)
                parameters.main_antenna_count = np.uint32(
                    len(rx_channel.rx_main_antennas)
                )
                parameters.num_ranges = np.uint32(rx_channel.num_ranges)
                parameters.num_sequences = aveperiod_meta.num_sequences
                parameters.num_slices = len(aveperiod_meta.sequences) * len(
                    sqn_meta.rx_channels
                )
                parameters.pulse_phase_offset = encodings
                parameters.pulses = np.array(rx_channel.ptab, dtype=np.uint32)
                parameters.range_sep = np.float32(rx_channel.range_sep)
                parameters.rx_center_freq = aveperiod_meta.rx_ctr_freq
                parameters.rx_sample_rate = sqn_meta.output_sample_rate
                parameters.rx_main_phases = rx_channel.rx_main_phases
                parameters.rx_intf_phases = rx_channel.rx_intf_phases
                parameters.samples_data_type = "complex float"
                parameters.scan_start_marker = aveperiod_meta.scan_flag
                parameters.scheduling_mode = aveperiod_meta.scheduling_mode
                parameters.slice_comment = rx_channel.slice_comment
                parameters.slice_id = np.uint32(rx_channel.slice_id)
                parameters.slice_interfacing = rx_channel.interfacing
                parameters.sqn_timestamps = data_parsing.timestamps
                parameters.station = self.options.site_id
                parameters.tau_spacing = np.uint32(rx_channel.tau_spacing)
                parameters.tx_antenna_phases = np.complex64(
                    rx_channel.tx_antenna_phases
                )
                parameters.tx_pulse_len = np.uint32(rx_channel.pulse_len)

                all_slice_data[rx_channel.slice_id] = parameters

        if write_rawacf and data_parsing.mainacfs_available:
            self._write_correlations(all_slice_data, data_parsing)
        if write_bfiq and data_parsing.bfiq_available:
            self._write_bfiq_params(all_slice_data, data_parsing)
        if write_antenna_iq and data_parsing.antenna_iq_available:
            self._write_antenna_iq_params(
                all_slice_data, data_parsing, aveperiod_meta.sequences
            )
        if data_parsing.rawrf_available:
            if write_raw_rf:
                # Just need first available slice parameters.
                one_slice_data = next(iter(all_slice_data.values()))
                self._write_raw_rf_params(
                    one_slice_data, data_parsing, aveperiod_meta.input_sample_rate
                )
            else:
                for rf_samples_location in data_parsing.rawrf_locations:
                    if rf_samples_location is not None:
                        shm = shared_memory.SharedMemory(name=rf_samples_location)
                        shm.close()
                        shm.unlink()
        if write_tx:
            self._write_tx_data(aveperiod_meta.sequences)

        write_time = time.perf_counter() - start
        log.info(
            "wrote record",
            write_time=write_time * 1e3,
            time_units="ms",
            dataset_name=self.timestamp,
        )

    def _write_correlations(self, aveperiod_data: dict, parsed_data: Aggregator):
        """
        Gathers the per-sequence data from parsed_data, conducts averaging over the sequence
        dimension, and then writes the records for each slice to their respective files.

        :param  aveperiod_data:  Dict containing SliceData for each slice.
        :type   aveperiod_data:  dict
        :param  parsed_data:     Object containing the data accumulators and flags
        :type   parsed_data:     Aggregator
        """
        main_acfs = parsed_data.mainacfs_accumulator
        xcfs = parsed_data.xcfs_accumulator
        intf_acfs = parsed_data.intfacfs_accumulator

        def find_expectation_value(x):
            """
            Get the mean or median of all correlations from all sequences in the integration
            period - only this will be recorded.
            This is effectively 'averaging' all correlations over the integration time, using a
            specified method for combining them.
            """
            # array_2d is num_sequences x (num_beams*num_ranges*num_lags)
            # so we get median of all sequences.
            averaging_method = slice_data.averaging_method
            array_2d = np.array(x, dtype=np.complex64)
            num_beams, num_ranges, num_lags = np.array(
                [
                    len(slice_data.beam_nums),
                    slice_data.num_ranges,
                    slice_data.lags.shape[0],
                ],
                dtype=np.uint32,
            )

            # First range offset in samples
            sample_off = slice_data.first_range_rtt * 1e-6 * slice_data.rx_sample_rate
            sample_off = np.uint32(sample_off)

            # Find sample number which corresponds with second pulse in sequence
            tau_in_samples = slice_data.tau_spacing * 1e-6 * slice_data.rx_sample_rate
            second_pulse_sample_num = (
                np.uint32(tau_in_samples) * slice_data.pulses[1] - sample_off - 1
            )

            # Average the data
            if averaging_method == "mean":
                array_expectation_value = np.mean(array_2d, axis=0)
            elif averaging_method == "median":
                array_expectation_value = np.median(
                    np.real(array_2d), axis=0
                ) + 1j * np.median(np.imag(array_2d), axis=0)
            else:
                log.error("wrong averaging method [mean, median]")
                raise

            # Reshape array to be 3d so we can replace lag0 far ranges that are cluttered with those
            # from alternate lag0 which have no clutter.
            array_3d = array_expectation_value.reshape(
                (num_beams, num_ranges, num_lags)
            )
            array_3d[:, second_pulse_sample_num:, 0] = array_3d[
                :, second_pulse_sample_num:, -1
            ]

            return array_3d

        for slice_num in main_acfs:
            slice_data = aveperiod_data[slice_num]
            slice_data.main_acfs = find_expectation_value(main_acfs[slice_num]["data"])

        for slice_num in xcfs:
            slice_data = aveperiod_data[slice_num]
            if parsed_data.xcfs_available:
                slice_data.xcfs = find_expectation_value(xcfs[slice_num]["data"])
            else:
                slice_data.xcfs = np.array([], np.complex64)

        for slice_num in intf_acfs:
            slice_data = aveperiod_data[slice_num]
            if parsed_data.intfacfs_available:
                slice_data.intf_acfs = find_expectation_value(
                    intf_acfs[slice_num]["data"]
                )
            else:
                slice_data.intf_acfs = np.array([], np.complex64)

        for slice_num, slice_data in aveperiod_data.items():
            two_hr_file_with_type = self.slice_filenames[slice_num].format(ext="rawacf")
            self._write_file(slice_data, two_hr_file_with_type, "rawacf")

            # Send rawacf data to realtime (if there is any)
            full_dict = {self.timestamp: slice_data.to_dmap(self.timestamp)}
            so.send_pyobj(
                self.realtime_socket, self.options.rt_to_dw_identity, full_dict
            )

    def _write_bfiq_params(self, aveperiod_data: dict, parsed_data: Aggregator):
        """
        Write out any possible beamformed IQ data that has been parsed. Adds additional slice
        info to each parameter dict.

        :param  aveperiod_data:  Dict of SliceData for each slice.
        :type   aveperiod_data:  dict
        :param  parsed_data:     Object containing the data accumulators and flags
        :type   parsed_data:     Aggregator
        """

        bfiq = parsed_data.bfiq_accumulator
        slice_id_list = [x for x in bfiq.keys() if isinstance(x, int)]

        for slice_num in slice_id_list:
            slice_data = aveperiod_data[slice_num]
            slice_data.channels = []

            all_data = []
            num_antenna_arrays = 1
            slice_data.channels.append("main")
            all_data.append(bfiq[slice_num]["main_data"])
            if "intf" in bfiq[slice_num]:
                num_antenna_arrays += 1
                slice_data.channels.append("intf")
                all_data.append(bfiq[slice_num]["intf_data"])

            slice_data.data = np.stack(all_data, axis=0)
            slice_data.num_samps = np.uint32(slice_data.data.shape[-1])

        for slice_num, slice_data in aveperiod_data.items():
            two_hr_file_with_type = self.slice_filenames[slice_num].format(ext="bfiq")
            self._write_file(slice_data, two_hr_file_with_type, "bfiq")

    def _write_antenna_iq_params(
        self, aveperiod_data: dict, parsed_data: Aggregator, sequences: list
    ):
        """
        Writes out any pre-beamformed IQ that has been parsed. Adds additional slice info
        to each parameter dict. Pre-beamformed iq is the individual antenna received data.
        ``channels`` will list the antennas' order.

        :param  aveperiod_data:  Dict that holds SliceData for each slice.
        :type   aveperiod_data:  dict
        :param  parsed_data:     Object containing the data accumulators and flags
        :type   parsed_data:     Aggregator
        :param  sequences:       List of ProcessedSequenceMetadata messages
        :type   sequences:       ProcessedSequenceMessage
        """

        antenna_iq = parsed_data.antenna_iq_accumulator
        slice_id_list = [x for x in antenna_iq.keys() if isinstance(x, int)]

        # Parse the antennas from message
        rx_main_antennas = {}
        rx_intf_antennas = {}

        for sqn in sequences:
            for rx_channel in sqn.rx_channels:
                rx_main_antennas[rx_channel.slice_id] = list(
                    rx_channel.rx_main_antennas
                )
                rx_intf_antennas[rx_channel.slice_id] = list(
                    rx_channel.rx_intf_antennas
                )

        # Build strings from antennas used in the message. This will be used to know
        # what antennas were recorded on since we sample all available USRP channels
        # and some channels may not be transmitted on, or connected.
        for slice_num in rx_main_antennas:
            rx_main_antennas[slice_num] = [
                f"antenna_{x}" for x in rx_main_antennas[slice_num]
            ]
            rx_intf_antennas[slice_num] = [
                f"antenna_{x + self.options.main_antenna_count}"
                for x in rx_intf_antennas[slice_num]
            ]

        final_data_params = {}
        for slice_num in slice_id_list:
            final_data_params[slice_num] = {}

            for stage in antenna_iq[slice_num]:
                stage_data = copy.deepcopy(aveperiod_data[slice_num])
                stage_data.num_samps = np.uint32(
                    antenna_iq[slice_num][stage].get("num_samps", None)
                )
                stage_data.channels = (
                    rx_main_antennas[slice_num] + rx_intf_antennas[slice_num]
                )

                data = []
                for k, data_dict in antenna_iq[slice_num][stage].items():
                    if k in stage_data.channels:
                        data.append(data_dict["data"])

                stage_data.data = np.stack(data, axis=0)
                final_data_params[slice_num][stage] = stage_data

        for slice_num, slice_ in final_data_params.items():
            for stage, params in slice_.items():
                two_hr_file_with_type = self.slice_filenames[slice_num].format(
                    ext=f"{stage}_iq"
                )
                self._write_file(params, two_hr_file_with_type, "antennas_iq")

    def _write_raw_rf_params(
        self, slice_data, parsed_data: Aggregator, sample_rate: float
    ):
        """
        Opens the shared memory location in the message and writes the samples out to file.
        Write medium must be able to sustain high write bandwidth. Shared memory is destroyed
        after write. It's expected that the user will have knowledge of what they are looking
        for when working with this data.

        Note that because this data is not slice-specific a lot of slice-specific data (ex.
        pulses, beam_nums, beam_azms) is not included (user must look at the experiment they
        ran).

        :param  slice_data:  Parameters for a single slice during the averaging period.
        :type   slice_data:  SliceData
        :param  parsed_data: Object containing the data accumulators and flags
        :type   parsed_data: Aggregator
        :param  sample_rate: Sampling rate of the data, in Hz.
        :type   sample_rate: float
        """

        raw_rf = parsed_data.rawrf_locations
        num_rawrf_samps = parsed_data.rawrf_num_samps

        samples_list = []
        shared_memory_locations = []
        total_ants = len(self.options.rx_main_antennas) + len(
            self.options.rx_intf_antennas
        )

        for raw in raw_rf:
            shared_mem = shared_memory.SharedMemory(name=raw)
            rawrf_array = np.ndarray(
                (total_ants, num_rawrf_samps),
                dtype=np.complex64,
                buffer=shared_mem.buf,
            )
            samples_list.append(rawrf_array)
            shared_memory_locations.append(shared_mem)

        slice_data.data = np.stack(samples_list, axis=0)
        slice_data.rx_sample_rate = np.float32(sample_rate)
        slice_data.num_samps = np.uint32(len(samples_list[0]) / total_ants)
        slice_data.main_antenna_count = np.uint32(self.options.main_antenna_count)
        slice_data.intf_antenna_count = np.uint32(self.options.intf_antenna_count)

        self._write_file(slice_data, self.raw_rf_two_hr_name, "rawrf")

        # Can only close mapped memory after it's been written to disk.
        for shared_mem in shared_memory_locations:
            shared_mem.close()
            shared_mem.unlink()

    def _write_tx_data(self, sequences: list):
        """
        Writes out the tx samples and metadata for debugging purposes.
        Does not use same parameters of other writes.

        :param  sequences: List of ProcessedSequenceMetadata messages
        :type   sequences: ProcessedSequenceMessage
        """
        tx_data = SliceData()
        for f in SliceData.type_fields("txdata"):
            setattr(tx_data, f, [])  # initialize to a list for all fields

        has_tx_data = [sqn.tx_data is not None for sqn in sequences]

        if True in has_tx_data:  # If any sequence has tx data, write to file
            for sqn in sequences:
                if sqn.tx_data is not None:
                    meta_data = sqn.tx_data
                    tx_data.tx_rate.append(meta_data.tx_rate)
                    tx_data.tx_center_freq.append(meta_data.tx_ctr_freq)
                    tx_data.pulse_timing.append(meta_data.pulse_timing_us)
                    tx_data.pulse_sample_start.append(meta_data.pulse_sample_start)
                    tx_data.dm_rate.append(meta_data.dm_rate)
                    tx_data.tx_samples.append(meta_data.tx_samples)
                    tx_data.decimated_tx_samples.append(meta_data.decimated_tx_samples)

                self._write_file(tx_data, self.tx_data_two_hr_name, "txdata")


def dw_parser():
    parser = ap.ArgumentParser(description="Write processed SuperDARN data to file")
    parser.add_argument(
        "--enable-raw-acfs", help="Enable raw acf writing", action="store_true"
    )
    parser.add_argument(
        "--enable-bfiq", help="Enable beamformed iq writing", action="store_true"
    )
    parser.add_argument(
        "--enable-antenna-iq",
        help="Enable individual antenna iq writing",
        action="store_true",
    )
    parser.add_argument(
        "--enable-raw-rf",
        help="Save raw, unfiltered IQ samples. Requires HDF5.",
        action="store_true",
    )
    parser.add_argument(
        "--enable-tx",
        help="Save tx samples and metadata. Requires HDF5.",
        action="store_true",
    )
    parser.add_argument(
        "--rawacf-format",
        choices=["hdf5", "dmap"],
        help="Format to store rawacf files in.",
    )
    return parser


def main():
    faulthandler.enable()
    args = dw_parser().parse_args()

    options = Options()
    if args.rawacf_format is None:
        rawacf_format = options.rawacf_format
    else:
        rawacf_format = args.rawacf_format

    sockets = so.create_sockets(
        options.router_address,
        options.dw_to_dsp_identity,
        options.dw_to_radctrl_identity,
        options.dw_cfs_identity,
    )

    dsp_to_data_write = sockets[0]
    radctrl_to_data_write = sockets[1]
    cfs_sequence_socket = sockets[2]

    poller = zmq.Poller()
    poller.register(dsp_to_data_write, zmq.POLLIN)
    poller.register(radctrl_to_data_write, zmq.POLLIN)
    poller.register(cfs_sequence_socket, zmq.POLLIN)

    log.debug("socket connected")

    aggregator = Aggregator(options=options)

    current_experiment = None
    data_write = None
    first_time = True
    expected_sqn_num = 0
    queued_sqns = []
    cfs_nums = []
    aveperiod_metadata_dict = dict()
    while True:
        try:
            socks = dict(poller.poll())
        except KeyboardInterrupt:
            log.info("keyboard interrupt exit")
            sys.exit(0)

        if (
            radctrl_to_data_write in socks
            and socks[radctrl_to_data_write] == zmq.POLLIN
        ):
            aveperiod_meta = so.recv_pyobj(
                radctrl_to_data_write,
                options.radctrl_to_dw_identity,
                log,
                expected_type=AveperiodMetadataMessage,
            )
            aveperiod_metadata_dict[aveperiod_meta.last_sqn_num] = aveperiod_meta

        if cfs_sequence_socket in socks and socks[cfs_sequence_socket] == zmq.POLLIN:
            cfs_sqn_num = so.recv_pyobj(
                cfs_sequence_socket,
                options.radctrl_cfs_identity,
                log,
                expected_type=int,
            )
            log.debug(
                "Received CFS sequence, increasing expected_sqn_num",
                cfs_sqn_num=cfs_sqn_num,
            )
            cfs_nums.append(cfs_sqn_num)

        if expected_sqn_num in cfs_nums:
            # If the current expected sqn num was a CFS sequence, increase the expected
            # sqn num by 1 to skip over the CFS sequence.
            cfs_nums.remove(expected_sqn_num)
            expected_sqn_num += 1

        if dsp_to_data_write in socks and socks[dsp_to_data_write] == zmq.POLLIN:
            data = so.recv_bytes_from_any_iden(dsp_to_data_write)
            processed_data = pickle.loads(data)
            queued_sqns.append(processed_data)
            log.debug("Received from DSP", sequence_num=processed_data.sequence_num)

            # Check if any data processing finished out of order.
            if processed_data.sequence_num != expected_sqn_num:
                continue

            sorted_q = sorted(queued_sqns, key=lambda x: x.sequence_num)

            # This is needed to check that if we have a backlog, there are no more
            # skipped sequence numbers we are still waiting for.
            break_now = False
            for i, pd in enumerate(sorted_q):
                if pd.sequence_num != expected_sqn_num + i:
                    expected_sqn_num += i
                    break_now = True
                    break
            if break_now:
                try:
                    if len(sorted_q) <= 20:
                        raise AssertionError(
                            f"len(sorted_q) ({len(sorted_q)}) is not <= 20"
                        )
                except Exception as e:
                    log.error("lost sequences", sequence_num=expected_sqn_num, error=e)
                    log.exception("lost sequences", exception=e)
                    sys.exit(1)
                continue

            expected_sqn_num = sorted_q[-1].sequence_num + 1

            for pd in sorted_q:
                if not first_time:
                    if aggregator.sequence_num in aveperiod_metadata_dict:
                        aggregator.finalize()
                        aveperiod_metadata = aveperiod_metadata_dict.pop(
                            aggregator.sequence_num
                        )

                        if aveperiod_metadata.experiment_name != current_experiment:
                            data_write = DataWrite(options, rawacf_format)
                            current_experiment = aveperiod_metadata.experiment_name

                        kwargs = dict(
                            write_bfiq=args.enable_bfiq,
                            write_antenna_iq=args.enable_antenna_iq,
                            write_raw_rf=args.enable_raw_rf,
                            write_tx=args.enable_tx,
                            aveperiod_meta=aveperiod_metadata,
                            data_parsing=aggregator,
                            write_rawacf=args.enable_raw_acfs,
                        )
                        thread = threading.Thread(
                            target=data_write.output_data, kwargs=kwargs
                        )
                        thread.daemon = True
                        thread.start()
                        aggregator = Aggregator(options=options)

                first_time = False

                start = time.perf_counter()
                aggregator.update(pd)
                parse_time = time.perf_counter() - start
                log.info(
                    f"parsed sequence {pd.sequence_num}",
                    parse_time=parse_time * 1e3,
                    time_units="ms",
                    slice_ids=[dset.slice_id for dset in pd.output_datasets],
                )

            queued_sqns = []


if __name__ == "__main__":
    from utils import log_config

    log = log_config.log()
    log.info("DATA_WRITE BOOTED")
    try:
        main()
        log.info("DATA_WRITE EXITED")
    except Exception as main_exception:
        log.critical("DATA_WRITE CRASHED", error=main_exception)
        log.exception("DATA_WRITE CRASHED", exception=main_exception)
