#!/usr/bin/python3

"""
    data_write package
    ~~~~~~~~~~~~~~~~~~
    This package contains utilities to parse protobuf packets containing antennas_iq data, bfiq
    data, rawacf data, etc. and write that data to HDF5 or JSON files.

    :copyright: 2017 SuperDARN Canada
"""

import sys
import os
import datetime
import json
import collections
import warnings
import time
import threading
import errno
from multiprocessing import shared_memory
import subprocess as sp
import argparse as ap
import numpy as np
import deepdish as dd
import tables
import zmq
import faulthandler
from scipy.constants import speed_of_light
import copy
import pickle
import utils.options.data_write_options as dwo
from utils import socket_operations as so


DATA_TEMPLATE = {
    "borealis_git_hash"         : None, # Identifies the version of Borealis that made this data.
    "experiment_id"             : None, # Number used to identify experiment.
    "experiment_name"           : None, # Name of the experiment file.
    "experiment_comment"        : None, # Comment about the whole experiment
    "slice_comment"             : None, # Additional text comment that describes the slice.
    "slice_id"                  : None, # the slice id of the file and dataset.
    "slice_interfacing"         : None, # the interfacing of this slice to other slices.
    "num_slices"                : None, # Number of slices in the experiment at this integration
                                        # time.
    "station"                   : None, # Three letter radar identifier.
    "num_sequences"             : None, # Number of sampling periods in the integration time.
    "num_ranges"                : None, # Number of ranges to calculate correlations for
    "range_sep"                 : None, # range gate separation (equivalent distance between
                                        # samples) in km.
    "first_range_rtt"           : None, # Round trip time of flight to first range in microseconds.
    "first_range"               : None, # Distance to first range in km.
    "rx_sample_rate"            : None, # Sampling rate of the samples being written to file in Hz.
    "scan_start_marker"         : None, # Designates if the record is the first in a scan.
    "int_time"                  : None, # Integration time in seconds.
    "tx_pulse_len"              : None, # Length of the pulse in microseconds.
    "tau_spacing"               : None, # The minimum spacing between pulses in microseconds.
                                        # Spacing between pulses is always a multiple of this.
    "main_antenna_count"        : None, # Number of main array antennas.
    "intf_antenna_count"        : None, # Number of interferometer array antennas.
    "freq"                      : None, # The frequency used for this experiment slice in kHz.
    # "filtered_3db_bandwidth"    : None, # Bandwidth of the output iq data types? can add later
    "rx_center_freq"            : None, # the center frequency of this data (for rawrf), kHz
    "samples_data_type"         : None, # C data type of the samples such as complex float.
    "pulses"                    : None, # The pulse sequence in units of the tau_spacing.
    "pulse_phase_offset"        : None, # For pulse encoding phase. Contains an encoding per pulse.
                                        # Each encoding can either be a single value or one value
                                        # for each sample.
    "lags"                      : None, # The lags created from two pulses in the pulses array.
    "blanked_samples"           : None, # Samples that have been blanked because they occurred
                                        # during transmission times. Can differ from the pulses
                                        # array due to multiple slices in a single sequence.
    "sqn_timestamps"            : None, # A list of GPS timestamps of the beginning of transmission
                                        # for each sampling period in the integration time. Seconds
                                        # since epoch.
    "beam_nums"                 : None, # A list of beam numbers used in this slice.
    "beam_azms"                 : None, # A list of the beams azimuths for each beam in degrees off
                                        # boresite.
    "noise_at_freq"             : None, # Noise at the receive frequency, should be an array
                                        # (one value per sequence) (TODO units??)
    # (TODO document FFT resolution bandwidth for this value, should be = output_sample rate?)
    # "noise_in_raw_band"         : None, # Average noise in the sampling band (input sample rate) (TODO units??)
    # "rx_bandwidth"              : None, # if the noise_in_raw_band is provided, the rx_bandwidth should be provided!
    "num_samps"                 : None, # Number of samples in the sampling period.
    "antenna_arrays_order"      : None, # States what order the data is in. Describes the data
                                        # layout.
    "data_descriptors"          : None, # Denotes what each data dimension represents.
    "data_dimensions"           : None, # The dimensions in which to reshape the data.
    "data_normalization_factor" : None, # The scale of all of the filters, multiplied, for a total
                                        # scaling factor to normalize by.
    "data"                      : [],   # A contiguous set of samples (complex float) at given
                                        # sample rate
    "correlation_descriptors"   : None, # Denotes what each acf/xcf dimension represents.
    "correlation_dimensions"    : None, # The dimensions in which to reshape the acf/xcf data.
    "averaging_method"          : None, # A string describing the averaging method, ex. mean, median
    "scheduling_mode"           : None, # A string describing the type of scheduling time at the
                                        # time of this dataset.
    "main_acfs"                 : [],   # Main array autocorrelations
    "intf_acfs"                 : [],   # Interferometer array autocorrelations
    "xcfs"                      : [],   # Crosscorrelations between main and interferometer arrays
    "gps_locked"                : None, # Boolean True if the GPS was locked during the entire
                                        # integration period
    "gps_to_system_time_diff"   : None, # Max time diff in seconds between GPS and system/NTP time
                                        # during the integration period.
    "agc_status_word"           : None, # 32 bits, a '1' in bit position corresponds to an AGC
                                        # fault on that transmitter
    "lp_status_word"            : None  # 32 bits, a '1' in bit position corresponds to a low power
                                        # condition on that transmitter
}

TX_TEMPLATE = {
    "tx_rate"               : [],
    "tx_center_freq"        : [],
    "pulse_timing_us"       : [],
    "pulse_sample_start"    : [],
    "tx_samples"            : [],
    "dm_rate"               : [],
    "decimated_tx_samples"  : [],
}


class ParseData(object):
    """
    Parse message data from sockets into file writable types, such as hdf5, json, dmap, etc.

    :param  nested_dict:    alias to a nested defaultdict
    :type   nested_dict:    dict
    :param  processed_data: Contains a message from dsp socket.
    :type   processed_data: ProcessedSequenceMessage
    """

    def __init__(self, data_write_options):
        super(ParseData, self).__init__()

        self.options = data_write_options

        # defaultdict will populate non-specified entries in the dictionary with the default
        # value given as an argument, in this case a dictionary. Nesting it in a lambda lets you
        # create arbitrarily deep dictionaries.
        self.nested_dict = lambda: collections.defaultdict(self.nested_dict)

        self.processed_data = None

        self._rx_rate = 0.0
        self._output_sample_rate = 0.0

        self._bfiq_available = False
        self._bfiq_accumulator = self.nested_dict()

        self._antenna_iq_accumulator = self.nested_dict()
        self._antenna_iq_available = False

        self._mainacfs_available = False
        self._mainacfs_accumulator = self.nested_dict()

        self._xcfs_available = False
        self._xcfs_accumulator = self.nested_dict()

        self._intfacfs_available = False
        self._intfacfs_accumulator = self.nested_dict()

        self._slice_ids = set()
        self._timestamps = []

        self._gps_locked = True  # init True so that logical AND works properly in update() method
        self._gps_to_system_time_diff = 0.0

        self._agc_status_word = 0b0
        self._lp_status_word = 0b0

        self._rawrf_locations = []
        self._rawrf_num_samps = 0
        self._raw_rf_available = False

    def parse_correlations(self):
        """
        Parses out the possible correlation data from the message. Runs on every new
        ProcessedSequenceMessage (contains all sampling period data). The expectation value is
        calculated at the end of a sampling period by a different function.
        """

        for data_set in self.processed_data.output_datasets:
            slice_id = data_set.slice_id

            data_shape = (data_set.num_beams, data_set.num_ranges, data_set.num_lags)

            def accumulate_data(holder, message_data):
                """
                Opens a numpy array from shared memory into the 'holder' accumulator.

                :param  holder:         accumulator to hold data
                :type   holder:         dict
                :param  message_data:   unique message field for parsing
                :type   message_data:   str
                """

                # Open the shared memory
                shm = shared_memory.SharedMemory(name=message_data)
                acf_data = np.ndarray(data_shape, dtype=np.complex64, buffer=shm.buf)

                # Put the data in the accumulator
                if 'data' not in holder[slice_id]:
                    holder[slice_id]['data'] = []
                holder[slice_id]['data'].append(acf_data.copy())
                shm.close()
                shm.unlink()

            if data_set.main_acf_shm:
                self._mainacfs_available = True
                accumulate_data(self._mainacfs_accumulator, data_set.main_acf_shm)

            if data_set.xcf_shm:
                self._xcfs_available = True
                accumulate_data(self._xcfs_accumulator, data_set.xcf_shm)

            if data_set.intf_acf_shm:
                self._intfacfs_available = True
                accumulate_data(self._intfacfs_accumulator, data_set.intf_acf_shm)

    def parse_bfiq(self):
        """
        Parses out any possible beamformed IQ data from the message. Runs on every
        ProcessedSequenceMessage (contains all sampling period data). All variables are captured
        from outer scope.
        """

        self._bfiq_accumulator['data_descriptors'] = ['num_antenna_arrays', 'num_sequences',
                                                      'num_beams', 'num_samps']

        num_slices = len(self.processed_data.output_datasets)
        max_num_beams = self.processed_data.max_num_beams
        num_samps = self.processed_data.num_samps

        main_shm = shared_memory.SharedMemory(name=self.processed_data.bfiq_main_shm)
        temp_data = np.ndarray((num_slices, max_num_beams, num_samps),
                               dtype=np.complex64,
                               buffer=main_shm.buf)
        main_data = temp_data.copy()
        main_shm.close()
        main_shm.unlink()

        intf_available = False
        if self.processed_data.bfiq_intf_shm != '':
            intf_available = True
            intf_shm = shared_memory.SharedMemory(name=self.processed_data.bfiq_intf_shm)
            temp_data = np.ndarray((num_slices, max_num_beams, num_samps),
                                   dtype=np.complex64,
                                   buffer=intf_shm.buf)
            intf_data = temp_data.copy()
            intf_shm.close()
            intf_shm.unlink()

        self._bfiq_available = True

        for i, data_set in enumerate(self.processed_data.output_datasets):
            slice_id = data_set.slice_id
            num_beams = data_set.num_beams

            self._bfiq_accumulator[slice_id]['num_samps'] = num_samps

            if 'main_data' not in self._bfiq_accumulator[slice_id]:
                self._bfiq_accumulator[slice_id]['main_data'] = []
            self._bfiq_accumulator[slice_id]['main_data'].append(main_data[i, :num_beams, :])

            if intf_available:
                if 'intf_data' not in self._bfiq_accumulator[slice_id]:
                    self._bfiq_accumulator[slice_id]['intf_data'] = []
                self._bfiq_accumulator[slice_id]['intf_data'].append(intf_data[i, :num_beams, :])

    def parse_antenna_iq(self):
        """
        Parses out any pre-beamformed IQ if available. Runs on every ProcessedSequenceMessage
        (contains all sampling period data). All variables are captured from outer scope.
        """

        self._antenna_iq_accumulator['data_descriptors'] = ['num_antennas', 'num_sequences', 'num_samps']

        # Get data dimensions for reading in the shared memory
        num_slices = len(self.processed_data.output_datasets)
        num_main_antennas = len(self.options.main_antennas)
        num_intf_antennas = len(self.options.intf_antennas)

        stages = []
        # Loop through all the filter stage data
        for debug_stage in self.processed_data.debug_data:
            stage_samps = debug_stage.num_samps
            stage_main_shm = shared_memory.SharedMemory(name=debug_stage.main_shm)
            stage_main_data = np.ndarray((num_slices, num_main_antennas, stage_samps),
                                         dtype=np.complex64,
                                         buffer=stage_main_shm.buf)
            stage_data = stage_main_data.copy()  # Move data out of shared memory so we can close it
            stage_main_shm.close()
            stage_main_shm.unlink()

            if debug_stage.intf_shm:
                stage_intf_shm = shared_memory.SharedMemory(name=debug_stage.intf_shm)
                stage_intf_data = np.ndarray((num_slices, num_intf_antennas, stage_samps),
                                             dtype=np.complex64,
                                             buffer=stage_intf_shm.buf)
                stage_data = np.hstack((stage_data, stage_intf_data.copy()))
                stage_intf_shm.close()
                stage_intf_shm.unlink()

            stage_dict = {'stage_name': debug_stage.stage_name,
                          'stage_samps': debug_stage.num_samps,
                          'main_shm': debug_stage.main_shm,
                          'intf_shm': debug_stage.intf_shm,
                          'data': stage_data}
            stages.append(stage_dict)

        self._antenna_iq_available = True

        # Iterate over every data set, one data set per slice
        for i, data_set in enumerate(self.processed_data.output_datasets):
            slice_id = data_set.slice_id

            # non beamformed IQ samples are available
            for debug_stage in stages:
                stage_name = debug_stage['stage_name']

                if stage_name not in self._antenna_iq_accumulator[slice_id]:
                    self._antenna_iq_accumulator[slice_id][stage_name] = collections.OrderedDict()

                antenna_iq_stage = self._antenna_iq_accumulator[slice_id][stage_name]

                antennas_data = debug_stage['data'][i]
                antenna_iq_stage["num_samps"] = antennas_data.shape[-1]

                # Loops over antenna data within stage
                for ant_num in range(antennas_data.shape[0]):
                    ant_str = f"antenna_{ant_num}"

                    if ant_str not in antenna_iq_stage:
                        antenna_iq_stage[ant_str] = {}

                    if 'data' not in antenna_iq_stage[ant_str]:
                        antenna_iq_stage[ant_str]['data'] = []
                    antenna_iq_stage[ant_str]['data'].append(antennas_data[ant_num, :])

    def numpify_arrays(self):
        """
        Consolidates data for each data type to one array.

        In parse_[type](), new data arrays are appended to a list for speed considerations.
        This function converts these lists into numpy arrays.
        """
        for slice_id, slice_data in self._antenna_iq_accumulator.items():
            if isinstance(slice_id, int):  # filtering out 'data_descriptors'
                for param_data in slice_data.values():
                    for array_name, array_data in param_data.items():
                        if array_name != 'num_samps':
                            array_data['data'] = np.array(array_data['data'], dtype=np.complex64)

        for slice_id, slice_data in self._bfiq_accumulator.items():
            if isinstance(slice_id, int):  # filtering out 'data_descriptors'
                for param_name, param_data in slice_data.items():
                    slice_data[param_name] = np.array(param_data, dtype=np.complex64)

        for slice_data in self._mainacfs_accumulator.values():
            slice_data['data'] = np.array(slice_data['data'], np.complex64)

        for slice_data in self._intfacfs_accumulator.values():
            slice_data['data'] = np.array(slice_data['data'], np.complex64)

        for slice_data in self._xcfs_accumulator.values():
            slice_data['data'] = np.array(slice_data['data'], np.complex64)

    def update(self, data):
        """
        Parses the message and updates the accumulator fields with the new data.

        :param  data: Processed sequence metadata.
        :type   data: ProcessedSequenceMessage
        """

        self.processed_data = data
        self._timestamps.append(data.sequence_start_time)

        self._rx_rate = data.rx_sample_rate
        self._output_sample_rate = data.output_sample_rate

        for data_set in data.output_datasets:
            self._slice_ids.add(data_set.slice_id)

        if data.rawrf_shm != '':
            self._raw_rf_available = True
            self._rawrf_num_samps = data.rawrf_num_samps
            self._rawrf_locations.append(data.rawrf_shm)

        # Logical AND to catch any time the GPS may have been unlocked during the integration period
        self._gps_locked = self._gps_locked and data.gps_locked

        # Find the max time diff between GPS and system time to report for this integration period
        if abs(self._gps_to_system_time_diff) < abs(data.gps_to_system_time_diff):
            self._gps_to_system_time_diff = data.gps_to_system_time_diff

        # Bitwise OR to catch any AGC faults during the integration period
        self._agc_status_word = self._agc_status_word | data.agc_status_bank_h

        # Bitwise OR to catch any low power conditions during the integration period
        self._lp_status_word = self._lp_status_word | data.lp_status_bank_h

        # TODO(keith): Parallelize?
        procs = []

        self.parse_correlations()
        self.parse_bfiq()
        self.parse_antenna_iq()

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

    @property
    def sequence_num(self):
        """
        Gets the sequence num of the latest processeddata packet.

        :returns:   sequence number
        :rtype:     int
        """

        return self.processed_data.sequence_num

    @property
    def bfiq_available(self):
        """
        Gets the bfiq available flag.

        :returns:   bfiq available flag
        :rtype:     bool
        """

        return self._bfiq_available

    @property
    def antenna_iq_available(self):
        """
        Gets the pre-bfiq available flag.

        :returns:   pre-bfiq available flag
        :rtype:     bool
        """

        return self._antenna_iq_available

    @property
    def mainacfs_available(self):
        """
        Gets the mainacfs available flag.

        :returns:   mainacfs available flag
        :rtype:     bool
        """

        return self._mainacfs_available

    @property
    def xcfs_available(self):
        """
        Gets the xcfs available flag.

        :returns:   xcfs available flag
        :rtype:     bool
        """

        return self._xcfs_available

    @property
    def intfacfs_available(self):
        """
        Gets the intfacfs available flag.

        :returns:   intfacfs available flag
        :rtype:     bool
        """

        return self._intfacfs_available

    @property
    def bfiq_accumulator(self):
        """
        Returns the nested default dictionary with complex stage data for each antenna array as well
        as some metadata.

        :returns:   bfiq_accumulator containing beamform data for each slice
        :rtype:     dict
        """

        return self._bfiq_accumulator

    @property
    def antenna_iq_accumulator(self):
        """
        Returns the nested default dictionary with complex stage data for each antenna as well
        as some metadata for each slice.

        :returns:   antenna_iq_accumulator containing data for each antenna and slice
        :rtype:     dict
        """

        return self._antenna_iq_accumulator

    @property
    def mainacfs_accumulator(self):
        """
        Returns the default dict containing a list of main acf data for each slice. There is an
        array of data for each sampling period.

        :returns:   mainacfs_accumulator containing main acf data for each slice
        :rtype:     dict
        """

        return self._mainacfs_accumulator

    @property
    def xcfs_accumulator(self):
        """
        Returns the default dict containing a list of xcf data for each slice. There is an
        array of data for each sampling period.

        :returns:   xcfs_accumulator containing xcf data for each slice
        :rtype:     dict
        """

        return self._xcfs_accumulator

    @property
    def intfacfs_accumulator(self):
        """
        Returns the default dict containing a list of intf acf data for each slice. There is an
        array of data for each sampling period.

        :returns:   intfacfs_accumulator containing intf acf data for each slice
        :rtype:     dict
        """

        return self._intfacfs_accumulator

    @property
    def timestamps(self):
        """
        Return the python list of sequence timestamps (when the sampling period begins)
        from the processsed data packets

        :returns:   sequence timestamps from the processed data packets
        :rtype:     list
        """

        return self._timestamps

    @property
    def rx_rate(self):
        """
        Return the rx_rate of the data in the data packet

        :returns:   sampling rate in Hz
        :rtype:     float
        """

        return self._rx_rate

    @property
    def output_sample_rate(self):
        """
        Return the output rate of the filtered, decimated data in the data packet.

        :returns:   output sampling rate in Hz
        :rtype:     float
        """

        return self._output_sample_rate

    @property
    def slice_ids(self):
        """
        Return the slice ids in python set so they are guaranteed unique

        :returns:   slice id numbers
        :rtype:     set
        """

        return self._slice_ids

    @property
    def raw_rf_available(self):
        """
        Gets the raw_rf available flag.

        :returns:   raw_rf available flag
        :rtype:     bool
        """

        return self._raw_rf_available

    @property
    def rawrf_locations(self):
        """
        Gets the list of raw rf memory locations.

        :returns:   raw rf memory locations
        :rtype:     list of strings
        """

        return self._rawrf_locations

    @property
    def rawrf_num_samps(self):
        """
        Gets the number of rawrf samples per antenna.

        :returns:   number of rawrf samples per antenna
        :rtype:     int
        """

        return self._rawrf_num_samps

    @property
    def gps_locked(self):
        """
        Return the boolean value indicating if the GPS was locked during the entire int period

        :returns:   gps_locked flag
        :rtype:     bool
        """

        return self._gps_locked

    @property
    def gps_to_system_time_diff(self):
        """
        Gets the maximum time diff in seconds between the GPS (box_time) and system (NTP) during
        the integration period. Negative if GPS time is ahead of system/NTP time.

        :returns:   gps to system time diff
        :rtype:     double
        """

        return self._gps_to_system_time_diff

    @property
    def agc_status_word(self):
        """
        AGC Status, a '1' in bit position corresponds to an AGC fault on that transmitter

        :returns:   agc_status_word
        :rtype:     int
        """

        return self._agc_status_word

    @property
    def lp_status_word(self):
        """
        Low Power, a '1' in bit position corresponds to a low power condition on that transmitter

        :returns:   lp_status_word
        :rtype:     int
        """

        return self._lp_status_word


class DataWrite(object):
    """
    This class contains the functions used to write out processed data to files.

    :param  data_write_options: The data write options from config file
    :type   data_write_options: DataWriteOptions
    """

    def __init__(self, data_write_options):
        super(DataWrite, self).__init__()

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

    def write_json_file(self, filename, data_dict):
        """
        Write out data to a json file. If the file already exists it will be overwritten.

        :param  filename:   The path to the file to write out
        :type   filename:   str
        :param  data_dict:  Python dictionary to write out to the JSON file.
        :type   data_dict:  dict
        """

        with open(filename, 'w+') as f:
            f.write(json.dumps(data_dict))

    def write_hdf5_file(self, filename, data_dict, dt_str):
        """
        Write out data to an HDF5 file. If the file already exists it will be overwritten.

        :param  filename:   The path to the file to write out
        :type   filename:   str
        :param  data_dict:  Python dictionary to write out to the HDF5 file.
        :type   data_dict:  dict
        :param  dt_str:     A datetime timestamp of the first transmission time in the record
        :type   dt_str:     str
        """

        def convert_to_numpy(dd):
            """
            Converts lists stored in dict into numpy array. Recursive.

            :param  dd: Dictionary with lists to convert to numpy arrays.
            :type   dd: dict
            """
            for k, v in dd.items():
                if isinstance(v, dict):
                    convert_to_numpy(v)
                elif isinstance(v, list):
                    dd[k] = np.array(v)
                else:
                    continue

        convert_to_numpy(data_dict)

        time_stamped_dd = {}
        time_stamped_dd[dt_str] = data_dict

        # Ignoring warning that arises from using integers as the keys of the data dictionary.
        warnings.simplefilter('ignore', tables.NaturalNameWarning)

        try:
            dd.io.save(filename, time_stamped_dd, compression=None)
        except Exception as e:
            if "No space left on device" in str(e):
                log.critical("no space left on device", error=e)
                log.exception("no space left on device", exception=e)
                sys.exit(-1)
            else:
                log.critical("unknown error when saving to file", error=e)
                log.exception("unknown error when saving to file", exception=e)
                sys.exit(-1)

    def write_dmap_file(self, filename, data_dict):
        """
        Write out data to a dmap file. If the file already exists it will be overwritten.

        :param  filename:   The path to the file to write out
        :type   filename:   str
        :param  data_dict:  Python dictionary to write out to the dmap file.
        :type   data_dict:  dict
        """

        # TODO: Complete this by parsing through the dictionary and write out to proper dmap format

        raise NotImplementedError

    def output_data(self, write_bfiq, write_antenna_iq, write_raw_rf, write_tx, file_ext,
                    aveperiod_meta, data_parsing, rt_dw, write_rawacf=True):
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
        :param  file_ext:           Type of file extention to use
        :type   file_ext:           str
        :param  aveperiod_meta:     Metadata from radar control about averaging period
        :type   aveperiod_meta:     AveperiodMetadataMessage
        :param  data_parsing:       All parsed and concatenated data from averaging period
        :type   data_parsing:       ParseData
        :param  rt_dw:              Pair of socket and iden for RT purposes.
        :type   rt_dw:              dict
        :param  write_rawacf:       Should rawacfs be written to file? Defaults to True.
        :type   write_rawacf:       bool, optional
        """

        start = time.perf_counter()
        try:
            assert file_ext in ['hdf5', 'json', 'dmap']
        except Exception as e:
            log.error("wrong file format [hdf5, json, dmap]", error=e)
            log.exception("wrong file format [hdf5, json, dmap]", exception=e)
            sys.exit(1)

        # Format the name and location for the dataset
        time_now = datetime.datetime.utcfromtimestamp(data_parsing.timestamps[0])

        today_string = time_now.strftime("%Y%m%d")
        datetime_string = time_now.strftime("%Y%m%d.%H%M.%S.%f")
        epoch = datetime.datetime.utcfromtimestamp(0)
        epoch_milliseconds = str(int((time_now - epoch).total_seconds() * 1000))
        dataset_directory = f"{self.options.data_directory}/{today_string}"
        dataset_name = f"{datetime_string}.{self.options.site_id}.{{sliceid}}.{{dformat}}.{file_ext}"
        dataset_location = f"{dataset_directory}/{{name}}"

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

            while time_now > boundary_time:
                boundary_time += datetime.timedelta(hours=2)

            return boundary_time

        if self.first_time:
            self.raw_rf_two_hr_name = self.raw_rf_two_hr_format.format(
                dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                site=self.options.site_id)
            self.tx_data_two_hr_name = self.tx_data_two_hr_format.format(
                dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                site=self.options.site_id)
            self.next_boundary = two_hr_ceiling(time_now)
            self.first_time = False

        for slice_id in data_parsing.slice_ids:
            if slice_id not in self.slice_filenames:
                two_hr_str = self.two_hr_format.format(dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                                                       sliceid=slice_id,
                                                       site=self.options.site_id)
                self.slice_filenames[slice_id] = two_hr_str

        if time_now > self.next_boundary:
            self.raw_rf_two_hr_name = self.raw_rf_two_hr_format.format(
                dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                site=self.options.site_id)
            self.tx_data_two_hr_name = self.tx_data_two_hr_format.format(
                dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                site=self.options.site_id)
            for slice_id in self.slice_filenames.keys():
                two_hr_str = self.two_hr_format.format(dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                                                       sliceid=slice_id,
                                                       site=self.options.site_id)
                self.slice_filenames[slice_id] = two_hr_str

            self.next_boundary = two_hr_ceiling(time_now)

        def write_file(tmp_file, final_data_dict, two_hr_file_with_type):
            """
            Writes the final data out to the location based on the type of file extension required

            :param  tmp_file:               File path and name to write single record
            :type   tmp_file:               str
            :param  final_data_dict:        Data dict parsed out from message
            :type   final_data_dict:        dict
            :param  two_hr_file_with_type:  Name of the two hour file with data type added
            :type   two_hr_file_with_type:  str
            """

            try:
                os.makedirs(dataset_directory, exist_ok=True)
            except OSError as e:
                if e.args[0] == errno.ENOSPC:
                    log.critical("no space left on device", error=e)
                    log.exception("no space left on device", exception=e)
                    sys.exit(-1)
                else:
                    log.critical("unknown error when making dirs", error=e)
                    log.exception("unknown error when making dirs", exception=e)
                    sys.exit(-1)

            if file_ext == 'hdf5':
                full_two_hr_file = f"{dataset_directory}/{two_hr_file_with_type}.hdf5.site"

                try:
                    fd = os.open(full_two_hr_file, os.O_CREAT)
                    os.close(fd)
                except OSError as e:
                    if e.args[0] == errno.ENOSPC:
                        log.critical("no space left on device", error=e)
                        log.exception("no space left on device", exception=e)
                        sys.exit(-1)
                    else:
                        log.critical("unknown error when opening file", error=e)
                        log.exception("unknown error when opening file", exception=e)
                        sys.exit(-1)

                self.write_hdf5_file(tmp_file, final_data_dict, epoch_milliseconds)

                # Use external h5copy utility to move new record into 2hr file.
                cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
                cmd = cmd.format(newfile=tmp_file, twohr=full_two_hr_file, dtstr=epoch_milliseconds)

                # TODO(keith): improve call to subprocess.
                sp.call(cmd.split())
                so.send_data(rt_dw['socket'], rt_dw['iden'], tmp_file)
                # Temp file is removed in real time module.

            elif file_ext == 'json':
                self.write_json_file(tmp_file, final_data_dict)
            elif file_ext == 'dmap':
                self.write_dmap_file(tmp_file, final_data_dict)

        def write_correlations(parameters_holder):
            """
            Parses out any possible correlation data from message and writes to file. Some variables
            are captured from outer scope.

            main_acfs, intf_acfs, and xcfs are all passed to data_write for all sequences
            individually. At this point, they will be combined into data for a single integration
            time via averaging.

            :param  parameters_holder:  A dict that hold dicts of parameters for each slice.
            :type   parameters_holder:  dict
            """

            needed_fields = [
            "borealis_git_hash",        "experiment_id",        "experiment_name",
            "experiment_comment",       "num_slices",           "slice_comment",
            "station",                  "num_sequences",        "range_sep",
            "first_range_rtt",          "first_range",          "rx_sample_rate",
            "scan_start_marker",        "int_time",             "tx_pulse_len",
            "tau_spacing",              "main_antenna_count",   "intf_antenna_count",
            "freq",                     "samples_data_type",    "pulses",
            "lags",                     "blanked_samples",      "sqn_timestamps",
            "beam_nums",                "beam_azms",            "correlation_descriptors",
            "correlation_dimensions",   "main_acfs",            "intf_acfs",
            "xcfs",                     "noise_at_freq",        "data_normalization_factor",
            "slice_id",                 "slice_interfacing",    "averaging_method",
            "scheduling_mode",          "gps_locked",           "gps_to_system_time_diff",
            "agc_status_word",          "lp_status_word"
            ]
            # Note: num_ranges not in needed_fields but is used to make correlation_dimensions

            main_acfs = data_parsing.mainacfs_accumulator
            xcfs = data_parsing.xcfs_accumulator
            intf_acfs = data_parsing.intfacfs_accumulator

            def find_expectation_value(x, parameters, field_name):
                """
                Get the mean or median of all correlations from all sequences in the integration
                period - only this will be recorded.
                This is effectively 'averaging' all correlations over the integration time, using a
                specified method for combining them.
                """

                # array_2d is num_sequences x (num_beams*num_ranges*num_lags)
                # so we get median of all sequences.
                averaging_method = parameters['averaging_method']
                array_2d = np.array(x, dtype=np.complex64)
                num_beams, num_ranges, num_lags = np.array([len(parameters["beam_nums"]),
                                                            parameters["num_ranges"],
                                                            parameters["lags"].shape[0]],
                                                           dtype=np.uint32)

                # First range offset in samples
                sample_off = parameters['first_range_rtt'] * 1e-6 * parameters['rx_sample_rate']
                sample_off = np.uint32(sample_off)

                # Find sample number which corresponds with second pulse in sequence
                tau_in_samples = parameters['tau_spacing'] * 1e-6 * parameters['rx_sample_rate']
                second_pulse_sample_num = np.uint32(tau_in_samples) * parameters['pulses'][1] - sample_off - 1

                # Average the data
                try:
                    assert averaging_method in ['mean', 'median']
                    if averaging_method == 'mean':
                        array_expectation_value = np.mean(array_2d, axis=0)
                    elif averaging_method == 'median':
                        array_expectation_value = np.median(np.real(array_2d), axis=0) + \
                                                  1j * np.median(np.imag(array_2d), axis=0)
                except Exception as e:
                    log.error("wrong averaging method [mean, median]", error=e)
                    log.exception("wrong averaging method [mean, median]", exception=e)
                    sys.exit(1)


                # Reshape array to be 3d so we can replace lag0 far ranges that are cluttered with those
                # from alternate lag0 which have no clutter.
                array_3d = array_expectation_value.reshape((num_beams, num_ranges, num_lags))
                array_3d[:, second_pulse_sample_num:, 0] = array_3d[:, second_pulse_sample_num:, -1]

                # Flatten back to a list
                parameters[field_name] = array_3d.flatten()

            for slice_id in main_acfs:
                parameters = parameters_holder[slice_id]
                find_expectation_value(main_acfs[slice_id]['data'], parameters, 'main_acfs')

            for slice_id in xcfs:
                parameters = parameters_holder[slice_id]
                find_expectation_value(xcfs[slice_id]['data'], parameters, 'xcfs')

            for slice_id in intf_acfs:
                parameters = parameters_holder[slice_id]
                find_expectation_value(intf_acfs[slice_id]['data'], parameters, 'intf_acfs')

            for slice_id, parameters in parameters_holder.items():
                parameters['correlation_descriptors'] = ['num_beams', 'num_ranges', 'num_lags']
                parameters['correlation_dimensions'] = np.array([len(parameters["beam_nums"]),
                                                                 parameters["num_ranges"], parameters["lags"].shape[0]],
                                                                dtype=np.uint32)
                for field in list(parameters.keys()):
                    if field not in needed_fields:
                        parameters.pop(field, None)

                name = dataset_name.format(sliceid=slice_id, dformat="rawacf")
                output_file = dataset_location.format(name=name)

                two_hr_file_with_type = self.slice_filenames[slice_id].format(ext="rawacf")

                write_file(output_file, parameters, two_hr_file_with_type)

        def write_bfiq_params(parameters_holder):
            """
            write out any possible beamformed IQ data that has been parsed. Adds additional slice
            info to each parameter dict. Some variables are captured from outer scope.

            :param  parameters_holder:  A dict that hold dicts of parameters for each slice.
            :type   parameters_holder:  dict
            """

            needed_fields = [
            "borealis_git_hash",        "experiment_id",                "experiment_name",
            "experiment_comment",       "num_slices",                   "slice_comment",
            "station",                  "num_sequences",                "rx_sample_rate",
            "pulse_phase_offset",       "scan_start_marker",            "int_time",
            "tx_pulse_len",             "tau_spacing",                  "main_antenna_count",
            "intf_antenna_count",       "freq",                         "samples_data_type",
            "pulses",                   "blanked_samples",              "sqn_timestamps",
            "beam_nums",                "beam_azms",                    "data_dimensions",
            "data_descriptors",         "antenna_arrays_order",         "data",
            "num_samps",                "noise_at_freq",                "range_sep",
            "first_range_rtt",          "first_range",                  "lags",
            "num_ranges",               "data_normalization_factor",    "slice_id",
            "slice_interfacing",        "scheduling_mode",              "gps_locked",
            "gps_to_system_time_diff",  "agc_status_word", "            lp_status_word"
            ]

            bfiq = data_parsing.bfiq_accumulator
            slice_id_list = [x for x in bfiq.keys() if isinstance(x, int)]

            # Pop these off so we dont include them in later iteration.
            data_descriptors = bfiq.pop('data_descriptors', None)

            for slice_id in slice_id_list:
                parameters = parameters_holder[slice_id]
                parameters['data_descriptors'] = data_descriptors
                parameters['antenna_arrays_order'] = []

                flattened_data = []
                num_antenna_arrays = 1
                parameters['antenna_arrays_order'].append("main")
                flattened_data.append(bfiq[slice_id]['main_data'].flatten())
                if "intf" in bfiq[slice_id]:
                    num_antenna_arrays += 1
                    parameters['antenna_arrays_order'].append("intf")
                    flattened_data.append(bfiq[slice_id]['intf_data'].flatten())

                flattened_data = np.concatenate(flattened_data)
                parameters['data'] = flattened_data

                parameters['num_samps'] = np.uint32(bfiq[slice_id]['num_samps'])
                parameters['data_dimensions'] = np.array([num_antenna_arrays,
                                                          aveperiod_meta.num_sequences,
                                                          len(parameters['beam_nums']),
                                                          parameters['num_samps']], dtype=np.uint32)

                for field in list(parameters.keys()):
                    if field not in needed_fields:
                        parameters.pop(field, None)

            for slice_id, parameters in parameters_holder.items():
                name = dataset_name.format(sliceid=slice_id, dformat="bfiq")
                output_file = dataset_location.format(name=name)

                two_hr_file_with_type = self.slice_filenames[slice_id].format(ext="bfiq")

                write_file(output_file, parameters, two_hr_file_with_type)

        def write_antenna_iq_params(parameters_holder):
            """
            Writes out any pre-beamformed IQ that has been parsed. Adds additional slice info
            to each paramater dict. Some variables are captured from outer scope. Pre-beamformed iq
            is the individual antenna received data. Antenna_arrays_order will list the antennas' order.

            :param  parameters_holder:  A dict that hold dicts of parameters for each slice.
            :type   parameters_holder:  dict
            """

            needed_fields = [
            "borealis_git_hash",    "experiment_id",                "experiment_name",
            "experiment_comment",   "num_slices",                   "slice_comment",
            "station",              "num_sequences",                "rx_sample_rate",
            "scan_start_marker",    "int_time",                     "tx_pulse_len",
            "tau_spacing",          "main_antenna_count",           "intf_antenna_count",
            "freq",                 "samples_data_type",            "pulses",
            "sqn_timestamps",       "beam_nums",                    "beam_azms",
            "data_dimensions",      "data_descriptors",             "antenna_arrays_order",
            "data",                 "num_samps",                    "pulse_phase_offset",
            "noise_at_freq",        "data_normalization_factor",    "blanked_samples",
            "slice_id",             "slice_interfacing",            "scheduling_mode",
            "gps_locked",           "gps_to_system_time_diff",      "agc_status_word",
            "lp_status_word"
            ]

            antenna_iq = data_parsing.antenna_iq_accumulator
            slice_id_list = [x for x in antenna_iq.keys() if isinstance(x, int)]

            # Pop these so we don't include them in later iteration.
            data_descriptors = antenna_iq.pop('data_descriptors', None)

            # Parse the antennas from message
            rx_main_antennas = {}
            rx_intf_antennas = {}

            for meta in aveperiod_meta.sequences:
                for rx_freq in meta.rx_channels:
                    rx_main_antennas[rx_freq.slice_id] = list(rx_freq.rx_main_antennas)
                    rx_intf_antennas[rx_freq.slice_id] = list(rx_freq.rx_intf_antennas)

            # Build strings from antennas used in the message. This will be used to know
            # what antennas were recorded on since we sample all available USRP channels
            # and some channels may not be transmitted on, or connected.
            main_ant_str = lambda x: f"antenna_{x}"
            intf_ant_str = lambda x: f"antenna_{x + self.options.main_antenna_count}"
            for slice_id in rx_main_antennas:
                rx_main_antennas[slice_id] = [main_ant_str(x) for x in rx_main_antennas[slice_id]]
                rx_intf_antennas[slice_id] = [intf_ant_str(x) for x in rx_intf_antennas[slice_id]]

            final_data_params = {}
            for slice_id in slice_id_list:
                final_data_params[slice_id] = {}

                for stage in antenna_iq[slice_id]:
                    parameters = parameters_holder[slice_id].copy()

                    parameters['data_descriptors'] = data_descriptors
                    parameters['num_samps'] = np.uint32(
                        antenna_iq[slice_id][stage].pop('num_samps', None))

                    parameters['antenna_arrays_order'] = rx_main_antennas[slice_id] + \
                                                         rx_intf_antennas[slice_id]

                    num_ants = len(parameters['antenna_arrays_order'])

                    parameters['data_dimensions'] = np.array([num_ants,
                                                              aveperiod_meta.num_sequences,
                                                              parameters['num_samps']],
                                                             dtype=np.uint32)

                    data = []
                    for k, data_dict in antenna_iq[slice_id][stage].items():
                        if k in parameters['antenna_arrays_order']:
                            data.append(data_dict['data'].flatten())

                    flattened_data = np.concatenate(data)
                    parameters['data'] = flattened_data

                    for field in list(parameters.keys()):
                        if field not in needed_fields:
                            parameters.pop(field, None)

                    final_data_params[slice_id][stage] = parameters

            for slice_id, slice_ in final_data_params.items():
                for stage, params in slice_.items():
                    name = dataset_name.format(sliceid=slice_id, dformat=f"{stage}_iq")
                    output_file = dataset_location.format(name=name)

                    ext = f"{stage}_iq"
                    two_hr_file_with_type = self.slice_filenames[slice_id].format(ext=ext)

                    write_file(output_file, params, two_hr_file_with_type)

        def write_raw_rf_params(param):
            """
            Opens the shared memory location in the message and writes the samples out to file.
            Write medium must be able to sustain high write bandwidth. Shared memory is destroyed
            after write. Some variables are captured in scope.

            :param  param:  A dict of parameters to write. Some will be removed.
            :type   param:  dict
            """

            needed_fields = [
            "borealis_git_hash",    "experiment_id",            "experiment_name",
            "experiment_comment",   "num_slices",               "station",
            "num_sequences",        "rx_sample_rate",           "scan_start_marker",
            "int_time",             "main_antenna_count",       "intf_antenna_count",
            "samples_data_type",    "sqn_timestamps",           "data_dimensions",
            "data_descriptors",     "data",                     "num_samps",
            "rx_center_freq",       "blanked_samples",          "scheduling_mode",
            "gps_locked",           "gps_to_system_time_diff",  "agc_status_word",
            "lp_status_word"
            ]

            # Some fields don't make much sense when working with the raw rf. It's expected that the
            # user will have knowledge of what they are looking for when working with this data.
            # Note that because this data is not slice-specific a lot of slice-specific data (ex.
            # pulses, beam_nums, beam_azms) is not included (user must look at the experiment they
            # ran)

            raw_rf = data_parsing.rawrf_locations
            num_rawrf_samps = data_parsing.rawrf_num_samps

            # Don't need slice id here
            name = dataset_name.replace('{sliceid}.', '').format(dformat='rawrf')
            output_file = dataset_location.format(name=name)

            samples_list = []
            shms = []
            total_ants = len(self.options.main_antennas) + len(self.options.intf_antennas)

            for raw in raw_rf:
                shm = shared_memory.SharedMemory(name=raw)
                rawrf_array = np.ndarray((total_ants, num_rawrf_samps), dtype=np.complex64, buffer=shm.buf)

                samples_list.append(rawrf_array.flatten())

                shms.append(shm)

            param['data'] = np.concatenate(samples_list)

            param['rx_sample_rate'] = np.float32(data_parsing.rx_rate)

            param['num_samps'] = np.uint32(len(samples_list[0]) / total_ants)

            param['data_descriptors'] = ["num_sequences", "num_antennas", "num_samps"]
            param['data_dimensions'] = np.array([param['num_sequences'], total_ants,
                                                 param['num_samps']],
                                                dtype=np.uint32)
            param['main_antenna_count'] = np.uint32(self.options.main_antenna_count)
            param['intf_antenna_count'] = np.uint32(self.options.intf_antenna_count)

            for field in list(param.keys()):
                if field not in needed_fields:
                    param.pop(field, None)

            write_file(output_file, param, self.raw_rf_two_hr_name)

            # Can only close mapped memory after its been written to disk.
            for shm in shms:
                shm.close()
                shm.unlink()

        def write_tx_data():
            """
            Writes out the tx samples and metadata for debugging purposes.
            Does not use same parameters of other writes.
            """
            tx_data = None
            for meta in aveperiod_meta.sequences:
                if meta.tx_data is not None:
                    tx_data = copy.deepcopy(TX_TEMPLATE)
                    break

            if tx_data is not None:
                for meta in aveperiod_meta.sequences:
                    meta_data = meta.tx_data
                    tx_data['tx_rate'].append(meta_data.tx_rate)
                    tx_data['tx_center_freq'].append(meta_data.tx_ctr_freq)
                    tx_data['pulse_timing_us'].append(meta_data.pulse_timing_us)
                    tx_data['pulse_sample_start'].append(meta_data.pulse_sample_start)
                    tx_data['dm_rate'].append(meta_data.dm_rate)
                    tx_data['tx_samples'].append(meta_data.tx_samples)
                    tx_data['decimated_tx_samples'].append(meta_data.decimated_tx_samples)

                name = dataset_name.replace('{sliceid}.', '').format(dformat='txdata')
                output_file = dataset_location.format(name=name)

                write_file(output_file, tx_data, self.tx_data_two_hr_name)

        parameters_holder = {}
        for meta in aveperiod_meta.sequences:
            for rx_freq in meta.rx_channels:
                parameters = DATA_TEMPLATE.copy()
                parameters['borealis_git_hash'] = self.git_hash.decode('utf-8')
                parameters['experiment_id'] = np.int16(aveperiod_meta.experiment_id)
                parameters['experiment_name'] = aveperiod_meta.experiment_name
                parameters['experiment_comment'] = aveperiod_meta.experiment_comment
                parameters['scheduling_mode'] = aveperiod_meta.scheduling_mode
                parameters['slice_comment'] = rx_freq.slice_comment
                parameters['slice_id'] = np.uint32(rx_freq.slice_id)
                parameters['averaging_method'] = rx_freq.averaging_method    # string
                parameters['slice_interfacing'] = rx_freq.interfacing        # string
                parameters['num_slices'] = len(aveperiod_meta.sequences) * len(meta.rx_channels)
                parameters['station'] = self.options.site_id
                parameters['num_sequences'] = aveperiod_meta.num_sequences
                parameters['num_ranges'] = np.uint32(rx_freq.num_ranges)
                parameters['range_sep'] = np.float32(rx_freq.range_sep)
                # time to first range and back. convert to meters, div by c then convert to us
                rtt = (rx_freq.first_range * 2 * 1.0e3 / speed_of_light) * 1.0e6
                parameters['first_range_rtt'] = np.float32(rtt)
                parameters['first_range'] = np.float32(rx_freq.first_range)
                parameters['rx_sample_rate'] = data_parsing.output_sample_rate  # this applies to pre-bf and bfiq
                parameters['scan_start_marker'] = aveperiod_meta.scan_flag  # Should this change to scan_start_marker?
                parameters['int_time'] = np.float32(aveperiod_meta.aveperiod_time)
                parameters['tx_pulse_len'] = np.uint32(rx_freq.pulse_len)
                parameters['tau_spacing'] = np.uint32(rx_freq.tau_spacing)
                parameters['main_antenna_count'] = np.uint32(len(rx_freq.rx_main_antennas))
                parameters['intf_antenna_count'] = np.uint32(len(rx_freq.rx_intf_antennas))
                parameters['freq'] = np.uint32(rx_freq.rx_freq)
                parameters['rx_center_freq'] = aveperiod_meta.rx_ctr_freq
                parameters['samples_data_type'] = "complex float"
                parameters['pulses'] = np.array(rx_freq.ptab, dtype=np.uint32)

                encodings = []
                for encoding in rx_freq.sequence_encodings:
                    encoding = np.array(encoding, dtype=np.float32)
                    encodings.append(encoding)

                encodings = np.array(encodings, dtype=np.float32)
                parameters['pulse_phase_offset'] = encodings
                parameters['data_normalization_factor'] = aveperiod_meta.data_normalization_factor

                lags = []
                for lag in rx_freq.ltabs:
                    lags.append([lag.pulse_position[0], lag.pulse_position[1]])

                parameters['lags'] = np.array(lags, dtype=np.uint32)

                parameters['blanked_samples'] = np.array(meta.blanks, dtype=np.uint32)
                parameters['sqn_timestamps'] = data_parsing.timestamps

                parameters['beam_nums'] = []
                parameters['beam_azms'] = []
                for beam in rx_freq.beams:
                    parameters['beam_nums'].append(np.uint32(beam.beam_num))
                    parameters['beam_azms'].append(beam.beam_azimuth)

                parameters['noise_at_freq'] = [0.0] * aveperiod_meta.num_sequences  # TODO update. should come from data_parsing

                parameters['gps_locked'] = data_parsing.gps_locked
                parameters['gps_to_system_time_diff'] = data_parsing.gps_to_system_time_diff

                parameters['agc_status_word'] = np.uint32(data_parsing.agc_status_word)
                parameters['lp_status_word'] = np.uint32(data_parsing.lp_status_word)

                # num_samps, antenna_arrays_order, data_descriptors, data_dimensions, data
                # correlation_descriptors, correlation_dimensions, main_acfs, intf_acfs, xcfs
                # all get set within the separate write functions.

                parameters_holder[rx_freq.slice_id] = parameters

        if write_rawacf and data_parsing.mainacfs_available:
            write_correlations(copy.deepcopy(parameters_holder))

        if write_bfiq and data_parsing.bfiq_available:
            write_bfiq_params(copy.deepcopy(parameters_holder))

        if write_antenna_iq and data_parsing.antenna_iq_available:
            write_antenna_iq_params(copy.deepcopy(parameters_holder))

        if data_parsing.raw_rf_available:
            if write_raw_rf:
                # Just need first available slice paramaters.
                one_slice_params = copy.deepcopy(next(iter(parameters_holder.values())))
                write_raw_rf_params(one_slice_params)
            else:
                for rf_samples_location in data_parsing.rawrf_locations:
                    if rf_samples_location is not None:
                        shm = shared_memory.SharedMemory(name=rf_samples_location)
                        shm.close()
                        shm.unlink()

        if write_tx:
            write_tx_data()

        write_time = time.perf_counter() - start
        log.info("write time",
                 write_time=write_time * 1e3,
                 write_time_units='ms',
                 dataset_name=dataset_name)


def main():
    faulthandler.enable()
    parser = ap.ArgumentParser(description='Write processed SuperDARN data to file')
    parser.add_argument('--file-type', help='Type of output file: hdf5, json, or dmap',
                        default='hdf5')
    parser.add_argument('--enable-raw-acfs', help='Enable raw acf writing',
                        action='store_true')
    parser.add_argument('--enable-bfiq', help='Enable beamformed iq writing',
                        action='store_true')
    parser.add_argument('--enable-antenna-iq', help='Enable individual antenna iq writing',
                        action='store_true')
    parser.add_argument('--enable-raw-rf', help='Save raw, unfiltered IQ samples. Requires HDF5.',
                        action='store_true')
    parser.add_argument('--enable-tx', help='Save tx samples and metadata. Requires HDF5.',
                        action='store_true')
    args = parser.parse_args()

    options = dwo.DataWriteOptions()
    sockets = so.create_sockets([options.dw_to_dsp_identity,
                                 options.dw_to_radctrl_identity,
                                 options.dw_to_rt_identity],
                                options.router_address)

    dsp_to_data_write = sockets[0]
    radctrl_to_data_write = sockets[1]
    realtime_to_data_write = sockets[2]

    poller = zmq.Poller()
    poller.register(dsp_to_data_write, zmq.POLLIN)
    poller.register(radctrl_to_data_write, zmq.POLLIN)

    log.debug("socket connected")

    data_parsing = ParseData(options)

    current_experiment = None
    data_write = None
    first_time = True
    expected_sqn_num = 0
    queued_sqns = []
    aveperiod_metadata_dict = dict()
    while True:
        try:
            socks = dict(poller.poll())
        except KeyboardInterrupt:
            log.info("keyboard interrupt exit")
            sys.exit(0)

        if radctrl_to_data_write in socks and socks[radctrl_to_data_write] == zmq.POLLIN:
            data = so.recv_bytes(radctrl_to_data_write, options.radctrl_to_dw_identity, log)
            aveperiod_meta = pickle.loads(data)
            aveperiod_metadata_dict[aveperiod_meta.last_sqn_num] = aveperiod_meta

        if dsp_to_data_write in socks and socks[dsp_to_data_write] == zmq.POLLIN:
            data = so.recv_bytes_from_any_iden(dsp_to_data_write)
            processed_data = pickle.loads(data)
            queued_sqns.append(processed_data)

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
                    assert len(sorted_q) <= 20
                except Exception as e:
                    log.error("lost sequences", sequence_num=expected_sqn_num, error=e)
                    log.exception("lost sequences", exception=e)
                    sys.exit(1)
                continue

            expected_sqn_num = sorted_q[-1].sequence_num + 1

            for pd in sorted_q:
                if not first_time:
                    if data_parsing.sequence_num in aveperiod_metadata_dict:
                        data_parsing.numpify_arrays()
                        aveperiod_metadata = aveperiod_metadata_dict.pop(data_parsing.sequence_num)

                        if aveperiod_metadata.experiment_name != current_experiment:
                            data_write = DataWrite(options)
                            current_experiment = aveperiod_metadata.experiment_name

                        kwargs = dict(write_bfiq=args.enable_bfiq,
                                      write_antenna_iq=args.enable_antenna_iq,
                                      write_raw_rf=args.enable_raw_rf,
                                      write_tx=args.enable_tx,
                                      file_ext=args.file_type,
                                      aveperiod_meta=aveperiod_metadata,
                                      data_parsing=data_parsing,
                                      rt_dw={"socket": realtime_to_data_write,
                                             "iden": options.rt_to_dw_identity},
                                      write_rawacf=args.enable_raw_acfs)
                        thread = threading.Thread(target=data_write.output_data, kwargs=kwargs)
                        thread.daemon = True
                        thread.start()
                        data_parsing = ParseData(options)

                first_time = False

                start = time.perf_counter()
                data_parsing.update(pd)
                parse_time = time.perf_counter() - start
                log.info("parse time",
                         parse_time=parse_time * 1e3,
                         parse_time_units='ms')

            queued_sqns = []


if __name__ == '__main__':
    from utils import log_config
    log = log_config.log()
    log.info(f"DATA_WRITE BOOTED")
    try:
        main()
        log.info(f"DATA_WRITE EXITED")
    except Exception as main_exception:
        log.critical("DATA_WRITE CRASHED", error=main_exception)
        log.exception("DATA_WRITE CRASHED", exception=main_exception)
