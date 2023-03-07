#!/usr/bin/python3

"""
    data_write package
    ~~~~~~~~~~~~~~~~~~
    This package contains utilities to parse protobuf packets containing antennas_iq data, bfiq
    data, rawacf data, etc. and write that data to HDF5 or JSON files.

    :copyright: 2017 SuperDARN Canada
"""
# built-in
import argparse as ap
import collections
import datetime
from dataclasses import dataclass, field, fields
import errno
import faulthandler
import json
from multiprocessing import shared_memory
import os
import pickle
import subprocess as sp
import sys
import threading
import time
from typing import List, Dict, Set
import warnings

# third-party
import numpy as np
import deepdish as dd
import tables
import zmq
from scipy.constants import speed_of_light

# local
from utils import socket_operations as so
from utils.message_formats import ProcessedSequenceMessage
import utils.options.data_write_options as dwo


@dataclass(init=False)
class SequenceData:
    """
    This class defines all fields that need to be written by any type of data file. The 'groups' metadata lists
    the applicable file types for each field.
    """
    agc_status_word: int = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    antenna_arrays_order: List[str] = field(
        metadata={'groups': ['antennas_iq', 'bfiq']})
    averaging_method: str = field(
        metadata={'groups': ['rawacf']})
    beam_azms: List[float] = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    beam_nums: List[int] = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    blanked_samples: np.ndarray = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    borealis_git_hash: str = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    correlation_descriptors: List[str] = field(
        metadata={'groups': ['rawacf']})
    correlation_dimensions: np.ndarray = field(
        metadata={'groups': ['rawacf']})
    data: np.ndarray = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawrf']})
    data_descriptors: List[str] = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawrf']})
    data_dimensions: np.ndarray = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawrf']})
    data_normalization_factor: float = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    decimated_tx_samples: List = field(
        metadata={'groups': ['txdata']})
    dm_rate: List[int] = field(
        metadata={'groups': ['txdata']})
    experiment_comment: str = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    experiment_id: int = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    experiment_name: str = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    first_range: int = field(
        metadata={'groups': ['bfiq', 'rawacf']})
    first_range_rtt: float = field(
        metadata={'groups': ['bfiq', 'rawacf']})
    freq: float = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    gps_locked: bool = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    gps_to_system_time_diff: float = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    int_time: float = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    intf_acfs: np.ndarray = field(
        metadata={'groups': ['rawacf']})
    intf_antenna_count: int = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    lags: np.ndarray = field(
        metadata={'groups': ['bfiq', 'rawacf']})        # TODO: Should this be in antennas_iq too? Or removed from bfiq?
    lp_status_word: int = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    main_acfs: np.ndarray = field(
        metadata={'groups': ['rawacf']})
    main_antenna_count: int = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    noise_at_freq: np.ndarray = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    num_ranges: int = field(
        metadata={'groups': ['bfiq']})                              # TODO: Does this need to be in more file types?
    num_samps: int = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawrf']})
    num_sequences: int = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    num_slices: int = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    pulse_phase_offset: np.ndarray = field(
        metadata={'groups': ['antennas_iq', 'bfiq']})
    pulse_sample_start: List[float] = field(
        metadata={'groups': ['txdata']})
    pulse_timing_us: List[float] = field(
        metadata={'groups': ['txdata']})
    pulses: np.ndarray = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    range_sep: int = field(
        metadata={'groups': ['bfiq', 'rawacf']})                      # TODO: Does this need to be in antennas_iq?
    rx_center_freq: float = field(
        metadata={'groups': 'rawrf'})
    rx_sample_rate: float = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    samples_data_type: str = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    scan_start_marker: bool = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    scheduling_mode: str = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    slice_comment: str = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    slice_id: int = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    slice_interfacing: dict = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    sqn_timestamps: List[float] = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    station: str = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf', 'rawrf']})
    tau_spacing: float = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    tx_center_freq: List[float] = field(
        metadata={'groups': ['txdata']})
    tx_pulse_len: float = field(
        metadata={'groups': ['antennas_iq', 'bfiq', 'rawacf']})
    tx_rate: List[float] = field(
        metadata={'groups': ['txdata']})
    tx_samples: List = field(
        metadata={'groups': ['txdata']})
    xcfs: np.ndarray = field(
        metadata={'groups': ['rawacf']})

    @classmethod
    def type_fields(cls, file_type: str):
        """
        Returns a list of names for all fields which belong in 'file_type' files.
        """
        matching_fields = []
        for f in fields(cls):
            if file_type in f.metadata.get('groups'):
                matching_fields.append(f.name)
        return matching_fields


@dataclass
class ParseData(object):
    """
    This class is for aggregating data during an averaging period.
    """
    agc_status_word: int = 0b0
    antenna_iq_accumulator: Dict = field(default_factory=dict)
    antenna_iq_available: bool = False
    bfiq_accumulator: Dict = field(default_factory=dict)
    bfiq_available: bool = False
    intfacfs_available: bool = False
    gps_locked: bool = True     # init True so that logical AND works properly in update() method
    gps_to_system_time_diff: float = 0.0
    intfacfs_accumulator: Dict = field(default_factory=dict)
    lp_status_word: int = 0b0
    mainacfs_accumulator: Dict = field(default_factory=dict)
    mainacfs_available: bool = False
    options: dwo.DataWriteOptions = None
    output_sample_rate: float = 0.0
    processed_data: ProcessedSequenceMessage = field(init=False)
    rawrf_available: bool = False
    rawrf_locations: List[str] = field(default_factory=list)
    rawrf_num_samps: int = 0
    rx_rate: float = 0.0
    sequence_num: int = field(init=False)
    slice_ids: Set = field(default_factory=set)
    timestamps: List[float] = field(default_factory=list)
    xcfs_accumulator: Dict = field(default_factory=dict)
    xcfs_available: bool = False

    def parse_correlations(self):
        """
        Parses out the possible correlation (acf/xcf) data from the message. Runs on every new
        ProcessedSequenceMessage (contains all sampling period data). The expectation value is
        calculated at the end of an averaging period by a different function.
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
                self.mainacfs_available = True
                accumulate_data(self.mainacfs_accumulator, data_set.main_acf_shm)

            if data_set.xcf_shm:
                self.xcfs_available = True
                accumulate_data(self.xcfs_accumulator, data_set.xcf_shm)

            if data_set.intf_acf_shm:
                self.intfacfs_available = True
                accumulate_data(self.intfacfs_accumulator, data_set.intf_acf_shm)

    def parse_bfiq(self):
        """
        Parses out any possible beamformed IQ data from the message. Runs on every
        ProcessedSequenceMessage (contains all sampling period data).
        """

        self.bfiq_accumulator['data_descriptors'] = ['num_antenna_arrays', 'num_sequences', 'num_beams', 'num_samps']

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

        self.bfiq_available = True

        for i, data_set in enumerate(self.processed_data.output_datasets):
            slice_id = data_set.slice_id
            num_beams = data_set.num_beams

            self.bfiq_accumulator[slice_id]['num_samps'] = num_samps

            if 'main_data' not in self.bfiq_accumulator[slice_id]:
                self.bfiq_accumulator[slice_id]['main_data'] = []
            self.bfiq_accumulator[slice_id]['main_data'].append(main_data[i, :num_beams, :])

            if intf_available:
                if 'intf_data' not in self.bfiq_accumulator[slice_id]:
                    self.bfiq_accumulator[slice_id]['intf_data'] = []
                self.bfiq_accumulator[slice_id]['intf_data'].append(intf_data[i, :num_beams, :])

    def parse_antenna_iq(self):
        """
        Parses out any pre-beamformed IQ if available. Runs on every ProcessedSequenceMessage
        (contains all sampling period data).
        """

        self.antenna_iq_accumulator['data_descriptors'] = ['num_antennas', 'num_sequences', 'num_samps']

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

        self.antenna_iq_available = True

        # Iterate over every data set, one data set per slice
        for i, data_set in enumerate(self.processed_data.output_datasets):
            slice_id = data_set.slice_id

            # non beamformed IQ samples are available
            for debug_stage in stages:
                stage_name = debug_stage['stage_name']

                if stage_name not in self.antenna_iq_accumulator[slice_id]:
                    self.antenna_iq_accumulator[slice_id][stage_name] = collections.OrderedDict()

                antenna_iq_stage = self.antenna_iq_accumulator[slice_id][stage_name]

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
        for slice_id, slice_data in self.antenna_iq_accumulator.items():
            if isinstance(slice_id, int):  # filtering out 'data_descriptors'
                for param_data in slice_data.values():
                    for array_name, array_data in param_data.items():
                        if array_name != 'num_samps':
                            array_data['data'] = np.array(array_data['data'], dtype=np.complex64)

        for slice_id, slice_data in self.bfiq_accumulator.items():
            if isinstance(slice_id, int):  # filtering out 'data_descriptors'
                for param_name, param_data in slice_data.items():
                    slice_data[param_name] = np.array(param_data, dtype=np.complex64)

        for slice_data in self.mainacfs_accumulator.values():
            slice_data['data'] = np.array(slice_data['data'], np.complex64)

        for slice_data in self.intfacfs_accumulator.values():
            slice_data['data'] = np.array(slice_data['data'], np.complex64)

        for slice_data in self.xcfs_accumulator.values():
            slice_data['data'] = np.array(slice_data['data'], np.complex64)

    def update(self, data):
        """
        Parses the message and updates the accumulator fields with the new data.

        :param  data: Processed sequence from rx_signal_processing module.
        :type   data: ProcessedSequenceMessage
        """
        self.processed_data = data
        self.timestamps.append(data.sequence_start_time)
        self.rx_rate = data.rx_sample_rate
        self.output_sample_rate = data.output_sample_rate

        for data_set in data.output_datasets:
            self.slice_ids.add(data_set.slice_id)

        if data.rawrf_shm != '':
            self.rawrf_available = True
            self.rawrf_num_samps = data.rawrf_num_samps
            self.rawrf_locations.append(data.rawrf_shm)

        # Logical AND to catch any time the GPS may have been unlocked during the integration period
        self.gps_locked = self.gps_locked and data.gps_locked

        # Find the max time diff between GPS and system time to report for this integration period
        if abs(self.gps_to_system_time_diff) < abs(data.gps_to_system_time_diff):
            self.gps_to_system_time_diff = data.gps_to_system_time_diff

        # Bitwise OR to catch any AGC faults during the integration period
        self.agc_status_word = self.agc_status_word | data.agc_status_bank_h

        # Bitwise OR to catch any low power conditions during the integration period
        self.lp_status_word = self.lp_status_word | data.lp_status_bank_h

        self.parse_correlations()
        self.parse_bfiq()
        self.parse_antenna_iq()


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

    @staticmethod
    def write_json_file(filename, data_dict):
        """
        Write out data to a json file. If the file already exists it will be overwritten.

        :param  filename:   The path to the file to write out
        :type   filename:   str
        :param  data_dict:  Python dictionary to write out to the JSON file.
        :type   data_dict:  dict
        """

        with open(filename, 'w+') as f:
            f.write(json.dumps(data_dict))

    @staticmethod
    def write_hdf5_file(filename, data_dict, dt_str):
        """
        Write out data to an HDF5 file. If the file already exists it will be overwritten.

        :param  filename:   The path to the file to write out
        :type   filename:   str
        :param  data_dict:  Python dictionary to write out to the HDF5 file.
        :type   data_dict:  dict
        :param  dt_str:     A datetime timestamp of the first transmission time in the record
        :type   dt_str:     str
        """

        def convert_to_numpy(dict_of_lists):
            """
            Converts lists stored in dict into numpy array. Recursive.

            :param  dict_of_lists: Dictionary with lists to convert to numpy arrays.
            :type   dict_of_lists: dict
            """
            for k, v in dict_of_lists.items():
                if isinstance(v, dict):
                    convert_to_numpy(v)
                elif isinstance(v, list):
                    dict_of_lists[k] = np.array(v)
                else:
                    continue

        convert_to_numpy(data_dict)

        time_stamped_dd = {dt_str: data_dict}

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

    @staticmethod
    def write_dmap_file(filename, data_dict):
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
        :param  file_ext:           Type of file extension to use
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
            log.error(f"wrong file format {file_ext} not in [hdf5, json, dmap]", error=e)
            log.exception(f"wrong file format {file_ext} not in [hdf5, json, dmap]", exception=e)
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

        def write_file(tmp_file, aveperiod_data, two_hr_file_with_type, file_type):
            """
            Writes the final data out to the location based on the type of file extension required

            :param  tmp_file:               File path and name to write single record
            :type   tmp_file:               str
            :param  aveperiod_data:         Collection of data from sequences
            :type   aveperiod_data:         SequenceData
            :param  two_hr_file_with_type:  Name of the two hour file with data type added
            :type   two_hr_file_with_type:  str
            :param  file_type:              Data type, e.g. 'antennas_iq', 'bfiq', etc.
            :type   file_type:              str
            """

            try:
                os.makedirs(dataset_directory, exist_ok=True)
            except OSError as err:
                if err.args[0] == errno.ENOSPC:
                    log.critical("no space left on device", error=err)
                    log.exception("no space left on device", exception=err)
                    sys.exit(-1)
                else:
                    log.critical("unknown error when making dirs", error=err)
                    log.exception("unknown error when making dirs", exception=err)
                    sys.exit(-1)

            data_dict = {}
            for f in SequenceData.type_fields(file_type):
                data_dict[f] = getattr(aveperiod_data, f)

            if file_ext == 'hdf5':
                full_two_hr_file = f"{dataset_directory}/{two_hr_file_with_type}.hdf5.site"

                try:
                    fd = os.open(full_two_hr_file, os.O_CREAT)
                    os.close(fd)
                except OSError as err:
                    if e.args[0] == errno.ENOSPC:
                        log.critical("no space left on device", error=err)
                        log.exception("no space left on device", exception=err)
                        sys.exit(-1)
                    else:
                        log.critical("unknown error when opening file", error=err)
                        log.exception("unknown error when opening file", exception=err)
                        sys.exit(-1)

                self.write_hdf5_file(tmp_file, data_dict, epoch_milliseconds)

                # Use external h5copy utility to move new record into 2hr file.
                cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
                cmd = cmd.format(newfile=tmp_file, twohr=full_two_hr_file, dtstr=epoch_milliseconds)

                # TODO(keith): improve call to subprocess.
                sp.call(cmd.split())
                so.send_data(rt_dw['socket'], rt_dw['iden'], tmp_file)
                # Temp file is removed in real time module.

            elif file_ext == 'json':
                self.write_json_file(tmp_file, data_dict)
            elif file_ext == 'dmap':
                self.write_dmap_file(tmp_file, data_dict)

        def write_correlations(aveperiod_data):
            """
            Parses out any possible correlation data from message and writes to file. Some variables
            are captured from outer scope.

            main_acfs, intf_acfs, and xcfs are all passed to data_write for all sequences
            individually. At this point, they will be combined into data for a single integration
            time via averaging.

            :param  aveperiod_data:  Dict of SequenceData for each slice.
            :type   aveperiod_data:  dict
            """

            main_acfs = data_parsing.mainacfs_accumulator
            xcfs = data_parsing.xcfs_accumulator
            intf_acfs = data_parsing.intfacfs_accumulator

            def find_expectation_value(x, template):
                """
                Get the mean or median of all correlations from all sequences in the integration
                period - only this will be recorded.
                This is effectively 'averaging' all correlations over the integration time, using a
                specified method for combining them.
                """

                # array_2d is num_sequences x (num_beams*num_ranges*num_lags)
                # so we get median of all sequences.
                averaging_method = template.averaging_method
                array_2d = np.array(x, dtype=np.complex64)
                num_beams, num_ranges, num_lags = np.array([len(template.beam_nums),
                                                            template.num_ranges,
                                                            template.lags.shape[0]],
                                                           dtype=np.uint32)

                # First range offset in samples
                sample_off = template.first_range_rtt * 1e-6 * template.rx_sample_rate
                sample_off = np.uint32(sample_off)

                # Find sample number which corresponds with second pulse in sequence
                tau_in_samples = template.tau_spacing * 1e-6 * template.rx_sample_rate
                second_pulse_sample_num = np.uint32(tau_in_samples) * template.pulses[1] - sample_off - 1

                # Average the data
                try:
                    assert averaging_method in ['mean', 'median']
                    if averaging_method == 'mean':
                        array_expectation_value = np.mean(array_2d, axis=0)
                    elif averaging_method == 'median':
                        array_expectation_value = np.median(np.real(array_2d), axis=0) + \
                                                  1j * np.median(np.imag(array_2d), axis=0)
                except Exception as err:
                    log.error("wrong averaging method [mean, median]", error=err)
                    log.exception("wrong averaging method [mean, median]", exception=err)
                    sys.exit(1)

                # Reshape array to be 3d so we can replace lag0 far ranges that are cluttered with those
                # from alternate lag0 which have no clutter.
                # TODO: Can we refactor so that pycharm doesn't get mad at this?
                array_3d = array_expectation_value.reshape((num_beams, num_ranges, num_lags))
                array_3d[:, second_pulse_sample_num:, 0] = array_3d[:, second_pulse_sample_num:, -1]

                # Flatten back to a list
                return array_3d.flatten()

            for slice_num in main_acfs:
                slice_data = aveperiod_data[slice_num]
                slice_data.main_acfs = find_expectation_value(main_acfs[slice_num]['data'], slice_data)

            for slice_num in xcfs:
                slice_data = aveperiod_data[slice_num]
                slice_data.xcfs = find_expectation_value(xcfs[slice_num]['data'], slice_data)

            for slice_num in intf_acfs:
                slice_data = aveperiod_data[slice_num]
                slice_data.intf_acfs = find_expectation_value(intf_acfs[slice_num]['data'], slice_data)

            for slice_num, slice_data in aveperiod_data.items():
                slice_data.correlation_descriptors = ['num_beams', 'num_ranges', 'num_lags']
                slice_data.correlation_dimensions = np.array([len(slice_data.beam_nums), slice_data.num_ranges,
                                                              slice_data.lags.shape[0]], dtype=np.uint32)

                name = dataset_name.format(sliceid=slice_num, dformat="rawacf")
                output_file = dataset_location.format(name=name)
                two_hr_file_with_type = self.slice_filenames[slice_num].format(ext="rawacf")

                write_file(output_file, slice_data, two_hr_file_with_type, "rawacf")

        def write_bfiq_params(aveperiod_data):
            """
            write out any possible beamformed IQ data that has been parsed. Adds additional slice
            info to each parameter dict. Some variables are captured from outer scope.

            :param  aveperiod_data:  Dict of SequenceData for each slice.
            :type   aveperiod_data:  dict
            """

            bfiq = data_parsing.bfiq_accumulator
            slice_id_list = [x for x in bfiq.keys() if isinstance(x, int)]

            # Pop these off so we don't include them in later iteration.
            data_descriptors = bfiq.pop('data_descriptors', None)

            for slice_num in slice_id_list:
                slice_data = aveperiod_data[slice_num]
                slice_data.data_descriptors = data_descriptors
                slice_data.antenna_arrays_order = []

                flattened_data = []
                num_antenna_arrays = 1
                slice_data.antenna_arrays_order.append("main")
                flattened_data.append(bfiq[slice_num]['main_data'].flatten())
                if "intf" in bfiq[slice_num]:
                    num_antenna_arrays += 1
                    slice_data.antenna_arrays_order.append("intf")
                    flattened_data.append(bfiq[slice_num]['intf_data'].flatten())

                flattened_data = np.concatenate(flattened_data)
                slice_data.data = flattened_data

                slice_data.num_samps = np.uint32(bfiq[slice_num]['num_samps'])
                slice_data.data_dimensions = np.array([num_antenna_arrays, aveperiod_meta.num_sequences,
                                                       len(slice_data.beam_nums), slice_data.num_samps],
                                                      dtype=np.uint32)

            for slice_num, slice_data in aveperiod_data.items():
                name = dataset_name.format(sliceid=slice_num, dformat="bfiq")
                output_file = dataset_location.format(name=name)
                two_hr_file_with_type = self.slice_filenames[slice_num].format(ext="bfiq")

                write_file(output_file, slice_data, two_hr_file_with_type, "bfiq")

        def write_antenna_iq_params(aveperiod_data):
            """
            Writes out any pre-beamformed IQ that has been parsed. Adds additional slice info
            to each paramater dict. Some variables are captured from outer scope. Pre-beamformed iq
            is the individual antenna received data. Antenna_arrays_order will list the antennas' order.

            :param  aveperiod_data:  Dict that holds SequenceData for each slice.
            :type   aveperiod_data:  dict
            """

            antenna_iq = data_parsing.antenna_iq_accumulator
            slice_id_list = [x for x in antenna_iq.keys() if isinstance(x, int)]

            # Pop these so we don't include them in later iteration.    TODO: Is this necessary? What iteration?
            data_descriptors = antenna_iq.pop('data_descriptors', None)

            # Parse the antennas from message
            rx_main_antennas = {}
            rx_intf_antennas = {}

            for sqn in aveperiod_meta.sequences:
                for rx_channel in sqn.rx_channels:
                    rx_main_antennas[rx_channel.slice_id] = list(rx_channel.rx_main_antennas)
                    rx_intf_antennas[rx_channel.slice_id] = list(rx_channel.rx_intf_antennas)

            # Build strings from antennas used in the message. This will be used to know
            # what antennas were recorded on since we sample all available USRP channels
            # and some channels may not be transmitted on, or connected.
            def main_ant_str(x):
                return f"antenna_{x}"

            def intf_ant_str(x):
                return f"antenna_{x + self.options.main_antenna_count}"

            for slice_num in rx_main_antennas:
                rx_main_antennas[slice_num] = [main_ant_str(x) for x in rx_main_antennas[slice_num]]
                rx_intf_antennas[slice_num] = [intf_ant_str(x) for x in rx_intf_antennas[slice_num]]

            final_data_params = {}
            for slice_num in slice_id_list:
                final_data_params[slice_num] = {}

                for stage in antenna_iq[slice_num]:
                    stage_data = aveperiod_data[slice_num].copy()

                    stage_data.data_descriptors = data_descriptors
                    # TODO: Why do we pop?
                    stage_data.num_samps = np.uint32(antenna_iq[slice_num][stage].pop('num_samps', None))
                    stage_data.antenna_arrays_order = rx_main_antennas[slice_num] + rx_intf_antennas[slice_num]
                    num_ants = len(stage_data.antenna_arrays_order)

                    stage_data.data_dimensions = np.array([num_ants, aveperiod_meta.num_sequences,
                                                           stage_data.num_samps], dtype=np.uint32)

                    data = []
                    for k, data_dict in antenna_iq[slice_num][stage].items():
                        if k in stage_data.antenna_arrays_order:
                            data.append(data_dict['data'].flatten())

                    flattened_data = np.concatenate(data)
                    stage_data.data = flattened_data

                    final_data_params[slice_num][stage] = stage_data

            for slice_num, slice_ in final_data_params.items():
                for stage, params in slice_.items():
                    name = dataset_name.format(sliceid=slice_num, dformat=f"{stage}_iq")
                    output_file = dataset_location.format(name=name)

                    ext = f"{stage}_iq"
                    two_hr_file_with_type = self.slice_filenames[slice_num].format(ext=ext)

                    write_file(output_file, params, two_hr_file_with_type, "antennas_iq")

        def write_raw_rf_params(slice_data):
            """
            Opens the shared memory location in the message and writes the samples out to file.
            Write medium must be able to sustain high write bandwidth. Shared memory is destroyed
            after write. Some variables are captured in scope. It's expected that the
            user will have knowledge of what they are looking for when working with this data.
            Note that because this data is not slice-specific a lot of slice-specific data (ex.
            pulses, beam_nums, beam_azms) is not included (user must look at the experiment they
            ran).

            :param  slice_data:  Parameters for a single slice during the averaging period.
            :type   slice_data:  SequenceData
            """

            raw_rf = data_parsing.rawrf_locations
            num_rawrf_samps = data_parsing.rawrf_num_samps

            # Don't need slice id here
            name = dataset_name.replace('{sliceid}.', '').format(dformat='rawrf')
            output_file = dataset_location.format(name=name)

            samples_list = []
            shared_memory_locations = []
            total_ants = len(self.options.main_antennas) + len(self.options.intf_antennas)

            for raw in raw_rf:
                shared_mem = shared_memory.SharedMemory(name=raw)
                rawrf_array = np.ndarray((total_ants, num_rawrf_samps), dtype=np.complex64, buffer=shared_mem.buf)

                samples_list.append(rawrf_array.flatten())

                shared_memory_locations.append(shared_mem)

            slice_data.data = np.concatenate(samples_list)
            slice_data.rx_sample_rate = np.float32(data_parsing.rx_rate)
            slice_data.num_samps = np.uint32(len(samples_list[0]) / total_ants)
            slice_data.data_descriptors = ["num_sequences", "num_antennas", "num_samps"]
            slice_data.data_dimensions = np.array([slice_data.num_sequences, total_ants,
                                                   slice_data.num_samps], dtype=np.uint32)
            slice_data.main_antenna_count = np.uint32(self.options.main_antenna_count)
            slice_data.intf_antenna_count = np.uint32(self.options.intf_antenna_count)

            write_file(output_file, slice_data, self.raw_rf_two_hr_name, "rawrf")

            # Can only close mapped memory after it's been written to disk.
            for shared_mem in shared_memory_locations:
                shared_mem.close()
                shared_mem.unlink()

        def write_tx_data():
            """
            Writes out the tx samples and metadata for debugging purposes.
            Does not use same parameters of other writes.
            """
            tx_data = SequenceData()
            for f in SequenceData.type_fields('txdata'):
                setattr(tx_data, f, [])     # initialize to a list for all fields

            has_tx_data = [sqn.tx_data is not None for sqn in aveperiod_meta.sequences]

            if True in has_tx_data:     # If any sequence has tx data, write to file
                for sqn in aveperiod_meta.sequences:
                    if sqn.tx_data is not None:
                        meta_data = sqn.tx_data
                        tx_data.tx_rate.append(meta_data.tx_rate)
                        tx_data.tx_center_freq.append(meta_data.tx_ctr_freq)
                        tx_data.pulse_timing_us.append(meta_data.pulse_timing_us)
                        tx_data.pulse_sample_start.append(meta_data.pulse_sample_start)
                        tx_data.dm_rate.append(meta_data.dm_rate)
                        tx_data.tx_samples.append(meta_data.tx_samples)
                        tx_data.decimated_tx_samples.append(meta_data.decimated_tx_samples)

                    name = dataset_name.replace('{sliceid}.', '').format(dformat='txdata')
                    output_file = dataset_location.format(name=name)

                    write_file(output_file, tx_data, self.tx_data_two_hr_name, 'txdata')

        all_slice_data = {}
        for meta in aveperiod_meta.sequences:
            for rx_freq in meta.rx_channels:

                # time to first range and back. convert to meters, div by c then convert to us
                rtt = (rx_freq.first_range * 2 * 1.0e3 / speed_of_light) * 1.0e6

                encodings = []
                for encoding in rx_freq.sequence_encodings:
                    encoding = np.array(encoding, dtype=np.float32)
                    encodings.append(encoding)
                encodings = np.array(encodings, dtype=np.float32)

                lags = [[lag.pulse_position[0], lag.pulse_position[1]] for lag in rx_freq.ltabs]

                parameters = SequenceData()
                parameters.agc_status_word = np.uint32(data_parsing.agc_status_word),
                parameters.averaging_method = rx_freq.averaging_method,
                parameters.beam_azms = [beam.beam_azimuth for beam in rx_freq.beams],
                parameters.beam_nums = [np.uint32(beam.beam_num) for beam in rx_freq.beams],
                parameters.blanked_samples = np.array(meta.blanks, dtype=np.uint32),
                parameters.borealis_git_hash = self.git_hash.decode('utf-8'),
                parameters.data_normalization_factor = aveperiod_meta.data_normalization_factor,
                parameters.experiment_comment = aveperiod_meta.experiment_comment,
                parameters.experiment_id = np.int16(aveperiod_meta.experiment_id),
                parameters.experiment_name = aveperiod_meta.experiment_name,
                parameters.first_range = np.float32(rx_freq.first_range),
                parameters.first_range_rtt = np.float32(rtt),
                parameters.freq = np.uint32(rx_freq.rx_freq),
                parameters.gps_locked = data_parsing.gps_locked,
                parameters.gps_to_system_time_diff = data_parsing.gps_to_system_time_diff,
                parameters.int_time = np.float32(aveperiod_meta.aveperiod_time),
                parameters.intf_antenna_count = np.uint32(len(rx_freq.rx_intf_antennas)),
                parameters.lags = np.array(lags, dtype=np.uint32),
                parameters.lp_status_word = np.uint32(data_parsing.lp_status_word),
                parameters.main_antenna_count = np.uint32(len(rx_freq.rx_main_antennas)),
                parameters.noise_at_freq = [0.0] * aveperiod_meta.num_sequences,  # TODO: should come from data_parsing
                parameters.num_ranges = np.uint32(rx_freq.num_ranges),
                parameters.num_sequences = aveperiod_meta.num_sequences,
                parameters.num_slices = len(aveperiod_meta.sequences) * len(meta.rx_channels),
                parameters.pulse_phase_offset = encodings,
                parameters.pulses = np.array(rx_freq.ptab, dtype=np.uint32),
                parameters.range_sep = np.float32(rx_freq.range_sep),
                parameters.rx_center_freq = aveperiod_meta.rx_ctr_freq,
                parameters.rx_sample_rate = data_parsing.output_sample_rate,
                parameters.samples_data_type = "complex float",
                parameters.scan_start_marker = aveperiod_meta.scan_flag,
                parameters.scheduling_mode = aveperiod_meta.scheduling_mode,
                parameters.slice_comment = rx_freq.slice_comment,
                parameters.slice_id = np.uint32(rx_freq.slice_id),
                parameters.slice_interfacing = rx_freq.interfacing,
                parameters.sqn_timestamps = data_parsing.timestamps,
                parameters.station = self.options.site_id,
                parameters.tau_spacing = np.uint32(rx_freq.tau_spacing),
                parameters.tx_pulse_len = np.uint32(rx_freq.pulse_len),

                all_slice_data[rx_freq.slice_id] = parameters

        # We no longer need to deepcopy. The following fields are changed in each write_{type} method:
        # rawacf:      [main_acfs, intf_acfs, xcfs, correlation_descriptors, correlation_dimensions]
        # bfiq:        [data_descriptors, num_samps, antenna_arrays_order, data_dimensions, data]
        # antennas_iq: [data_descriptors, num_samps, antenna_arrays_order, data_dimensions, data]
        # rawrf:       [data_descriptors, num_samps, data, rx_sample_rate, data_dimensions, main_antenna_count,
        #               intf_antenna_count]
        # txdata: completely uses its own fields
        if write_rawacf and data_parsing.mainacfs_available:
            write_correlations(all_slice_data)
        if write_bfiq and data_parsing.bfiq_available:
            write_bfiq_params(all_slice_data)
        if write_antenna_iq and data_parsing.antenna_iq_available:
            write_antenna_iq_params(all_slice_data)
        if data_parsing.rawrf_available:
            if write_raw_rf:
                # Just need first available slice parameters.
                one_slice_data = next(iter(all_slice_data.values()))
                write_raw_rf_params(one_slice_data)
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

    data_parsing = ParseData(options=options)

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
                        data_parsing = ParseData(options=options)

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
