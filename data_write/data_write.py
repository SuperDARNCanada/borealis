#!/usr/bin/python3

# Copyright 2017 SuperDARN Canada
#
# data_write.py
# 2018-05-14
# Data writing functionality to write iq/raw and other files


import sys
import os
import datetime
import json
import collections
import mmap
import warnings
import time
import threading
import multiprocessing as mp
import subprocess as sp
import argparse as ap
import numpy as np
import deepdish as dd
import posix_ipc as ipc
import zmq
import faulthandler
from scipy.constants import speed_of_light
import copy

borealis_path = os.environ['BOREALISPATH']
if not borealis_path:
    raise ValueError("BOREALISPATH env variable not set")

if __debug__:
    sys.path.append(borealis_path + '/build/debug/utils/protobuf')
else:
    sys.path.append(borealis_path + '/build/release/utils/protobuf')

import processeddata_pb2
import datawritemetadata_pb2

sys.path.append(borealis_path + '/utils/')
import data_write_options.data_write_options as dwo
from zmq_borealis_helpers import socket_operations as so


def printing(msg):
    """
    Pretty print function for the Data Write module.
    :param msg: The string to format nicely for printing
    """
    DATA_WRITE = "\033[96m" + "DATA WRITE: " + "\033[0m"
    sys.stdout.write(DATA_WRITE + msg + "\n")

DATA_TEMPLATE = {
    "borealis_git_hash" : None, # Identifies the version of Borealis that made this data.
    "experiment_id" : None, # Number used to identify experiment.
    "experiment_name" : None, # Name of the experiment file.
    "experiment_comment" : None,  # Comment about the whole experiment
    "slice_comment" : None, # Additional text comment that describes the slice.
    "num_slices" : None, # Number of slices in the experiment at this integration time.
    "station" : None, # Three letter radar identifier.
    "num_sequences": None, # Number of sampling periods in the integration time.
    "num_ranges": None, # Number of ranges to calculate correlations for
    "range_sep": None, # range gate separation (equivalent distance between samples) in km.
    "first_range_rtt" : None, # Round trip time of flight to first range in microseconds.
    "first_range" : None, # Distance to first range in km.
    "rx_sample_rate" : None, # Sampling rate of the samples being written to file in Hz.
    "scan_start_marker" : None, # Designates if the record is the first in a scan.
    "int_time" : None, # Integration time in seconds.
    "tx_pulse_len" : None, # Length of the pulse in microseconds.
    "tau_spacing" : None, # The minimum spacing between pulses in microseconds.
                          # Spacing between pulses is always a multiple of this.
    "main_antenna_count" : None, # Number of main array antennas.
    "intf_antenna_count" : None, # Number of interferometer array antennas.
    "freq" : None, # The frequency used for this experiment slice in kHz.
    #"filtered_3db_bandwidth" : None, # Bandwidth of the output iq data types? can add later
    "rx_center_freq" : None, # the center frequency of this data (for rawrf), kHz
    "samples_data_type" : None, # C data type of the samples such as complex float.
    "pulses" : None, # The pulse sequence in units of the tau_spacing.
    "pulse_phase_offset" : None, # For pulse encoding phase. Contains one phase offset per pulse in pulses.
    "lags" : None, # The lags created from two pulses in the pulses array.
    "blanked_samples" : None, # Samples that have been blanked because they occurred during transmission times.
                              # Can differ from the pulses array due to multiple slices in a single sequence.
    "sqn_timestamps" : None, # A list of GPS timestamps of the beginning of transmission for each
                             # sampling period in the integration time. Seconds since epoch.
    "beam_nums" : None, # A list of beam numbers used in this slice.
    "beam_azms" : None, # A list of the beams azimuths for each beam in degrees off boresite.
    "noise_at_freq" : None, # Noise at the receive frequency, should be an array (one value per sequence) (TODO units??) (TODO document FFT resolution bandwidth for this value, should be = output_sample rate?)
    #"noise_in_raw_band" : None, # Average noise in the sampling band (input sample rate) (TODO units??)
    #"rx_bandwidth" : None, # if the noise_in_raw_band is provided, the rx_bandwidth should be provided!
    "num_samps" : None, # Number of samples in the sampling period.
    "antenna_arrays_order" : None, # States what order the data is in. Describes the data layout.
    "data_descriptors" : None, # Denotes what each data dimension represents.
    "data_dimensions" : None, # The dimensions in which to reshape the data.
    "data_normalization_factor" : None, # The scale of all of the filters, multiplied, for a total scaling factor to normalize by.
    "data" : [], # A contiguous set of samples (complex float) at given sample rate
    "correlation_descriptors" : None, # Denotes what each acf/xcf dimension represents.
    "correlation_dimensions" : None, # The dimensions in which to reshape the acf/xcf data.
    "main_acfs" : [], # Main array autocorrelations
    "intf_acfs" : [], # Interferometer array autocorrelations
    "xcfs" : [] # Crosscorrelations between main and interferometer arrays
}

TX_TEMPLATE = {
    "tx_rate" : [],
    "tx_centre_freq" : [],
    "pulse_sequence_timing_us" : [],
    "pulse_offset_error_us" : [],
    "tx_samples" : [],
    "dm_rate" : [],
    "dm_rate_error" : [],
    "decimated_tx_samples" : [],
    "tx_antennas" : [],
    "decimated_tx_antennas" : [],
}


class ParseData(object):
    """Parse protobuf data from sockets into file writable types, such as hdf5, json, dmap, etc.

    Attributes:
        nested_dict (Python default nested dictionary): alias to a nested defaultdict
        processed_data (Protobuf packet): Contains a processeddata protobuf from dsp socket in
                                          protobuf_pb2 format.
    """

    def __init__(self):
        super(ParseData, self).__init__()

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

        self._rawrf_locations = []


    def parse_correlations(self):
        """
        Parses out the possible correlation data from the protobuf. Runs on every new processeddata
        packet(contains all sampling period data). The expectation value is calculated at the end
        of a sampling period by a different function.
        """

        for data_set in self.processed_data.outputdataset:
            slice_id = data_set.slice_id

            def accumulate_data(holder, proto_data):
                cmplx = np.ones(len(proto_data), dtype=np.complex64)

                for i, cf in enumerate(proto_data):
                    cmplx[i] = cf.real + 1.0j * cf.imag

                if 'data' not in holder[slice_id]:
                    holder[slice_id]['data'] = []
                holder[slice_id]['data'].append(cmplx)


            if data_set.mainacf:
                self._mainacfs_available = True
                accumulate_data(self._mainacfs_accumulator, data_set.mainacf)

            if data_set.xcf:
                self._xcfs_available = True
                accumulate_data(self._xcfs_accumulator, data_set.xcf)

            if data_set.intacf:
                self._intfacfs_available = True
                accumulate_data(self._intfacfs_accumulator, data_set.intacf)



    def parse_bfiq(self):
        """
        Parses out any possible beamformed IQ data from the protobuf. Runs on every processeddata
        packet(contains all sampling period data). All variables are captured from outer scope.

        """

        self._bfiq_accumulator['data_descriptors'] = ['num_antenna_arrays', 'num_sequences',
                                                      'num_beams', 'num_samps']

        for data_set in self.processed_data.outputdataset:
            slice_id = data_set.slice_id

            # Find out what is available in the data to determine what to write out
            if data_set.beamformedsamples:
                self._bfiq_available = True

                for beam in data_set.beamformedsamples:
                    self._bfiq_accumulator[slice_id]['num_samps'] = len(beam.mainsamples)
                    def add_samples(samples, antenna_arr_type):
                        """Takes samples from protobuf and converts them to Numpy. Samples
                        are then concatenated to previous data.

                        Args:
                            samples (Protobuf): ProcessedData protobuf samples
                            antenna_arr_type (String): Denotes "Main" or "Intf" arrays.
                        """

                        cmplx = np.ones(len(beam.mainsamples), dtype=np.complex64)
                        # builds complex samples from protobuf sample (which contains real and
                        # imag floats)
                        for i, sample in enumerate(samples):
                            cmplx[i] = sample.real + 1.0j * sample.imag

                        # Assign if data does not exist, else concatenate to whats already there.
                        if 'data' not in self._bfiq_accumulator[slice_id][antenna_arr_type]:
                            self._bfiq_accumulator[slice_id][antenna_arr_type]['data'] = cmplx
                        else:
                            arr = self._bfiq_accumulator[slice_id][antenna_arr_type]
                            arr['data'] = np.concatenate((arr['data'], cmplx))

                    add_samples(beam.mainsamples, "main")

                    if beam.intfsamples:
                        add_samples(beam.intfsamples, "intf")


    def parse_antenna_iq(self):
        """
        Parses out any pre-beamformed IQ if available. Runs on every processeddata
        packet(contains all sampling period data). All variables are captured from outer scope.
        """

        self._antenna_iq_accumulator['data_descriptors'] = ['num_antennas', 'num_sequences',
                                                          'num_samps']
        # Iterate over every data set, one data set per slice
        for data_set in self.processed_data.outputdataset:
            slice_id = data_set.slice_id

            # non beamformed IQ samples are available
            if data_set.debugsamples:
                self._antenna_iq_available = True

                # Loops over all filter stage data, one set per stage
                for debug_samples in data_set.debugsamples:
                    stage_name = debug_samples.stagename

                    if stage_name not in self._antenna_iq_accumulator[slice_id]:
                        self._antenna_iq_accumulator[slice_id][stage_name] = collections.OrderedDict()

                    antenna_iq_stage = self._antenna_iq_accumulator[slice_id][stage_name]
                    # Loops over antenna data within stage
                    for ant_num, ant_data in enumerate(debug_samples.antennadata):
                        ant_str = "antenna_{0}".format(ant_num)

                        cmplx = np.empty(len(ant_data.antennasamples), dtype=np.complex64)
                        antenna_iq_stage["num_samps"] = len(ant_data.antennasamples)

                        for i, sample in enumerate(ant_data.antennasamples):
                            cmplx[i] = sample.real + 1.0j * sample.imag

                        if ant_str not in antenna_iq_stage:
                            antenna_iq_stage[ant_str] = {}

                        if 'data' not in antenna_iq_stage[ant_str]:
                            antenna_iq_stage[ant_str]['data'] = cmplx
                        else:
                            arr = antenna_iq_stage[ant_str]
                            arr['data'] = np.concatenate((arr['data'], cmplx))

    def update(self, data):
        """ Parses the protobuf and updates the accumulator fields with the new data.

        Args:
            data (Protobuf): deserialized ProcessedData protobuf.
        """
        self.processed_data = data
        self._timestamps.append(self.processed_data.sequence_start_time)

        self._rx_rate = self.processed_data.rx_sample_rate
        self._output_sample_rate = self.processed_data.output_sample_rate

        for data_set in self.processed_data.outputdataset:
            self._slice_ids.add(data_set.slice_id)

        self._rawrf_locations.append(self.processed_data.rf_samples_location)

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
        """ Gets the sequence num of the latest processeddata packet.

        Returns:
            TYPE: Int
        """
        return self.processed_data.sequence_num

    @property
    def bfiq_available(self):
        """ Gets the bfiq available flag.

        Returns:
            TYPE: Bool
        """
        return self._bfiq_available

    @property
    def antenna_iq_available(self):
        """ Gets the pre-bfiq available flag.

        Returns:
            TYPE: Bool
        """
        return self._antenna_iq_available

    @property
    def mainacfs_available(self):
        """Gets the mainacfs available flag.

        Returns:
            TYPE: Bool
        """
        return self._mainacfs_available

    @property
    def xcfs_available(self):
        """Gets the xcfs available flag.

        Returns:
            TYPE: Bool
        """
        return self._xcfs_available

    @property
    def intfacfs_available(self):
        """Gets the intfacfs available flag.

        Returns:
            TYPE: Bool
        """
        return self._intfacfs_available


    @property
    def bfiq_accumulator(self):
        """ Returns the nested default dictionary with complex stage data for each antenna array as
        well as some metadata.

        Returns:
            TYPE: Nested default dict: Contains beamform data for each slice.
        """
        return self._bfiq_accumulator

    @property
    def antenna_iq_accumulator(self):
        """Returns the nested default dictionary with complex stage data for each antenna as well
        as some metadata for each slice.

        Returns:
            Nested default dict: Contains stage data for each antenna and slice.
        """
        return self._antenna_iq_accumulator

    @property
    def mainacfs_accumulator(self):
        """Returns the default dict containing a list of main acf data for each slice. There is an
        array of data for each sampling period.

        Returns:
            TYPE: Default dict: Contains main acf data for each slice.
        """
        return self._mainacfs_accumulator

    @property
    def xcfs_accumulator(self):
        """Returns the default dict containing a list of xcf data for each slice. There is an
        array of data for each sampling period.

        Returns:
            TYPE: Default dict: Contains xcf data for each slice.
        """
        return self._xcfs_accumulator

    @property
    def intfacfs_accumulator(self):
        """Returns the default dict containing a list of intf acf data for each slice. There is an
        array of data for each sampling period.

        Returns:
            TYPE: Default dict: Contains intf acf data for each slice.
        """
        return self._intfacfs_accumulator

    @property
    def timestamps(self):
        """Return the python list of sequence timestamps (when the sampling period begins)
        from the processsed data packets

        Returns:
            python list: A list of sequence timestamps from the processed data packets
        """
        return self._timestamps

    @property
    def rx_rate(self):
        """Return the rx_rate of the data in the data packet

        Returns:
            float: sampling rate in Hz.
        """
        return self._rx_rate

    @property
    def output_sample_rate(self):
        """Return the output rate of the filtered, decimated data in the data packet.

        Returns:
            float: output sampling rate in Hz.
        """
        return self._output_sample_rate

    @property
    def slice_ids(self):
        """Return the slice ids in python set so they are guaranteed unique

        Returns:
            set: slice id numbers
        """
        return self._slice_ids

    @property
    def rawrf_locations(self):
        """ Gets the list of raw rf memory locations.

        Returns:
            TYPE: List of strings.
        """
        return self._rawrf_locations



class DataWrite(object):
    """This class contains the functions used to write out processed data to files.

    Args:
        data_write_options (DataWriteOptions): The data write options from config.
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

        # The next two hour boundary for files.
        self.next_boundary = None

        # Default this to true so we know if we are running for the first time.
        self.first_time = True

    def write_json_file(self, filename, data_dict):
        """
        Write out data to a json file. If the file already exists it will be overwritten.
        :param filename: The path to the file to write out. String
        :param data_dict: Python dictionary to write out to the JSON file.
        """
        with open(filename, 'w+') as f:
            f.write(json.dumps(data_dict))


    def write_hdf5_file(self, filename, data_dict, dt_str):
        """
        Write out data to an HDF5 file. If the file already exists it will be overwritten.
        :param filename: The path to the file to write out. String
        :param data_dict: Python dictionary to write out to the HDF5 file.
        :param dt_str: A datetime timestamp of the first transmission time in the record as string.
        """

        def convert_to_numpy(dd):
            """Converts lists stored in dict into numpy array. Recursive.

            Args:
                dd (Python dictionary): Dictionary with lists to convert to numpy arrays.
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

        # TODO(keith): Investigate warning.
        warnings.simplefilter("ignore") #ignore NaturalNameWarning
        dd.io.save(filename, time_stamped_dd, compression=None)


    def write_dmap_file(self, filename, data_dict):
        """
        Write out data to a dmap file. If the file already exists it will be overwritten.
        :param filename: The path to the file to write out. String
        :param data_dict: Python dictionary to write out to the dmap file.
        """
        # TODO: Complete this by parsing through the dictionary and write out to proper dmap format
        pass

    def output_data(self, write_bfiq, write_antenna_iq, write_raw_rf, write_tx, file_ext,
                    integration_meta, data_parsing, rt_dw, write_rawacf=True):
        """
        Parse through samples and write to file.

        A file will be created using the file extention for each requested data product.

        :param write_bfiq:          Should beamformed IQ be written to file? Bool.
        :param write_antenna_iq:      Should pre-beamformed IQ be written to file? Bool.
        :param write_raw_rf:        Should raw rf samples be written to file? Bool.
        :param file_ext:            Type of file extention to use. String
        :param integration_meta:    Metadata from radar control about integration period. Protobuf
        :param data_parsing:        All parsed and concatenated data from integration period stored
                                    in DataParsing object.
        :param rt_dw_socket:        Pair of socket and iden for RT purposes.
        :param write_rawacf:        Should rawacfs be written to file? Bool, default True.
        """


        start = time.time()
        if file_ext not in ['hdf5', 'json', 'dmap']:
            raise ValueError("File format selection required (hdf5, json, dmap), none given")

        # Format the name and location for the dataset
        time_now = datetime.datetime.utcfromtimestamp(data_parsing.timestamps[0])

        today_string = time_now.strftime("%Y%m%d")
        datetime_string = time_now.strftime("%Y%m%d.%H%M.%S.%f")
        epoch = datetime.datetime.utcfromtimestamp(0)
        epoch_milliseconds = str(int((time_now - epoch).total_seconds() * 1000))
        dataset_directory = "{0}/{1}".format(self.options.data_directory, today_string)
        dataset_name = "{dt}.{site}.{{sliceid}}.{{dformat}}.{fformat}".format(dt=datetime_string,
                                                                        site=self.options.site_id,
                                                                        fformat=file_ext)
        dataset_location = "{dir}/{{name}}".format(dir=dataset_directory)


        def two_hr_ceiling(dt):
            """Finds the next 2hr boundary starting from midnight

            Args:
                dt (TYPE): A datetime to find the next 2hr boundary.

            Returns:
                TYPE: 2hr aligned datetime
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
                                                       sliceid=slice_id, site=self.options.site_id)
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

            :param tmp_file:                File path and name to write single record. String
            :param final_data_dict:         Data dict parsed out from protobuf. Dict
            :param two_hr_file_with_type:   Name of the two hour file with data type added. String

            """
            if not os.path.exists(dataset_directory):
                # Don't try-catch this, because we want it to fail hard if we can't write files
                os.makedirs(dataset_directory)


            if file_ext == 'hdf5':
                full_two_hr_file = "{0}/{1}.hdf5.site".format(dataset_directory, two_hr_file_with_type)

                try:
                    fd = os.open(full_two_hr_file, os.O_CREAT | os.O_EXCL)
                    os.close(fd)
                except FileExistsError:
                    pass

                self.write_hdf5_file(tmp_file, final_data_dict, epoch_milliseconds)

                # use external h5copy utility to move new record into 2hr file.
                cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
                cmd = cmd.format(newfile=tmp_file, twohr=full_two_hr_file, dtstr=epoch_milliseconds)

                # TODO(keith): improve call to subprocess.
                sp.call(cmd.split())
                #os.remove(tmp_file)
                so.send_data(rt_dw['socket'], rt_dw['iden'], tmp_file)

            elif file_ext == 'json':
                self.write_json_file(tmp_file, final_data_dict)
            elif file_ext == 'dmap':
                self.write_dmap_file(tmp_file, final_data_dict)

        def write_correlations(parameters_holder):
            """
            Parses out any possible correlation data from protobuf and writes to file. Some variables
            are captured from outer scope.

            main_acfs, intf_acfs, and xcfs are all passed to data_write for all sequences
            individually. At this point, they will be combined into data for a single integration
            time via averaging.
            """

            needed_fields = ["borealis_git_hash", "experiment_id",
            "experiment_name", "experiment_comment", "num_slices", "slice_comment", "station",
            "num_sequences", "range_sep", "first_range_rtt", "first_range", "rx_sample_rate",
            "scan_start_marker", "int_time", "tx_pulse_len", "tau_spacing",
            "main_antenna_count", "intf_antenna_count", "freq", "samples_data_type",
            "pulses", "lags", "blanked_samples", "sqn_timestamps", "beam_nums", "beam_azms",
            "correlation_descriptors", "correlation_dimensions", "main_acfs", "intf_acfs",
            "xcfs", "noise_at_freq", "data_normalization_factor"]
            # note num_ranges not in needed_fields but are used to make
            # correlation_dimensions

            #unneeded_fields = ['data_dimensions', 'data_descriptors', 'antenna_arrays_order',
            #'data', 'num_ranges', 'num_samps', 'rx_center_freq', 'pulse_phase_offset']

            main_acfs = data_parsing.mainacfs_accumulator
            xcfs = data_parsing.xcfs_accumulator
            intf_acfs = data_parsing.intfacfs_accumulator

            def find_expectation_value(x, parameters, field_name):
                """
                Get the median of all correlations from all sequences in the
                integration period - only this will be recorded.
                This is effectively 'averaging' all correlations over the integration
                time.
                """
                # array_2d is num_sequences x (num_beams*num_ranges*num_lags)
                # so we get median of all sequences.
                array_2d = np.array(x, dtype=np.complex64)
                array_expectation_value = np.mean(array_2d, axis=0) # or use np.median?
                parameters[field_name] = array_expectation_value

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
                    parameters["num_ranges"], parameters["lags"].shape[0]],dtype=np.uint32)
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

            Args:
                parameters_holder (Dict): A dict that hold dicts of parameters for each slice.
            """
            needed_fields = ["borealis_git_hash", "experiment_id",
            "experiment_name", "experiment_comment", "num_slices", "slice_comment", "station",
            "num_sequences", "rx_sample_rate", "pulse_phase_offset",
            "scan_start_marker", "int_time", "tx_pulse_len", "tau_spacing",
            "main_antenna_count", "intf_antenna_count", "freq", "samples_data_type",
            "pulses", "blanked_samples", "sqn_timestamps", "beam_nums", "beam_azms",
            "data_dimensions", "data_descriptors", "antenna_arrays_order", "data",
            "num_samps", "noise_at_freq", "range_sep", "first_range_rtt", "first_range",
            "lags", "num_ranges", "data_normalization_factor"]

            bfiq = data_parsing.bfiq_accumulator

            # Pop these off so we dont include them in later iteration.
            data_descriptors = bfiq.pop('data_descriptors', None)

            for slice_id in bfiq:
                parameters = parameters_holder[slice_id]

                parameters['data_descriptors'] = data_descriptors
                parameters['antenna_arrays_order'] = []

                flattened_data = []
                num_antenna_arrays = 1
                parameters['antenna_arrays_order'].append("main")
                flattened_data.append(bfiq[slice_id]['main']['data'])
                if "intf" in bfiq[slice_id]:
                    num_antenna_arrays += 1
                    parameters['antenna_arrays_order'].append("intf")
                    flattened_data.append(bfiq[slice_id]['intf']['data'])

                flattened_data = np.concatenate(flattened_data)
                parameters['data'] = flattened_data

                parameters['num_samps'] = np.uint32(bfiq[slice_id]['num_samps'])
                parameters['data_dimensions'] = np.array([num_antenna_arrays,
                                                          integration_meta.num_sequences,
                                                          len(parameters['beam_nums']),
                                                          parameters['num_samps']], dtype=np.uint32)


                for field in list(parameters.keys()):
                    if field not in needed_fields:
                        parameters.pop(field, None)

                # for field in unneeded_fields:
                #     parameters.pop(field, None)

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

            Args:
                parameters_holder (Dict): A dict that hold dicts of parameters for each slice.

            """

            needed_fields = ["borealis_git_hash", "experiment_id",
            "experiment_name", "experiment_comment", "num_slices", "slice_comment", "station",
            "num_sequences", "rx_sample_rate", "scan_start_marker", "int_time", "tx_pulse_len", "tau_spacing",
            "main_antenna_count", "intf_antenna_count", "freq", "samples_data_type",
            "pulses", "sqn_timestamps", "beam_nums", "beam_azms", "data_dimensions", "data_descriptors",
            "antenna_arrays_order", "data", "num_samps", "pulse_phase_offset", "noise_at_freq",
            "data_normalization_factor"]

            antenna_iq = data_parsing.antenna_iq_accumulator

            # Pop these so we don't include them in later iteration.
            data_descriptors = antenna_iq.pop('data_descriptors', None)

            # Parse the antennas from protobuf
            rx_main_antennas = {}
            rx_intf_antennas = {}
            for meta in integration_meta.sequences:
                for rx_freq in meta.rxchannel:
                    rx_main_antennas[rx_freq.slice_id] = list(rx_freq.rx_main_antennas)
                    rx_intf_antennas[rx_freq.slice_id] = list(rx_freq.rx_intf_antennas)

            # Build strings from antennas used in the protobuf. This will be used to know
            # what antennas were recorded on since we sample all available USRP channels
            # and some channels may not be transmitted on, or connected.
            main_ant_str = lambda x: "antenna_{}".format(x)
            intf_ant_str = lambda x: "antenna_{}".format(x + self.options.main_antenna_count)
            for slice_id in rx_main_antennas:
                rx_main_antennas[slice_id] = [main_ant_str(x) for x in rx_main_antennas[slice_id]]
                rx_intf_antennas[slice_id] = [intf_ant_str(x) for x in rx_intf_antennas[slice_id]]


            final_data_params = {}
            for slice_id in antenna_iq:
                final_data_params[slice_id] = {}

                for stage in antenna_iq[slice_id]:
                    parameters = parameters_holder[slice_id].copy()

                    parameters['data_descriptors'] = data_descriptors
                    parameters['num_samps'] = np.uint32(
                        antenna_iq[slice_id][stage].pop('num_samps', None))


                    parameters['antenna_arrays_order'] = rx_main_antennas[slice_id] +\
                                                         rx_intf_antennas[slice_id]

                    num_ants = len(parameters['antenna_arrays_order'])

                    parameters['data_dimensions'] = np.array([num_ants,
                                                              integration_meta.num_sequences,
                                                              parameters['num_samps']],
                                                             dtype=np.uint32)

                    data = []
                    for k, data_dict in antenna_iq[slice_id][stage].items():
                        if k in parameters['antenna_arrays_order']:
                            data.append(data_dict['data'])

                    flattened_data = np.concatenate(data)
                    parameters['data'] = flattened_data


                    for field in list(parameters.keys()):
                        if field not in needed_fields:
                            parameters.pop(field, None)

                    final_data_params[slice_id][stage] = parameters


            for slice_id, slice_ in final_data_params.items():
                for stage, params in slice_.items():
                    name = dataset_name.format(sliceid=slice_id, dformat="{}_iq".format(stage))
                    output_file = dataset_location.format(name=name)

                    ext = "{}_iq".format(stage)
                    two_hr_file_with_type = self.slice_filenames[slice_id].format(ext=ext)

                    write_file(output_file, params, two_hr_file_with_type)


        def write_raw_rf_params(param):
            """
            Opens the shared memory location in the protobuf and writes the samples out to file.
            Write medium must be able to sustain high write bandwidth. Shared memory is destroyed
            after write. Some variables are captured in scope.

            Args:
                param (Dict): A dict of parameters to write. Some will be removed.


            """

            needed_fields = ["borealis_git_hash", "experiment_id",
            "experiment_name", "experiment_comment", "num_slices", "station",
            "num_sequences", "rx_sample_rate", "scan_start_marker", "int_time",
            "main_antenna_count", "intf_antenna_count", "samples_data_type",
            "sqn_timestamps", "data_dimensions", "data_descriptors", "data", "num_samps",
            "rx_center_freq"]

            # Some fields don't make much sense when working with the raw rf. It's expected
            # that the user will have knowledge of what they are looking for when working with
            # this data. Note that because this data is not slice-specific a lot of slice-specific
            # data (ex. pulses, beam_nums, beam_azms) is not included (user must look
            # at the experiment they ran)

            raw_rf = data_parsing.rawrf_locations

            # Don't need slice id here
            name = dataset_name.replace('{sliceid}.', '').format(dformat='rawrf')
            output_file = dataset_location.format(name=name)

            samples_list = []
            shms = []
            mapfiles = []

            for raw in raw_rf:
                shm = ipc.SharedMemory(raw)
                mapfile = mmap.mmap(shm.fd, shm.size)

                samples_list.append(np.frombuffer(mapfile, dtype=np.complex64))

                shms.append(shm)
                mapfiles.append(mapfile)

            param['data'] = np.concatenate(samples_list)

            param['rx_sample_rate'] = np.float32(data_parsing.rx_rate)

            total_ants = self.options.main_antenna_count + self.options.intf_antenna_count
            param['num_samps'] = np.uint32(len(samples_list[0])/total_ants)

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
            for shm, mapfile in zip(shms, mapfiles):
                shm.close_fd()
                shm.unlink()
                mapfile.close()

        def write_tx_data():
            """
            Writes out the tx samples and metadata for debugging purposes.
            Does not use same parameters of other writes.

            """
            tx_data = None
            for meta in integration_meta.sequences:
                if meta.HasField('tx_data'):
                    tx_data = TX_TEMPLATE.copy()
                    break


            if tx_data is not None:
                for meta in integration_meta.sequences:
                    tx_data['tx_rate'].append(meta.tx_data.txrate)
                    tx_data['tx_centre_freq'].append(meta.tx_data.txctrfreq)
                    tx_data['pulse_sequence_timing_us'].append(
                        meta.tx_data.pulse_sequence_timing_us)
                    tx_data['pulse_offset_error_us'].append(meta.tx_data.pulse_offset_error_us)
                    tx_data['dm_rate'].append(meta.tx_data.dmrate)
                    tx_data['dm_rate_error'].append(meta.tx_data.dmrate_error)

                    tx_samples = []
                    decimated_tx_samples = []
                    decimated_tx_antennas = []
                    tx_antennas = []

                    for ant in meta.tx_data.tx_samples:
                        tx_antennas.append(ant.tx_antenna_number)
                        real = np.array(ant.real, dtype=np.float32)
                        imag = np.array(ant.imag, dtype=np.float32)

                        cmplx = np.array(real + 1j*imag, dtype=np.complex64)
                        tx_samples.append(cmplx)


                    for ant in meta.tx_data.decimated_tx_samples:
                        decimated_tx_antennas.append(ant.tx_antenna_number)
                        real = np.array(ant.real, dtype=np.float32)
                        imag = np.array(ant.imag, dtype=np.float32)

                        cmplx = np.array(real + 1j*imag, dtype=np.complex64)
                        decimated_tx_samples.append(cmplx)

                    tx_data['tx_antennas'].append(tx_antennas)
                    tx_data['decimated_tx_antennas'].append(decimated_tx_antennas)
                    tx_data['tx_samples'].append(tx_samples)
                    tx_data['decimated_tx_samples'].append(decimated_tx_samples)


                tx_data['tx_antennas'] = np.array(tx_data['tx_antennas'], dtype=np.uint32)
                tx_data['decimated_tx_antennas'] = np.array(tx_data['decimated_tx_antennas'],
                                                            dtype=np.uint32)
                tx_data['tx_samples'] = np.array(tx_data['tx_samples'], dtype=np.complex64)
                tx_data['decimated_tx_samples'] = np.array(tx_data['decimated_tx_samples'],
                                                           dtype=np.complex64)

                name = dataset_name.replace('{sliceid}.', '').format(dformat='txdata')
                output_file = dataset_location.format(name=name)

                write_file(output_file, tx_data, self.tx_data_two_hr_name)



        parameters_holder = {}
        for meta in integration_meta.sequences:
            for rx_freq in meta.rxchannel:
                parameters = DATA_TEMPLATE.copy()
                parameters['borealis_git_hash'] = self.git_hash.decode('utf-8')
                parameters['experiment_id'] = np.int64(integration_meta.experiment_id)
                parameters['experiment_name'] = integration_meta.experiment_name
                parameters['experiment_comment'] = integration_meta.experiment_comment
                parameters['slice_comment'] = rx_freq.slice_comment
                parameters['num_slices'] = len(integration_meta.sequences) * len(meta.rxchannel)
                parameters['station'] = self.options.site_id
                parameters['num_sequences'] = integration_meta.num_sequences
                parameters['num_ranges'] = np.uint32(rx_freq.num_ranges)
                parameters['range_sep'] = np.float32(rx_freq.range_sep)
                #time to first range and back. convert to meters, div by c then convert to us
                rtt = (rx_freq.first_range * 2 * 1.0e3 / speed_of_light) * 1.0e6
                parameters['first_range_rtt'] = np.float32(rtt)
                parameters['first_range'] = np.float32(rx_freq.first_range)
                parameters['rx_sample_rate'] = data_parsing.output_sample_rate # this applies to pre-bf and bfiq
                parameters['scan_start_marker'] = integration_meta.scan_flag # Should this change to scan_start_marker?
                parameters['int_time'] = np.float32(integration_meta.integration_time)
                parameters['tx_pulse_len'] = np.uint32(rx_freq.pulse_len)
                parameters['tau_spacing'] = np.uint32(rx_freq.tau_spacing)
                parameters['main_antenna_count'] = np.uint32(len(rx_freq.rx_main_antennas))
                parameters['intf_antenna_count'] = np.uint32(len(rx_freq.rx_intf_antennas))
                parameters['freq'] = np.uint32(rx_freq.rxfreq)
                parameters['rx_center_freq'] = integration_meta.rx_centre_freq # Sorry, we'll convert to US English here
                parameters['samples_data_type'] = "complex float"
                parameters['pulses'] = np.array(rx_freq.ptab.pulse_position, dtype=np.uint32)
                parameters['pulse_phase_offset'] = np.array(rx_freq.pulse_phase_offsets.pulse_phase, dtype=np.float32)
                parameters['data_normalization_factor'] = integration_meta.data_normalization_factor

                lags = []
                for lag in rx_freq.ltab.lag:
                    lags.append([lag.pulse_position[0], lag.pulse_position[1]])

                parameters['lags'] = np.array(lags, dtype=np.uint32)

                parameters['blanked_samples'] = np.array(meta.blanks, dtype=np.uint32)
                parameters['sqn_timestamps'] = data_parsing.timestamps

                parameters['beam_nums'] = []
                parameters['beam_azms'] = []
                for beam in rx_freq.beams:
                    parameters['beam_nums'].append(np.uint32(beam.beamnum))
                    parameters['beam_azms'].append(beam.beamazimuth)

                parameters['noise_at_freq'] = [0.0] * integration_meta.num_sequences # TODO update. should come from data_parsing

                # num_samps, antenna_arrays_order, data_descriptors, data_dimensions, data
                # correlation_descriptors, correlation_dimensions, main_acfs, intf_acfs, xcfs
                # all get set within the separate write functions.

                parameters_holder[rx_freq.slice_id] = parameters

        if write_rawacf:

            write_correlations(copy.deepcopy(parameters_holder))
            pass

        if write_bfiq and data_parsing.bfiq_available:
            write_bfiq_params(copy.deepcopy(parameters_holder))

        if write_antenna_iq and data_parsing.antenna_iq_available:
            write_antenna_iq_params(copy.deepcopy(parameters_holder))

        if write_raw_rf:
            # Just need first available slice paramaters.
            one_slice_params = copy.deepcopy(next(iter(parameters_holder.values())))
            write_raw_rf_params(one_slice_params)
        else:
            for rf_samples_location in data_parsing.rawrf_locations:
                shm = ipc.SharedMemory(rf_samples_location)
                shm.close_fd()
                shm.unlink()

        if write_tx:
            write_tx_data()


        end = time.time()
        printing("Time to write: {} ms".format((end-start)*1000))



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
    sockets = so.create_sockets([options.dw_to_dsp_identity, options.dw_to_radctrl_identity,
                                 options.dw_to_rt_identity],
                                options.router_address)

    dsp_to_data_write = sockets[0]
    radctrl_to_data_write = sockets[1]
    realtime_to_data_write = sockets[2]

    poller = zmq.Poller()
    poller.register(dsp_to_data_write, zmq.POLLIN)
    poller.register(radctrl_to_data_write, zmq.POLLIN)

    if __debug__:
        printing("Socket connected")

    data_parsing = ParseData()
    final_integration = sys.maxsize
    integration_meta = None

    current_experiment = None
    data_write = None
    first_time = True
    expected_sqn_num = 0
    queued_sqns = []
    abandon_list = []
    while True:

        try:
            socks = dict(poller.poll())
        except KeyboardInterrupt:
            sys.exit()

        if radctrl_to_data_write in socks and socks[radctrl_to_data_write] == zmq.POLLIN:
            data = so.recv_bytes(radctrl_to_data_write, options.radctrl_to_dw_identity, printing)

            integration_meta = datawritemetadata_pb2.IntegrationTimeMetadata()
            integration_meta.ParseFromString(data)

            final_integration = integration_meta.last_seqn_num

        if dsp_to_data_write in socks and socks[dsp_to_data_write] == zmq.POLLIN:
            data = so.recv_bytes_from_any_iden(dsp_to_data_write)

            processed_data = processeddata_pb2.ProcessedData()
            processed_data.ParseFromString(data)

            queued_sqns.append(processed_data)
            # Check if any data processing finished out of order.

            if processed_data.sequence_num != expected_sqn_num:
                continue

            sorted_q = sorted(queued_sqns, key=lambda x:x.sequence_num)

            # This is needed to check that if we have a backlog, there are no more
            # skipped sequence numbers we are still waiting for.
            break_now = False
            for i, pd in enumerate(sorted_q):
                if pd.sequence_num != expected_sqn_num + i:
                    expected_sqn_num += i
                    break_now = True
                    break
            if break_now:
                if len(sorted_q) > 20:
                    #TODO error out correctly
                    printing("Lost sequence #{}. Exiting.".format(expected_sqn_num))
                    sys.exit()
                continue

            expected_sqn_num = sorted_q[-1].sequence_num + 1

            for pd in sorted_q:
                if not first_time:
                    if data_parsing.sequence_num == final_integration:

                        if integration_meta.experiment_name != current_experiment:
                            data_write = DataWrite(options)
                            current_experiment = integration_meta.experiment_name

                        kwargs = dict(write_bfiq=args.enable_bfiq,
                                               write_antenna_iq=args.enable_antenna_iq,
                                               write_raw_rf=args.enable_raw_rf,
                                               write_tx=args.enable_tx,
                                               file_ext=args.file_type,
                                               integration_meta=integration_meta,
                                               data_parsing=data_parsing,
                                               rt_dw={"socket":realtime_to_data_write,
                                                        "iden":options.rt_to_dw_identity},
                                               write_rawacf=args.enable_raw_acfs)
                        thread = threading.Thread(target=data_write.output_data, kwargs=kwargs)
                        thread.daemon = True
                        thread.start()
                        data_parsing = ParseData()

                first_time = False

                start = time.time()
                data_parsing.update(pd)
                end = time.time()
                printing("Time to parse: {} ms".format((end-start)*1000))

            queued_sqns = []


if __name__ == '__main__':

    main()
