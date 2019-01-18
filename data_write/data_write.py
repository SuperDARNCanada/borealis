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
from scipy.constants import speed_of_light


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
    "timestamp_of_write" : None, # Timestamp of when the record was written. Seconds since epoch.
    "experiment_id" : None, # Number used to identify experiment.
    "experiment_string" : None, # Name of the experiment file.
    "station" : None, # Three letter radar identifier.
    "num_sequences": None, # Number of sampling periods in the integration time.
    "first_range_rtt" : None, # Round trip time of flight to first range in microseconds.
    "first_range" : None, # Distance to first range in km.
    "rx_sample_rate" : None, # Sampling rate of the samples being written to file in Hz.
    "range_separation" : None, # Range gate separation in km.
    "scan_start_marker" : None, # Designates if the record is the first in a scan.
    "int_time" : None, # Integration time in seconds.
    "tx_pulse_len" : None, # Length of the pulse in microseconds.
    "tau_spacing" : None, # The minimum spacing between pulses, spacing between pulses is always a
                          # multiple of this in microseconds.
    "num_pulses" : None, # Number of pulses in sequence.
    "num_lags" : None, # Number of lags in the lag table.
    "num_ranges" : None, # Number of ranges.
    "main_antenna_count" : None, # Number of main array antennas.
    "intf_antenna_count" : None, # Number of interferometer array antennas.
    "freq" : None, # The frequency used for this experiment slice in kHz.
    "tx_center_freq" : None, # The center frequency the USRP uses for digital up conversion in Hz.
    "rx_center_freq" : None, # The center frequency the USRP uses for digital down conversion in Hz.
    "comment" : None, # Additional text comment that describes the slice.
    "num_samps" : None, # Number of samples in the sampling period.
    "antenna_arrays_order" : None, # States what order the data is in. Describes the data layout.
    "samples_data_type" : None, # C data type of the samples such as complex float.
    "pulses" : None, # The pulse sequence in units of the tau_spacing.
    "lags" : None, # The lags created from combined pulses.
    "blanked_samples" : None, # Samples that have been blanked during TR switching.
    "sqn_timestamps" : None, # A list of GPS timestamps of the beginning of transmission for each
                             # sampling period in the integration time. Seconds since epoch.
    "beam_nums" : None, # A list of beam numbers used in this slice.
    "beam_azms" : None, # A list of the beams azimuths for each beam in degrees.
    "data_descriptors" : None, # Denotes what each data dimension represents.
    "data_dimensions" : None, # The dimensions in which to reshape the data.
    "data" : None # A contiguous set of data.
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

        self._bfiq_available = False
        self._bfiq_accumulator = self.nested_dict()

        self._pre_bfiq_accumulator = self.nested_dict()
        self._pre_bfiq_available = False

        self._slice_ids = set()
        self._timestamps = []

        self._rawrf_locations = []

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



    def parse_pre_bfiq(self):
        """
        Parses out any pre-beamformed IQ if available. Runs on every processeddata
        packet(contains all sampling period data). All variables are captured from outer scope.
        """

        self._pre_bfiq_accumulator['data_descriptors'] = ['num_antennas', 'num_sequences',
                                                          'num_samps']
        # Iterate over every data set, one data set per slice
        for data_set in self.processed_data.outputdataset:
            slice_id = data_set.slice_id

            # non beamformed IQ samples are available
            if data_set.debugsamples:
                self._pre_bfiq_available = True

                # Loops over all filter stage data, one set per stage
                for debug_samples in data_set.debugsamples:
                    stage_name = debug_samples.stagename

                    if stage_name not in self._pre_bfiq_accumulator[slice_id]:
                        self._pre_bfiq_accumulator[slice_id][stage_name] = collections.OrderedDict()

                    pre_bfiq_stage = self._pre_bfiq_accumulator[slice_id][stage_name]
                    # Loops over antenna data within stage
                    for ant_num, ant_data in enumerate(debug_samples.antennadata):
                        ant_str = "antenna_{0}".format(ant_num)

                        cmplx = np.empty(len(ant_data.antennasamples), dtype=np.complex64)
                        pre_bfiq_stage["num_samps"] = len(ant_data.antennasamples)

                        for i, sample in enumerate(ant_data.antennasamples):
                            cmplx[i] = sample.real + 1.0j * sample.imag

                        if ant_str not in pre_bfiq_stage:
                            pre_bfiq_stage[ant_str] = {}

                        if 'data' not in pre_bfiq_stage[ant_str]:
                            pre_bfiq_stage[ant_str]['data'] = cmplx
                        else:
                            arr = pre_bfiq_stage[ant_str]
                            arr['data'] = np.concatenate((arr['data'], cmplx))

    def update(self, data):
        """ Parses the protobuf and updates the accumulator fields with the new data.

        Args:
            data (bytes): Serialized ProcessedData protobuf
        """
        self.processed_data = processeddata_pb2.ProcessedData()
        self.processed_data.ParseFromString(data)

        self._timestamps.append(self.processed_data.sequence_start_time)

        for data_set in self.processed_data.outputdataset:
            self._slice_ids.add(data_set.slice_id)

        self._rawrf_locations.append(self.processed_data.rf_samples_location)

        self.parse_bfiq()
        self.parse_pre_bfiq()



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
    def pre_bfiq_available(self):
        """ Gets the pre-bfiq available flag.

        Returns:
            TYPE: Bool
        """
        return self._pre_bfiq_available

    @property
    def bfiq_accumulator(self):
        """ Returns the nested default dictionary with complex stage data for each antenna array as
        well as some metadata.

        Returns:
            TYPE: Nested default dict: Contains beamform data for each slice.
        """
        return self._bfiq_accumulator

    @property
    def pre_bfiq_accumulator(self):
        """Returns the nested default dictionary with complex stage data for each antenna as well
        as some metadata for each slice.

        Returns:
            Nested default dict: Contains stage data for each antenna and slice.
        """
        return self._pre_bfiq_accumulator

    @property
    def timestamps(self):
        """Return the python list of sequence timestamps (when the sampling period begins)
        from the processsed data packets

        Returns:
            python list: A list of sequence timestamps from the processed data packets
        """
        return self._timestamps

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

    def output_data(self, write_bfiq, write_pre_bfiq, write_raw_rf, write_tx, file_ext,
                    integration_meta, data_parsing, write_rawacf=True):
        """
        Parse through samples and write to file.

        A file will be created using the file extention for each requested data product.

        :param write_bfiq:          Should beamformed IQ be written to file? Bool.
        :param write_pre_bfiq:      Should pre-beamformed IQ be written to file? Bool.
        :param write_raw_rf:        Should raw rf samples be written to file? Bool.
        :param file_ext:            Type of file extention to use. String
        :param integration_meta:    Metadata from radar control about integration period. Protobuf
        :param data_parsing:        All parsed and concatenated data from integration period stored
                                    in DataParsing object.
        :param write_rawacf:        Should rawacfs be written to file? Bool, default True.
        """


        start = time.time()
        if file_ext not in ['hdf5', 'json', 'dmap']:
            raise ValueError("File format selection required (hdf5, json, dmap), none given")

        # Format the name and location for the dataset
        write_time = datetime.datetime.utcnow()
        time_now = datetime.datetime.fromtimestamp(data_parsing.timestamps[0])

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
                full_two_hr_file = "{0}/{1}.hdf5".format(dataset_directory, two_hr_file_with_type)

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
                os.remove(tmp_file)

            elif file_ext == 'json':
                self.write_json_file(tmp_file, final_data_dict)
            elif file_ext == 'dmap':
                self.write_dmap_file(tmp_file, final_data_dict)

        def do_acf():
            """
            Parses out any possible ACF data from protobuf and writes to file. All variables are
            captured.

            """

            # TODO



        def write_bfiq_params(parameters_holder):
            """
            write out any possible beamformed IQ data that has been parsed. Adds additional slice
            info to each parameter dict. Some variables are captured from outer scope.

            Args:
                parameters_holder (Dict): A dict that hold dicts of parameters for each slice.

            """

            bfiq = data_parsing.bfiq_accumulator

            # Pop these off so we dont include them in later iteration.
            data_descriptors = bfiq.pop('data_descriptors', None)

            for slice_id in bfiq:
                parameters = parameters_holder[slice_id]

                parameters['data_descriptors'] = data_descriptors
                parameters['antenna_arrays_order'] = []

                flattened_data = []
                if "main" in bfiq[slice_id]:
                    parameters['antenna_arrays_order'].append("main")
                    flattened_data.append(bfiq[slice_id]['main']['data'])
                if "intf" in bfiq[slice_id]:
                    parameters['antenna_arrays_order'].append("intf")
                    flattened_data.append(bfiq[slice_id]['intf']['data'])

                flattened_data = np.concatenate(flattened_data)
                parameters['data'] = flattened_data

                parameters['num_samps'] = np.uint32(bfiq[slice_id].pop('num_samps', None))
                parameters['data_dimensions'] = np.array([len(bfiq[slice_id].keys()),
                                                          integration_meta.nave,
                                                          len(parameters['beam_nums']),
                                                          parameters['num_samps']], dtype=np.uint32)


            for slice_id, parameters in parameters_holder.items():
                name = dataset_name.format(sliceid=slice_id, dformat="bfiq")
                output_file = dataset_location.format(name=name)

                two_hr_file_with_type = self.slice_filenames[slice_id].format(ext="bfiq")

                write_file(output_file, parameters, two_hr_file_with_type)


        def write_pre_bfiq_params(parameters_holder):
            """
            Writes out any pre-beamformed IQ that has been parsed. Adds additional slice info
            to each paramater dict. Some variables are captured from outer scope.

            Args:
                parameters_holder (Dict): A dict that hold dicts of parameters for each slice.

            """

            pre_bfiq = data_parsing.pre_bfiq_accumulator

            # Pop these so we don't include them in later iteration.
            data_descriptors = pre_bfiq.pop('data_descriptors', None)

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
            for slice_id in pre_bfiq:
                final_data_params[slice_id] = {}

                for stage in pre_bfiq[slice_id]:
                    parameters = parameters_holder[slice_id].copy()

                    parameters['data_descriptors'] = data_descriptors
                    parameters['num_samps'] = np.uint32(
                        pre_bfiq[slice_id][stage].pop('num_samps', None))


                    parameters['antenna_arrays_order'] = rx_main_antennas[slice_id] +\
                                                         rx_intf_antennas[slice_id]

                    num_ants = len(parameters['antenna_arrays_order'])

                    parameters['data_dimensions'] = np.array([num_ants,
                                                              integration_meta.nave,
                                                              parameters['num_samps']],
                                                             dtype=np.uint32)


                    data = []
                    for k, data_dict in pre_bfiq[slice_id][stage].items():
                        if k in parameters['antenna_arrays_order']:
                            data.append(data_dict['data'])

                    flattened_data = np.concatenate(data)
                    parameters['data'] = flattened_data

                    final_data_params[slice_id][stage] = parameters


            for slice_id, slice_ in final_data_params.items():
                for stage, params in slice_.items():
                    name = dataset_name.format(sliceid=slice_id, dformat="{}_iq".format(stage))
                    output_file = dataset_location.format(name=name)

                    ext = "{}_iq".format(stage)
                    two_hr_file_with_type = self.slice_filenames[slice_id].format(ext=ext)

                    write_file(output_file, params, two_hr_file_with_type)


        def write_raw_rf_params(param, freqs):
            """
            Opens the shared memory location in the protobuf and writes the samples out to file.
            Write medium must be able to sustain high write bandwidth. Shared memory is destroyed
            after write. Some variables are captured in scope. Some new members are added to the
            parameter dict.


            Args:
                param (Dict): A dict of parameters to write. Some will be removed.
                freqs (List): A list of the slice frequencies contained in this data.


            """
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

            param['rx_sample_rate'] = np.float32(self.options.rx_sample_rate)

            total_ants = self.options.main_antenna_count + self.options.intf_antenna_count
            param['num_samps'] = np.uint32(len(samples_list[0])/total_ants)

            param['data_descriptors'] = ["num_sequences", "num_antennas", "num_samps"]
            param['data_dimensions'] = np.array([param['num_sequences'], total_ants,
                                                 param['num_samps']],
                                                dtype=np.uint32)
            param['main_antenna_count'] = np.uint32(self.options.main_antenna_count)
            param['intf_antenna_count'] = np.uint32(self.options.intf_antenna_count)
            param['rx_frequencies'] = np.array(freqs, dtype=np.uint32)
            # These fields don't make much sense when working with the raw rf. It's expected
            # that the user will have knowledge of what they are looking for when working with
            # this data.
            unneeded_fields = ['first_range', 'first_range_rtt', 'tx_pulse_len', 'tau_spacing',
                               'num_pulses', 'num_lags', 'freq', 'pulses', 'lags',
                               'blanked_samples', 'beam_nums', 'beam_azms', 'antenna_arrays_order',
                               'range_separation', 'nrang']

            for field in unneeded_fields:
                param.pop(field, None)


            write_file(output_file, param, self.raw_rf_two_hr_name)

            # Can only close mapped memory after its been written to disk.
            for shm, mapfile in zip(shms, mapfiles):
                shm.close_fd()
                shm.unlink()
                mapfile.close()

        def write_tx_data():
            """Writes out the tx samples and metadata for debugging purposes.

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

                parameters['timestamp_of_write'] = (write_time - epoch).total_seconds()
                parameters['experiment_id'] = np.uint32(integration_meta.experiment_id)
                parameters['experiment_string'] = integration_meta.experiment_string
                parameters['station'] = self.options.site_id
                parameters['num_sequences'] = integration_meta.nave

                #time to first range and back. convert to meters, div by c then convert to us
                rtt = (rx_freq.frang * 2 * 1.0e3 / speed_of_light) * 1.0e6
                parameters['first_range_rtt'] = np.float32(rtt)
                parameters['first_range'] = np.float32(rx_freq.frang)
                parameters['rx_sample_rate'] = np.float32(self.options.third_stage_sample_rate)
                parameters['range_separation'] = np.float32(rx_freq.rsep)
                parameters['scan_start_marker'] = integration_meta.scan_flag # Should this change to scan_start_marker?
                parameters['int_time'] = np.float32(integration_meta.integration_time)
                parameters['tx_pulse_len'] = np.uint32(rx_freq.pulse_len)
                parameters['tau_spacing'] = np.uint32(rx_freq.tau_spacing)
                parameters['num_pulses'] = np.uint32(len(rx_freq.ptab.pulse_position))
                parameters['num_ranges'] = np.uint32(rx_freq.nrang)
                parameters['num_lags'] = np.uint32(len(rx_freq.ltab.lag))
                parameters['main_antenna_count'] = np.uint32(len(rx_freq.rx_main_antennas))
                parameters['intf_antenna_count'] = np.uint32(len(rx_freq.rx_intf_antennas))
                parameters['freq'] = np.uint32(rx_freq.rxfreq)
                parameters['tx_center_freq'] = np.float32(integration_meta.tx_center_freq)
                parameters['rx_center_freq'] = np.float32(integration_meta.rx_center_freq)
                parameters['comment'] = rx_freq.comment
                parameters['samples_data_type'] = "complex float"
                parameters['pulses'] = np.array(rx_freq.ptab.pulse_position, dtype=np.uint32)

                lags = []
                for lag in rx_freq.ltab.lag:
                    lags.append([lag.pulse_position[0], lag.pulse_position[1]])

                parameters['lags'] = np.array(lags, dtype=np.uint32)

                parameters['blanked_samples'] = np.array(meta.blanks, dtype=np.uint32)

                parameters['beam_nums'] = []
                parameters['beam_azms'] = []

                for beam in rx_freq.beams:
                    parameters['beam_nums'].append(np.uint32(beam.beamnum))
                    parameters['beam_azms'].append(beam.beamazimuth)

                parameters['sqn_timestamps'] = data_parsing.timestamps

                parameters_holder[rx_freq.slice_id] = parameters

        # Use multiprocessing to speed up writing. Each data type can be parsed and written by a
        # separate process in order to parallelize the work.
        procs = []

        if write_rawacf:
            procs.append(mp.Process(target=do_acf))

        if write_bfiq and data_parsing.bfiq_available:
            procs.append(mp.Process(target=write_bfiq_params, args=(parameters_holder.copy(), )))

        if write_pre_bfiq and data_parsing.pre_bfiq_available:
            procs.append(mp.Process(target=write_pre_bfiq_params,
                                    args=(parameters_holder.copy(), )))

        if write_raw_rf:
            # Just need first available slice paramaters.
            freqs = []
            for meta in integration_meta.sequences:
                for rx_freq in meta.rxchannel:
                    freqs.append(rx_freq.rxfreq)

            one_slice_params = next(iter(parameters_holder.values())).copy()
            procs.append(mp.Process(target=write_raw_rf_params, args=(one_slice_params, freqs)))
        else:
            for rf_samples_location in data_parsing.rawrf_locations:
                shm = ipc.SharedMemory(rf_samples_location)
                shm.close_fd()
                shm.unlink()

        if write_tx:
            procs.append(mp.Process(target=write_tx_data))

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()

        end = time.time()
        printing("Time to write: {} ms".format((end-start)*1000))



def main():

    parser = ap.ArgumentParser(description='Write processed SuperDARN data to file')
    parser.add_argument('--file-type', help='Type of output file: hdf5, json, or dmap',
                        default='hdf5')
    parser.add_argument('--enable-bfiq', help='Enable beamformed iq writing',
                        action='store_true')
    parser.add_argument('--enable-pre-bfiq', help='Enable individual antenna iq writing',
                        action='store_true')
    parser.add_argument('--enable-raw-rf', help='Save raw, unfiltered IQ samples. Requires HDF5.',
                        action='store_true')
    parser.add_argument('--enable-tx', help='Save tx samples and metadata. Requires HDF5.',
                        action='store_true')
    args = parser.parse_args()


    options = dwo.DataWriteOptions()
    sockets = so.create_sockets([options.dw_to_dsp_identity, options.dw_to_radctrl_identity],
                                options.router_address)

    dsp_to_data_write = sockets[0]
    radctrl_to_data_write = sockets[1]

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
            data = so.recv_bytes(dsp_to_data_write, options.dsp_to_dw_identity, printing)

            if not first_time:
                if data_parsing.sequence_num == final_integration:

                    if integration_meta.experiment_string != current_experiment:
                        data_write = DataWrite(options)
                        current_experiment = integration_meta.experiment_string

                    kwargs = dict(write_bfiq=args.enable_bfiq,
                                           write_pre_bfiq=args.enable_pre_bfiq,
                                           write_raw_rf=args.enable_raw_rf,
                                           write_tx=args.enable_tx,
                                           file_ext=args.file_type,
                                           integration_meta=integration_meta,
                                           data_parsing=data_parsing,
                                           write_rawacf=False)
                    thread = threading.Thread(target=data_write.output_data, kwargs=kwargs)
                    thread.daemon = True
                    thread.start()
                    data_parsing = ParseData()


            first_time = False

            start = time.time()
            data_parsing.update(data)
            end = time.time()
            printing("Time to parse: {} ms".format((end-start)*1000))






if __name__ == '__main__':

    main()
