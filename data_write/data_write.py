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
import multiprocessing as mp
import subprocess as sp
import argparse as ap
import numpy as np
import deepdish as dd
import posix_ipc as ipc
import zmq
import data_file_classes

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
    :param msg: The string to format nicely for printingw
    """
    DATA_WRITE = "\033[96m" + "DATA WRITE: " + "\033[0m"
    sys.stdout.write(DATA_WRITE + msg + "\n")


def write_json_file(filename, data_dict):
    """
    Write out data to a json file. If the file already exists it will be overwritten.
    :param filename: The path to the file to write out. String
    :param data_dict: Python dictionary to write out to the JSON file.
    """
    with open(filename, 'w+') as f:
        f.write(json.dumps(data_dict))


def write_hdf5_file(filename, data_dict, dt_str):
    """
    Write out data to an HDF5 file. If the file already exists it will be overwritten.
    :param filename: The path to the file to write out. String
    :param data_dict: Python dictionary to write out to the HDF5 file.
    """

    def convert_to_numpy(dd):
        """Converts an input dictionary type into numpy array. Recursive.

        Args:
            dd (Python dictionary): Dictionary to convert to numpy array, can contain nested dicts.
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

    warnings.simplefilter("ignore") #ignore NaturalNameWarning
    dd.io.save(filename, time_stamped_dd, compression=None)


def write_dmap_file(filename, data_dict):
    """
    Write out data to a dmap file. If the file already exists it will be overwritten.
    :param filename: The path to the file to write out. String
    :param data_dict: Python dictionary to write out to the dmap file.
    """
    # TODO: Complete this by parsing through the dictionary and write out to proper dmap format
    pass

class ParseData(object):
    """Parse protobuf data from sockets into file writable types, such as hdf5, json, dmap, etc.

    Attributes:
        nested_dict (Python default nested dictionary): alias to a nested defaultdict
        processed_data (Protobuf packet): Contains a packet from a socket in protobuf_pb2 format
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

    def do_bfiq(self):
        """
        Parses out any possible beamformed IQ data from the protobuf and writes it to file.
        All variables are captured.

        """

        self._bfiq_accumulator['data_descriptors'] = ['num_antenna_arrays', 'num_sequences',
                                                      'num_beams', 'num_samps']

        for data_set in self.processed_data.outputdataset:
            slice_id = data_set.slice_id

            # Find out what is available in the data to determine what to write out
            if data_set.beamformedsamples:
                self._bfiq_available = True

                for beam in data_set.beamformedsamples:
                    cmplx = np.empty(len(beam.mainsamples), dtype=np.complex64)
                    self._bfiq_accumulator[slice_id]['num_samps'] = len(beam.mainsamples)

                    def add_samples(samples, antenna_arr_type):
                        """Summary TODO

                        Args:
                            samples (TYPE): Description TODO
                            antenna_arr_type (TYPE): Description TODO
                        """

                        for i, sample in enumerate(samples):
                            cmplx[i] = sample.real + 1.0j * sample.imag

                        # only need to test either real or imag to see if data exists
                        if not 'data' in self._bfiq_accumulator[slice_id][antenna_arr_type]:
                            self._bfiq_accumulator[slice_id][antenna_arr_type]['data'] = cmplx
                        else:
                            arr = self._bfiq_accumulator[slice_id][antenna_arr_type]
                            arr['data'] = np.concatenate((arr['data'], cmplx))

                    add_samples(beam.mainsamples, "main")

                    if beam.intfsamples:
                        add_samples(beam.intfsamples, "intf")



    def do_pre_bfiq(self):
        """
        Parses out any pre-beamformed IQ if available and writes it out to file.
        All variables are captured.
        """
        self._pre_bfiq_accumulator['data_descriptors'] = ['num_antenna_arrays', 'num_sequences',
                                                          'num_antennas', 'num_samps']
        # Iterate over every data set, one data set per slice
        for data_set in self.processed_data.outputdataset:
            slice_id = data_set.slice_id

            # non beamformed IQ samples are available
            if data_set.debugsamples:
                self._pre_bfiq_available = True

                # Loops over all filter stage data, one set per stage
                for debug_samples in data_set.debugsamples:
                    stage_name = debug_samples.stagename
                    self._pre_bfiq_accumulator[slice_id][stage_name] = collections.OrderedDict()
                    pre_bfiq_stage = self._pre_bfiq_accumulator[slice_id][stage_name]
                    # Loops over antenna data within stage
                    for ant_num, ant_data in enumerate(debug_samples.antennadata):
                        ant_str = "antenna_{0}".format(ant_num)

                        cmplx = np.empty(len(ant_data.antennasamples), dtype=np.complex64)
                        pre_bfiq_stage["num_samps"] = len(ant_data.antennasamples)

                        for i, sample in enumerate(ant_data.antennasamples):
                            cmplx[i] = sample.real + 1.0j * sample.imag

                        pre_bfiq_stage[ant_str] = {}
                        if not 'data' in pre_bfiq_stage[ant_str]:
                            pre_bfiq_stage[ant_str]['data'] = cmplx
                        else:
                            arr = pre_bfiq_stage[ant_str]
                            arr['data'] = np.concatenate((arr['data'], cmplx))


        # if pre_bf_iq_available:
        #     name = dataset_name.format(dformat="iq")
        #     output_file = dataset_location.format(name=name)

        #     two_hr_file_with_type = two_hr_file.format(ext="iq")
        #     write_file(output_file, iq_pre_bf_data_dict, two_hr_file_with_type)

    def update(self, data):
        """ TODO

        Args:
            data (TYPE): Description TODO
        """
        self.processed_data = processeddata_pb2.ProcessedData()
        self.processed_data.ParseFromString(data)

        self._timestamps.append(self.processed_data.sequence_start_time)

        for data_set in self.processed_data.outputdataset:
            self._slice_ids.add(data_set.slice_id)

        self._rawrf_locations.append(self.processed_data.rf_samples_location)

        procs = []

        self.do_bfiq()
        self.do_pre_bfiq()

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()


    @property
    def sequence_num(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return self.processed_data.sequence_num

    @property
    def bfiq_available(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return self._bfiq_available

    @property
    def pre_bfiq_available(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return self._pre_bfiq_available

    @property
    def bfiq_accumulator(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return self._bfiq_accumulator

    @property
    def pre_bfiq_accumulator(self):
        """Returns the nested default dictionary with complex stage data for each antenna as well
        as some metadata

        Returns:
            Nested default dict: Contains stage data for each antenna
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
        return self._rawrf_locations

    # def parse_acf(self):
    #     """
    #     Parses out any possible ACF data from protobuf and writes to file. All variables are
    #     captured.

    #     """
    #     rawacf_available = False
    #     rawacf_data_dict = self.nested_dict()

    #     for freq_num, data_set in enumerate(self.processed_data.outputdataset):
    #         freq_str = "frequency_{0}".format(freq_num)

    #         # Find out what is available in the data to determine what to write out
    #         # Main acfs were calculated
    #         if len(data_set.mainacf) > 0:
    #             rawacf_available = True
    #             rawacf_data_dict[freq_str]['mainacf'] = {'real': [], 'imag': []}
    #             for complex_sample in data_set.mainacf:
    #                 rawacf_data_dict[freq_str]['mainacf']['real'].append(complex_sample.real)
    #                 rawacf_data_dict[freq_str]['mainacf']['imag'].append(complex_sample.imag)

    #         # Interferometer acfs were calculated
    #         if len(data_set.intacf) > 0:
    #             rawacf_available = True
    #             rawacf_data_dict[freq_str]['intacf'] = {'real': [], 'imag': []}
    #             for complex_sample in data_set.intacf:
    #                 rawacf_data_dict[freq_str]['intacf']['real'].append(complex_sample.real)
    #                 rawacf_data_dict[freq_str]['intacf']['imag'].append(complex_sample.imag)

    #         # Cross correlations were calculated
    #         if len(data_set.xcf) > 0:
    #             rawacf_available = True
    #             rawacf_data_dict[freq_str]['xcf'] = {'real': [], 'imag': []}
    #             for complex_sample in data_set.xcf:
    #                 rawacf_data_dict[freq_str]['xcf']['real'].append(complex_sample.real)
    #                 rawacf_data_dict[freq_str]['xcf']['imag'].append(complex_sample.imag)

    #     if rawacf_available:
    #         name = dataset_name.format(dformat="rawacf")
    #         output_file = dataset_location.format(name=name)

    #         two_hr_file_with_type = two_hr_file.format(ext="rawacf")
    #         write_file(output_file, rawacf_data_dict, two_hr_file_with_type)


    # def do_bfiq():
    #     """
    #     Parses out any possible beamformed IQ data from the protobuf and writes it to file.
    #     All variables are captured.

    #     """

    #     bfiq_available = False
    #     for freq_num, data_set in enumerate(self.processed_data.outputdataset):
    #         slice_id = "{0}".format(data_set.slice_id)

    #         # Find out what is available in the data to determine what to write out
    #         if len(data_set.beamformedsamples) > 0:
    #             bfiq_available = True

    #             for beam in data_set.beamformedsamples:
    #                 beam_str = "beam_{}".format(beam.beamnum)

    #                 real = np.empty(len(beam.mainsamples))
    #                 imag = np.empty(len(beam.mainsamples))

    #                 def add_samples(samples, arr_type):
    #                     for i, sample in enumerate(samples):
    #                         real[i] = sample.real
    #                         imag[i] = sample.imag

    #                     # only need to test either real or imag to see if data exists
    #                     if not 'real' in iq_data_dict[slice_id][beam_str][arr_type]:
    #                         iq_accumulator[slice_id][beam_str][arr_type]['real'] = real
    #                         iq_accumulator[slice_id][beam_str][arr_type]['imag'] = imag
    #                     else:
    #                         arr = iq_accumulator[slice_id][beam_str][arr_type]
    #                         arr['real'] = np.concatenate(arr_type['real'], real)
    #                         arr['imag'] = np.concatenate(arr_type['imag'], imag)

    #                 add_samples(beam.mainsamples, "main")

    #                 if len(beam.intfsamples) > 0:
    #                     add_samples(beam.intfsamples, "intf")







    # def do_raw_rf():
    #     """
    #     Opens the shared memory location in the protobuf and writes the samples out to file.
    #     Write medium must be able to sustain high write bandwidth. Shared memory is destroyed
    #     after write. All variables are captured.
    #     """
    #     raw_rf_dict = nested_dict()
    #     name = dataset_name.format(dformat='rawrf')
    #     output_file = dataset_location.format(name=name)

    #     shm = ipc.SharedMemory(self.processed_data.rf_samples_location)
    #     mapfile = mmap.mmap(shm.fd,shm.size)

    #     rf_samples = np.frombuffer(mapfile,dtype=np.complex64)

    #     total_antennas = self.options.main_antenna_count + self.options.intf_antenna_count
    #     rf_samples = np.reshape(rf_samples,(total_antennas,-1))
    #     for ant in range(total_antennas):
    #         ant_str = "antenna_{0}".format(ant)
    #         raw_rf_dict[ant_str] = rf_samples[ant]

    #     write_file(output_file, raw_rf_dict)

    #     shm.close_fd()
    #     shm.unlink()
    #     mapfile.close()



class DataWrite(object):
    """This class contains the functions used to write out processed data to files.

    """
    def __init__(self, data_write_options):
        super(DataWrite, self).__init__()
        self.options = data_write_options

        self.two_hr_format = "{dt}.{site}.{sliceid}.{{ext}}"

        self.slice_filenames = {}

        self.git_hash = sp.check_output("git describe --always".split()).strip()

        self.next_boundary = None


    def output_data(self, write_bfiq, write_pre_bfiq, write_raw_rf, file_ext, integration_meta,
                    data_parsing, write_rawacf=True):
        """
        Parse through samples and write to file.

        A file will be created using the file extention for each requested data product.

        :param write_bfiq:          Should beamformed IQ be written to file? Bool.
        :param write_pre_bfiq:     Should pre-beamformed IQ be written to file? Bool.
        :param write_raw_rf:        Should raw rf samples be written to file? Bool.
        :param file_ext:            Type of file extention to use. String
        :param integration_meta:    Metadata from radar control about integration period. Protobuf
        :param data_parsing:        All parsed and concatenated data from integration period. Dict
        :param write_rawacf:        Should rawacfs be written to file? Bool, default True.
        """

        if file_ext not in ['hdf5', 'json', 'dmap']:
            raise ValueError("File format selection required (hdf5, json, dmap), none given")

        # Format the name and location for the dataset
        time_now = datetime.datetime.utcnow()
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

        time_now = datetime.datetime.utcnow()

        for slice_id in data_parsing.slice_ids:
            if slice_id not in self.slice_filenames:
                two_hr_str = self.two_hr_format.format(dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                                                       sliceid=slice_id, site=self.options.site_id)
                self.slice_filenames[slice_id] = two_hr_str

                self.next_boundary = two_hr_ceiling(time_now)

        if time_now > self.next_boundary:
            for slice_id in self.slice_filenames.keys():
                two_hr_str = self.two_hr_format.format(dt=time_now.strftime("%Y%m%d.%H%M.%S"),
                                                        sliceid=slice_id,
                                                        site=self.options.site_id)
                self.slice_filenames[slice_id] = two_hr_str
                self.next_boundary = two_hr_ceiling(time_now)



        def write_file(location, final_data_dict, two_hr_file_with_type):
            """
            Writes the final data out to the location based on the type of file extention required

            :param location:                File path and name to write to. String
            :param final_data_dict:         Data dict parsed out from protobuf. Dict
            :param two_hr_file_with_type:   Name of the two hour file with data type added. String

            """
            if not os.path.exists(dataset_directory):
                try:
                    os.makedirs(dataset_directory)
                except os.error:
                    pass

            if file_ext == 'hdf5':
                full_two_hr_file = "{0}/{1}.hdf5".format(dataset_directory, two_hr_file_with_type)

                try:
                    fd = os.open(full_two_hr_file, os.O_CREAT | os.O_EXCL)
                    os.close(fd)
                except FileExistsError:
                    pass

                write_hdf5_file(location, final_data_dict, epoch_milliseconds)

                # use external h5copy utility to move new record into 2hr file.
                cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
                cmd = cmd.format(newfile=location, twohr=full_two_hr_file, dtstr=epoch_milliseconds)
                sp.call(cmd.split())
                os.remove(location)

            elif file_ext == 'json':
                write_json_file(location, final_data_dict)
            elif file_ext == 'dmap':
                write_dmap_file(location, final_data_dict)

        def do_acf():
            """
            Parses out any possible ACF data from protobuf and writes to file. All variables are
            captured.

            """

            if rawacf_available:
                name = dataset_name.format(dformat="rawacf")
                output_file = dataset_location.format(name=name)

                two_hr_file_with_type = two_hr_file.format(ext="rawacf")
                write_file(output_file, rawacf_data_dict, two_hr_file_with_type)


        def do_bfiq():
            """
            Parses out any possible beamformed IQ data from the protobuf and writes it to file.
            All variables are captured.

            """

            bfiq = data_parsing.bfiq_accumulator

            parameters_holder = {}
            for meta in integration_meta.sequences:
                for rx_freq in meta.rxchannel:
                    parameters = data_file_classes.iq_data.copy()

                    parameters['borealis_git_hash'] = self.git_hash.decode('utf-8')

                    parameters['timestamp_of_write'] = (time_now - epoch).total_seconds()
                    parameters['experiment_id'] = np.uint32(integration_meta.experiment_id)
                    parameters['experiment_string'] = integration_meta.experiment_string
                    parameters['station'] = self.options.site_id
                    parameters['timestamp_of_scan'] = data_parsing.timestamps[0]
                    parameters['num_sequences'] = integration_meta.nave

                    speed_of_light = 299792458 #m/s
                    #time to first range and back. convert to meters, div by c then convert to us
                    rtt = (rx_freq.frang * 2 * 1.0e3 / speed_of_light) * 1.0e6
                    parameters['first_range_rtt'] = np.uint32(rtt)
                    parameters['first_range'] = np.uint32(rx_freq.frang)
                    parameters['rx_sample_rate'] = 0
                    parameters['scan_start_marker'] = integration_meta.scan_flag # Should this change to scan_start_marker?
                    parameters['int_time'] = np.float32(integration_meta.integration_time)
                    parameters['tx_pulse_len'] = np.uint32(rx_freq.pulse_len)
                    parameters['tau_spacing'] = np.uint32(rx_freq.tau_spacing)
                    parameters['num_pulses'] = np.uint32(len(rx_freq.ptab.pulse_position))
                    parameters['num_lags'] = np.uint32(len(rx_freq.ltab.lag))
                    parameters['main_antenna_count'] = np.uint32(self.options.main_antenna_count)
                    parameters['intf_antenna_count'] = np.uint32(self.options.intf_antenna_count)
                    parameters['freq'] = np.uint32(rx_freq.rxfreq)
                    parameters['comment'] = rx_freq.comment_buffer
                    parameters['samples_data_type'] = "complex float"
                    parameters['pulses'] = np.array(rx_freq.ptab.pulse_position, dtype=np.uint32)

                    lags = []
                    for lag in rx_freq.ltab.lag:
                        lags.append([lag.pulse_position[0], lag.pulse_position[1]])

                    parameters['lags'] = np.array(lags,dtype=np.uint32)

                    parameters['blanked_samples'] = np.array(meta.blanks,dtype=np.uint32)

                    parameters['beam_nums'] = []
                    parameters['beam_azms'] = []

                    for beam in rx_freq.beams:
                        parameters['beam_nums'].append(np.uint32(beam.beamnum))
                        parameters['beam_azms'].append(beam.beamazimuth)

                    parameters['data_descriptors'] = bfiq.pop('data_descriptors', None)
                    parameters_holder[rx_freq.slice_id] = parameters


            for slice_id in bfiq:
                parameters = parameters_holder[slice_id]

                parameters['antenna_arrays_order'] = []
                if "main" in bfiq[slice_id]:
                    parameters['antenna_arrays_order'].append("main")
                if "intf" in bfiq[slice_id]:
                    parameters['antenna_arrays_order'].append("intf")

                parameters['sqn_timestamps'] = data_parsing.timestamps

                parameters['num_samps'] = np.uint32(bfiq[slice_id].pop('num_samps', None))
                parameters['data_dimensions'] = np.array([len(bfiq[slice_id].keys()),
                                                 integration_meta.nave,
                                                 len(parameters['beam_nums']),
                                                 parameters['num_samps']], dtype=np.uint32)

                if bfiq[slice_id]['intf']:
                    flattened_data = np.concatenate((bfiq[slice_id]['main']['data'],
                                                    bfiq[slice_id]['intf']['data']))
                else:
                    flattened_data = bfiq[slice_id]['main']['data']

                parameters['data'] = flattened_data


            for slice_id, parameters in parameters_holder.items():
                name = dataset_name.format(sliceid=slice_id, dformat="bfiq")
                output_file = dataset_location.format(name=name)

                two_hr_file_with_type = self.slice_filenames[slice_id].format(ext="bfiq")

                write_file(output_file, parameters, two_hr_file_with_type)


        def do_pre_bfiq():
            """
            Parses out any pre-beamformed IQ if available and writes it out to file.
            All variables are captured.
            """

            pre_bfiq = data_parsing.pre_bfiq_accumulator

            parameters_holder = {}
            for meta in integration_meta.sequences:
                for rx_freq in meta.rxchannel:
                    parameters = data_file_classes.iq_data.copy()

                    parameters['borealis_git_hash'] = self.git_hash.decode('utf-8')

                    parameters['timestamp_of_write'] = (time_now - epoch).total_seconds()
                    parameters['experiment_id'] = np.uint32(integration_meta.experiment_id)
                    parameters['experiment_string'] = integration_meta.experiment_string
                    parameters['station'] = self.options.site_id
                    parameters['timestamp_of_scan'] = data_parsing.timestamps[0]
                    parameters['num_sequences'] = integration_meta.nave

                    speed_of_light = 299792458 #m/s
                    #time to first range and back. convert to meters, div by c then convert to us
                    rtt = (rx_freq.frang * 2 * 1.0e3 / speed_of_light) * 1.0e6
                    parameters['first_range_rtt'] = np.uint32(rtt)
                    parameters['first_range'] = np.uint32(rx_freq.frang)
                    parameters['rx_sample_rate'] = 0
                    parameters['scan_start_marker'] = integration_meta.scan_flag # Should this change to scan_start_marker?
                    parameters['int_time'] = np.float32(integration_meta.integration_time)
                    parameters['tx_pulse_len'] = np.uint32(rx_freq.pulse_len)
                    parameters['tau_spacing'] = np.uint32(rx_freq.tau_spacing)
                    parameters['num_pulses'] = np.uint32(len(rx_freq.ptab.pulse_position))
                    parameters['num_lags'] = np.uint32(len(rx_freq.ltab.lag))
                    parameters['main_antenna_count'] = np.uint32(self.options.main_antenna_count)
                    parameters['intf_antenna_count'] = np.uint32(self.options.intf_antenna_count)
                    parameters['freq'] = np.uint32(rx_freq.rxfreq)
                    parameters['comment'] = rx_freq.comment_buffer
                    parameters['samples_data_type'] = "complex float"
                    parameters['pulses'] = np.array(rx_freq.ptab.pulse_position, dtype=np.uint32)

                    lags = []
                    for lag in rx_freq.ltab.lag:
                        lags.append([lag.pulse_position[0], lag.pulse_position[1]])

                    parameters['lags'] = np.array(lags,dtype=np.uint32)

                    parameters['blanked_samples'] = np.array(meta.blanks,dtype=np.uint32)

                    parameters['beam_nums'] = []
                    parameters['beam_azms'] = []

                    for beam in rx_freq.beams:
                        parameters['beam_nums'].append(np.uint32(beam.beamnum))
                        parameters['beam_azms'].append(beam.beamazimuth)

                    parameters['data_descriptors'] = pre_bfiq.pop('data_descriptors', None)
                    parameters_holder[rx_freq.slice_id] = parameters


            final_data_params = {}
            for slice_id in pre_bfiq:
                final_data_params[slice_id] = {}

                for stage in pre_bfiq[slice_id]:
                    parameters = parameters_holder[slice_id].copy()

                    parameters['sqn_timestamps'] = data_parsing.timestamps

                    parameters['num_samps'] = np.uint32(
                                                pre_bfiq[slice_id][stage].pop('num_samps', None))
                    parameters['data_dimensions'] = np.array([len(pre_bfiq[slice_id][stage].keys()),
                                                     integration_meta.nave,
                                                     len(parameters['beam_nums']),
                                                     parameters['num_samps']], dtype=np.uint32)

                    parameters['antenna_arrays_order'] = list(pre_bfiq[slice_id][stage].keys())

                    data = []
                    for ant, data_dict in pre_bfiq[slice_id][stage].items():
                        data.append(data_dict['data'])

                    flattened_data = np.concatenate(data)
                    parameters['data'] = flattened_data

                    final_data_params[slice_id][stage] = parameters


            for slice_id, slices in final_data_params.items():
                for stage, params in slices.items():
                    name = dataset_name.format(sliceid=slice_id, dformat="{}.iq".format(stage))
                    output_file = dataset_location.format(name=name)

                    ext = "{}.iq".format(stage)
                    two_hr_file_with_type = self.slice_filenames[slice_id].format(ext=ext)

                    write_file(output_file, params, two_hr_file_with_type)




        def do_raw_rf():
            """
            Opens the shared memory location in the protobuf and writes the samples out to file.
            Write medium must be able to sustain high write bandwidth. Shared memory is destroyed
            after write. All variables are captured.
            """
            raw_rf_dict = nested_dict()
            name = dataset_name.format(dformat='rawrf')
            output_file = dataset_location.format(name=name)

            shm = ipc.SharedMemory(self.processed_data.rf_samples_location)
            mapfile = mmap.mmap(shm.fd,shm.size)

            rf_samples = np.frombuffer(mapfile,dtype=np.complex64)

            total_antennas = self.options.main_antenna_count + self.options.intf_antenna_count
            rf_samples = np.reshape(rf_samples,(total_antennas,-1))
            for ant in range(total_antennas):
                ant_str = "antenna_{0}".format(ant)
                raw_rf_dict[ant_str] = rf_samples[ant]

            write_file(output_file, raw_rf_dict)

            shm.close_fd()
            shm.unlink()
            mapfile.close()

        # Use multiprocessing to speed up writing. Each data type can be parsed and written by a
        # separate process in order to parallelize the work. Parsing the protobuf is not a fast
        # operation.
        procs = []

        if write_rawacf:
            procs.append(mp.Process(target=do_acf))

        if write_bfiq and data_parsing.bfiq_available:
            procs.append(mp.Process(target=do_bfiq))

        if write_pre_bfiq and data_parsing.pre_bfiq_available:
            procs.append(mp.Process(target=do_pre_bfiq))

        if write_raw_rf:
            procs.append(mp.Process(target=do_raw_rf))
        else:
            for rf_samples_location in data_parsing.rawrf_locations:
                shm = ipc.SharedMemory(rf_samples_location)
                shm.close_fd()
                shm.unlink()

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()



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
            # Send a request for data to dsp. The actual message doesn't matter, so use 'Request'
            # After that, receive the processed data from dsp, blocking.
            socks = dict(poller.poll())
        except KeyboardInterrupt:
            sys.exit()


        if dsp_to_data_write in socks and socks[dsp_to_data_write] == zmq.POLLIN:
            data = so.recv_bytes(dsp_to_data_write, options.dsp_to_dw_identity, printing)

            if not first_time:
                if data_parsing.sequence_num == final_integration:

                    if integration_meta.experiment_string != current_experiment:
                        data_write = DataWrite(options)
                        current_experiment = integration_meta.experiment_string

                    start = time.time()
                    data_write.output_data(write_bfiq=args.enable_bfiq,
                                           write_pre_bfiq=args.enable_pre_bfiq,
                                           write_raw_rf=args.enable_raw_rf,
                                           file_ext=args.file_type,
                                           integration_meta=integration_meta,
                                           data_parsing=data_parsing,
                                           write_rawacf=False)
                    end = time.time()
                    printing("Time to write: {} ms".format((end-start)*1000))

                    data_parsing = ParseData()




            first_time = False

            start = time.time()
            data_parsing.update(data)
            end = time.time()
            printing("Time to parse: {} ms".format((end-start)*1000))


        if radctrl_to_data_write in socks and socks[radctrl_to_data_write] == zmq.POLLIN:
            data = so.recv_data(radctrl_to_data_write, options.radctrl_to_dw_identity, printing)

            integration_meta = datawritemetadata_pb2.IntegrationTimeMetadata()
            integration_meta.ParseFromString(data)

            final_integration = integration_meta.last_seqn_num



if __name__ == '__main__':

    main()

