#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# data_write.py
# 2018-05-14
# Data writing functionality to write iq/raw and other files

import zmq
import sys
import datetime
import json
import os
import h5py
import collections
import threading

borealis_path = os.environ['BOREALISPATH']
if not borealis_path:
    raise ValueError("BOREALISPATH env variable not set")

if __debug__:
    sys.path.append(borealis_path + '/build/debug/utils/protobuf')
else:
    sys.path.append(borealis_path + '/build/release/utils/protobuf')
import processeddata_pb2

sys.path.append(borealis_path + '/utils/')
import data_write_options.data_write_options as dwo
from zmq_borealis_helpers import socket_operations as so


def printing(msg):
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


def write_hdf5_file(filename, data_dict):
    """
    Write out data to an HDF5 file. If the file already exists it will be overwritten.
    :param filename: The path to the file to write out. String
    :param data_dict: Python dictionary to write out to the HDF5 file.
    """
    hdf5_file = h5py.File(filename, "w+")
    # TODO: Complete this by parsing through the dictionary and write out to proper HDF5 format


def write_dmap_file(filename, data_dict):
    """
    Write out data to a dmap file. If the file already exists it will be overwritten.
    :param filename: The path to the file to write out. String
    :param data_dict: Python dictionary to write out to the dmap file.
    """
    # TODO: Complete this by parsing through the dictionary and write out to proper dmap format
    pass


class DataWrite(object):
    """This class contains the functions used to write out processed data to files.

    """
    def __init__(self, processed_data, data_write_options):
        super(DataWrite, self).__init__()
        self.options = data_write_options
        self.processed_data = processed_data

    def output_debug_data(self):
        """
        Writes out to file a JSON representation of each stage of filtering for debug analysis.
        WARNING: This takes a while
        """
        debug_data = {}
        for set_num, data_set in enumerate(self.processed_data.outputdataset):
            set_str = "dataset_{0}".format(set_num)
            debug_data[set_str] = {}
            for stage_num, debug_samples in enumerate(data_set.debugsamples):
                stage_str = debug_samples.stagename
                debug_data[set_str][stage_str] = {}
                for antenna_num, antenna_data in enumerate(debug_samples.antennadata):
                    ant_str = "antenna_{0}".format(antenna_num)
                    debug_data[set_str][stage_str][ant_str] = {"real": [], "imag": []}
                    for antenna_sample in antenna_data.antennasamples:
                        debug_data[set_str][stage_str][ant_str]["real"].append(antenna_sample.real)
                        debug_data[set_str][stage_str][ant_str]["imag"].append(antenna_sample.imag)

        write_json_file(self.options.debug_file, debug_data)

    def output_data(self, write_rawacf=True, write_iq=False, write_pre_bf_iq=False,
                    use_hdf5=False, use_json=True, use_dmap=False):
        """
        Parse through samples and write to file. Note that only one data type will be written out, 
        and only to one type of file. If you specify all three types, the order of preference is: 
        1) pre_bf_iq
        2) iq
        3) rawacf
        The file format order of preference is:
        1) hdf5
        2) json
        3) dmap
        :param write_rawacf: Should rawacfs be written to file? Bool, default True. 
        :param write_iq: Should IQ be written to file? Bool, default False.
        :param write_pre_bf_iq: Should pre-beamformed IQ be written to file? Bool. Default False
        :param use_hdf5: Write data out to hdf5 file. Default False
        :param use_json: Write data out to json file. Default True
        :param use_dmap: Write data out to dmap file. Default False
        """
        # Find out what file format to write out
        file_format_string = None
        if use_hdf5:
            file_format_string = 'hdf5'
        elif use_json:
            file_format_string = 'json'
        elif use_dmap:
            file_format_string = 'dmap'

        if not file_format_string:
            raise ValueError("File format selection required (hdf5, json, dmap), none given")

        iq_available = False
        rawacf_available = False
        pre_bf_iq_available = False
        data_format_string = None

        # defaultdict will populate non-specified entries in the dictionary with the default
        # value given as an argument, in this case a dictionary.
        iq_pre_bf_data_dict = collections.defaultdict(dict)
        rawacf_data_dict = {}
        iq_data_dict = {}
        final_data_dict = {}

        # Iterate over every data set, one data set per frequency
        for freq_num, data_set in enumerate(self.processed_data.outputdataset):
            freq_str = "frequency_{0}".format(freq_num)

            # Find out what is available in the data to determine what to write out
            # Main acfs were calculated
            if len(data_set.mainacf) > 0:
                rawacf_available = True
                rawacf_data_dict[freq_str]['mainacf'] = {'real': [], 'imag': []}
                for complex_sample in data_set.mainacf:
                    rawacf_data_dict[freq_str]['mainacf']['real'].append(complex_sample.real)
                    rawacf_data_dict[freq_str]['mainacf']['imag'].append(complex_sample.imag)

            # Interferometer acfs were calculated
            if len(data_set.intacf) > 0:
                rawacf_available = True
                rawacf_data_dict[freq_str]['intacf'] = {'real': [], 'imag': []}
                for complex_sample in data_set.intacf:
                    rawacf_data_dict[freq_str]['intacf']['real'].append(complex_sample.real)
                    rawacf_data_dict[freq_str]['intacf']['imag'].append(complex_sample.imag)

            # Cross correlations were calculated
            if len(data_set.xcf) > 0:
                rawacf_available = True
                rawacf_data_dict[freq_str]['xcf'] = {'real': [], 'imag': []}
                for complex_sample in data_set.xcf:
                    rawacf_data_dict[freq_str]['xcf']['real'].append(complex_sample.real)
                    rawacf_data_dict[freq_str]['xcf']['imag'].append(complex_sample.imag)

            # IQ samples were beamformed
            if len(data_set.beamformediqsamples) > 0:
                iq_available = True
                iq_data_dict[freq_str] = {'real': [], 'imag': []}
                for complex_sample in data_set.beamformediqsamples:
                    iq_data_dict[freq_str]['real'].append(complex_sample.real)
                    iq_data_dict[freq_str]['imag'].append(complex_sample.imag)

            # non beamformed IQ samples are available
            if len(data_set.debugsamples) > 0:
                for stage_num, debug_samples in enumerate(data_set.debugsamples):
                    if debug_samples.stagename == 'output_samples':
                        # Final stage, so write these samples only to file
                        pre_bf_iq_available = True
                        for ant_num, ant_data in enumerate(debug_samples.antennadata):
                            ant_str = "antenna_{0}".format(ant_num)
                            iq_pre_bf_data_dict[freq_str][ant_str] = {'real': [], 'imag': []}
                            for ant_samp in ant_data.antennasamples:
                                iq_pre_bf_data_dict[freq_str][ant_str]['real'].append(ant_samp.real)
                                iq_pre_bf_data_dict[freq_str][ant_str]['imag'].append(ant_samp.imag)
                    else:
                        continue

        # Note that only one data type will be written out, and only to one type of file.
        # If you specify all three types, the order of preference is: pre_bf_iq, iq, then rawacf
        if write_rawacf and rawacf_available:
            data_format_string = "rawacf"
            final_data_dict = rawacf_data_dict
        if write_iq and iq_available:
            data_format_string = "bfiq"
            final_data_dict = iq_data_dict
        if write_pre_bf_iq and pre_bf_iq_available:
            data_format_string = "iq"
            final_data_dict = iq_pre_bf_data_dict

        # Format the name and location for the dataset
        today_string = datetime.datetime.today().strftime("%Y%m%d")
        datetime_string = datetime.datetime.today().strftime("%Y%m%d.%H%M.%S.%f")
        dataset_name = "{0}.{1}.{2}.{3}".format(datetime_string, self.options.site_id,
                                                data_format_string, file_format_string)
        dataset_directory = "{0}/{1}".format(self.options.data_directory, today_string)
        dataset_location = "{0}/{1}".format(dataset_directory, dataset_name)

        if not os.path.exists(dataset_directory):
            os.makedirs(dataset_directory)

        # Finally write out the appropriate file type
        if use_hdf5:
            write_hdf5_file(dataset_location, final_data_dict)
        elif use_json:
            write_json_file(dataset_location, final_data_dict)
        elif use_dmap:
            write_dmap_file(dataset_location, final_data_dict)


if __name__ == '__main__':
    options = dwo.DataWriteOptions()
    sockets = so.create_sockets([options.dw_to_dsp_identity], options.router_address)
    dsp_to_data_write = sockets[0]

    if __debug__:
        printing("Socket connected")

    while True:
        try:
            # Send a request for data to dsp. The actual message doesn't matter, so use 'Request'
            # After that, receive the processed data from dsp, blocking.
            #so.send_request(dsp_to_data_write, options.dsp_to_dw_identity, "Request")
            data = so.recv_data(dsp_to_data_write, options.dsp_to_dw_identity, printing)
        except KeyboardInterrupt:
            sys.exit()

        def make_file(data_data):
            if __debug__:
                printing("Data received from dsp")
                start = datetime.datetime.now()

            pd = processeddata_pb2.ProcessedData()
            pd.ParseFromString(data_data)

            dw = DataWrite(pd, options)

            if __debug__:
                dw.output_debug_data()
            else:
                start = datetime.datetime.now()
                dw.output_data(write_pre_bf_iq=True)

            
            end = datetime.datetime.now()
            diff = end - start
            time = diff.total_seconds() * 1000
            print("Sequence number: {0}".format(pd.sequence_num))
            print("Time to process samples: {0} s".format(pd.processing_time))
            print("Time to parse + write: {0} ms".format(time))

        thread = threading.Thread(target=make_file,args=(data,))
        thread.daemon = True
        thread.start()
