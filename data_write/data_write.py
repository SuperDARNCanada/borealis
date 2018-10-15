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
import numpy as np
import deepdish as dd
import argparse as ap
import posix_ipc as ipc
import mmap 
import multiprocess

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
    #hdf5_file = h5py.File(filename, "w+")

    def convert_to_numpy(dd):
        for k,v in dd.items():
            if isinstance(v,dict):
                convert_to_numpy(v)
            elif isinstance(v,list):
                dd[k] = np.array(v)
            else:
                continue

    convert_to_numpy(data_dict)

    a = datetime.datetime.now()
    dd.io.save(filename, data_dict, compression=None)
    b = datetime.datetime.now()
    diff= b-a
    printing("Time to write: {}".format(diff))
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

    def output_data(self, write_iq, write_pre_bf_iq, write_raw_rf, file_ext, write_rawacf=True):
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

        if file_ext not in ['hdf5','json','dmap']:
            raise ValueError("File format selection required (hdf5, json, dmap), none given")

        main_iq_available = False
        intf_iq_available = False
        rawacf_available = False
        pre_bf_iq_available = False
        data_format_string = None

        # defaultdict will populate non-specified entries in the dictionary with the default
        # value given as an argument, in this case a dictionary.

        nested_dict = lambda: collections.defaultdict(nested_dict)
        iq_pre_bf_data_dict = nested_dict()
        rawacf_data_dict = nested_dict()
        iq_data_dict = nested_dict()
        raw_rf_dict = nested_dict()

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

            if len(data_set.beamformedsamples) > 0:
                iq_available = True
                for beam in data_set.beamformedsamples:
                    beam_str = "beam_{}".format(beam.beamnum)
                    iq_data_dict[freq_str][beam_str] = {'main' : {'real' : [], 'imag' : []},
                                                        'intf' : {'real' : [], 'imag' : []}
                                                        }
                    for main_sample in beam.mainsamples:
                        iq_data_dict[freq_str][beam_str]['main']['real'].append(main_sample.real)
                        iq_data_dict[freq_str][beam_str]['main']['imag'].append(main_sample.imag)

                    if len(beam.intfsamples) > 0:
                        for intf_sample in beam.intfsamples:
                            dict_to_add = iq_data_dict[freq_str][beam_str]['intf']
                            dict_to_add['real'].append(intf_sample.real)
                            dict_to_add['imag'].append(intf_sample.imag)

            # non beamformed IQ samples are available
            if len(data_set.debugsamples) > 0:
                for stage_num, debug_samples in enumerate(data_set.debugsamples):
                    pre_bf_iq_available = True
                    for ant_num, ant_data in enumerate(debug_samples.antennadata):
                        ant_str = "antenna_{0}".format(ant_num)
                        stage_name = debug_samples.stagename
                        #iq_pre_bf_data_dict[stage_name] = collections.defaultdict(dict)
                        ipbdd = iq_pre_bf_data_dict[stage_name][freq_str][ant_str]
                        ipbdd = {'real': [], 'imag': []}
                        for ant_samp in ant_data.antennasamples:
                            ipbdd['real'].append(ant_samp.real)
                            ipbdd['imag'].append(ant_samp.imag)



        # Format the name and location for the dataset
        today_string = datetime.datetime.today().strftime("%Y%m%d")
        datetime_string = datetime.datetime.today().strftime("%Y%m%d.%H%M.%S.%f")
        dataset_directory = "{0}/{1}".format(self.options.data_directory, today_string)
        dataset_name = "{dt}.{site}.{{dformat}}.{fformat}".format(dt=datetime_string,
                                                                site=self.options.site_id,
                                                                fformat=file_ext)
        dataset_location = "{dir}/{{name}}".format(dir=dataset_directory)


        def write_file(location, final_data_dict):
            if not os.path.exists(dataset_directory):
                os.makedirs(dataset_directory)

            # Finally write out the appropriate file type
            if file_ext == 'hdf5':
                write_hdf5_file(location, final_data_dict)
            elif file_ext == 'json':
                write_json_file(location, final_data_dict)
            elif file_ext == 'dmap':
                write_dmap_file(location, final_data_dict)


        if write_rawacf and rawacf_available:
            name = dataset_name.format(dformat="rawacf")
            output_file = dataset_location.format(name=name)

            write_file(output_file, rawacf_data_dict)

        if write_iq and iq_available:
            name = dataset_name.format(dformat="bfiq")
            output_file = dataset_location.format(name=name)

            write_file(output_file, iq_data_dict)

        if write_pre_bf_iq and pre_bf_iq_available:
            name = dataset_name.format(dformat="iq")
            output_file = dataset_location.format(name=name)

            write_file(output_file, iq_pre_bf_data_dict)

        if write_raw_rf:
            name = dataset_name.format(dformat='rawrf')
            output_file = dataset_location.format(name=name)

            shm = ipc.SharedMemory(self.processed_data.rf_samples_location)
            mapfile = mmap.mmap(shm.fd,shm.size)

            rf_samples = np.frombuffer(mapfile,dtype=np.complex64)
            


            total_antennas = self.options.main_antenna_count + self.options.intf_antenna_count
            rf_samples = np.reshape(rf_samples,(total_antennas,-1))
            #rf_samples.tofile(output_file)
            for ant in range(total_antennas):
                ant_str = "antenna_{0}".format(ant)
                raw_rf_dict[ant_str] = rf_samples[ant]

            write_file(output_file, raw_rf_dict)

            shm.close_fd()
            shm.unlink()
            mapfile.close()





if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Write processed SuperDARN data to file')
    parser.add_argument('--file-type', help='Type of output file: hdf5, json, or dmap',
                        default='hdf5')
    parser.add_argument('--enable-bfiq', help='Enable beamformed iq writing',
                        action='store_true')
    parser.add_argument('--enable-pre-bf-iq', help='Enable individual antenna iq writing',
                        action='store_true')
    parser.add_argument('--enable-raw-rf', help='Save the raw, unfiltered IQ samples',
                        action='store_true')
    args=parser.parse_args()

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

            start = datetime.datetime.now()
            dw.output_data(write_iq=args.enable_bfiq, write_pre_bf_iq=args.enable_pre_bf_iq,
                            write_raw_rf=args.enable_raw_rf,file_ext=args.file_type, 
                            write_rawacf=False)

            end = datetime.datetime.now()
            diff = end - start
            time = diff.total_seconds() * 1000
            print("Sequence number: {0}".format(pd.sequence_num))
            print("Time to process samples: {0} s".format(pd.processing_time))
            print("Time to parse + write: {0} ms".format(time))

        thread = threading.Thread(target=make_file,args=(data,))
        thread.daemon = True
        thread.start()
