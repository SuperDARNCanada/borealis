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
import multiprocessing as mp

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

    def convert_to_numpy(dd):
        for k,v in dd.items():
            if isinstance(v,dict):
                convert_to_numpy(v)
            elif isinstance(v,list):
                dd[k] = np.array(v)
            else:
                continue

    convert_to_numpy(data_dict)

    dd.io.save(filename, data_dict, compression=None)


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

    def output_data(self, write_iq, write_pre_bf_iq, write_raw_rf, file_ext, write_rawacf=True):
        """
        Parse through samples and write to file. 

        A file will be created using the file extention for each requested data product.

        :param write_iq:        Should IQ be written to file? Bool.
        :param write_pre_bf_iq: Should pre-beamformed IQ be written to file? Bool.
        :param write_raw_rf:    Should raw rf samples be written to file? Bool.
        :param file_ext:        Type of file extention to use. String
        :param write_rawacf: Should rawacfs be written to file? Bool, default True.
        """

        if file_ext not in ['hdf5','json','dmap']:
            raise ValueError("File format selection required (hdf5, json, dmap), none given")

        # Format the name and location for the dataset
        today_string = datetime.datetime.today().strftime("%Y%m%d")
        datetime_string = datetime.datetime.today().strftime("%Y%m%d.%H%M.%S.%f")
        dataset_directory = "{0}/{1}".format(self.options.data_directory, today_string)
        dataset_name = "{dt}.{site}.{{dformat}}.{fformat}".format(dt=datetime_string,
                                                                site=self.options.site_id,
                                                                fformat=file_ext)
        dataset_location = "{dir}/{{name}}".format(dir=dataset_directory)


        def write_file(location, final_data_dict):     
            """
            Writes the final data out to the location based on the type of file extention required

            :param location:        File path and name to write to. String
            :param final_data_dict: Data dict parsed out from protobuf. Dict
                
            """
            if not os.path.exists(dataset_directory):
                try:
                    os.makedirs(dataset_directory)
                except:
                    pass

            if file_ext == 'hdf5':
                write_hdf5_file(location, final_data_dict)
            elif file_ext == 'json':
                write_json_file(location, final_data_dict)
            elif file_ext == 'dmap':
                write_dmap_file(location, final_data_dict)

        # defaultdict will populate non-specified entries in the dictionary with the default
        # value given as an argument, in this case a dictionary. Nesting it in a lambda lets you
        # create arbitrarily deep dictionaries.
        nested_dict = lambda: collections.defaultdict(nested_dict)

        def do_acf():
            """
            Parses out any possible ACF data from protobuf and writes to file. All variables are
            captured.

            """
            rawacf_available = False
            rawacf_data_dict = nested_dict()

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
            
            if rawacf_available:
                name = dataset_name.format(dformat="rawacf")
                output_file = dataset_location.format(name=name)

                write_file(output_file, rawacf_data_dict)


        def do_bfiq():
            """
            Parses out any possible beamformed IQ data from the protobuf and writes it to file.
            All variables are captured.

            """
            iq_data_dict = nested_dict()
            iq_available = False
            for freq_num, data_set in enumerate(self.processed_data.outputdataset):
                freq_str = "frequency_{0}".format(freq_num)

                # Find out what is available in the data to determine what to write out
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
            
            if iq_available:
                name = dataset_name.format(dformat="bfiq")
                output_file = dataset_location.format(name=name)

                write_file(output_file, iq_data_dict)


        def do_pre_bfiq():
            """
            Parses out any pre-beamformed IQ if available and writes it out to file.
            All variables are captured.
            """
            iq_pre_bf_data_dict = nested_dict()
            pre_bf_iq_available = False
            # Iterate over every data set, one data set per frequency
            for freq_num, data_set in enumerate(self.processed_data.outputdataset):
                freq_str = "frequency_{0}".format(freq_num)

                # non beamformed IQ samples are available
                if len(data_set.debugsamples) > 0:
                    for stage_num, debug_samples in enumerate(data_set.debugsamples):
                        pre_bf_iq_available = True
                        for ant_num, ant_data in enumerate(debug_samples.antennadata):
                            ant_str = "antenna_{0}".format(ant_num)
                            stage_name = debug_samples.stagename
                            #iq_pre_bf_data_dict[stage_name] = collections.defaultdict(dict)
                            iq_pre_bf_data_dict[stage_name][freq_str][ant_str] = {'real': [],
                                                                                  'imag': []}
                            ipbdd = iq_pre_bf_data_dict[stage_name][freq_str][ant_str] 
                            for ant_samp in ant_data.antennasamples:
                                ipbdd['real'].append(ant_samp.real)
                                ipbdd['imag'].append(ant_samp.imag)
        
            if pre_bf_iq_available:
                name = dataset_name.format(dformat="iq")
                output_file = dataset_location.format(name=name)

                write_file(output_file, iq_pre_bf_data_dict)



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

        if write_iq:
            procs.append(mp.Process(target=do_bfiq))

        if write_pre_bf_iq:
            procs.append(mp.Process(target=do_pre_bfiq))

        if write_raw_rf:
            procs.append(mp.Process(target=do_raw_rf))
        else:
            shm = ipc.SharedMemory(self.processed_data.rf_samples_location)
            shm.close_fd()
            shm.unlink()

        for proc in procs:
            proc.start()

        for proc in procs:
            proc.join()





if __name__ == '__main__':

    parser = ap.ArgumentParser(description='Write processed SuperDARN data to file')
    parser.add_argument('--file-type', help='Type of output file: hdf5, json, or dmap',
                        default='hdf5')
    parser.add_argument('--enable-bfiq', help='Enable beamformed iq writing',
                        action='store_true')
    parser.add_argument('--enable-pre-bf-iq', help='Enable individual antenna iq writing',
                        action='store_true')
    parser.add_argument('--enable-raw-rf', help='Save the raw, unfiltered IQ samples. Requires HDF5.',
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
                            write_raw_rf=args.enable_raw_rf, file_ext=args.file_type, 
                            write_rawacf=False)

            end = datetime.datetime.now()
            diff = end - start
            time = diff.total_seconds() * 1000
            printing("Sequence number: {0}".format(pd.sequence_num))
            printing("Time to process samples: {0} ms".format(pd.processing_time))
            printing("Time to parse + write: {0} ms".format(time))

        thread = threading.Thread(target=make_file,args=(data,))
        thread.daemon = True
        thread.start()
