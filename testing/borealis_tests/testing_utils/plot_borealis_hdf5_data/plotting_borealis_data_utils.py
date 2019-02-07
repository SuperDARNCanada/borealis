#!/usr/bin/env python3

import sys
import deepdish
import random
import numpy as np
import matplotlib.pyplot as plt


def plot_output_samples_iq_data(record_dict, record_info_string, sequence=0, real_only=True, antenna_indices=None):
    """
    :param record_dict: dict of the hdf5 data of a given record of output_samples_iq, ie deepdish.io.load(filename)[record_name]
    :param record_info_string: a string indicating the type of data being plotted (to be used on the plot legend). Should 
     be output_samples_iq type, but there might be multiple slices.
    """

    # data dimensions are num_antennas, num_sequences, num_samps
    number_of_antennas = len(record_dict['antenna_arrays_order'])

    flat_data = np.array(record_dict['data'])  
    # reshape to number of antennas (M0..... I3) x nave x number_of_samples
    output_samples_iq_data = np.reshape(flat_data, (number_of_antennas, record_dict['num_sequences'], record_dict['num_samps']))
    record_dict['data'] = output_samples_iq_data
    antennas_present = [int(i.split('_')[-1]) for i in record_dict['antenna_arrays_order']]

    print('Sequence number: {}'.format(sequence))

    if antenna_indices is None:
        indices = range(0, record_dict['data'].shape[0])
    else:
        indices = antenna_indices    

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)    
    for index in indices:
        antenna = antennas_present[index]
        ax1.set_title('Main Antennas {}'.format(record_info_string))
        ax2.set_title('Intf Antennas {}'.format(record_info_string))
        if antenna < record_dict['main_antenna_count']:
            ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].real, label='Real {}'.format(antenna))
            if not real_only:
                ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].imag, label="Imag {}".format(antenna))
            ax1.legend()
        else:
            ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].real, label='Real {}'.format(antenna))
            if not real_only:
                ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].imag, label="Imag {}".format(antenna))
            ax2.legend()                       
    plt.show()


def plot_output_raw_data(record_dict, record_info_string, sequence=0, real_only=True, start_sample=18000, end_sample=20000, antenna_indices=None):
    """
    :param record_dict: dict of the hdf5 data of a given record of rawrf, ie deepdish.io.load(filename)[record_name]
    :param record_info_string: a string indicating the type of data being plotted (to be used on the plot legend). Should 
     be raw_rf_data type, but you may also want to distinguish what data this is.
    """

    # data dimensions are num_sequences, num_antennas, num_samps

    number_of_antennas = record_dict['main_antenna_count'] + record_dict['intf_antenna_count']
    flat_data = np.array(record_dict['data'])  
    # reshape to nave x number of antennas (M0..... I3) x number_of_samples

    raw_rf_data_data = np.reshape(flat_data, (record_dict['num_sequences'], number_of_antennas, record_dict['num_samps']))
    record_dict['data'] = raw_rf_data_data

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_title('Main Antennas {}'.format(record_info_string))
    ax2.set_title('Intf Antennas {}'.format(record_info_string))


    if antenna_indices is None:
        indices = range(0, record_dict['data'].shape[1])
    else:
        indices = antenna_indices    

    for antenna in indices:
        #if max(abs(record_dict['data'][index,antenna,18000:20000])) < 0.05:
        #    continue
        if antenna < record_dict['main_antenna_count']:
            ax1.plot(np.arange(end_sample-start_sample), record_dict['data'][sequence,antenna,start_sample:end_sample].real, label='Real antenna {}'.format(antenna))
            if not real_only:
                ax1.plot(np.arange(end_sample-start_sample), record_dict['data'][sequence,antenna,start_sample:end_sample].imag, label="Imag {}".format(antenna))
        else:
            ax2.plot(np.arange(end_sample-start_sample), record_dict['data'][sequence,antenna,start_sample:end_sample].real, label='Real antenna {}'.format(antenna))
            if not real_only:
                ax2.plot(np.arange(end_sample-start_sample), record_dict['data'][sequence,antenna,start_sample:end_sample].imag, label="Imag {}".format(antenna))
    ax1.legend()   
    ax2.legend()     
    plt.show()


def plot_bf_iq_data(record_dict, record_info_string, beam=0, sequence=0):
    """
    :param record_dict: dict of the hdf5 data of a given record of bfiq, ie deepdish.io.load(filename)[record_name]
    :param record_info_string: a string indicating the type of data being plotted (to be used on the plot legend). Should 
     be bf_iq type, but there might be multiple slices or you may wish to otherwise identify
    :param beam: The beam number, indexed from 0. Assumed only one beam and plotting the first.
    """

    # data dimensions are num_antenna_arrays, num_sequences, num_beams, num_samps
    number_of_beams = len(record_dict['beam_azms'])
    number_of_arrays = len(record_dict['antenna_arrays_order'])
    flat_data = np.array(record_dict['data'])  
    # reshape to 2 (main, intf) x nave x number_of_beams x number_of_samples
    bf_iq_data = np.reshape(flat_data, (number_of_arrays, record_dict['num_sequences'], number_of_beams, record_dict['num_samps']))
    record_dict['data'] = bf_iq_data

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    for index in range(0, record_dict['data'].shape[0]):
        antenna_array = record_dict['antenna_arrays_order'][index]

        if antenna_array == 'main':
            ax1.set_title('Main Array {} sequence {}'.format(record_info_string, sequence))
            ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,sequence,beam,:].real, label='Real {}'.format(antenna_array))
            ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,sequence,beam,:].imag, label="Imag {}".format(antenna_array))
            ax1.legend()
        else:
            ax2.set_title('Intf Array {} sequence {}'.format(record_info_string, sequence))
            ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,sequence,beam,:].real, label='Real {}'.format(antenna_array))
            ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,sequence,beam,:].imag, label="Imag {}".format(antenna_array))
            ax2.legend()                       
    plt.show()


def plot_output_tx_data(record_dict, record_info_string, sequence=0, real_only=True, antenna_indices=None):
    """
    :param record_dict: dict of the hdf5 data of a given record of txdata, ie deepdish.io.load(filename)[record_name]
    :param record_info_string): a string indicating the type of data being plotted (to be used on the plot legend). Should 
     be raw_rf_data type, but you may want to otherwise identify the data.
    """

    # data dimensions are num_sequences, num_antennas, num_samps
    record_dict['rx_sample_rate'] = int(record_dict['tx_rate'][0]/record_dict['dm_rate'])
    print('Decimation rate error: {}'.format(record_dict['dm_rate_error']))
    print(record_dict['rx_sample_rate'])
    record_dict['data_descriptors'] = ['num_sequences', 'num_antennas', 'num_samps']
    record_dict['data'] = record_dict['tx_samples']#tx['decimated_tx_samples']
    record_dict['antennas_present'] = record_dict['decimated_tx_antennas'][0]
    record_dict['dm_start_sample'] = 0

    if antenna_indices is None:
        indices = range(0, record_dict['data'].shape[1])
    else:
        indices = antenna_indices    

    fig, (ax1) = plt.subplots(1, 1, sharex=True)
    plt.title(record_info_string)
    for antenna in indices:
        #if max(abs(record_dict['data'][index,antenna,18000:20000])) < 0.05:
        #    continue
        ax1.plot(np.arange(record_dict['data'].shape[2]), record_dict['data'][sequence,antenna,:].real, label='Real antenna {}'.format(antenna))
        if not real_only:
            ax1.plot(np.arange(record_dict['data'].shape[2]), record_dict['data'][sequence,antenna,:].imag, label='Imag antenna {}'.format(antenna))
    ax1.legend()               
    plt.show()