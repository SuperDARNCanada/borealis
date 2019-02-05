#!/usr/bin/env python3

import sys
import deepdish
import random
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]


def plot_output_tx_data(record_dict, record_filetype):
    """
    :param record_dict: a record containing data_dimensions, data_descriptors, antenna_array_names, and data (reshaped)
    :param record_filetype: a string indicating the type of data being plotted (to be used on the plot legend). Should 
     be raw_rf_data type, but there might be multiple slices.
    """

    # data dimensions are num_antennas, num_sequences, num_samps


    for index in range(0,record_dict['data'].shape[0]): 
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for antenna in range(0, 4):#record_dict['data'].shape[1]):
            #if max(abs(record_dict['data'][index,antenna,18000:20000])) < 0.05:
            #    continue
            ax1.plot(np.arange(record_dict['data'].shape[2]), record_dict['data'][index,antenna,:].real, label='Real antenna {}'.format(antenna))
            #ax1.plot(np.arange(2000), record_dict['data'][index,antenna,18000:20000].imag, label="Imag {}".format(antenna))
        ax1.legend()               
        plt.show()


data = deepdish.io.load(filename)

record_name = random.choice(list(data.keys()))
print(record_name)

tx = data[record_name]
tx['rx_sample_rate'] = int(tx['tx_rate'][0]/tx['dm_rate'])
print('Decimation rate error: {}'.format(tx['dm_rate_error']))
print(tx['rx_sample_rate'])
tx['data_descriptors'] = ['num_sequences', 'num_antennas', 'num_samps']
tx['data'] = tx['tx_samples']#tx['decimated_tx_samples']
tx['antennas_present'] = tx['decimated_tx_antennas'][0]
tx['dm_start_sample'] = 0

plot_output_tx_data(tx, 'tx_data')