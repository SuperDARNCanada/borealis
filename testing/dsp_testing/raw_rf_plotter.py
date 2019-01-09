#!/usr/bin/env python3

import sys
import deepdish
import random
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]


def plot_output_raw_data(record_dict, record_filetype, main_antenna_count):
    """
    :param record_dict: a record containing data_dimensions, data_descriptors, antenna_array_names, and data (reshaped)
    :param record_filetype: a string indicating the type of data being plotted (to be used on the plot legend). Should 
     be raw_rf_data type, but there might be multiple slices.
    """

    # data dimensions are num_sequences, num_antennas, num_samps


    for index in range(0,record_dict['data'].shape[0]): 
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        for antenna in range(0,record_dict['data'].shape[1]):#record_dict['data'].shape[1]):
            #if max(abs(record_dict['data'][index,antenna,18000:20000])) < 0.05:
            #    continue
            if antenna < main_antenna_count:
                ax1.plot(np.arange(2000), record_dict['data'][index,antenna,18000:20000].real, label='Real antenna {}'.format(antenna))
                ax2.plot(np.arange(2000), record_dict['data'][index,antenna,18000:20000].imag, label="Imag {}".format(antenna))
        ax1.legend()   
        ax2.legend()            
        plt.show()


data = deepdish.io.load(filename)

record_name = random.choice(list(data.keys()))
#record_name = '1545153334070'
print(record_name)

raw_rf_data = data[record_name]
number_of_antennas = raw_rf_data['main_antenna_count'] + raw_rf_data['intf_antenna_count']
flat_data = np.array(raw_rf_data['data'])  
# reshape to nave x number of antennas (M0..... I3) x number_of_samples

print(raw_rf_data['num_samps'])

raw_rf_data_data = np.reshape(flat_data, (raw_rf_data['num_sequences'], number_of_antennas, raw_rf_data['num_samps']))
raw_rf_data['data'] = raw_rf_data_data


plot_output_raw_data(raw_rf_data, 'raw_rf_data', raw_rf_data['main_antenna_count'])