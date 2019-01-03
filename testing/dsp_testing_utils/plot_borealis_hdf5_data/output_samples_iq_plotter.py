#!/usr/bin/env python3

import sys
import deepdish
import random
import numpy as np
import matplotlib.pyplot as plt

filename = sys.argv[1]


def plot_output_samples_iq_data(record_dict, record_filetype):
    """
    :param record_dict: a record containing data_dimensions, data_descriptors, antenna_array_names, and data (reshaped)
    :param record_filetype: a string indicating the type of data being plotted (to be used on the plot legend). Should 
     be output_samples_iq type, but there might be multiple slices.
    """

    # data dimensions are num_antennas, num_sequences, num_samps

    antennas_present = [int(i.split('_')[-1]) for i in record_dict['antenna_arrays_order']]

    for sequence in range(0, record_dict['data'].shape[1]):
        print('Sequence number: {}'.format(sequence))

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)         
        for index in range(0, record_dict['data'].shape[0]):
            antenna = antennas_present[index]

            if antenna < record_dict['main_antenna_count']:
                if antenna == 7 or antenna == 8:
                    ax1.set_title('Main Antennas {}'.format(record_filetype))
                    ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].real, label='Real {}'.format(antenna))
                    #ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].imag, label="Imag {}".format(antenna))
                    ax1.legend()
            # else:
            #     ax2.set_title('Intf Antennas {}'.format(record_filetype))
            #     ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].real, label='Real {}'.format(antenna))
            #     ax2.plot(np.arange(record_dict['num_samps']), record_dict['data'][antenna,sequence,:].imag, label="Imag {}".format(antenna))
            #     ax2.legend()                       
        plt.show()
    # for index in range(0, record_dict['data'].shape[0]):
    #     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    #     ax1.set_title('Main Antennas {}'.format(record_filetype))
    #     ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,:].real, label='Real {}'.format(index))
    #     ax1.plot(np.arange(record_dict['num_samps']), record_dict['data'][index,:].imag, label="Imag {}".format(index))
    #     ax1.legend()               
    #     plt.show()


data = deepdish.io.load(filename)

record_name = random.choice(list(data.keys()))
print(record_name)
output_samples_iq = data[record_name]
number_of_antennas = len(output_samples_iq['antenna_arrays_order'])

flat_data = np.array(output_samples_iq['data'])  
# reshape to number of antennas (M0..... I3) x nave x number_of_samples
output_samples_iq_data = np.reshape(flat_data, (number_of_antennas, output_samples_iq['num_sequences'], output_samples_iq['num_samps']))
output_samples_iq['data'] = output_samples_iq_data
antennas_present = [int(i.split('_')[-1]) for i in output_samples_iq['antenna_arrays_order']]
output_samples_iq['antennas_present'] = antennas_present

plot_output_samples_iq_data(output_samples_iq, 'output_samples_iq')