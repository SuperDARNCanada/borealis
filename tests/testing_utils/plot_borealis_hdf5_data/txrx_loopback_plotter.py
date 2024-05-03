#!/usr/bin/env python3

# this file is for plotting data taken using the UHD example for txrx loopback.

import sys
import numpy as np
import matplotlib.pyplot as plt

filename_template = sys.argv[1]

channels = ["00", "01", "02", "03"]
file_ext = ".bin"

# filenames = [filename_template + file_ext]
filenames = [filename_template + "." + i + file_ext for i in channels]


def plot_uhd_txrx_loopback_data(record_array, record_filetype):
    """
    :param record_array: num_antennas x num_samples
    :param record_filetype: a string indicating the type of data being plotted (to be used on the plot legend). Should
     be raw_rf_data type, but there might be multiple slices.
    """

    # data dimensions are num_antennas, num_samps

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    # plt.settitle(record_filetype)
    for antenna in range(record_array.shape[0]):
        # if max(abs(record_dict['data'][index,antenna,18000:20000])) < 0.05:
        #    continue
        ax1.plot(
            np.arange(record_array.shape[1]),
            record_array[antenna, :].real,
            label="Real {}".format(antenna),
        )
        ax2.plot(
            np.arange(record_array.shape[1]),
            record_array[antenna, :].imag,
            label="Imag {}".format(antenna),
        )
    ax1.legend()
    plt.show()


data_list = []
for filename in filenames:
    print(filename)
    with open(filename, "rb") as f:
        data = f.read()
        data_list.append(np.frombuffer(data, dtype=np.complex64)[1750000:1760000])

data_array = np.array(data_list, dtype=np.complex64)


number_of_antennas = 4

# fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
# #plt.settitle(record_filetype)
# ax1.plot(np.arange(record_array.shape[0]), record_array[:].real, label='Real')
# ax2.plot(np.arange(record_array.shape[0]), record_array[:].imag, label="Imag")
# ax1.legend()
# plt.show()

plot_uhd_txrx_loopback_data(data_array, "txrx_loopback_data")
