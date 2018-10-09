# Copyright 2018 SuperDARN Canada
#
# Marci Detwiller
#
# Functions to plot samples and plot the fft of samples.

import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

def plot_samples(filename, samplesa, **kwargs):
    """
    Plot samples to a file for testing.
    
    :param filename: the filename to save the plot as.
    :param samplesa: Some samples to plot
    :param kwargs: any more sample arrays to plot.
    """

    more_samples = dict(kwargs)

    fig, smpplot = plt.subplots(1, 1)
    smpplot.plot(range(0, samplesa.shape[0]), samplesa)
    for sample_name, sample_array in more_samples.items():
        smpplot.plot(range(0, sample_array.shape[0]), sample_array)
    # plt.ylim([-1, 1])
    # plt.xlim([0, len(samplesa)])
    fig.savefig(filename)
    plt.close(fig)


def plot_fft(filename, samplesa, rate):
    """
    For plotting the fft of test samples. 
    
    Plot the double-sided FFT of the samples in Hz (-fs/2 to +fs/2)
    
    :param filename: The filename to save the plot as. 
    :param samplesa: The time-domain samples to take the fft of.
    :param rate: The sampling rate that the samples were taken at (Hz).
    """

    fft_samps = fft(samplesa)
    T = 1.0 / float(rate)
    num_samps = len(samplesa)
    xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)
    # print len(xf), len(fft_samps)
    fig, smpplt = plt.subplots(1, 1)
    fft_to_plot = np.empty([num_samps], dtype=np.complex64)
    if num_samps % 2 == 1:
        halfway = (num_samps + 1) / 2
        for sample in range(halfway, num_samps):
            fft_to_plot[sample - halfway] = fft_samps[sample]
            # Move negative samples to start for plot
        for sample in range(0, halfway):
            fft_to_plot[sample + halfway - 1] = fft_samps[sample]
            # Move positive samples at end
    else:
        halfway = num_samps / 2
        for sample in range(halfway, num_samps):
            fft_to_plot[sample - halfway] = fft_samps[sample]
            # Move negative samples to start for plot
        for sample in range(0, halfway):
            fft_to_plot[sample + halfway] = fft_samps[sample]
            # Move positive samples at end
    smpplt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot))
    #    plt.xlim([-2500000,-2000000])
    fig.savefig(filename)
    plt.close(fig)
    return None


