# Copyright 2018 SuperDARN Canada
#
# Marci Detwiller
#
# Functions to plot samples and plot the fft of samples.
from scipy import signal
import numpy as np

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math


def plot_samples(filename, samplesa, **kwargs):
    """
    Plot samples to a file for testing. Can be any samples including filter taps to
    plot in the time domain.

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


def get_samples(rate, wave_freq, sampleslen):
    rate = float(rate)
    wave_freq = float(wave_freq)

    sampling_freq = 2 * math.pi * wave_freq / rate
    samples = np.empty([sampleslen], dtype=complex)
    for i in range(0, sampleslen):
        amp = 1
        rads = math.fmod(sampling_freq * i, 2 * math.pi)
        samples[i] = amp * math.cos(rads) + amp * math.sin(rads) * 1j
    return samples


def plot_fft(samplesa, rate, filename=None):
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
        halfway = int(num_samps / 2)
        for sample in range(halfway, num_samps):
            fft_to_plot[sample - halfway] = fft_samps[sample]
            # Move negative samples to start for plot
        for sample in range(0, halfway):
            fft_to_plot[sample + halfway] = fft_samps[sample]
            # Move positive samples at end
    smpplt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot))
    #    plt.xlim([-2500000,-2000000])
    if filename is not None:
        fig.savefig(filename)
        plt.close(fig)
        return None
    else:
        return fig


def create_remez_filter(num_taps, freq_s, cutoff, transition, maxiteration=5000000):
    """
    Create a remez filter using scipy and return the filter taps. If decimating, cutoff must be
    at or below the new sampling frequency after decimation in order to avoid aliasing (with complex samples).
    If the samples are not complex, then the cutoff should be the new sampling frequency /2.
    :param num_taps: number of taps for the filter, int
    :param freq_s: current sampling frequency of the data
    :param cutoff: cutoff for the filter, where the passband for the low pass filter ends.
    :param transition: transition bandwidth from cutoff of passband to stopband
    :param maxiteration: max iteration, optional, default 5000000.
    :returns filter_taps: the filter taps of the resolved remez filter.
    """
    filter_taps = signal.remez(
        num_taps,
        [
            x * freq_s
            for x in [0.0, cutoff / freq_s, (cutoff + transition) / freq_s, 0.5]
        ],
        [1, 0],
        Hz=freq_s,
        maxiter=maxiteration,
    )
    return filter_taps


def create_impulse_boxcar(decimation_rates, offset):
    """
    Create a boxcar function to evaluate the impulse response of cascading filters and decimation.
    The boxcar is the impulse (once decimated) The offset typically determined by the
    max lengths of the filters.
    :param decimation_rates: list of decimation rates, to determine boxcar length
    :param offset: number of zeros to pad at the beginning and end for full convolution response.
    :returns signal: real only signal with boxcar.
    """
    length_of_impulse = 1
    for decimation in decimation_rates:
        length_of_impulse = length_of_impulse * decimation
    boxcar = [0.0] * offset
    boxcar.extend([1.0] * length_of_impulse)
    boxcar.extend([0.0] * offset)
    return boxcar


def plot_filter_response(filter_taps, title_identifier, sampling_freq):
    """
    Plot filter response given filter taps
    sampling_freq : Hz
    """

    w, h = signal.freqz(filter_taps, whole=True)

    w = w * sampling_freq / (2 * math.pi)  # w now in Hz

    fig = plt.figure()
    plt.title("Digital filter frequency response {}".format(title_identifier))
    ax1 = fig.add_subplot(111)
    plt.plot(w, 20 * np.log10(abs(h)), "b")
    plt.ylabel("Amplitude [dB]", color="b")
    plt.xlabel("Frequency [Hz]")
    ax2 = ax1.twinx()  # fig.add_subplot(111)
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles, "g")
    plt.ylabel("Angle (radians)", color="g")
    plt.grid()
    plt.axis("tight")

    plt.show()


def get_num_taps_for_remez_filter(freq_s, transition_band, k):
    """
    Calculates number of filter taps according to Lyon's Understanding Digital
    Signal Processing(1st edition). Uses Eqn 7-6 to calculate how many filter taps should be used
    for a given stage. The choice in k=3 was used in the book seems to minimize the amount of
    ripple in filter. The number of taps will always truncate down to an int.
    :param freq_s: sampling frequency of the current data to be filtered.
    :param transition_band: desired transition band for the filter
    :param k: a const multiplier to increase FIR filter order, if desired to reduce ripple.
    """
    return int(k * (freq_s / transition_band))


def create_blackman_window(N):
    """
    N = length of window
    """
    M = (N - 1) / 2
    blackman = []
    for n in range(0, N):
        blackman.append(
            0.42
            + 0.5 * math.cos((2 * math.pi * (n - M)) / (2 * M + 1))
            + 0.08 * math.cos((4 * math.pi * (n - M)) / (2 * M - 1))
        )
    return blackman


def shift_to_bandpass(lpass_taps, shift_freq, rate):
    """
    Take a numpy array of lowpass taps and shift it to turn it to a bandpass filter.

    :param lpass_taps: numpy array of lowpass filter taps.
    :param shift_freq: frequency to shift the taps at.
    :param rate: rate that this filter will be applied to.
    """
    shift_wave = get_samples(rate, shift_freq, len(lpass_taps))
    bpass_taps = np.array([l * i for l, i in zip(lpass_taps, shift_wave)])
    return bpass_taps
