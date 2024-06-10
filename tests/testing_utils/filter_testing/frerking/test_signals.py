import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift


def create_signal_1(freq1, freq2, num_samps, rate):

    f1 = freq1 + 2 * np.random.randn(num_samps)
    f2 = freq2 + 2 * np.random.randn(num_samps)

    t = np.arange(num_samps) / rate

    sig = 10 * np.exp(1j * 2 * np.pi * f1 * t) + 10 * np.exp(1j * 2 * np.pi * f2 * t)

    return sig


def plot_fft(samplesa, rate):
    fft_samps = fft(samplesa)
    T = 1.0 / float(rate)
    num_samps = len(samplesa)
    if num_samps % 2 == 1:
        xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)
    else:
        # xf = np.arange(-1.0/(2.0*T), 1.0/(2.0*T),1.0/(T*num_samps))
        xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)
    print(num_samps)
    print(len(fft_samps))
    print(len(xf))
    fig, smpplt = plt.subplots(1, 1)
    fft_to_plot = np.empty([num_samps], dtype=complex)
    fft_to_plot = fftshift(fft_samps)
    smpplt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot))
    #    plt.xlim([-2500000,-2000000])
    return fig


# sig1 = create_signal_1(10000,25e3)

# fig_1 = plot_fft(sig1,25e3)

# plt.show()
