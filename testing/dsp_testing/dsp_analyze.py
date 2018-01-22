import json
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import math

plt.rcParams['agg.path.chunksize'] = 20000

def ascii_encode_dict(data):
    ascii_encode = lambda x: x.encode('ascii')
    return dict(map(ascii_encode, pair) for pair in data.items())

class AnalysisUtils(object):

    @staticmethod
    def fft_and_plot(samples, rate):
        fft_samps = fft(samples)
        T = 1.0/float(rate)
        num_samps = len(samples)
        xf = np.linspace(-1.0/(2.0*T),1.0/(2.0*T),num_samps)

        fig, smpplt = plt.subplots(1,1)

        fft_to_plot = np.empty([num_samps],dtype=np.complex64)
        halfway = int(math.ceil(float(num_samps)/2))
        fft_to_plot = np.concatenate([fft_samps[halfway:], fft_samps[:halfway]])
        xf = xf[699900:700100]
        fft_to_plot = fft_to_plot[699900:700100]
        smpplt.plot(xf, 1.0/num_samps * np.abs(fft_to_plot))
        return fig

    @staticmethod
    def make_nco(rate,wave_freqs,num_samps):
        rate = float(rate)
        wave_freqs = [float(wave_freq) for wave_freq in wave_freqs]

        sampling_freqs = [2*math.pi*wave_freq/rate for wave_freq in wave_freqs]
        samples = np.zeros(num_samps,dtype=complex)

        x = np.arange(num_samps)
        rads = np.fmod([[sampling_freq * i for sampling_freq in sampling_freqs] for i in x], 2 * np.pi)
        amp = 1
        for i,rad in enumerate(rads):
            real = amp * np.cos(rad)
            imag = amp * np.sin(rad) * 1j
            cmplx = real + imag
            samples[i] += np.sum(cmplx)

        return samples



class PlotData(object):

    def __init__(self, config):
        self.config = config



    def create_debug_plots(self,rf_samples):
        if not os.path.exists('debug_plots'):
            os.makedirs('debug_plots')

        rx_rate = float(self.config["rx_sample_rate"])

        for i in range(rf_samples.shape[0]):
            ant_samps = rf_samples[i]
            x = np.arange(len(ant_samps))
            fig, smpplt = plt.subplots(1,1)
            smpplt.plot(x,ant_samps.real)
            #smpplt.plot(x,ant_samps.imag)
            plt.show()
            fig.savefig("debug_plots/antenna_{0}_rf.png".format(i))
            plt.close(fig)
            fig = AnalysisUtils.fft_and_plot(ant_samps,rx_rate)
            fig.savefig("debug_plots/antenna_{0}_rf_fft.png".format(i))
            plt.close(fig)

        with open(self.config['filter_outputs_debug_file'],'r') as f:
            data = json.load(f)

        first_stage_rate = float(config["first_stage_sample_rate"])
        second_stage_rate = float(config["second_stage_sample_rate"])
        third_stage_rate = float(config["third_stage_sample_rate"])
        for dsetk,dsetv in data.iteritems():
            for stagek,stagev in dsetv.iteritems():
                if "stage_1" in stagek:
                    rate = first_stage_rate
                elif "stage_2" in stagek:
                    rate = second_stage_rate
                else:
                    rate = third_stage_rate

                for antk,antv in stagev.iteritems():
                    real = antv["real"]
                    imag = antv["imag"]
                    cmplx = np.array(real) + 1.0j * np.array(imag)

                    fig = AnalysisUtils.fft_and_plot(cmplx,rate)
                    output_name = "debug_plots/{0}.{1}.{2}.png".format(dsetk,stagek,antk)
                    fig.savefig(output_name)
                    plt.close(fig)

def open_binary_samples(file_path,num_antennas):
    samples = np.fromfile(file_path,dtype=np.complex64)
    samples = samples.reshape((num_antennas,-1))
    return samples




if __name__ == "__main__":

    file_path = sys.argv[1]


    if not os.environ["BOREALISPATH"]:
        raise ValueError("BOREALISPATH env variable not set")
    config_path = os.environ["BOREALISPATH"] + "/config.ini"

    with open(config_path, 'r') as f:
        config = json.load(f,object_hook=ascii_encode_dict)

    plotting = PlotData(config)

    total_antennas = int(config["main_antenna_count"]) + int(config["interferometer_antenna_count"])
    rf_samples = open_binary_samples(file_path, total_antennas)
    plotting.create_debug_plots(rf_samples)











