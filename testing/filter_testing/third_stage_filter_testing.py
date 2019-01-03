# investigation of the third stage filter to avoid 'intersymbol interference' 
# or in our case range gate interference due to the length of the filter

from scipy import signal
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import math
 
def plot_fft(samplesa, rate):
    fft_samps=fft(samplesa)
    T= 1.0 /float(rate)
    num_samps=len(samplesa)
    xf=np.linspace(-1.0/(2.0*T),1.0/(2.0*T),num_samps)
    #print len(xf), len(fft_samps)
    fig, smpplt = plt.subplots(1,1)
    fft_to_plot=np.empty([num_samps],dtype=complex)
    if num_samps%2==1:
        halfway=(num_samps+1)/2
        for sample in range(halfway,num_samps):
            fft_to_plot[sample-halfway]=fft_samps[sample]
            # Move negative samples to start for plot
        for sample in range(0,halfway):
            fft_to_plot[sample+halfway-1]=fft_samps[sample]
            # Move positive samples at end
    else:
        halfway=num_samps/2
        for sample in range(halfway,num_samps):
            fft_to_plot[sample-halfway]=fft_samps[sample]
            # Move negative samples to start for plot
        for sample in range(0,halfway):
            fft_to_plot[sample+halfway]=fft_samps[sample]
            # Move positive samples at end
    smpplt.plot(xf, 1.0/num_samps * np.abs(fft_to_plot))
#    plt.xlim([-2500000,-2000000])
    return fig

def get_samples(rate,wave_freq,filter_len):
    rate = float(rate)
    wave_freq = float(wave_freq)

    sampling_freq=2*math.pi*wave_freq/rate
    sampleslen=filter_len
    samples=np.empty([sampleslen],dtype=complex)
    for i in range(0,sampleslen):
        amp=1
        rads=math.fmod(sampling_freq*i,2*math.pi)
        samples[i]=amp*math.cos(rads)+amp*math.sin(rads)*1j
    return samples





filter_len=60
lpass = signal.remez(filter_len, [x * 10000 for x in [0, .1, .15, .5]], [1,0], Hz=10000, maxiter=50000000)
lpass = np.concatenate((lpass,np.array([0,0,0,0])))
shift_wave = get_samples(10000,-1000,filter_len)
bpass = np.array([l*i for l,i in zip(lpass,shift_wave)])


# w,h = signal.freqz(bpass, whole=True)

# fig4 = plt.figure()
# plt.plot(np.arange(len(bpass)),bpass)
# plt.plot(np.arange(len(lpass)),lpass)
# fig = plt.figure()
# plt.title('Digital filter frequency response')
# ax1 = fig.add_subplot(111)
# plt.plot(w, 20 * np.log10(abs(h)), 'b')
# plt.ylabel('Amplitude [dB]', color='b')
# plt.xlabel('Frequency [rad/sample]')
# ax2 = ax1.twinx()
# angles = np.unwrap(np.angle(h))
# plt.plot(w, angles, 'g')
# plt.ylabel('Angle (radians)', color='g')
# plt.grid()
# plt.axis('tight')

# fig2 = plot_fft(bpass,22050)
# fig3 = plot_fft(lpass,22050)

# plt.show()

boxcar = [0.0] * 360
boxcar.extend([1.0] * 30)
boxcar.extend([0.0] * 360)

output = signal.convolve(boxcar,filter_taps,mode='full')

# get all possible scenarios depending on the location of the pulse echo in the data
for start_sample in range(0, 30):
    decimated_output = output[start_sample::30]
    plt.plot(np.arange(len(decimated_output)), decimated_output)

plt.title('Decimated Pulse Response')
plt.show()