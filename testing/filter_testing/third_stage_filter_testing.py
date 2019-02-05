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

def get_samples(rate,wave_freq,sampleslen):
    rate = float(rate)
    wave_freq = float(wave_freq)

    sampling_freq=2*math.pi*wave_freq/rate
    samples=np.empty([sampleslen],dtype=complex)
    for i in range(0,sampleslen):
        amp=1
        rads=math.fmod(sampling_freq*i,2*math.pi)
        samples[i]=amp*math.cos(rads)+amp*math.sin(rads)*1j
    return samples


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
    return int(k * (freq_s/transition_band))


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
    filter_taps = signal.remez(num_taps, [x * freq_s for x in [0.0, cutoff/freq_s, (cutoff+ transition)/freq_s, 0.5]], [1,0], Hz=freq_s, maxiter=maxiteration)
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
        length_of_impulse  = length_of_impulse * decimation
    boxcar = [0.0] * offset
    boxcar.extend([1.0] * length_of_impulse)
    boxcar.extend([0.0] * offset)
    return boxcar


def plot_filter_response(filter_taps, title_identifier, sampling_freq):
    """
    Plot filter response given filter taps
    sampling_freq : Hz
    """

    w,h = signal.freqz(filter_taps, whole=True)

    w = w * sampling_freq/(2 * math.pi) # w now in Hz

    fig = plt.figure()
    plt.title('Digital filter frequency response {}'.format(title_identifier))
    ax1 = fig.add_subplot(111)
    plt.plot(w, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [rad/sample]')
    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')

    plt.show()


def create_blackman_window(N):
    """
    N = length of window
    """
    M=(N-1)/2
    blackman = []
    for n in range(0, N): 
        blackman.append(0.42 + 0.5*math.cos((2*math.pi*(n-M))/(2*M+1) ) + 0.08*math.cos((4*math.pi*(n-M))/(2*M-1)))
    return blackman


def plot_decimated_impulse_response(num_stages, numtaps, filter_taps, decimation):
    """
    num_stages: int
    numtaps: list, of ints of number of taps in each filter
    filter_taps: list of list, of filter taps for each stage
    decimation: list of ints of decimation for each stage.
    """

    fig, [filter_response_plots, output_plots] = plt.subplots(2, num_stages)

    for stage in range(0, num_stages):
        filter_response_plots[stage].plot(np.arange(numtaps[stage]), filter_taps[stage])
        filter_response_plots[stage].set_title('Filter Response Stage {}'.format(stage+1))

    for i in range(0, num_stages):
        for start_sample in range(0, decimation[i]):
            decimated_output = all_decimated_filter_outputs[i][start_sample]
            output_plots[i].plot(np.arange(len(decimated_output)), decimated_output)
        output_plots[i].set_title('After Stage {}'.format(stage+1))

    # get all possible scenarios depending on the location of the pulse echo in the data

    plt.show()


def create_original_filter_plots():
    k = 3
    rx_rate = 5000000.0
    decimation = [5, 10, 30]
    print('Decimation: {}'.format(decimation))
    num_stages = len(decimation)
    decimation_total = 1
    total_decimation_per_stage = []
    for stage in range(0, num_stages):
        total_decimation_per_stage.append(decimation_total)
        decimation_total = decimation_total * decimation[stage]
    freq_s = [rx_rate/i for i in total_decimation_per_stage]
    print('Sampling Freq per stage: {}'.format(freq_s))
    cutoff =     [1.0e6,   100.0e3,  3.333e3]
    transition = [500.0e3, 50.0e3,   0.833e3]

    numtaps = [get_num_taps_for_remez_filter(freq_s[0], transition[0], 3)]
    numtaps.append(get_num_taps_for_remez_filter(freq_s[1], transition[1], 3))
    numtaps.append(get_num_taps_for_remez_filter(freq_s[2], transition[2], 3))

    filter_taps = []
    for i in range(0, num_stages):
        filter_taps.append(create_remez_filter(numtaps[i], freq_s[i], cutoff[i], transition[i]))

    boxcar = create_impulse_boxcar(decimation, numtaps[0]) #numtaps[2] * decimation[0] * decimation[1])

    output = [boxcar]
    all_decimated_filter_outputs = []
    for i in range(0, num_stages):
        print('stage: {}'.format(i))
        print(len(output[i]), len(filter_taps[i]))
        filter_out = signal.convolve(output[i],filter_taps[i],mode='full')
        decimated_filter_out = []
        for start_sample in range(0, decimation[i]):
            decimated_output = filter_out[start_sample::decimation[i]]
            if start_sample == 0:
                # align samples, take this as the correct length (the max length)
                decimated_length = len(decimated_output)
            if len(decimated_output) != decimated_length:
                decimated_output = np.concatenate((np.array([0.0]), decimated_output))
            decimated_filter_out.append(decimated_output)
        output.append(decimated_filter_out[0]) # take the first one for now to carry over to next stage.
        all_decimated_filter_outputs.append(decimated_filter_out)

    fig, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3)
    ax1.plot(np.arange(numtaps[0]), filter_taps[0])
    ax2.plot(np.arange(numtaps[1]), filter_taps[1])
    ax3.plot(np.arange(numtaps[2]), filter_taps[2])
    ax1.set_title('Filter Response Stage 1')
    ax2.set_title('Filter Response Stage 2')
    ax3.set_title('Filter Response Stage 3')

    ax4.set_title('After Stage 1')
    ax5.set_title('After Stage 2')
    ax6.set_title('After Stage 3')

    output_plots = [ax4, ax5, ax6]

    for i in range(0, num_stages):
        for start_sample in range(0, decimation[i]):
            decimated_output = all_decimated_filter_outputs[i][start_sample]
            output_plots[i].plot(np.arange(len(decimated_output)), decimated_output)

    # get all possible scenarios depending on the location of the pulse echo in the data


    plt.show()


rx_rate = 5000000.0
decimation = [5, 10, 30]
print('Decimation: {}'.format(decimation))
num_stages = len(decimation)
decimation_total = 1
total_decimation_per_stage = [1]
for stage in range(0, num_stages):
    decimation_total = decimation_total * decimation[stage]
    total_decimation_per_stage.append(decimation_total)
freq_s = [rx_rate/i for i in total_decimation_per_stage[:-1]]
print('Sampling Freq per stage: {}'.format(freq_s))
#cutoff =     [1.0e6,   100.0e3,  3.333e3]
cutoff = [rx_rate/total_decimation_per_stage[1], rx_rate/total_decimation_per_stage[2], rx_rate/total_decimation_per_stage[3]]
#transition = [500.0e3,  50.0e3,  0.833e3]
transition = [cutoff[0]/2.0, cutoff[1]/2.0, cutoff[2]/4.0]

numtaps = [get_num_taps_for_remez_filter(freq_s[0], transition[0], 3)]
numtaps.append(get_num_taps_for_remez_filter(freq_s[1], transition[1], 3))
numtaps.append(get_num_taps_for_remez_filter(freq_s[2], transition[2], 3))

filter_taps = []
for i in range(0, num_stages):
    filter_taps.append(create_remez_filter(numtaps[i], freq_s[i], cutoff[i], transition[i]))

boxcar = create_impulse_boxcar(decimation, numtaps[0]) #numtaps[2] * decimation[0] * decimation[1])
cw_lp_wave = get_samples(rx_rate, cutoff[2], len(boxcar))
input_signal = np.array(cw_lp_wave) * np.array(boxcar)
output = [input_signal]
all_decimated_filter_outputs = []
for i in range(0, num_stages):
    print('stage: {}'.format(i))
    print(len(output[i]), len(filter_taps[i]))
    filter_out = signal.convolve(output[i],filter_taps[i],mode='full')
    decimated_filter_out = []
    for start_sample in range(0, decimation[i]):
        decimated_output = filter_out[start_sample::decimation[i]]
        # if start_sample == 0:
        #     # align samples, take this as the correct length (the max length)
        #     decimated_length = len(decimated_output)
        # if len(decimated_output) != decimated_length:
        #     decimated_output = np.concatenate((np.array([0.0]), decimated_output))
        decimated_filter_out.append(decimated_output)
    output.append(decimated_filter_out[0]) # take the first one for now to carry over to next stage.
    all_decimated_filter_outputs.append(decimated_filter_out)


#plot_decimated_impulse_response(num_stages, numtaps, filter_taps, decimation)

for stage in range(0, num_stages):
    plot_filter_response(filter_taps[stage], stage, freq_s[stage])

ros_cfir_filter = [-24, 74, 494, 548, -977, -3416, -3672, 1525, 13074, 26547, 32767] 
ros_cfir_filter = ros_cfir_filter + list(reversed(ros_cfir_filter.copy()[:-1]))
ros_cfir_freq = 40625000

plot_filter_response(ros_cfir_filter, "ROS CFIR", ros_cfir_freq)

ros_pfir_filter = [14, 30, 41, 27, -29, -118, -200, -212, -95, 150,
                     435, 598, 475, 5, -680, -1256, -1330, -653, 669,
                     2112, 2880, 2269, 101, -2996, -5632, 6103,
                     -3091, 3666, 13042, 22747, 30053, 32767]
ros_pfir_filter = ros_pfir_filter + list(reversed(ros_pfir_filter.copy()[:-1]))
ros_pfir_freq = ros_cfir_freq/2031.0

#plot_filter_response(ros_pfir_filter, "PFIR no window")

blackman = create_blackman_window(63)

def calculate_pfir(window):
    Fpass=Fstop=3333
    freq_in = 10000
    wp=math.pi*float(Fpass)/float(freq_in)
    ws=math.pi*float(Fstop)/float(freq_in)
    wc=(wp+ws)/2
    PFIRgain = 0
    pfircoeffs = []
    for n in range(0,31):
        pfircoeffs.append(int(32767*window[n]*(math.pi/wc)*math.sin(wc*(float(n-31)))/(math.pi*(float(n-31)))+.49999))
        PFIRgain = PFIRgain + pfircoeffs[n]

    pfircoeffs.append(32767)
    PFIRgain=2*PFIRgain+pfircoeffs[31] # this is the total sum of the coefficients once reversed list is concatenated to itself for full filter.
    PFIRgain=PFIRgain/65536

    if PFIRgain > 1:
        gaintemp=PFIRgain
        PFIRgain=0
        for n in range(0, 31):
            pfircoeffs[n]=int((1/gaintemp)*float(pfircoeffs[n])+0.499999)
    PFIRgain=PFIRgain+pfircoeffs[n]
    pfircoeffs[31]=int((1/gaintemp)*float(pfircoeffs[31])+0.49999 )
    PFIRgain=2*PFIRgain+pfircoeffs[31];
    PFIRgain=PFIRgain/65536;
    print('PFIR gain: {}'.format(PFIRgain))
    return pfircoeffs

calculated_pfir = calculate_pfir(blackman)
calculated_pfir = calculated_pfir + list(reversed(calculated_pfir.copy()[:-1]))

plot_filter_response(calculated_pfir, 'calculated blackman pfir', ros_pfir_freq)
