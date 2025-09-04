from scipy import signal
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
import math
import cmath

# import test_signals
import sys


def create_signal_1(freq1, freq2, num_samps, rate):

    f1 = freq1 + 2 * np.random.randn(num_samps)
    f2 = freq2 + 2 * np.random.randn(num_samps)

    t = np.arange(num_samps) / rate

    sig = 10 * np.exp(1j * 2 * np.pi * f1 * t) + 10 * np.exp(1j * 2 * np.pi * f2 * t)

    return sig


def create_signal_2(freq1, freq2, num_samps, rate):

    t = np.arange(num_samps) / rate

    sig = 10 * np.exp(1j * 2 * np.pi * freq1 * t) + 10 * np.exp(
        1j * 2 * np.pi * freq2 * t
    )

    return sig


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def plot_fft(samplesa, rate, title):
    fft_samps = fft(samplesa)
    T = 1.0 / float(rate)
    num_samps = len(samplesa)
    if num_samps % 2 == 1:
        xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)
    else:
        # xf = np.arange(-1.0/(2.0*T), 1.0/(2.0*T),1.0/(T*num_samps))
        xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fft_to_plot = np.empty([num_samps], dtype=complex)
    fft_to_plot = fftshift(fft_samps)
    plt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot))
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    ax2 = ax1.twinx()
    plt.ylabel("Phase [rads]", color="g")
    angles = np.angle(fft_to_plot)
    plt.plot(xf, angles, "m")
    return fig


def plot_all_ffts(bpass_filters, rate, title):
    fig, smpplt = plt.subplots(1, 1)
    for filt in bpass_filters:
        fft_samps = fft(filt)
        T = 1.0 / float(rate)
        num_samps = len(filt)
        if num_samps % 2 == 1:
            xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)
        else:
            # xf = np.arange(-1.0/(2.0*T), 1.0/(2.0*T),1.0/(T*num_samps))
            xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)
        fft_to_plot = np.empty([num_samps], dtype=complex)
        fft_to_plot = fftshift(fft_samps)
        smpplt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot))
    smpplt.set_title(title)
    smpplt.set_xlabel("Frequency (Hz)")
    smpplt.set_ylabel("Amplitude")
    return fig


def get_samples(rate, wave_freq, numberofsamps, start_rads):
    rate = float(rate)
    wave_freq = float(wave_freq)
    start_rads = float(start_rads)

    print(start_rads)
    sampling_freq = 2 * math.pi * wave_freq / rate
    sampleslen = int(numberofsamps)
    samples = np.empty([sampleslen], dtype=complex)
    for i in range(0, sampleslen):
        amp = 1
        rads = math.fmod(start_rads + (sampling_freq * i), 2 * math.pi)
        samples[i] = amp * math.cos(rads) + amp * math.sin(rads) * 1j
    return samples


def downsample(samples, rate):
    rate = int(rate)
    sampleslen = len(samples) / rate + 1  # should be an int
    samples_down = np.empty([sampleslen], dtype=complex)
    samples_down[0] = samples[0]
    print(sampleslen)
    for i in range(1, len(samples)):
        if i % rate == 0:
            # print(i/rate)
            samples_down[i / rate] = samples[i]
    return samples_down


def fftnoise(f):
    f = np.array(f, dtype="complex")
    Np = (len(f) - 1) // 2
    phases = np.random.rand(Np) * 2 * np.pi
    phases = np.cos(phases) + 1j * np.sin(phases)
    f[1 : Np + 1] *= phases
    f[-1 : -1 - Np : -1] = np.conj(f[1 : Np + 1])
    return np.fft.ifft(f).real


def band_limited_noise(min_freq, max_freq, samples=1024, samplerate=1):
    freqs = np.abs(np.fft.fftfreq(samples, 1 / samplerate))
    f = np.zeros(samples)
    idx = np.where(np.logical_and(freqs >= min_freq, freqs <= max_freq))[0]
    f[idx] = 1
    return fftnoise(f)


# SET VALUES
# Low-pass filter design parameters
fs = 12e6  # Sample rate, Hz
wave_freq = -0.96e6  # 1.8 MHz below centre freq (12.2 MHz if ctr = 14 MHz)
ctrfreq = 14000  # kHz
cutoff = 400e3  # Desired cutoff frequency, Hz
trans_width = 50e3  # Width of transition from pass band to stop band, Hz
numtaps = 1024  # Size of the FIR filter.

decimation_rate = 200.0
pmax = 100  # max value of P (integer) that we will run the script at.

# Set the number of output samples first so we can get correct number of
#   input samples to have our three frequencies all perfectly in the FFT
#   bins to check phase linearity.
num_output_samps = 5001

if numtaps > decimation_rate:
    num_input_samps = num_output_samps * decimation_rate + numtaps
    # num_output_samps = int((len(pulse_samples)-len(lpass))/decimation_rate)
else:
    num_input_samps = num_output_samps * decimation_rate
    # num_output_samps = int(len(pulse_samples)/decimation_rate)

fft_bin_increment = (fs / decimation_rate) / ((num_output_samps - 1) / 2)

# Set the frequencies depending on FFT bins so that we can prove
#   phase linearity.
freq_2 = wave_freq + 10 * fft_bin_increment
freq_3 = wave_freq - 10 * fft_bin_increment

# Calculate for Frerking's filter, Rf/fs which must be rational.

frerking = abs(decimation_rate * wave_freq / fs)
# find number of filter coefficients

for x in range(1, int(fs)):
    if x * frerking % 1 == 0:
        number_of_coeff_sets = x
        break
else:  # no break
    sys.exit(["Error: could not find number of coefficient sets, greater than fs"])

if number_of_coeff_sets > pmax:
    sys.exit(
        [
            "Error: number of coefficient sets required is too large: %d"
            % number_of_coeff_sets
        ]
    )

# pulse_samples = test_signals.create_signal_1(wave_freq,4.0e6,10000,fs) # with added noise
# pulse_samples = 0.008*np.asarray(random.sample(range(-10000,10000),10000))
# pulse_samples = band_limited_noise(-6000000,6000000,10000,fs)
pulse_samples = create_signal_2(
    wave_freq, 0, num_input_samps, fs
)  # without noise added
pulse_samples += create_signal_2(freq_2, freq_3, num_input_samps, fs)

print("Fs = %d" % fs)
print("F = %d" % wave_freq)
print("R = %d" % decimation_rate)
print("P = %d" % number_of_coeff_sets)

fig1 = plot_fft(pulse_samples, fs, "FFT of Original Pulse Samples")

lpass = signal.remez(
    numtaps, [0, cutoff, cutoff + trans_width, 0.5 * fs], [1, 0], Hz=fs
)

bpass = np.array([])

for i in range(0, number_of_coeff_sets):
    if i == 0:
        print(number_of_coeff_sets)
        start_rads = 0
        shift_wave = get_samples(fs, wave_freq, numtaps, start_rads)
        # we need a number of bpass filters depending on number_of_coeff_sets
        bpass = np.array([[l * i for l, i in zip(lpass, shift_wave)]])
    else:
        # shift wave needs to start in a different location
        # start at sampling rate * nth sample we are on (i * decimation_rate)
        start_rads = -math.fmod(
            (2 * math.pi * wave_freq / fs) * i * decimation_rate, 2 * math.pi
        )
        print(start_rads)
        shift_wave = get_samples(fs, wave_freq, numtaps, start_rads)
        bpass = np.append(bpass, [[l * i for l, i in zip(lpass, shift_wave)]], axis=0)


# have to implement special convolution with multiple filters.

#
#
#
# CALCULATE USING FRERKING'S METHOD OF MULTIPLE COEFF SETS.

if len(lpass) > decimation_rate:
    first_sample_index = len(lpass)
else:
    first_sample_index = decimation_rate

output1 = np.array([], dtype=complex)
for x in range(0, num_output_samps):
    bpass_filt_num = x % number_of_coeff_sets
    sum_array = np.array(
        [
            l * i
            for l, i in zip(
                pulse_samples[
                    (first_sample_index + x * decimation_rate - len(lpass)) : (
                        first_sample_index + x * decimation_rate
                    )
                ],
                bpass[bpass_filt_num][::-1],
            )
        ]
    )
    # sum_array = np.array([l*i for l,i in zip(pulse_samples[(x*len(lpass)):((x+1)*len(lpass))],bpass[bpass_filt_num][::-1])])
    output_sum = 0.0
    for element in sum_array:
        output_sum += element
    output1 = np.append(output1, output_sum)

print(num_output_samps)
# Uncomment to plot the fft after first filter stage.
# response1 = plot_fft(output,fs)


fig2 = plot_all_ffts(bpass, fs, "FFT of All Bandpass Filters Using Frerking's Method")
fig3 = plot_fft(lpass, fs, "FFT of Lowpass Filter")

fig4 = plt.figure()
plt.title("Frequency Responses of the P Bandpass Filters (Amp)")
plt.ylabel("Amplitude [dB]", color="b")
plt.xlabel("Frequency [rad/sample]")
plt.grid()
for i in range(0, number_of_coeff_sets):
    w, h = signal.freqz(bpass[i], whole=True)
    # ax1 = fig.add_subplot(111)
    plt.plot(w, 20 * np.log10(abs(h)))
plt.axis("tight")

fig5 = plt.figure()
plt.title("Frequency Responses of the P Bandpass Filters (Phase)")
plt.xlabel("Frequency [rad/sample]")
plt.ylabel("Angle (radians)", color="g")
plt.grid()
for i in range(0, number_of_coeff_sets):
    w, h = signal.freqz(bpass[i], whole=True)
    # ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w, angles)
plt.axis("tight")

#
#
#
# CALCULATE USING ALEX'S METHOD, TRANSLATING SAMPLES AFTER DECIMATION.

output2 = np.array([], dtype=complex)
for x in range(0, num_output_samps):
    bpass_filt_num = 0
    sum_array = np.array(
        [
            l * i
            for l, i in zip(
                pulse_samples[
                    (first_sample_index + x * decimation_rate - len(lpass)) : (
                        first_sample_index + x * decimation_rate
                    )
                ],
                bpass[bpass_filt_num][::-1],
            )
        ]
    )
    output_sum = 0.0
    for element in sum_array:
        output_sum += element
    output2 = np.append(output2, output_sum)

# fig10 = plot_fft(output2, fs/decimation_rate, 'FFT of New Method Before Phase Correction')

# Phase shift after Convolution.
for i in range(0, number_of_coeff_sets):
    # calculate the offset.
    start_rads = math.fmod(
        (2 * math.pi * wave_freq / fs) * i * decimation_rate, 2 * math.pi
    )
    # offset every nth + i sample
    n = i
    while n < num_output_samps:
        output2[n] = output2[n] * cmath.exp(-1j * start_rads)
        n += number_of_coeff_sets

# fig9 = plot_fft(output2, fs/decimation_rate, 'FFT of New Method Output')

#
#
#
# CALCULATE USING MIXING THEN DECIMATION, IN TWO STEPS.

# shifting the signal not the filter so we must shift in the other direction.
# we have to start the shift_wave in the right spot (offset by the first sample index that was used above)
shift_wave = get_samples(
    fs,
    -wave_freq,
    len(pulse_samples),
    (math.fmod((first_sample_index - 1) * 2 * math.pi * wave_freq / fs, 2 * math.pi)),
)
pulse_samples = [l * i for l, i in zip(pulse_samples, shift_wave)]

# filter before decimating to prevent aliasing
# fig7 = plot_fft(pulse_samples,fs, 'FFT of Mixed Pulse Samples Using Traditional Method, Before Filtering and Decimating')
output = signal.convolve(pulse_samples, lpass, mode="valid")  # / sum(lpass)

# OR, can convolve using the same method as above (which is using the valid method).
# output=np.array([],dtype=complex)
# for x in range(0,len(pulse_samples)-first_sample_index):
#    sum_array = np.array([l*i for l,i in zip(pulse_samples[(first_sample_index + x - len(lpass)):(first_sample_index + x)],lpass[::-1])])
#    output_sum = 0.0
#    for element in sum_array:
#        output_sum += element
#    output = np.append(output,output_sum)

# fig8 = plot_fft(output, fs, 'FFT of Filtered Output Using Traditional Method, Before Decimating')

# Decimate here.
output3 = np.array([], dtype=complex)
for x in range(0, num_output_samps):
    samp = output[x * decimation_rate]
    output3 = np.append(output3, samp)
# Plot the output using Frerking's method
new_fs = float(fs) / decimation_rate

#
#
#
#
# Plot FFTs and Phase responses of all methods
# fig6, smpplt = plt.subplots(1,1)
fig6 = plt.figure()
fft_samps1 = fft(output1)
fft_samps2 = fft(output2)
fft_samps3 = fft(output3)
T = 1.0 / float(new_fs)
num_samps = len(output1)
if num_samps % 2 == 1:
    xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)
else:
    # xf = np.arange(-1.0/(2.0*T), 1.0/(2.0*T),1.0/(T*num_samps))
    xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)
print(num_samps)
# print(len(fft_samps))
# print(len(xf))
ax1 = fig6.add_subplot(111)
plt.title("Response of All Filters")
plt.ylabel("Amplitude [dB]", color="r")
plt.xlabel("Frequency [rad/sample]")
plt.grid()
fft_to_plot1 = np.empty([num_samps], dtype=complex)
fft_to_plot1 = fftshift(fft_samps1)
fft_to_plot2 = np.empty([num_samps], dtype=complex)
fft_to_plot2 = fftshift(fft_samps2)
fft_to_plot3 = np.empty([num_samps], dtype=complex)
fft_to_plot3 = fftshift(fft_samps3)
plt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot1), "c")
plt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot2), "y")
plt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot3), "r")
# plt.plot(xf, 1.0/num_samps * np.abs( np.roll( fft_to_plot3, int(-len(output1) * wave_freq / (1.0/T)))), 'c')
ax2 = ax1.twinx()
plt.ylabel("Phase [rads]", color="g")
angles1 = np.angle(fft_to_plot1)
angles2 = np.angle(fft_to_plot2)
angles3 = np.angle(fft_to_plot3)
plt.plot(xf, angles1, "m")
plt.plot(xf, angles2, "b")
plt.plot(xf, angles3, "g")

for ind, freq in enumerate(xf):
    if freq == -10 * fft_bin_increment:
        angle1 = angles1[ind]
        print("Angle 1: %f Rads" % angle1)
    if freq == 0.0:
        angle2 = angles1[ind]
        print("Angle 2: %f Rads" % angle2)
    if freq == 10 * fft_bin_increment:
        angle3 = angles1[ind]
        print("Angle 3: %f Rads" % angle3)

angle_offset1 = angle2 - angle1
angle_offset2 = angle3 - angle2

print(
    "Angle2 - Angle1 = %f = Angle3 - Angle2 = %f" % (angle2 - angle1, angle3 - angle2)
)

print(
    "Error of : %f Degrees" % (abs(angle_offset2 - angle_offset1) * 360 / (2 * math.pi))
)

# in time domain, all filtered outputs:
fig7 = plt.figure()
plt.title("Three Filtered Outputs Of Different Methods, Time Domain")
plt.plot(range(0, len(output1)), output1)
plt.plot(range(0, len(output2)), output2)
plt.plot(range(0, len(output3)), output3)

# for n in range(0,len(output1)):
#    if not isclose(output1[n], output2[n]) or not isclose(output2[n], output3[n]) or not isclose(output1[n],output3[n]):
#        print "NOT EQUAL: %d" % n
#        print output1[n]
#        print output2[n]
#        print output3[n]
# WHY? The last output sample is not exactly equal but the rest are....
# Maybe a python float thing?
plt.show()
