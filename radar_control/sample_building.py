# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#
# Functions to process and build samples,
# including getting phase shifts from beam directions,
# and functions for creating samples
# as well as plotting them and shifting them

from scipy.fftpack import fft
from scipy.constants import speed_of_light
from scipy.signal import kaiserord, lfilter, firwin, freqz
import numpy as np
import math
import cmath
import sys
import matplotlib.pyplot as plt
from experiments.experiment_exception import ExperimentException


def get_phshift(beamdir, freq, antenna, pulse_shift, num_antennas, antenna_spacing):
    """
    Form the beam given the beam direction (degrees off boresite), the tx frequency, the antenna number,
    a specified extra phase shift if there is any, the number of antennas in the array, and the spacing 
    between antennas.
    
    :param beamdir: the direction of the beam off azimuth, in degrees, positive beamdir being to the right of 
        azimuth if looking down the azimuth
    :param freq: transmit frequency in kHz
    :param antenna: antenna number, INDEXED FROM ZERO, zero being the leftmost antenna if looking down the azimuth 
        and positive beamdir right of azimuth
    :param pulse_shift: in radians, for phase encoding
    :param num_antennas: number of antennas in this array
    :param antenna_spacing: distance between antennas in this array, in meters
    
    :return a phase shift for the samples for this antenna number, in radians.
    """

    freq = freq * 1000.0  # convert to Hz.

    if isinstance(beamdir, list):
        # TODO handle imaging/multiple beam directions
        beamdir = 0  # directly ahead for now
    else:  # is a float
        beamdir = float(beamdir)

    beamrad = math.pi * float(beamdir) / 180.0
    # Pointing to right of boresight, use point in middle (hypothetically antenna 7.5) as phshift=0
    #   so all channels have a non-zero phase shift
    phshift = 2 * math.pi * freq * ((num_antennas-1)/2 - antenna) * antenna_spacing * math.cos(
        math.pi / 2 - beamrad) / speed_of_light

    # Add an extra phase shift if there is any specified
    phshift = phshift + pulse_shift

    phshift = math.fmod(phshift, 2 * math.pi)

    return phshift


def get_wavetables(wavetype):
    """
    If there are ever any other types of wavetypes, set them up here. 
    
    NOTE: The wavetables should sample a single cycle of the waveform. Note that we will have to block frequencies 
    that could interfere with our license, which will affect the waveform. This blocking of frequencies is not 
    currently set up, so beware. Would have to get the spectrum of the wavetable waveform and then block frequencies 
    that when mixed with the centre frequency, result in the restricted frequencies.
    
    Also NOTE: wavetables create a fixed frequency resolution based on their length. This code is from get_samples:
    f_norm = wave_freq / rate
    sample_skip = int(f_norm * wave_table_len) # THIS MUST BE AN INT, WHICH DEFINES THE FREQUENCY RESOLUTION.
    actual_wave_freq = (float(sample_skip) / float(wave_table_len)) * rate 
    
    :param wavetype: A string descriptor of the wavetype.
    :return: an in-phase wavetable and a quadrature wavetable, or 2 Nones if it is a sine wavetype.
    """
    # TODO : See docstring above.

    if wavetype == "SINE":
        iwave_table = None
        qwave_table = None

    else:
        iwave_table = []
        qwave_table = []
        errmsg = "Wavetype %s not defined" % (wavetype)
        sys.exit(errmsg)  # TODO error handling

    # Example of a wavetable is below, if they were defined for SINE wavetypes.
    # wave_table_len=8192
    # for i in range(0, wave_table_len):
    #    iwave_table.append(math.cos(i*2*math.pi/wave_table_len))
    #    qwave_table.append(math.sin(i*2*math.pi/wave_table_len))

    return iwave_table, qwave_table


def get_samples(rate, wave_freq, pulse_len, ramp_time, max_amplitude, iwave_table=None,
                qwave_table=None):
    """
    Find the normalized sample array given the rate (Hz), frequency (Hz), pulse length (s), and wavetables (list
    containing single cycle of waveform). Will shift for beam later. No need to use wavetable if just as sine wave.
    :param rate: tx sampling rate, in Hz.
    :param wave_freq: frequency offset from the centre frequency on the USRP, given in Hz. To be mixed with the centre
        frequency before transmitting. (ex. centre = 12 MHz, wave_freq = + 1.2 MHz, output = 13.2 MHz.
    :param pulse_len : length of the pulse (in seconds)
    :param ramp_time : ramp up and ramp down time for the pulse, in seconds. Typical 0.00001 s from config.
    :param max_amplitude: USRP's max DAC amplitude. 1.0 causes overflow in testing (2016)
    :param iwave_table: i samples (in-phase) wavetable if a wavetable is required (ie. not a sine wave to be sampled)
    :param qwave_table: q samples (quadrature) wavetable if a wavetable is required (ie. not a sine wave to be sampled)
    
    :return a numpy array of complex samples, representing all samples needed for a pulse of length pulse_len sampled 
        at a rate of rate. 
    """

    wave_freq = float(wave_freq)
    rate = float(rate)

    if iwave_table is None and qwave_table is None:
        sampling_freq = 2 * math.pi * wave_freq / rate
        rampsampleslen = int(rate * ramp_time)  # number of samples for ramp-up and ramp-down of pulse.
        sampleslen = int(rate * pulse_len) + 2 * rampsampleslen

        rads = sampling_freq * np.arange(0, sampleslen)
        wave_form = np.exp(rads * 1j)

        amplitude_ramp_up = [ind * max_amplitude / rampsampleslen for ind in np.arange(0, rampsampleslen)]
        amplitude_ramp_down = np.fliplr(amplitude_ramp_up)  # flip left to right.
        amplitude = [max_amplitude for ind in np.arange(rampsampleslen, sampleslen - rampsampleslen)]
        all_amps = np.concatenate((amplitude_ramp_up, amplitude, amplitude_ramp_down))

        samples = [x * y for x, y in zip(wave_form, all_amps)]
        actual_wave_freq = wave_freq

    elif iwave_table is not None and qwave_table is not None:
        wave_table_len = len(iwave_table)
        rampsampleslen = int(rate * ramp_time)
        # Number of samples in ramp-up, ramp-down

        sampleslen = int(rate * pulse_len + 2 * rampsampleslen)
        samples = np.empty([sampleslen], dtype=np.complex64)

        # sample at wave_freq with given phase shift
        f_norm = wave_freq / rate
        sample_skip = int(f_norm * wave_table_len)
        # This must be an int to create perfect sine, and
        #   this int defines the frequency resolution of our generated
        #   waveform

        actual_wave_freq = (float(sample_skip) / float(wave_table_len)) * rate
        # This is the actual frequency given the sample_skip
        for i in range(0, rampsampleslen):
            amp = max_amplitude * float(i + 1) / float(rampsampleslen)  # rampup is linear
            if sample_skip < 0:
                ind = -1 * ((abs(sample_skip * i)) % wave_table_len)
            else:
                ind = (sample_skip * i) % wave_table_len
            samples[i] = (amp * iwave_table[ind] + amp * qwave_table[ind] * 1j)
            # qsamples[chi,i]=amp*qwave_table[ind]
        for i in range(rampsampleslen, sampleslen - rampsampleslen):
            amp = max_amplitude
            if sample_skip < 0:
                ind = -1 * ((abs(sample_skip * i)) % wave_table_len)
            else:
                ind = (sample_skip * i) % wave_table_len
            samples[i] = (amp * iwave_table[ind] + amp * qwave_table[ind] * 1j)
            # qsamples[chi,i]=qwave_table[ind]
        for i in range(sampleslen - rampsampleslen, sampleslen):
            amp = max_amplitude * float(sampleslen - i) / float(rampsampleslen)
            if sample_skip < 0:
                ind = -1 * ((abs(sample_skip * i)) % wave_table_len)
            else:
                ind = (sample_skip * i) % wave_table_len
            samples[i] = (amp * iwave_table[ind] + amp * qwave_table[ind] * 1j)
            # qsamples[chi,i]=amp*qwave_table[ind]

    else:
        errmsg = "Error: only one wavetable passed"
        sys.exit(errmsg)  # REVIEW #6 TODO Handle gracefully or something

    # Samples is an array of complex samples
    # NOTE: phasing will be done in shift_samples function
    return samples, actual_wave_freq


def shift_samples(basic_samples, phshift):
    """Take the samples and shift by given phase shift in rads."""

    samples = basic_samples * np.exp(1j * phshift)
    return samples


def plot_samples(filename, samplesa, samplesb=np.empty([2], dtype=np.complex64),
                 samplesc=np.empty([2], dtype=np.complex64)):
    """For testing only, plots samples to filename"""
    fig, smpplot = plt.subplots(1, 1)
    smpplot.plot(range(0, samplesa.shape[0]), samplesa)
    smpplot.plot(range(0, samplesb.shape[0]), samplesb)
    smpplot.plot(range(0, samplesc.shape[0]), samplesc)
    plt.ylim([-1, 1])
    plt.xlim([0, 100])
    fig.savefig(filename)
    plt.close(fig)
    return None


def plot_fft(filename, samplesa, rate):
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


def make_pulse_samples(pulse_list, exp_slices, beamdir, txctrfreq, txrate, power_divider, options, iwavetable=None,
                       qwavetable=None):
    """
    Make and phase shift samples, and combine them if there are multiple pulse types to send within this pulse.
    """

    txrate = float(txrate)
    txctrfreq = float(txctrfreq)
    samples_dict = {}

    for pulse in pulse_list:  # REVIEW #35 This loop is good candidate to make into function called create_samples or something TODO
        wave_freq = float(exp_slices[pulse[1]]['txfreq']) - txctrfreq
        samples_dict[tuple(pulse)] = []
        phase_array = []
        for antenna in range(0, options.main_antenna_count):
            # Get phase shifts for all channels
            phase_array.append(get_phshift(
                beamdir[pulse[1]],
                exp_slices[pulse[1]]['txfreq'], antenna,
                exp_slices[pulse[1]]['pulse_shift'][pulse[2]]))

        # Create samples for this frequency at this rate. Convert pulse_len to seconds and wave_freq to Hz.
        basic_samples, real_freq = get_samples(txrate, wave_freq * 1000, float(exp_slices[pulse[1]]['pulse_len']) / 1000000,
                                    options.pulse_ramp_time, iwavetable, qwavetable)

        if real_freq != wave_freq:
            errmsg = 'Actual Frequency {} is Not Equal to Intended Wave Freq {}'.format(real_freq,
                                                                                        wave_freq)
            raise ExperimentException(errmsg)  # TODO change to warning? only happens on non-SINE

        for antenna in range(0, options.main_antenna_count):
            # REVIEW #6 TODO: Handle different amplitudes necessary for imaging. Something like pulse_samples = shape_samples(basic_samples, amplitude_array[antenna]) and that function could just be a numpy array multiply
            if antenna in exp_slices[pulse[1]]['txantennas']:
                pulse_samples = shift_samples(basic_samples, phase_array[antenna])
                samples_dict[tuple(pulse)].append(pulse_samples)
                # Samples_dict[pulse] is a list of numpy arrays now.
            else:
                pulse_samples = np.zeros([len(basic_samples)], dtype=np.complex64)
                samples_dict[tuple(pulse)].append(pulse_samples)
                # Will be an empty array for that channel.

    # Combine samples given pulse timing in 'pulse' list.
    # Find timing of where the pulses start in comparison to the first
    #   pulse, and find out the total number of samples for this
    #   combined pulse.
    samples_begin = []
    total_length = len(samples_dict[tuple(pulse_list[0])][
                           0])  # REVIEW #26 This is difficult to debug, so maybe break this up into several lines - make variable for tuple(pulse_list[0]) and give meaningful name to key into samples_dict. Then you can do the same for samples_dict[key] [0] perhaps.
    for pulse in pulse_list:  # REVIEW #35 This for loop is good candidate to make into function called calculate_true_sample_length or similar
        # pulse_list is in order of timing
        start_samples = int(txrate * float(pulse[0] - pulse_list[0][
            0]) * 1e-6)  # REVIEW #26 #29 #1 same here difficult to debug, also magic number of 1e-6. Also, why do you use pulse_list[0][0] instead of just '0'? isn't the first pulse's timing always set to 0? pls explain
        # First value in samples_begin should be zero.
        samples_begin.append(start_samples)
        if start_samples + len(samples_dict[tuple(pulse)][0]) > total_length:
            total_length = start_samples + len(samples_dict[tuple(
                pulse)])  # REVIEW #26 - we think total_lenght should be called max_length_of_pulse or max_length or overlap_length or similar since it's actually the maximum length of any overlapping pulses or long pulse in the pulse_list
            # Timing from first sample + length of this pulse is max
    # print "Total Length : {}".format(total_length)

    # Now we have total length so make all pulse samples same length
    #   before combining them sample by sample.
    for pulse in pulse_list:  # REVIEW #35 can be refactored into a zero-pad function that creates a zero_pad prepend array, a zero-pad postpend array and the 'array' in the middle of actual samples. Continued below at the for i in range(0,total_length) : line
        start_samples = samples_begin[pulse_list.index(pulse)]  # REVIEW #39 Can also do 'for i, pulse in enumerate(pulse_list)' and use i as the index. OR use 'for start_sample, pulse in zip(start_samples, pulse_list)' which will give you the start sample, as well as the tuple from pulse_list that coincide
        # print start_samples
        for antenna in range(0,
                             16):  # REVIEW #29 Magic number 16 - is this equal to main antennas? should be a config option now
            array = samples_dict[tuple(pulse)][
                antenna]  # REVIEW #26 - array and new_array can be named something meaningful
            new_array = np.empty([total_length],
                                 dtype=np.complex_)  # REVIEW #28 We believe that the dtype for samples array should be complex float ( np.complex64 ) instead of complex (which seems to default to complex128  - or a double complex value) Also - what is the reason for np.complex_ ?
            for i in range(0,
                           total_length):  # REVIEW #39, continued from above: an example of how to do this: zero_prepend = np.zeros(start_sample,dtype=np.complex64); zero_append = np.zeros(max_length - len_samples_array - start_sample, dtype=np.complex64); pulse = samples_dict[tuple(pulse)][channel]; complete_new_array = np.concatenate((zero_prepend, pulse, zero_append))
                if i < start_samples:
                    new_array[i] = 0.0
                if i >= start_samples and i < (start_samples + len(array)):
                    new_array[i] = array[i - samples_begin[pulse_list.index(pulse)]]
                if i > start_samples + len(array):
                    new_array[i] = 0.0
            samples_dict[tuple(pulse)][antenna] = new_array
            # Sub in new array of right length for old array.

    total_samples = []  # REVIEW #26 maybe combined_samples instead of total_samples?
    # This is a list of arrays (one for each channel) with the combined
    #   samples in it (which will be transmitted).
    for antenna in range(0,
                         16):  # REVIEW #29 Magic number 16 - is this equal to main antennas? should be a config option now
        total_samples.append(samples_dict[tuple(pulse_list[0])][antenna])
        for samplen in range(0, total_length):
            try:
                total_samples[antenna][samplen] = (total_samples[antenna][
                                                       samplen] / power_divider)  # REVIEW #39 Can use np.array division. Don't need the for samplen loop, just do 'total_samples[channel] = total_samples[channel] / power_divider'
            except RuntimeWarning:  # REVIEW #3 What would cause this exception?
                print "RUNTIMEWARNING {} {}".format(total_samples[antenna][samplen], power_divider)
            for pulse in pulse_list:
                if pulse == pulse_list[0]:
                    continue
                total_samples[antenna][samplen] += (samples_dict[tuple(pulse)]
                                                        # REVIEW #39 if you take this out of the for samplen loop, can do 'total_samples[channel] +=  samples_dict[tuple(pulse)][channel]/power_divider'
                                                    [antenna][samplen]  # REVIEW #1 why do you have parenthesis around this? Is it a tuple?
                                                    / power_divider)  # REVIEW #0 are you dividing by power_divider twice?

    # Now get what channels we need to transmit on for this combined
    #   pulse.
    # print("First cpo: {}".format(pulse_list[0][1]))
    pulse_channels = exp_slices[pulse_list[0][1]]['txantennas']  # REVIEW #35 can make this a function
    for pulse in pulse_list:
        for chan in exp_slices[pulse[1]]['txantennas']:
            if chan not in pulse_channels:
                pulse_channels.append(chan)
    pulse_channels.sort()

    return total_samples, pulse_channels
