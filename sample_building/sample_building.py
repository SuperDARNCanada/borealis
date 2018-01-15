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


def resolve_imaging_directions(beamdirs_list, num_antennas, antenna_spacing):
    """
    This function will take in a list of directions and resolve that to a direction for each
    antenna. It will return a list of length num_antenna where each element is a direction off 
    orthogonal for that antenna.
    :param beamdirs_list: 
    :param num_antennas: 
    :param antenna_spacing: 
    :return: list of beam directions for each antenna.
    """
    # TODO. Note that we could make this a user-writeable custom function specific to an experiment
    # because you may want more power in certain directions, etc. ??
    # Or may prefer changing input params to a single beam direction and perhaps beamwidth?
    beamdirs = [beamdirs_list[ant % len(beamdirs_list)] for ant in range(0, num_antennas)]  # TODO fix
    amplitudes = [1.0 for ant in range(0, num_antennas)]  # TODO fix
    return beamdirs, amplitudes


def get_phshift(beamdir, freq, antenna, pulse_shift, num_antennas, antenna_spacing):
    """
    Form the beam given the beam direction (degrees off boresite), the tx frequency, the antenna number,
    a specified extra phase shift if there is any, the number of antennas in the array, and the spacing 
    between antennas.
    
    :param beamdir: the direction of the beam off boresight, in degrees, positive beamdir being to 
        the right of the boresight. This is for this antenna.
    :param freq: transmit frequency in kHz
    :param antenna: antenna number, INDEXED FROM ZERO, zero being the leftmost antenna if looking down the azimuth 
        and positive beamdir right of azimuth
    :param pulse_shift: in radians, for phase encoding
    :param num_antennas: number of antennas in this array
    :param antenna_spacing: distance between antennas in this array, in meters
    
    :return a phase shift for the samples for this antenna number, in radians.
    """

    freq = freq * 1000.0  # convert to Hz.

    beamdir = float(beamdir)

    beamrad = math.pi * float(beamdir) / 180.0
    # Pointing to right of boresight, use point in middle (hypothetically antenna 7.5) as phshift=0
    #   so all channels have a non-zero phase shift
    phshift = 2 * math.pi * freq * ((num_antennas-1)/2.0 - antenna) * antenna_spacing * math.cos(
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
        amplitude_ramp_down = np.flipud(amplitude_ramp_up)  # reverse
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
        raise ExperimentException(errmsg)  # REVIEW #6 TODO Handle gracefully or something

    # Samples is an array of complex samples
    # NOTE: phasing will be done in shift_samples function
    return samples, actual_wave_freq


def shape_samples(basic_samples, phshift, amplitude):
    """Take the samples and shift by given phase shift in rads and adjust amplitude as required
    for imaging.
    :param basic_samples : samples for this pulse
    :param phshift : phase for this antenna to offset by
    :param amplitude : amplitude for this antenna (= 1 if not imaging)
    :return samples, shaped for the antenna for the desired beam.
    """

    samples = [sample * amplitude * np.exp(1j * phshift) for sample in basic_samples]
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


def make_pulse_samples(pulse_list, power_divider, exp_slices, slice_to_beamdir_dict, txctrfreq,
                       txrate, options):
    """
    Make and phase shift samples, and combine them if there are multiple pulse types to send within this pulse.
    :param pulse_list: a list of dictionaries, each dict is a pulse. The list only contains pulses
    that will be sent as a single pulse (ie. have the same combined_pulse_index).
    :param power_divider: an integer for number of pulses combined (max) in the whole sequence, 
        so we can adjust the amplitude of each uncombined pulse accordingly. 
    :param exp_slices:
    :param slice_to_beamdir_dict: 
    :param txctrfreq: 
    :param txrate: 
    :param options: 
    :return: 
    
    """

    for pulse in pulse_list:
        try:
            assert pulse['combined_pulse_index'] == pulse_list[0]['combined_pulse_index']
            assert pulse['pulse_timing_us'] == pulse_list[0]['pulse_timing_us']
        except AssertionError:
            errmsg = 'Error building samples from pulse dictionaries'
            raise ExperimentException(errmsg, pulse, pulse_list[0])

    txrate = float(txrate)
    txctrfreq = float(txctrfreq)

    # make the uncombined pulses
    create_uncombined_pulses(pulse_list, power_divider, exp_slices, slice_to_beamdir_dict,
                             txctrfreq, txrate, options)
    # all pulse dictionaries in the pulse_list now have a 'samples' key which is a list of numpy
    # complex arrays (one for each tx antenna).

    # determine how long the combined pulse will be in number of samples, and add the key
    # 'sample_start_number' for all pulses in the pulse_list.
    combined_pulse_length = calculated_combined_pulse_samples_length(pulse_list, txrate)

    # Now we have total length so make all pulse samples same length
    #   before combining them sample by sample.
    for pulse in pulse_list:
        # print start_samples
        for antenna in range(0, options.main_antenna_count):
            pulse_array = pulse['samples'][antenna]
            # print(combined_pulse_length, len(pulse_array), pulse['sample_number_start'])
            zeros_prepend = np.zeros(pulse['sample_number_start'], dtype=np.complex64)
            zeros_append = np.zeros((combined_pulse_length - len(pulse_array) - pulse['sample_number_start']), dtype=np.complex64)

            corrected_pulse_array = np.concatenate((zeros_prepend, pulse_array, zeros_append))

            pulse['samples'][antenna] = corrected_pulse_array
            # Sub in new array of right length for old array.

    # initialize to correct length
    combined_samples = [np.zeros(combined_pulse_length, dtype=np.complex64) for ant in range(0, options.main_antenna_count)]
    # This is a list of arrays (one for each antenna) with the combined
    #   samples in it (which will be transmitted). Need to add together multiple pulses if there
    #   are multiple frequencies, for example.
    for antenna in range(0, options.main_antenna_count):
        for pulse in pulse_list:
            try:
                combined_samples[antenna] += pulse['samples'][antenna]
            except RuntimeWarning:  # REVIEW #3 What would cause this exception?  REPLY: I cannot remember actually....
                print("RUNTIMEWARNING {}".format(len(combined_samples[antenna])))

    # Now get what channels we need to transmit on for this combined
    #   pulse.
    # TODO : figure out - why did I do this I thought we were transmitting zeros on any channels not wanted
    pulse_channels = []
    for pulse in pulse_list:
        for ant in exp_slices[pulse['slice_id']]['tx_antennas']:
            if ant not in pulse_channels:
                pulse_channels.append(ant)
    pulse_channels.sort()

    return combined_samples, pulse_channels


def create_uncombined_pulses(pulse_list, power_divider, exp_slices, beamdir, txctrfreq, txrate,
                             options):
    """
    Creates a sample dictionary where the pulse is the key and the samples (in a list from 0th to 
    max antenna) are the value.
    :param pulse_list: a list of dictionaries, each dict is a pulse
    :param power_divider: an integer for number of pulses combined (max) in the whole sequence, 
        so we can adjust the amplitude of each uncombined pulse accordingly. 
    :param exp_slices: 
    :param beamdir: 
    :param txctrfreq: 
    :param txrate: 
    :param options: 
    :return: 
    """

    for pulse in pulse_list:
        # print exp_slices[pulse['slice_id']]
        wave_freq = float(exp_slices[pulse['slice_id']]['txfreq']) - txctrfreq  # TODO error will occur here if clrfrqrange because clrfrq search isn't completed yet.
        phase_array = []
        pulse['samples'] = []

        if len(beamdir[pulse['slice_id']]) > 1:  # todo move this somwhere for each slice_id, not pulse as unnecessary repetition
            # we have imaging. We need to figure out the direction and amplitude to give
            # each antenna
            beamdirs_for_antennas, amps_for_antennas = \
                resolve_imaging_directions(beamdir[pulse['slice_id']], options.main_antenna_count,
                                           options.main_antenna_spacing)
        else:  # not imaging, all antennas transmitting same direction.
            beamdirs_for_antennas = [beamdir[pulse['slice_id']][0] for ant in
                                     range(0, options.main_antenna_count)]
            amps_for_antennas = [1.0 for ant in range(0, options.main_antenna_count)]

        amplitude_array = [amplitude / float(power_divider) for amplitude in amps_for_antennas]
        # also adjust amplitudes for number of pulses transmitted at once. # TODO : review this as
        for antenna in range(0, options.main_antenna_count):
            # Get phase shifts for all channels off centre of array being phase = 0.
            phase_for_antenna = \
                get_phshift(beamdirs_for_antennas[antenna], exp_slices[pulse['slice_id']]['txfreq'],
                            antenna,
                            exp_slices[pulse['slice_id']]['pulse_shift'][pulse['slice_pulse_index']],
                            options.main_antenna_count, options.main_antenna_spacing)
            phase_array.append(phase_for_antenna)

        wave_freq_hz = wave_freq * 1000

        # Create samples for this frequency at this rate. Convert pulse_len to seconds and
        # wave_freq to Hz.
        basic_samples, real_freq = get_samples(txrate, wave_freq_hz,
                                               float(pulse['pulse_len']) / 1000000,
                                               options.pulse_ramp_time,
                                               options.max_usrp_dac_amplitude,
                                               exp_slices[pulse['slice_id']]['iwavetable'],
                                               exp_slices[pulse['slice_id']]['qwavetable'])

        if real_freq != wave_freq_hz:
            errmsg = 'Actual Frequency {} is Not Equal to Intended Wave Freq {}'.format(real_freq,
                                                                                        wave_freq_hz)
            raise ExperimentException(errmsg)  # TODO change to warning? only happens on non-SINE

        for antenna in range(0, options.main_antenna_count):
            if antenna in exp_slices[pulse['slice_id']]['tx_antennas']:
                pulse_samples = shape_samples(basic_samples, phase_array[antenna],
                                              amplitude_array[antenna])
                pulse['samples'].append(pulse_samples)
                # pulse_dict['samples'] is a list of numpy arrays now.
            else:
                pulse_samples = np.zeros([len(basic_samples)], dtype=np.complex64)
                pulse['samples'].append(pulse_samples)
                # Will be an empty array for that channel.


def calculated_combined_pulse_samples_length(pulse_list, txrate):
    """
    Determine the length of the combined pulse in number of samples before combining the samples, 
    and the starting sample number for each pulse to combine. (ie not all pulse frequencies may start
    at sample zero due to differing intra_pulse_start_times.)
    :param pulse_list list of pulse dictionaries that must be combined to one pulse.
    :param txrate - sampling rate of transmission going to DAC
    :return: 
    """

    combined_pulse_length = 0
    for pulse in pulse_list:
        # sample number to begin this pulse in the combined pulse. Must convert
        # intra_pulse_start_time to seconds from us.

        pulse['sample_number_start'] = int(txrate * float(pulse['intra_pulse_start_time']) * 1e-6)

        if (pulse['sample_number_start'] + len(pulse['samples'][0])) > combined_pulse_length:
            combined_pulse_length = pulse['sample_number_start'] + len(pulse['samples'][0])
            # Timing from first sample + length of this pulse is max
            # print "Total Length : {}".format(total_length)

    return combined_pulse_length


def azimuth_to_antenna_offset(beamdir, main_antenna_count, interferometer_antenna_count,
                              main_antenna_spacing, interferometer_antenna_spacing, txfreq):
    """
    
    :param beamdir: list of length 1 or more.
    :param main_antenna_count: 
    :param interferometer_antenna_count: 
    :param main_antenna_spacing: 
    :param interferometer_antenna_spacing: 
    :param txfreq: 
    :return: 
    """

    beams_antenna_phases = []
    for beam in beamdir:
        phase_array = []
        for channel in range(0, main_antenna_count):
            # Get phase shifts for all channels
            phase_array.append(get_phshift(beam, txfreq, channel, 0, main_antenna_count,
                main_antenna_spacing))
        for channel in range(0, interferometer_antenna_count):  # interferometer TODO interferometer offset ***
            # Get phase shifts for all channels
            phase_array.append(get_phshift(beam, txfreq, channel, 0, interferometer_antenna_count,
                interferometer_antenna_spacing))  # zero pulse shift b/w pulses when beamforming.
        beams_antenna_phases.append(phase_array)

    return beams_antenna_phases
