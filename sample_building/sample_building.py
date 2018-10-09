# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#
# Functions to process and build samples,
# including getting phase shifts from beam directions,
# and functions for creating samples
# as well as shifting them as required.

from scipy.fftpack import fft
from scipy.constants import speed_of_light
from scipy.signal import gaussian
import numpy as np
import math
import json
from datetime import datetime
from experiment_prototype.experiment_exception import ExperimentException

def resolve_imaging_directions(beamdirs_list, num_antennas, antenna_spacing):
    """
    Resolve imaging directions to direction per antenna.
    
    This function will take in a list of directions and resolve that to a direction for each
    antenna. It will return a list of length num_antenna where each element is a direction off 
    orthogonal for that antenna.
    
    :param beamdirs_list: The list of beam directions for this pulse sequence.
    :param num_antennas: The number of antennas to calculate direcitonrs for.
    :param antenna_spacing: The spacing between the antennas.
    :returns beamdirs: A list of beam directions for each antenna.
    :returns amplitudes: A list of amplitudes for each antenna
    """

    # TODO. Note that we could make this a user-writeable custom function specific to an experiment
    # because you may want more power in certain directions, etc. ??
    # Or may prefer changing input params to a single beam direction and perhaps beamwidth?
    beamdirs = [beamdirs_list[ant % len(beamdirs_list)] for ant in range(0, num_antennas)]  # TODO fix
    amplitudes = [1.0 for ant in range(0, num_antennas)]  # TODO fix
    return beamdirs, amplitudes


def get_phshift(beamdir, freq, antenna, pulse_shift, num_antennas, antenna_spacing):
    """
    Find the phase shift for a given antenna and beam direction.
    
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
    
    :returns phshift: a phase shift for the samples for this antenna number, in radians.
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
    Find the wavetable to sample from for a given wavetype.
    
    If there are ever any other types of wavetypes besides 'SINE', set them up here. 
    
    NOTE: The wavetables should sample a single cycle of the waveform. Note that we will have to block frequencies 
    that could interfere with our license, which will affect the waveform. This blocking of frequencies is not 
    currently set up, so beware. Would have to get the spectrum of the wavetable waveform and then block frequencies 
    that when mixed with the centre frequency, result in the restricted frequencies.
    
    Also NOTE: wavetables create a fixed frequency resolution based on their length. This code is from get_samples:
    
    f_norm = wave_freq / rate
    
    sample_skip = int(f_norm * wave_table_len) # THIS MUST BE AN INT, WHICH DEFINES 
    THE FREQUENCY RESOLUTION.
    
    actual_wave_freq = (float(sample_skip) / float(wave_table_len)) * rate 
    
    :param wavetype: A string descriptor of the wavetype.
    :returns iwavetable: an in-phase wavetable, or None if given 'SINE' wavetype.
    :returns qwavetable: a quadrature wavetable, or None if given 'SINE' wavetype.
    """

    # TODO : See docstring above.

    if wavetype == "SINE":
        iwave_table = None
        qwave_table = None

    else:
        iwave_table = []
        qwave_table = []
        errmsg = "Wavetype %s not defined" % (wavetype)
        raise ExperimentException(errmsg)

    # Example of a wavetable is below, if they were defined for SINE wavetypes.
    # wave_table_len=8192
    # for i in range(0, wave_table_len):
    #    iwave_table.append(math.cos(i*2*math.pi/wave_table_len))
    #    qwave_table.append(math.sin(i*2*math.pi/wave_table_len))

    return iwave_table, qwave_table


def get_samples(rate, wave_freq, pulse_len, ramp_time, max_amplitude, iwave_table=None, qwave_table=None):
    """
    Get basic (not phase-shifted) samples for a given pulse.
    
    Find the normalized sample array given the rate (Hz), frequency (Hz), pulse length 
    (s), and wavetables (list containing single cycle of waveform). Will shift for 
    beam later. No need to use wavetable if just a sine wave.
    
    :param rate: tx sampling rate, in Hz.
    :param wave_freq: frequency offset from the centre frequency on the USRP, given in 
     Hz. To be mixed with the centre frequency before transmitting. (ex. centre = 12 
     MHz, wave_freq = + 1.2 MHz, output = 13.2 MHz.
    :param pulse_len: length of the pulse (in seconds)
    :param ramp_time: ramp up and ramp down time for the pulse, in seconds. Typical 
     0.00001 s from config.
    :param max_amplitude: USRP's max DAC amplitude. N200 = 0.707 max
    :param iwave_table: i samples (in-phase) wavetable if a wavetable is required 
     (ie. not a sine wave to be sampled)
    :param qwave_table: q samples (quadrature) wavetable if a wavetable is required 
     (ie. not a sine wave to be sampled)
    :returns samples: a numpy array of complex samples, representing all samples needed 
     for a pulse of length pulse_len sampled at a rate of rate. 
    :returns actual_wave_freq: the frequency possible given the wavetable. If wavetype 
     != 'SINE' (i.e. calculated wavetables were used), then actual_wave_freq may not 
     be equal to the requested wave_freq param. 
    """

    wave_freq = float(wave_freq)
    rate = float(rate)

    if iwave_table is None and qwave_table is None:
        sampling_freq = 2 * math.pi * wave_freq / rate

        # for linear we used the below:
        linear_rampsampleslen = int(rate * ramp_time)  # number of samples for ramp-up and ramp-down of pulse.

        sampleslen = int(rate * pulse_len)

        rads = sampling_freq * np.arange(0, sampleslen)
        wave_form = np.exp(rads * 1j)

        amplitude_ramp_up = [ind * max_amplitude / linear_rampsampleslen for ind in np.arange(0, linear_rampsampleslen)]
        amplitude_ramp_down = np.flipud(amplitude_ramp_up)  # reverse
        amplitude = [max_amplitude for ind in np.arange(linear_rampsampleslen, sampleslen - linear_rampsampleslen)]
        linear_amps = np.concatenate((amplitude_ramp_up, amplitude, amplitude_ramp_down))

        samples = [x * y for x, y in zip(wave_form, linear_amps)]

        #gaussian_amps = max_amplitude * np.ones([sampleslen]) * gaussian(sampleslen, math.ceil(pulse_len/6.0))
        # TODO modify ramp_time input to this function because going Gaussian (after
        # ... TODO: testing this)
        #samples = [x * y for x, y in zip(wave_form, gaussian_amps)]
        samples = np.array(samples)
        actual_wave_freq = wave_freq

    elif iwave_table is not None and qwave_table is not None:
        wave_table_len = len(iwave_table)

        # TODO turn this into Gaussian ramp-up not linear!!
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
        raise ExperimentException(errmsg)

    # Samples is an array of complex samples
    # NOTE: phasing will be done in shift_samples function
    return samples, actual_wave_freq


def shift_samples(basic_samples, phshift, amplitude):
    """
    Shift samples for a pulse by a given phase shift.
    
    Take the samples and shift by given phase shift in rads and adjust amplitude as 
    required for imaging.
    
    :param basic_samples: samples for this pulse, numpy array
    :param phshift: phase for this antenna to offset by, float
    :param amplitude: amplitude for this antenna (= 1 if not imaging), float
    :returns samples: basic_samples that have been shaped for the antenna for the 
     desired beam.
    """

    #samples = [sample * amplitude * np.exp(1j * phshift) for sample in basic_samples]
    samples = amplitude * np.exp(1j * phshift) * basic_samples
    return samples


def make_pulse_samples(pulse_list, power_divider, exp_slices, slice_to_beamdir_dict, txctrfreq,
                       txrate, options):
    """
    Make all necessary samples for all antennas for this pulse.
    
    Given a pulse_list (list of dictionaries of pulses that must be combined), make and 
    phase shift samples for all antennas, and combine pulse dictionaries into one 
    pulse if there are multiple waveforms to combine (e.g., multiple frequencies). 
    
    :param pulse_list: a list of dictionaries, each dict is a pulse. The list only 
     contains pulses that will be sent as a single pulse (ie. have the same 
     combined_pulse_index).
    :param power_divider: an integer for number of pulses combined (max) in the whole 
     sequence, so we can adjust the amplitude of each uncombined pulse accordingly. 
    :param exp_slices: this is the slice dictionary containing the slices necessary for 
     the sequence.
    :param slice_to_beamdir_dict: a dictionary describing the beam directions for the 
     slice_ids.
    :param txctrfreq: the txctrfreq  which determines the wave_freq to build our pulses 
     at.
    :param txrate: the tx sample rate.
    :param options: the experiment options from the config, hdw.dat, and restrict.dat 
     files.
    :returns combined_samples: a list of arrays - each array corresponds to an antenna 
     (the samples are phased). All arrays are the same length for a single pulse on 
     that antenna. The length of the list is equal to main_antenna_count (all samples 
     are calculated). If we are not using an antenna, that index is a numpy array of 
     zeroes.
    :returns pulse_channels: The antennas to actually send the corresponding array. If 
     not all transmit antennas, then we will know that we are transmitting zeroes on 
     that antenna.
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
    # complex arrays (one for each possible tx antenna).

    #print type(pulse_list[0]), type(pulse_list[0]['samples']), type(pulse_list[0]['samples'][0])
    #plot_samples("samples.png", pulse_list[0]['samples'][0])

    # determine how long the combined pulse will be in number of samples, and add the key
    # 'sample_number_start' for all pulses in the pulse_list.
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
            except RuntimeWarning:
                raise ExperimentException("RUNTIMEWARNING {}".format(len(combined_samples[antenna])))
                # TODO determine if we can manage this overflow error to prevent this.

    tr_window_num_samps = int(math.ceil(options.tr_window_time * txrate))
    tr_window_samples = np.zeros(tr_window_num_samps, dtype=np.complex64)
    combined_samples_tr = []
    for cs in combined_samples:
        combined_samples_channel = np.concatenate((tr_window_samples, cs,
                                                   tr_window_samples))
        combined_samples_tr.append(combined_samples_channel)

    # print(len(combined_samples_tr[0]))
    # Now get what channels we need to transmit on for this combined
    #   pulse.
    # TODO : figure out - why did I do this I thought we were transmitting zeros on any channels not wanted
    # TODO : decide which to do. Currently filling the combined_samples[unused_antenna] with an array of zeroes and
    # also sending the channels that we want to send.
    pulse_channels = []
    for pulse in pulse_list:
        for ant in exp_slices[pulse['slice_id']]['tx_antennas']:
            if ant not in pulse_channels:
                pulse_channels.append(ant)
    pulse_channels.sort()

    return combined_samples_tr, pulse_channels


def create_uncombined_pulses(pulse_list, power_divider, exp_slices, beamdir, txctrfreq, txrate,
                             options):
    """
    Create the samples for a given pulse_list and append those samples to the pulse_list. 
    
    Creates a list of numpy arrays where each numpy array is the pulse samples for a 
    given pulse and a given transmit antenna (index of array in list provides antenna 
    number). Adds the list of samples to the pulse dictionary (in the pulse_list list) 
    under the key 'samples'. 
    
    :param pulse_list: a list of dictionaries, each dict is a pulse. The list includes 
     all pulses that will be combined together. All dictionaries in this list (all 
     'pulses') will be modified to include the 'samples' key which will be a list of 
     arrays where every array is a set of samples for a specific antenna.
    :param power_divider: an integer for number of pulses combined (max) in the whole 
     sequence, so we can adjust the amplitude of each uncombined pulse accordingly. 
    :param exp_slices: slice dictionary containing all necessary slice_ids for this 
     pulse.
    :param beamdir: the slice to beamdir dictionary to retrieve the phasing information 
     for each antenna in a certain slice's pulses.
    :param txctrfreq: centre frequency we are transmitting at.
    :param txrate: sampling rate we are transmitting at. 
    :param options: experiment options, from config.ini, hdw.dat, and restrict.dat. 
    """

    for pulse in pulse_list:
        # print exp_slices[pulse['slice_id']]
        wave_freq = float(exp_slices[pulse['slice_id']]['txfreq']) - txctrfreq  # TODO error will occur here if clrfrqrange because clrfrq search isn't completed yet. (when clrfrq, no txfreq)
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
                pulse_samples = shift_samples(basic_samples, phase_array[antenna],
                                              amplitude_array[antenna])
                pulse['samples'].append(pulse_samples)
                # pulse['samples'] is a list of numpy arrays now.
            else:
                pulse_samples = np.zeros([len(basic_samples)], dtype=np.complex64)
                pulse['samples'].append(pulse_samples)
                # Will be an empty array for that channel.


def calculated_combined_pulse_samples_length(pulse_list, txrate):
    """
    Get the total length of the array for the combined pulse. 
    
    Determine the length of the combined pulse in number of samples before combining the samples, 
    using the length of the samples arrays and the starting sample number for each pulse to combine. 
    (Not all pulse samples may start at sample zero due to differing intra_pulse_start_times.)
    
    :param pulse_list: list of pulse dictionaries that must be combined to one pulse.
    :param txrate: sampling rate of transmission going to DAC.
    :returns combined_pulse_length: the length of the pulse after combining slices if necessary.  
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
                              main_antenna_spacing, interferometer_antenna_spacing, freq):
    """
    Get all the necessary phase shifts for all antennas for all the beams for a pulse sequence.
  
    Take all beam directions and resolve into a list of phase offsets for all antennas given the
    spacing, frequency, and number of antennas to resolve for. 
    
    :param beamdir: list of length 1 or more.
    :param main_antenna_count: the number of main antennas to calculate the phase offset for.
    :param interferometer_antenna_count: the number of interferometer antennas to calculate the 
     phase offset for.
    :param main_antenna_spacing: the spacing between the main array antennas (m). 
    :param interferometer_antenna_spacing: the spacing between the interferometer antennas (m). 
    :param freq: the frequency we are transmitting/receiving at.
    :returns beams_antenna_phases: a list of length = beam directions, where each element is a list
     of length = number of antennas (main array followed by interferometer array). The inner list
     contains the phase shift for the corresponding antenna for the corresponding beam. 
    """

    beams_antenna_phases = []
    for beam in beamdir:
        phase_array = []
        for channel in range(0, main_antenna_count):
            # Get phase shifts for all channels
            phase_array.append(get_phshift(beam, freq, channel, 0, main_antenna_count,
                main_antenna_spacing))
        for channel in range(0, interferometer_antenna_count):  # interferometer TODO interferometer offset ***
            # Get phase shifts for all channels
            phase_array.append(get_phshift(beam, freq, channel, 0, interferometer_antenna_count,
                interferometer_antenna_spacing))  # zero pulse shift b/w pulses when beamforming.
        beams_antenna_phases.append(phase_array)

    return beams_antenna_phases


def write_samples_to_file(txrate, txctrfreq, list_of_pulse_dicts,
                          file_path, main_antenna_count, final_rx_sample_rate, ssdelay):
    """
    Write the samples and transmitted metadata to a json file for use in testing.

    :param txrate: The rate at which these samples will be transmitted at.
    :param txctrfreq: The centre frequency that the N200 is tuned to (and will mix with
     these samples).
    :param list_of_pulse_dicts: The list of all pulse dictionaries for pulses included
    in this sequence. Pulse dictionaries have all metadata and the samples for the
    pulse.
    :param file_path: location to place the json file.
    :param main_antenna_count: The number of antennas available for transmitting on.
    :param final_rx_sample_rate: The final sample rate after decimating on the receive
    side.
    :param ssdelay: Receiver time of flight for last echo. This is the time to continue
     receiving after the last pulse is transmitted.
    :return:
    """

    # Get full pulse sequence
    pulse_sequence_us = []
    sequence_of_samples = [[] for x in range(main_antenna_count)]
    for pulse_index, pulse_dict in enumerate(list_of_pulse_dicts):
        pulse_sequence_us.append(pulse_dict['timing'])
        # Determine the time difference and number of samples between each start of pulse.

    num_samples_list = []
    pulse_offset_error = []
    for pulse_index, pulse_time in enumerate(pulse_sequence_us):
        if pulse_index == 0:
            continue
        num_samples = (pulse_time - pulse_sequence_us[pulse_index - 1]) * 1.0e-6 * txrate
        error = (num_samples - int(num_samples)) / txrate  # in seconds
        num_samples = int(num_samples)
        num_samples_list.append(num_samples)
        pulse_offset_error.append(error)

    current_pulse_samples = []
    for pulse_index, pulse_dict in enumerate(list_of_pulse_dicts):
        if pulse_dict['startofburst'] or not pulse_dict['isarepeat']:
            current_pulse_samples = pulse_dict['samples_array']

        if pulse_index != len(list_of_pulse_dicts) - 1:  # not the last index
            # Add in zeros for the correct number of samples - all arrays in
            # current_pulse_samples are the same length.
            num_zero_samples = num_samples_list[pulse_index] - len(current_pulse_samples[0])
        else:
            num_zero_samples = int(ssdelay * 1.0e-6 * txrate)

        zeros_list = [0.0] * num_zero_samples

        for antenna, samples_array in enumerate(current_pulse_samples):
            sequence_of_samples[antenna].extend(samples_array)
            sequence_of_samples[antenna].extend(zeros_list)

    sequence_of_samples = [np.array(samples_array) for samples_array in
                           sequence_of_samples[:]]

    dm_rate = txrate/final_rx_sample_rate
    dm_rate_error = dm_rate - int(dm_rate)
    dm_rate = int(dm_rate)

    # Create a dictionary to encode as json
    write_dict = {
        'txrate': txrate,
        'txctrfreq': txctrfreq,
        'pulse_sequence_timing': pulse_sequence_us,
        'sequence_samples': {},
        'decimated_sequence': {},
        'pulse_offset_error': pulse_offset_error,
        'dm_rate_error': dm_rate_error,
        'dm_rate': dm_rate
    }

    for ant, samples in enumerate(sequence_of_samples):
        write_dict['sequence_samples'][ant] = {
            'real': samples.real.tolist(),
            'imag': samples.imag.tolist()
        }

    for ant, samples in enumerate(sequence_of_samples):
        decimated_samples = samples[::dm_rate]
        write_dict['decimated_sequence'][ant] = {
            'real': decimated_samples.real.tolist(),
            'imag': decimated_samples.imag.tolist()
        }

    write_time = datetime.now()
    string_time = write_time.strftime('%Y%m%d.%H%M')
    write_dict['samples_approx_time'] = string_time

    with open(file_path + string_time + '.json', 'w') as outfile:
        json.dump(write_dict, outfile)

