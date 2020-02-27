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


def get_phshift(beamdir, freq, antenna, pulse_shift, num_antennas, antenna_spacing,
        centre_offset=0.0):
    """
    Find the phase shift for a given antenna and beam direction.

    Form the beam given the beam direction (degrees off boresite), the tx frequency, the antenna number,
    a specified extra phase shift if there is any, the number of antennas in the array, and the spacing
    between antennas.

    :param beamdir: the azimuthal direction of the beam off boresight, in degrees, positive beamdir being to
        the right of the boresight (looking along boresight from ground). This is for this antenna.
    :param freq: transmit frequency in kHz
    :param antenna: antenna number, INDEXED FROM ZERO, zero being the leftmost antenna if looking down the boresight
        and positive beamdir right of boresight
    :param pulse_shift: in degrees, for phase encoding
    :param num_antennas: number of antennas in this array
    :param antenna_spacing: distance between antennas in this array, in meters
    :param centre_offset: the phase reference for the midpoint of the array. Default = 0.0, in metres.
     Important if there is a shift in centre point between arrays in the direction along the array.
     Positive is shifted to the right when looking along boresight (from the ground).
    :returns phshift: a phase shift for the samples for this antenna number, in radians.
    """

    freq = freq * 1000.0  # convert to Hz.

    beamdir = float(beamdir)

    beamrad = math.pi * float(beamdir) / 180.0

    # Pointing to right of boresight, use point in middle (hypothetically antenna 7.5) as phshift=0
    phshift = 2 * math.pi * freq * (((num_antennas-1)/2.0 - antenna) * \
        antenna_spacing + centre_offset) * math.cos(math.pi / 2.0 - beamrad) \
        / speed_of_light

    # Add an extra phase shift if there is any specified
    phshift = phshift + math.radians(pulse_shift)

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
        linear_rampsampleslen = round(rate * ramp_time)  # number of samples for ramp-up and ramp-down of pulse.
        sampleslen = round(rate * pulse_len)

        rads = sampling_freq * np.arange(sampleslen)
        wave_form = np.exp(rads * 1j)

        amplitude_ramp_up = np.arange(linear_rampsampleslen)/linear_rampsampleslen
        amplitude_ramp_down = np.flipud(amplitude_ramp_up)

        ramp_up_piece = wave_form[:linear_rampsampleslen]
        ramp_down_piece = wave_form[sampleslen - linear_rampsampleslen:]
        np.multiply(ramp_up_piece, amplitude_ramp_up, out=ramp_up_piece)
        np.multiply(ramp_down_piece, amplitude_ramp_down, out=ramp_down_piece)

        samples = wave_form * max_amplitude

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


def rx_azimuth_to_antenna_offset(beamdir, main_antenna_count, interferometer_antenna_count,
                              main_antenna_spacing, interferometer_antenna_spacing,
                              intf_offset, freq):
    """
    Get all the necessary phase shifts for all antennas for all the beams for a pulse sequence.

    Take all beam directions and resolve into a list of phase offsets for all antennas given the
    spacing, frequency, and number of antennas to resolve for (provided in config).

    If the experiment does not use all channels in config, that will be accounted for in the
    send_dsp_metadata function, where the phase rotation will instead = 0.0 so all samples from
    that receive channel will be multiplied by zero and therefore not included (in beamforming).

    :param beamdir: list of length 1 or more.
    :param main_antenna_count: the number of main antennas to calculate the phase offset for.
    :param interferometer_antenna_count: the number of interferometer antennas to calculate the
     phase offset for.
    :param main_antenna_spacing: the spacing between the main array antennas (m).
    :param interferometer_antenna_spacing: the spacing between the interferometer antennas (m).
    :param intf_offset: The interferometer offset from the main array (from centre to centre),
     in Cartesian coordinates. [x, y, z] where x is along line of antennas, y is along array
     normal and z is altitude difference, in m.
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
            # zero pulse shift b/w pulses when beamforming.
            phase_array.append(get_phshift(beam, freq, channel, 0, main_antenna_count,
                main_antenna_spacing))
        for channel in range(0, interferometer_antenna_count):
            # Get phase shifts for all channels, adding in the x - offset of the interferometer
            # from the main array.
            phase_array.append(get_phshift(beam, freq, channel, 0, interferometer_antenna_count,
                interferometer_antenna_spacing, centre_offset=intf_offset[0]))
        beams_antenna_phases.append(phase_array)

    return beams_antenna_phases


def create_debug_sequence_samples(txrate, txctrfreq, list_of_pulse_dicts,
                          main_antenna_count, final_rx_sample_rate, ssdelay):
    """
    Build the samples for the whole sequence, to be recorded in datawrite.

    :param txrate: The rate at which these samples will be transmitted at, Hz.
    :param txctrfreq: The centre frequency that the N200 is tuned to (and will mix with
     these samples, kHz).
    :param list_of_pulse_dicts: The list of all pulse dictionaries for pulses included
    in this sequence. Pulse dictionaries have all metadata and the samples for the
    pulse.
    :param file_path: location to place the json file.
    :param main_antenna_count: The number of antennas available for transmitting on.
    :param final_rx_sample_rate: The final sample rate after decimating on the receive
    side (Hz).
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
        num_samples = ((pulse_time - pulse_sequence_us[pulse_index - 1]) * txrate) * 1.0e-6
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

    # Create a dictionary to be written in datawrite
    write_dict = {
        'txrate': txrate,
        'txctrfreq': txctrfreq,
        'pulse_sequence_timing': pulse_sequence_us,
        'pulse_offset_error': pulse_offset_error,
        'sequence_samples': {},
        'decimated_sequence': {},
        'dmrate_error': dm_rate_error,
        'dmrate': dm_rate
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

    return write_dict
