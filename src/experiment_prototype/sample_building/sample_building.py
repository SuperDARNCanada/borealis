"""
    sample_building
    ~~~~~~~~~~~~~~~
    Functions to process and build samples, including getting phase shifts from beam directions, and
    functions for creating samples as well as shifting them as required.

    :copyright: 2017 SuperDARN Canada
    :author: Marci Detwiller
"""

from scipy.constants import speed_of_light
import numpy as np
import math
from experiment_prototype.experiment_exception import ExperimentException


def resolve_imaging_directions(beamdirs_list, num_antennas, antenna_spacing):   # TODO: Delete this? Unused
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


def get_phase_shift(beam_angle, freq, num_antennas, antenna_spacing, centre_offset=0.0):
    """
    Find the phase shift for a given antenna and beam direction.

    Form the beam given the beam direction (degrees off boresite), the tx frequency, the antenna
    number, a specified extra phase shift if there is any, the number of antennas in the array, and
    the spacing between antennas.

    :param  beam_angle:         list of azimuthal direction of the beam off boresight, in degrees,
                                positive beamdir being to the right of the boresight (looking along
                                boresight from ground). This is for this antenna.
    :type   beam_angle:         list
    :param  freq:               transmit frequency in kHz
    :type   freq:               float
    :param  num_antennas:       number of antennas in this array
    :type   num_antennas:       int
    :param  antenna_spacing:    distance between antennas in this array, in meters
    :type   antenna_spacing:    float
    :param  centre_offset:      the phase reference for the midpoint of the array. Default = 0.0, in
                                metres. Important if there is a shift in centre point between arrays
                                in the direction along the array. Positive is shifted to the right
                                when looking along boresight (from the ground).
    :type   centre_offset:      float

    :returns:   phase_shift     a 2D array of beam_phases x antennas in radians.
    :rtype:     phase_shift     ndarray
    """

    freq_hz = freq * 1000.0  # convert to Hz.

    # convert the beam angles to rads
    beam_rads = (np.pi / 180) * np.array(beam_angle, dtype=np.float32)

    antennas = np.arange(num_antennas)
    x = ((num_antennas - 1) / 2.0 - antennas) * antenna_spacing + centre_offset
    x *= 2 * np.pi * freq_hz

    y = np.cos(np.pi / 2.0 - beam_rads) / speed_of_light
    # split up the calculations for beams and antennas. Outer multiply of the two
    # vectors will yield all antenna phases needed for each beam.
    # If there are N antennas and M beams
    # Eventual matrix is now:
    # [antenna0beam0 .. antenna1beam0 .... ... antennaN-1beam0
    # antenna0beam1 ... antenna1beam1 .... ... antennaN-1beam1
    # ...
    # ...
    # antenna0beamM-1 ... antenna1beamM-1... ... anteannaN-1beamM-1]
    phase_shift = np.fmod(np.outer(y, x), 2.0 * np.pi) # beams by antenna
    phase_shift = np.exp(1j * phase_shift)

    # Pointing to right of boresight, use point in middle (hypothetically antenna 7.5) as phshift=0
    return phase_shift


def get_samples(rate, wave_freq, pulse_len, ramp_time, max_amplitude):
    """
    Get basic (not phase-shifted) samples for a given pulse.

    Find the normalized sample array given the rate (Hz), frequency (Hz), pulse length (s).
    Will shift for beam later.

    :param  rate:           tx sampling rate, in Hz.
    :type   rate:           float
    :param  wave_freq:      frequency offset from the centre frequency on the USRP, given in Hz. To
                            be mixed with the centre frequency before transmitting. (ex. centre = 12
                            MHz, wave_freq = + 1.2 MHz, output = 13.2 MHz.
    :type   wave_freq:      float
    :param  pulse_len:      length of the pulse (in seconds)
    :type   pulse_len:      float
    :param  ramp_time:      ramp up and ramp down time for the pulse, in seconds. Typical 0.00001 s
                            from config.
    :type   ramp_time:      float
    :param  max_amplitude:  USRP's max DAC amplitude. N200 = 0.707 max
    :type   max_amplitude:  float

    :returns:

            samples:        a numpy array of complex samples, representing all samples
                            needed for a pulse of length pulse_len sampled at a rate of rate.
    :rtype:     ndarray
    """

    wave_freq = float(wave_freq)
    rate = float(rate)

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

    # Samples is an array of complex samples
    # NOTE: phasing will be done in shift_samples function
    return samples


