"""
signals
~~~~~~~~~~~~~~~~~~~~~
This file contains the digital signal processing functionality of Borealis. This includes generation of pulses,
determination of antenna phases for beamforming, filtering and downsampling of received signals, beamforming of
filtered signals, and extraction of lag profiles from beamformed samples.

:copyright: 2024 SuperDARN Canada
:author: Remington Rohel
"""

from functools import reduce
import math
from multiprocessing import shared_memory

from scipy.constants import speed_of_light
import numpy as np
import numpy.fft as fft

try:
    import cupy as xp
except ImportError:
    cupy_available = False
    import numpy as xp
else:
    cupy_available = True


class DSP:
    """
    This class performs the digital signal processing functions of Borealis. Filtering and downsampling are specified
    by lists of filter taps and downsampling factors, which must have the same length. The first filter stage is always
    a bandpass filter, specified by a list of mixing frequencies which can be selected simultaneously. All subsequent
    filter stages are lowpass.

    Beamforming can also be conducted on the filtered and downsampled data, requiring a list of complex antenna
    phases for the beamforming operation. Multiple beams can be formed simultaneously.

    This class also supports extraction of lag profiles from multi-pulse sequences.
    """

    def __init__(
        self, rx_rate, filter_taps, mixing_freqs, dm_rates, use_shared_mem=True
    ):
        """
        Create the filters and initialize parameters for signal processing operations.

        :param      rx_rate:        The wideband rx rate.
        :type       rx_rate:        float
        :param      dm_rates:       The decimation rates at each stage.
        :type       dm_rates:       list
        :param      filter_taps:    The filter taps to use at each stage.
        :type       filter_taps:    ndarray
        :param      mixing_freqs:   The freqs used to mix to baseband.
        :type       mixing_freqs:   list
        :param      use_shared_mem: Flag to use shared memory for CPU arrays
        :type       use_shared_mem: bool
        """
        self.filters = None
        self.filter_outputs = []
        self.beamformed_samples = None
        self.antennas_iq_samples = None
        self.shared_mem = {}
        self.use_shared_mem = use_shared_mem
        self.rx_rate = rx_rate
        self.mixing_freqs = mixing_freqs
        self.dm_rates = dm_rates
        self.filters = self.create_filters(filter_taps, mixing_freqs, rx_rate)

    def apply_filters(self, input_samples):
        """
        Applies the multi-stage filter to input_samples. Filter results are kept on the same device as
        input_samples.

        :param  input_samples:  The wideband samples to operate on.
        :type   input_samples:  ndarray
        """
        # Apply the filtering and downsampling operations
        self.filter_outputs.append(
            self.apply_bandpass_decimate(
                input_samples,
                self.filters[0],
                self.mixing_freqs,
                self.dm_rates[0],
                self.rx_rate,
            )
        )
        for i in range(1, len(self.filters)):
            self.filter_outputs.append(
                self.apply_lowpass_decimate(
                    self.filter_outputs[i - 1], self.filters[i], self.dm_rates[i]
                )
            )

    def move_filter_results(self):
        """
        Move the final results of filtering (antennas_iq data) to the CPU, optionally in SharedMemory if
        specified for this instance.
        """
        # Create an array on the CPU for antennas_iq data
        antennas_iq_samples = self.filter_outputs[-1]
        if self.use_shared_mem:
            ant_shm = shared_memory.SharedMemory(
                create=True, size=antennas_iq_samples.nbytes
            )
            buffer = ant_shm.buf
            self.shared_mem["antennas_iq"] = ant_shm
        else:
            buffer = None
        self.antennas_iq_samples = np.ndarray(
            antennas_iq_samples.shape, dtype=np.complex64, buffer=buffer
        )

        # Move the antennas_iq samples to the CPU for beamforming
        if cupy_available:
            self.antennas_iq_samples[...] = xp.asnumpy(antennas_iq_samples)[...]
        else:
            self.antennas_iq_samples[...] = antennas_iq_samples[...]

    def beamform(self, beam_phases):
        """
        Applies the beamforming operation to the antennas_iq samples. Different beam directions are formed in parallel,
        and the results are stored in SharedMemory if so specified for this instance.

        :param  beam_phases:    The phases used to beamform the filtered samples.
        :type   beam_phases:    list
        """
        # beam_phases: [num_slices, num_beams, num_antennas]
        beam_phases = np.array(beam_phases)

        # Create shared memory for result of beamforming
        # final_shape: [num_slices, num_beams, num_samples]
        final_shape = (
            self.antennas_iq_samples.shape[0],
            beam_phases.shape[1],
            self.antennas_iq_samples.shape[2],
        )
        final_size = np.dtype(np.complex64).itemsize * reduce(
            lambda a, b: a * b, final_shape
        )

        if self.use_shared_mem:
            bf_shm = shared_memory.SharedMemory(create=True, size=final_size)
            buffer = bf_shm.buf
            self.shared_mem["bfiq"] = bf_shm
        else:
            buffer = None
        self.beamformed_samples = np.ndarray(
            final_shape, dtype=np.complex64, buffer=buffer
        )

        # Apply beamforming
        self.beamformed_samples[...] = self.beamform_samples(
            self.antennas_iq_samples, beam_phases
        )

    def cfs_freq_analysis(self, metadata, n):
        """
        Performs decimation and frequency analysis on clear frequency search data. Data will not be
        in shared memory.

        :param  metadata:   Clear frequency search sequence metadata
        :type   metadata:   dict
        :param  n:          Number of points used in the FFT. Determines the frequency resolution
                            of where df = (rx_rate / total decimation rate) / N
        :type   n:          int
        """
        fs = self.rx_rate / np.prod(self.dm_rates)  # Sampling frequency in Hz

        pulses = metadata["pulses"]
        tau = round(metadata["tau_spacing"] * 1e-6 * fs)  # puts into units of samples
        pulses_in_samples = [int(round(p * tau)) for p in pulses]
        pulse_dur = round(0.0006 * fs)  # TODO: determine way to derive 0.0006
        start_sample = int(round(pulses_in_samples[0] + pulse_dur / 2))
        if len(pulses_in_samples) > 1:
            end_sample = int(round(pulses_in_samples[1] - pulse_dur / 2))
        else:
            end_sample = self.antennas_iq_samples.shape[-1]

        num_intervals = int((end_sample - start_sample) / n)
        end_sample = start_sample + num_intervals * n
        data = self.beamformed_samples[:, :, start_sample:end_sample]
        data_chunks = np.reshape(data, data.shape[:-1] + (num_intervals, n))

        fft_data = fft.fftshift(fft.fft(data_chunks, axis=-1), axes=-1)
        cfs_data = 20 * np.log(np.sum(np.abs(np.average(fft_data, axis=2)), axis=1))
        cfs_freqs = fft.fftshift(fft.fftfreq(n, d=1 / fs))

        return cfs_data, cfs_freqs

    @staticmethod
    def create_filters(filter_taps, mixing_freqs, rx_rate):
        """
        Creates and shapes the filters arrays using the original sets of filter taps. The first
        stage filters are mixed to bandpass and the low pass filters are reshaped. The filters
        coefficients are typically symmetric, except for the first-stage bandpass filters.
        Mixing frequencies should be given as an offset from the center frequency.
        For example, with 12 MHz center frequency and a 10.5 MHz transmit
        frequency, the mixing frequency should be -1.5 MHz.

        :param      filter_taps:   The filter taps from the experiment decimation scheme.
        :type       filter_taps:   list
        :param      mixing_freqs:  The frequencies used to mix the first stage filter for bandpass.
        :type       mixing_freqs:  list
        :param      rx_rate:       The rf rx rate.
        :type       rx_rate:       float

        :returns:   List of stages of filter taps. First stage is bandpass, subsequent stages are lowpass.
        :rtype:     list[ndarray]
        """
        filters = []
        n = len(mixing_freqs)
        m = filter_taps[0].shape[0]
        bandpass = np.zeros((n, m), dtype=np.complex64)
        s = np.arange(m, dtype=np.complex64)
        for idx, f in enumerate(mixing_freqs):
            # Filtering is actually done via a correlation, not a convolution (the filter is not reversed).
            # Thus, the mixing frequency should be the negative of the frequency that is actually being extracted.
            bandpass[idx, :] = filter_taps[0] * np.exp(s * 2j * np.pi * (-f) / rx_rate)
        filters.append(xp.array(bandpass, dtype=xp.complex64))

        for t in filter_taps[1:]:
            filters.append(xp.array(t[np.newaxis, :], dtype=xp.complex64))

        return filters

    @staticmethod
    def windowed_view(ndarray, window_len, step):
        """
        Creates a strided and windowed view of the ndarray. This allows us to skip samples that will
        otherwise be dropped without missing samples needed for the convolutions windows. The strides
        will also not extend out of bounds meaning we do not need to pad extra samples and then drop bad
        samples after the fact.

        :param      ndarray:     The input ndarray
        :type       ndarray:     ndarray
        :param      window_len:  The window length(filter length)
        :type       window_len:  int
        :param      step:        The step(dm rate)
        :type       step:        int

        :returns:   The array with a new view.
        :rtype:     ndarray
        """

        nrows = ((ndarray.shape[-1] - window_len) // step) + 1
        last_dim_stride = ndarray.strides[-1]
        new_shape = ndarray.shape[:-1] + (nrows, window_len)
        new_strides = list(ndarray.strides + (last_dim_stride,))
        new_strides[-2] *= step
        return xp.lib.stride_tricks.as_strided(
            ndarray, shape=new_shape, strides=new_strides
        )

    @staticmethod
    def apply_bandpass_decimate(
        input_samples, bp_filters, mixing_freqs, dm_rate, rx_rate
    ):
        """
        Apply a Frerking bandpass filter to the input samples. Several different frequencies can be
        centered on simultaneously. Downsampling is done in parallel via a strided window view of
        the input samples.

        :param      input_samples:  The input raw rf samples for each antenna.
        :type       input_samples:  ndarray [num_antennas, num_samples]
        :param      bp_filters:     The bandpass filter(s).
        :type       bp_filters:     ndarray [num_slices, num_taps]
        :param      mixing_freqs:   The frequencies used to mix the first stage filter for bandpass.
        :type       mixing_freqs:   list
        :param      dm_rate:        The decimation rate of this stage
        :type       dm_rate:        int
        :param      rx_rate:        The rf rx rate.
        :type       rx_rate:        float

        :returns:   Samples after bandpass filter and downsampling operations. Shape [num_slices, num_antennas, samples]
        :rtype:     ndarray
        """
        # We need to force the input into the GPU to be float16, float32, or complex64 so that the einsum result is
        # complex64 and NOT complex128. The GPU is significantly slower (10x++) working with complex128 numbers.
        # We do not require the additional precision.
        bp_filters = xp.array(bp_filters, dtype=xp.complex64)
        input_samples = DSP.windowed_view(input_samples, bp_filters.shape[-1], dm_rate)

        # [num_slices, num_taps]
        # [num_antennas, num_output_samples, num_taps]
        filtered = xp.einsum(
            "ij,klj->ikl", bp_filters, input_samples, optimize="greedy"
        )

        # Apply the phase correction for the Frerking method.
        ph = xp.arange(filtered.shape[-1], dtype=np.float32)[xp.newaxis, :]
        freqs = xp.array(mixing_freqs)[:, xp.newaxis]

        # [1, num_output_samples]
        # [num_slices, 1]
        # ph: [num_slices, num_output_samples]
        ph = xp.fmod(ph * 2.0 * xp.pi * (-freqs) / rx_rate * dm_rate, 2.0 * xp.pi)
        ph = xp.exp(1j * ph.astype(xp.float32))

        # [num_slices, num_antennas, num_output_samples]
        # [num_slices, 1, num_output_samples]
        # corrected: [num_slices, num_antennas, num_output_samples]
        corrected = filtered * ph[:, xp.newaxis, :]

        return corrected

    @staticmethod
    def apply_lowpass_decimate(input_samples, lp_filter, dm_rate):
        """
        Apply a lowpass filter to the baseband input samples. Downsampling is done in parallel via a
        strided window view of the input samples.

        :param      input_samples:  Baseband input samples
        :type       input_samples:  ndarray [num_slices, num_antennas, num_samples]
        :param      lp_filter:      Lowpass filter taps
        :type       lp_filter:      ndarray [1, num_taps]
        :param      dm_rate:        The decimation rate of this stage.
        :type       dm_rate:        int

        :returns:   Samples after lowpass filter and downsampling operations. Shape [num_slices, num_antennas, samples]
        :rtype:     ndarray
        """
        # We need to force the input into the GPU to be float16, float32, or complex64 so that the einsum result is
        # complex64 and NOT complex128. The GPU is significantly slower (10x++) working with complex128 numbers.
        # We do not require the additional precision.
        lp_filter = xp.array(lp_filter, dtype=xp.complex64)
        input_samples = DSP.windowed_view(input_samples, lp_filter.shape[-1], dm_rate)

        # [1, num_taps]
        # [num_slices, num_antennas, num_output_samples, num_taps]
        # filtered: [num_slices, num_antennas, num_output_samples]
        filtered = xp.einsum("ij,klmj->klm", lp_filter, input_samples)

        return filtered

    @staticmethod
    def beamform_samples(filtered_samples, beam_phases):
        """
        Beamform the filtered samples for multiple beams simultaneously.

        :param      filtered_samples:  The filtered input samples.
        :type       filtered_samples:  ndarray [num_slices, num_antennas, num_samples]
        :param      beam_phases:       The beam phases used to phase each antenna's samples before
                                       combining.
        :type       beam_phases:       ndarray [num_slices, num_beams, num_antennas]

        :returns:   Beamformed samples of shape [num_slices, num_beams, num_samples]
        :rtype:     np.ndarray
        """
        # [num_slices, num_beams, num_antennas]
        beam_phases = np.array(beam_phases)

        # result: [num_slices, num_beams, num_samples]
        return np.einsum("ijk,ilj->ilk", filtered_samples, beam_phases)

    @staticmethod
    def correlations_from_samples(
        beamformed_samples_1,
        beamformed_samples_2,
        output_sample_rate,
        slice_index_details,
    ):
        """
        Correlate two sets of beamformed samples together. Correlation matrices are used and indices
        corresponding to lag pulse pairs are extracted.

        :param      beamformed_samples_1:   The first beamformed samples.
        :type       beamformed_samples_1:   ndarray [num_slices, num_beams, num_samples]
        :param      beamformed_samples_2:   The second beamformed samples.
        :type       beamformed_samples_2:   ndarray [num_slices, num_beams, num_samples]
        :param      output_sample_rate:     Sampling rate of data.
        :type       output_sample_rate:     float
        :param      slice_index_details:    Details used to extract indices for each slice.
        :type       slice_index_details:    list

        :returns:   Correlations for slices. List of length num_slices, with each entry having shape
                    [num_beams, num_range_gates, num_lags].
        :rtype:     list[ndarray]
        """
        values = []
        for s, slice_info in enumerate(slice_index_details):
            if slice_info["lags"].size == 0:
                values.append(np.array([]))
                continue

            range_off = (
                np.arange(slice_info["num_range_gates"], dtype=np.int32)
                + slice_info["first_range_off"]
            )
            tau_in_samples = slice_info["tau_spacing"] * 1e-6 * output_sample_rate
            lag_pulses_as_samples = np.array(slice_info["lags"], np.int32) * np.int32(
                tau_in_samples
            )

            # [num_range_gates, 1, 1]
            # [1, num_lags, 2]
            samples_for_all_range_lags = (
                range_off[..., np.newaxis, np.newaxis]
                + lag_pulses_as_samples[np.newaxis, :, :]
            )

            # [num_range_gates, num_lags, 2]
            row = samples_for_all_range_lags[..., 1].astype(np.int32)

            # [num_range_gates, num_lags, 2]
            col = samples_for_all_range_lags[..., 0].astype(np.int32)

            values_for_slice = np.empty(
                (beamformed_samples_1.shape[1], row.shape[0], row.shape[1]),
                dtype=np.complex64,
            )

            for lag in range(row.shape[1]):
                values_for_slice[:, :, lag] = np.einsum(
                    "ij,ij->ji",
                    beamformed_samples_1[s, :, row[:, lag]],
                    beamformed_samples_2[s, :, col[:, lag]].conj(),
                )

            # [num_beams, num_range_gates, num_lags]
            values_for_slice = np.einsum(
                "ijk,k->ijk", values_for_slice, slice_info["lag_phase_offsets"]
            )

            values.append(values_for_slice)

        return values


def get_phase_shift(
    beam_angle: list[float],
    freq_khz: float,
    antenna_locations: np.ndarray,
    centre_offset=(0.0, 0.0, 0.0),
):
    """
    Find the complex excitation for all antennas to make beams in all given directions.

    :param  beam_angle:         list of azimuthal direction of the beam off boresight, in degrees,
                                positive beamdir being to the right of the boresight (looking along
                                boresight from ground).
    :type   beam_angle:         list
    :param  freq_khz:           transmit frequency in kHz
    :type   freq_khz:           float
    :param  antenna_locations:  [x, y, z] coordinates of each antenna in the array, in meters. Shape [antennas, 3]
    :type   antenna_locations:  np.ndarray
    :param  centre_offset:      the phase reference for the midpoint of the array. Default = (0.0, 0.0, 0.0), in
                                metres. Important if there is a shift in centre point between arrays
                                in the direction along the array. Positive is shifted to the right
                                when looking along boresight (from the ground).
    :type   centre_offset:      list[float]

    :returns:   phase_shift     a 2D array of shape [beams, antennas] giving the complex excitation for each
                                antenna required to form each beam.
    :rtype:     phase_shift     ndarray
    """

    freq_hz = freq_khz * 1000.0  # convert to Hz.
    k = 2 * np.pi * freq_hz / speed_of_light  # 2pi / wavelength
    beam_rads = np.deg2rad(np.array(beam_angle, dtype=np.float32))

    # Get the x-position of each antenna relative to the array midpoint
    x = antenna_locations[:, 0] - centre_offset[0]

    # phase shift = 0 at array midpoint (by convention), so this is the displacement in x of each beam from the array
    # midpoint after the wave traverses one wavelength. Essentially, the component along x of the beam, normalized by
    # the wavelength.
    beam_displacements = -1 * np.sin(beam_rads) * k

    phase_shift = np.einsum("i,j->ij", beam_displacements, x)
    phase_shift = np.exp(1j * phase_shift)

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

    :returns samples:       a numpy array of complex samples, representing all samples
                            needed for a pulse of length pulse_len sampled at a rate of rate.
    :rtype:  ndarray
    """

    wave_freq = float(wave_freq)
    rate = float(rate)

    sampling_freq = 2 * math.pi * wave_freq / rate

    # for linear we used the below:
    linear_rampsampleslen = round(
        rate * ramp_time
    )  # number of samples for ramp-up and ramp-down of pulse.
    sampleslen = round(rate * pulse_len)

    rads = sampling_freq * np.arange(sampleslen)
    wave_form = np.exp(rads * 1j)

    amplitude_ramp_up = np.arange(linear_rampsampleslen) / linear_rampsampleslen
    amplitude_ramp_down = np.flipud(amplitude_ramp_up)

    ramp_up_piece = wave_form[:linear_rampsampleslen]
    ramp_down_piece = wave_form[sampleslen - linear_rampsampleslen :]
    np.multiply(ramp_up_piece, amplitude_ramp_up, out=ramp_up_piece)
    np.multiply(ramp_down_piece, amplitude_ramp_down, out=ramp_down_piece)

    samples = wave_form * max_amplitude

    return samples


def basic_pulse_phase_offset(exp_slice):
    """
    Calculate the phase difference of each pulse with respect to the first
    pulse based on the transmit frequency and the pulse separation.

    :param      exp_slice:  The experiment slice information
    :type       exp_slice:  class

    :returns:   Pulse phase offsets
    :rtype:     array (rad)
    """
    freq_hz = exp_slice.freq * 1e3
    tau_s = exp_slice.tau_spacing / 1e6
    omega = -2 * np.pi * freq_hz
    pulse_sequence = exp_slice.pulse_sequence

    num_pulses = len(pulse_sequence)
    pulse_phases = np.zeros(num_pulses)
    for p in range(num_pulses):
        pulse_time = pulse_sequence[p] * tau_s
        pulse_phases[p] = np.angle(np.exp(1j * omega * pulse_time))

    return pulse_phases
