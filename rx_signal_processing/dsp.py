import numpy as np
from scipy.fftpack import fft
import math
import time
from multiprocessing import shared_memory

try:
    import cupy as xp
except ImportError:
    import numpy as xp

    cupy_available = False
else:
    cupy_available = True


def windowed_view(ndarray, window_len, step):
    """
    Creates a strided and windowed view of the ndarray. This allows us to skip samples that will
    otherwise be dropped without missing samples needed for the convolutions windows. The
    strides will also not extend out of bounds meaning we do not need to pad extra samples and
    then drop bad samples after the fact.

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
    return xp.lib.stride_tricks.as_strided(ndarray, shape=new_shape, strides=new_strides)


class DSP(object):
    """
    This class performs the DSP functions of Borealis

    :param      input_samples: The wideband samples to operate on.
    :type       input_samples: ndarray
    :param      rx_rate: The wideband rx rate.
    :type       rx_rate: float
    :param      dm_rates: The decimation rates at each stage.
    :type       dm_rates: list
    :param      filter_taps: The filter taps to use at each stage.
    :type       filter_taps: ndarray
    :param      mixing_freqs: The freqs used to mix to baseband.
    :type       mixing_freqs: list
    :param      beam_phases: The phases used to beamform the final decimated samples.
    :type       beam_phases: list
    """

    def __init__(self, input_samples, rx_rate, dm_rates, filter_taps, mixing_freqs, beam_phases):
        super(DSP, self).__init__()
        self.filters = None
        self.filter_outputs = []
        self.beamformed_samples = None
        self.shared_mem = {}

        self.create_filters(filter_taps, mixing_freqs, rx_rate)

        self.apply_bandpass_decimate(input_samples, self.filters[0], mixing_freqs, dm_rates[0], rx_rate)

        for i in range(1, len(self.filters)):
            self.apply_lowpass_decimate(self.filter_outputs[i - 1], self.filters[i], dm_rates[i])

        # Create shared memory for antennas_iq data
        antennas_iq_samples = self.filter_outputs[-1]
        ant_shm = shared_memory.SharedMemory(create=True, size=antennas_iq_samples.nbytes)
        self.antennas_iq_samples = np.ndarray(antennas_iq_samples.shape, dtype=np.complex64, buffer=ant_shm.buf)

        # Move the antennas_iq samples to the CPU for beamforming
        if cupy_available:
            self.antennas_iq_samples = xp.asnumpy(self.antennas_iq_samples)
        else:
            self.antennas_iq_samples = antennas_iq_samples
        self.shared_mem['antennas_iq'] = ant_shm.name

        self.beamform_samples(self.antennas_iq_samples, beam_phases)
        ant_shm.close()
        
    def create_filters(self, filter_taps, mixing_freqs, rx_rate):
        """
        Creates and shapes the filters arrays using the original sets of filter taps. The first
        stage filters are mixed to bandpass and the low pass filters are reshaped.
        The filters coefficients are typically symmetric, with the exception of the first-stage bandpass
        filters. As a result, the mixing frequency should be the negative of the frequency that is actually
        being extracted. For example, with 12 MHz center frequency and a 10.5 MHz transmit frequency,
        the mixing frequency should be 1.5 MHz.

        :param      filter_taps:   The filters taps from the experiment decimation scheme.
        :type       filter_taps:   list
        :param      mixing_freqs:  The frequencies used to mix the first stage filter for bandpass. Calculated
                                   as (center freq - rx freq), as filter coefficients are cross-correlated with
                                   samples instead of convolved.
        :type       mixing_freqs:  list
        :param      rx_rate:       The rf rx rate.
        :type       rx_rate:       float

        """
        filters = []
        n = len(mixing_freqs)
        m = filter_taps[0].shape[0]
        bandpass = np.zeros((n, m), dtype=np.complex64)
        s = np.arange(m, dtype=np.complex64)
        for idx, f in enumerate(mixing_freqs):
            bandpass[idx, :] = filter_taps[0] * np.exp(s * 2j * np.pi * f / rx_rate)
        filters.append(bandpass)

        for t in filter_taps[1:]:
            filters.append(t[np.newaxis, :])

        self.filters = filters

    def apply_bandpass_decimate(self, input_samples, bp_filters, mixing_freqs, dm_rate, rx_rate):
        """
        Apply a Frerking bandpass filter to the input samples. Several different frequencies can
        be centered on simultaneously. Downsampling is done in parallel via a strided window
        view of the input samples.

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

        """
        # We need to force the input into the GPU to be float16, float32, or complex64 so that the einsum result is
        # complex64 and NOT complex128. The GPU is significantly slower (10x++) working with complex128 numbers.
        # We do not require the additional precision.
        bp_filters = xp.array(bp_filters, dtype=xp.complex64)
        input_samples = windowed_view(input_samples, bp_filters.shape[-1], dm_rate)

        # [num_slices, num_taps]
        # [num_antennas, num_output_samples, num_taps]
        filtered = xp.einsum('ij,klj->ikl', bp_filters, input_samples)

        ph = xp.arange(filtered.shape[-1], dtype=np.float64)[xp.newaxis, :]
        freqs = xp.array(mixing_freqs)[:, xp.newaxis]

        # [1, num_output_samples]
        # [num_slices, 1]
        ph = xp.fmod(ph * 2.0 * xp.pi * freqs / rx_rate * dm_rate, 2.0 * xp.pi)
        ph = xp.exp(1j * ph.astype(xp.complex64))

        # [num_slices, num_antennas, num_output_samples]
        # [num_slices, 1, num_output_samples]
        corrected = filtered * ph[:, xp.newaxis, :]

        self.filter_outputs.append(corrected)

    def apply_lowpass_decimate(self, input_samples, lp_filter, dm_rate):
        """
        Apply a lowpass filter to the baseband input samples. Downsampling is done in parallel via a
        strided window view of the input samples.

        :param      input_samples:  Baseband input samples
        :type       input_samples:  ndarray [num_slices, num_antennas, num_samples]
        :param      lp:             Lowpass filter taps
        :type       lp:             ndarray [1, num_taps]
        :param      dm_rate:        The decimation rate of this stage.
        :type       dm_rate:        int

        """
        # We need to force the input into the GPU to be float16, float32, or complex64 so that the einsum result is
        # complex64 and NOT complex128. The GPU is significantly slower (10x++) working with complex128 numbers.
        # We do not require the additional precision.
        lp_filter = xp.array(lp_filter, dtype=xp.complex64)
        input_samples = windowed_view(input_samples, lp_filter.shape[-1], dm_rate)

        # [1, num_taps]
        # [num_slices, num_antennas, num_output_samples, num_taps]
        filtered = xp.einsum('ij,klmj->klm', lp_filter, input_samples)

        self.filter_outputs.append(filtered)

    def beamform_samples(self, filtered_samples, beam_phases):
        """
        Beamform the filtered samples for multiple beams simultaneously.

        :param      filtered_samples:  The filtered input samples.
        :type       filtered_samples:  ndarray [num_slices, num_antennas, num_samples]
        :param      beam_phases:       The beam phases used to phase each antenna's samples before
                                       combining.
        :type       beam_phases:       list

        """
        # [num_slices, num_beams, num_antennas]
        beam_phases = np.array(beam_phases)

        # [num_slices, num_beams, num_samples]
        final_shape = (filtered_samples.shape[0], beam_phases.shape[2], filtered_samples.shape[2])
        final_size = np.dtype(np.complex64).itemsize * filtered_samples.shape[0] * filtered_samples.shape[2] * beam_phases.shape[2]
        bf_shm = shared_memory.SharedMemory(create=True, size=final_size)
        self.beamformed_samples = np.ndarray(final_shape, dtype=np.complex64, buffer=bf_shm.buf)
        self.beamformed_samples = np.einsum('ijk,ilj->ilk', filtered_samples, beam_phases)

        self.shared_mem['bfiq'] = bf_shm.name
        bf_shm.close()

    @staticmethod
    def correlations_from_samples(beamformed_samples_1, beamformed_samples_2, output_sample_rate, slice_index_details):
        """
        Correlate two sets of beamformed samples together. Correlation matrices are used and
        indices corresponding to lag pulse pairs are extracted.

        :param      beamformed_samples_1:  The first beamformed samples.
        :type       beamformed_samples_1:  ndarray [num_slices, num_beams, num_samples]
        :param      beamformed_samples_2:  The second beamformed samples.
        :type       beamformed_samples_2:  ndarray [num_slices, num_beams, num_samples]
        :param      slice_index_details:   Details used to extract indices for each slice.
        :type       slice_index_details:   list

        :returns:   Correlations for slices.
        :rtype:     list
        """

        # [num_slices, num_beams, num_samples]
        # [num_slices, num_beams, num_samples]
        correlated = np.einsum('ijk,ijl->ijkl', beamformed_samples_1,
                               beamformed_samples_2.conj())

        values = []
        for s in slice_index_details:
            if s['lags'].size == 0:
                values.append(np.array([]))
                continue
            range_off = np.arange(s['num_range_gates'], dtype=np.int32) + s['first_range_off']

            tau_in_samples = s['tau_spacing'] * 1e-6 * output_sample_rate

            lag_pulses_as_samples = np.array(s['lags'], np.int32) * np.int32(tau_in_samples)

            # [num_range_gates, 1, 1]
            # [1, num_lags, 2]
            samples_for_all_range_lags = (range_off[..., np.newaxis, np.newaxis] +
                                          lag_pulses_as_samples[np.newaxis, :, :])

            # [num_range_gates, num_lags, 2]
            row = samples_for_all_range_lags[..., 1].astype(np.int32)

            # [num_range_gates, num_lags, 2]
            column = samples_for_all_range_lags[..., 0].astype(np.int32)

            values_for_slice = correlated[s['slice_num'], :, row, column]

            # [num_range_gates, num_lags, num_beams]
            values_for_slice = np.einsum('ijk,j->kij', values_for_slice, s['lag_phase_offsets'])

            values.append(values_for_slice)

        return values


def fft_and_plot(samples, rate):
    import matplotlib.pyplot as plt
    fft_samps = fft(samples)
    T = 1.0 / float(rate)
    num_samps = len(samples)
    xf = np.linspace(-1.0 / (2.0 * T), 1.0 / (2.0 * T), num_samps)

    fig, smpplt = plt.subplots(1, 1)

    fft_to_plot = np.empty([num_samps], dtype=np.complex64)
    halfway = int(math.ceil(float(num_samps) / 2))
    fft_to_plot = np.concatenate([fft_samps[halfway:], fft_samps[:halfway]])
    smpplt.plot(xf, 1.0 / num_samps * np.abs(fft_to_plot))
    plt.show()


def quick_test(n):
    dm_rates = np.array([30, 10, 5])
    freqs = np.array([1e6, 2e6, 4e6])
    F_s = 5e6
    t = 100e-3

    filter_taps = []
    filter_taps.append(np.array([1.02221e-08, 1.58724e-08, 2.29334e-08, 3.16247e-08, 4.21887e-08, 5.48931e-08, 7.00324e-08, 8.79294e-08, 1.08938e-07, 1.33443e-07, 1.61867e-07, 1.94666e-07, 2.32336e-07, 2.75416e-07, 3.24486e-07, 3.80174e-07, 4.43157e-07, 5.14162e-07, 5.9397e-07, 6.8342e-07, 7.83409e-07, 8.94899e-07, 1.01891e-06, 1.15655e-06, 1.30897e-06, 1.47743e-06, 1.66322e-06, 1.86777e-06, 2.09255e-06, 2.33913e-06, 2.60918e-06, 2.90446e-06, 3.22683e-06, 3.57824e-06, 3.96076e-06, 4.37657e-06, 4.82794e-06, 5.31729e-06, 5.84713e-06, 6.42011e-06, 7.039e-06, 7.7067e-06, 8.42624e-06, 9.2008e-06, 1.00337e-05, 1.09284e-05, 1.18885e-05, 1.29177e-05, 1.40199e-05, 1.51993e-05, 1.64601e-05, 1.78065e-05, 1.92433e-05, 2.0775e-05, 2.24066e-05, 2.41432e-05, 2.59899e-05, 2.79522e-05, 3.00358e-05, 3.22462e-05, 3.45896e-05, 3.7072e-05, 3.96998e-05, 4.24795e-05, 4.54177e-05, 4.85213e-05, 5.17974e-05, 5.52533e-05, 5.88963e-05, 6.2734e-05, 6.67744e-05, 7.10253e-05, 7.54949e-05, 8.01916e-05, 8.51238e-05, 9.03003e-05, 9.57298e-05, 0.000101422, 0.000107385, 0.000113628, 0.000120162, 0.000126996, 0.000134139, 0.000141602, 0.000149394, 0.000157526, 0.000166009, 0.000174851, 0.000184065, 0.000193661, 0.000203649, 0.000214041, 0.000224846, 0.000236077, 0.000247743, 0.000259857, 0.000272429, 0.00028547, 0.000298991, 0.000313004, 0.00032752, 0.000342549, 0.000358102, 0.000374191, 0.000390827, 0.000408019, 0.00042578, 0.00044412, 0.000463048, 0.000482576, 0.000502713, 0.000523471, 0.000544857, 0.000566883, 0.000589558, 0.00061289, 0.000636889, 0.000661563, 0.000686922, 0.000712972, 0.000739722, 0.00076718, 0.000795351, 0.000824244, 0.000853865, 0.000884219, 0.000915312, 0.000947149, 0.000979735, 0.00101307, 0.00104717, 0.00108203, 0.00111765, 0.00115403, 0.00119118, 0.0012291, 0.00126779, 0.00130725, 0.00134747, 0.00138846, 0.00143022, 0.00147274, 0.00151602, 0.00156005, 0.00160483, 0.00165036, 0.00169664, 0.00174364, 0.00179137, 0.00183982, 0.00188898, 0.00193884, 0.0019894, 0.00204063, 0.00209253, 0.00214509, 0.00219828, 0.00225211, 0.00230656, 0.0023616, 0.00241723, 0.00247342, 0.00253017, 0.00258744, 0.00264523, 0.00270351, 0.00276227, 0.00282147, 0.00288111, 0.00294115, 0.00300158, 0.00306237, 0.00312349, 0.00318492, 0.00324664, 0.00330862, 0.00337083, 0.00343325, 0.00349584, 0.00355858, 0.00362144, 0.00368438, 0.00374739, 0.00381043, 0.00387346, 0.00393646, 0.00399939, 0.00406222, 0.00412493, 0.00418747, 0.00424981, 0.00431192, 0.00437377, 0.00443532, 0.00449654, 0.0045574, 0.00461785, 0.00467787, 0.00473742, 0.00479647, 0.00485498, 0.00491292, 0.00497025, 0.00502694, 0.00508295, 0.00513826, 0.00519282, 0.0052466, 0.00529958, 0.00535171, 0.00540297, 0.00545332, 0.00550273, 0.00555117, 0.00559861, 0.00564502, 0.00569036, 0.00573461, 0.00577774, 0.00581973, 0.00586053, 0.00590013, 0.0059385, 0.00597562, 0.00601145, 0.00604599, 0.00607919, 0.00611104, 0.00614152, 0.00617061, 0.00619829, 0.00622454, 0.00624934, 0.00627267, 0.00629452, 0.00631487, 0.00633372, 0.00635103, 0.00636682, 0.00638105, 0.00639373, 0.00640484, 0.00641439, 0.00642235, 0.00642872, 0.00643351, 0.0064367, 0.00643829, 0.00643829, 0.0064367, 0.00643351, 0.00642872, 0.00642235, 0.00641439, 0.00640484, 0.00639373, 0.00638105, 0.00636682, 0.00635103, 0.00633372, 0.00631487, 0.00629452, 0.00627267, 0.00624934, 0.00622454, 0.00619829, 0.00617061, 0.00614152, 0.00611104, 0.00607919, 0.00604599, 0.00601145, 0.00597562, 0.0059385, 0.00590013, 0.00586053, 0.00581973, 0.00577774, 0.00573461, 0.00569036, 0.00564502, 0.00559861, 0.00555117, 0.00550273, 0.00545332, 0.00540297, 0.00535171, 0.00529958, 0.0052466, 0.00519282, 0.00513826, 0.00508295, 0.00502694, 0.00497025, 0.00491292, 0.00485498, 0.00479647, 0.00473742, 0.00467787, 0.00461785, 0.0045574, 0.00449654, 0.00443532, 0.00437377, 0.00431192, 0.00424981, 0.00418747, 0.00412493, 0.00406222, 0.00399939, 0.00393646, 0.00387346, 0.00381043, 0.00374739, 0.00368438, 0.00362144, 0.00355858, 0.00349584, 0.00343325, 0.00337083, 0.00330862, 0.00324664, 0.00318492, 0.00312349, 0.00306237, 0.00300158, 0.00294115, 0.00288111, 0.00282147, 0.00276227, 0.00270351, 0.00264523, 0.00258744, 0.00253017, 0.00247342, 0.00241723, 0.0023616, 0.00230656, 0.00225211, 0.00219828, 0.00214509, 0.00209253, 0.00204063, 0.0019894, 0.00193884, 0.00188898, 0.00183982, 0.00179137, 0.00174364, 0.00169664, 0.00165036, 0.00160483, 0.00156005, 0.00151602, 0.00147274, 0.00143022, 0.00138846, 0.00134747, 0.00130725, 0.00126779, 0.0012291, 0.00119118, 0.00115403, 0.00111765, 0.00108203, 0.00104717, 0.00101307, 0.000979735, 0.000947149, 0.000915312, 0.000884219, 0.000853865, 0.000824244, 0.000795351, 0.00076718, 0.000739722, 0.000712972, 0.000686922, 0.000661563, 0.000636889, 0.00061289, 0.000589558, 0.000566883, 0.000544857, 0.000523471, 0.000502713, 0.000482576, 0.000463048, 0.00044412, 0.00042578, 0.000408019, 0.000390827, 0.000374191, 0.000358102, 0.000342549, 0.00032752, 0.000313004, 0.000298991, 0.00028547, 0.000272429, 0.000259857, 0.000247743, 0.000236077, 0.000224846, 0.000214041, 0.000203649, 0.000193661, 0.000184065, 0.000174851, 0.000166009, 0.000157526, 0.000149394, 0.000141602, 0.000134139, 0.000126996, 0.000120162, 0.000113628, 0.000107385, 0.000101422, 9.57298e-05, 9.03003e-05, 8.51238e-05, 8.01916e-05, 7.54949e-05, 7.10253e-05, 6.67744e-05, 6.2734e-05, 5.88963e-05, 5.52533e-05, 5.17974e-05, 4.85213e-05, 4.54177e-05, 4.24795e-05, 3.96998e-05, 3.7072e-05, 3.45896e-05, 3.22462e-05, 3.00358e-05, 2.79522e-05, 2.59899e-05, 2.41432e-05, 2.24066e-05, 2.0775e-05, 1.92433e-05, 1.78065e-05, 1.64601e-05, 1.51993e-05, 1.40199e-05, 1.29177e-05, 1.18885e-05, 1.09284e-05, 1.00337e-05, 9.2008e-06, 8.42624e-06, 7.7067e-06, 7.039e-06, 6.42011e-06, 5.84713e-06, 5.31729e-06, 4.82794e-06, 4.37657e-06, 3.96076e-06, 3.57824e-06, 3.22683e-06, 2.90446e-06, 2.60918e-06, 2.33913e-06, 2.09255e-06, 1.86777e-06, 1.66322e-06, 1.47743e-06, 1.30897e-06, 1.15655e-06, 1.01891e-06, 8.94899e-07, 7.83409e-07, 6.8342e-07, 5.9397e-07, 5.14162e-07, 4.43157e-07, 3.80174e-07, 3.24486e-07, 2.75416e-07, 2.32336e-07, 1.94666e-07, 1.61867e-07, 1.33443e-07, 1.08938e-07, 8.79294e-08, 7.00324e-08, 5.48931e-08, 4.21887e-08, 3.16247e-08, 2.29334e-08, 1.58724e-08, 1.02221e-08,]))
    filter_taps.append(np.array([5.54226e-05, 9.37939e-05, 0.00014319, 0.000205215, 0.000281536, 0.000373868, 0.00048396, 0.000613567, 0.000764438, 0.000938283, 0.00113676, 0.00136143, 0.00161376, 0.00189508, 0.00220654, 0.00254912, 0.00292359, 0.00333047, 0.00377003, 0.00424226, 0.00474686, 0.00528323, 0.00585043, 0.0064472, 0.00707195, 0.00772275, 0.00839736, 0.0090932, 0.00980737, 0.0105367, 0.0112777, 0.0120268, 0.0127799, 0.0135329, 0.0142816, 0.0150215, 0.0157483, 0.0164572, 0.0171439, 0.0178038, 0.0184326, 0.0190259, 0.0195798, 0.0200903, 0.0205538, 0.020967, 0.021327, 0.0216312, 0.0218772, 0.0220633, 0.0221882, 0.0222509, 0.0222509, 0.0221882, 0.0220633, 0.0218772, 0.0216312, 0.021327, 0.020967, 0.0205538, 0.0200903, 0.0195798, 0.0190259, 0.0184326, 0.0178038, 0.0171439, 0.0164572, 0.0157483, 0.0150215, 0.0142816, 0.0135329, 0.0127799, 0.0120268, 0.0112777, 0.0105367, 0.00980737, 0.0090932, 0.00839736, 0.00772275, 0.00707195, 0.0064472, 0.00585043, 0.00528323, 0.00474686, 0.00424226, 0.00377003, 0.00333047, 0.00292359, 0.00254912, 0.00220654, 0.00189508, 0.00161376, 0.00136143, 0.00113676, 0.000938283, 0.000764438, 0.000613567, 0.00048396, 0.000373868, 0.000281536, 0.000205215, 0.00014319, 9.37939e-05, 5.54226e-05,]))
    filter_taps.append(np.array([0.0120348, 0.0160089, 0.0202755, 0.0247398, 0.029294, 0.0338208, 0.0381971, 0.0422994, 0.0460081, 0.0492122, 0.0518141, 0.0537332, 0.0549093, 0.0553056, 0.0549093, 0.0537332, 0.0518141, 0.0492122, 0.0460081, 0.0422994, 0.0381971, 0.0338208, 0.029294, 0.0247398, 0.0202755, 0.0160089, 0.0120348,]))

    beam_phases = np.ones((3, 1, 20), dtype=np.complex64)
    s = np.arange(int(F_s * t), dtype=np.complex64)
    signal = np.zeros(int(F_s * t), dtype=np.complex64)

    for f in freqs:
        signal += np.exp(2j * np.pi * f / F_s * s).astype(np.complex64)

    signals = xp.array(np.repeat(signal[np.newaxis, :], 20, axis=0))
    details = [{"num_range_gates": 75,
                "first_range_off": 4,
                "slice_num": 0,
                "tau_spacing": 2400,
                "lags": np.array([[0, 0],
                                  [26, 27],
                                  [20, 22],
                                  [9, 12],
                                  [22, 26],
                                  [22, 27],
                                  [20, 26],
                                  [20, 27],
                                  [0, 9],
                                  [12, 22],
                                  [9, 20],
                                  [0, 12],
                                  [9, 22],
                                  [12, 26],
                                  [12, 27],
                                  [9, 26],
                                  [9, 27],
                                  [27, 27],
                                  [0, 0]]),
                "lag_phase_offsets": np.zeros(19)}]

    # samp_test = xp.arange(processed_samples.beamformed_samples.shape[-1])[np.newaxis, np.newaxis,:]
    average_time = np.zeros(n)
    for i in range(n):
        a = time.time()
        processed_samples = DSP(signals, F_s, dm_rates, filter_taps, freqs, beam_phases)
        acf = DSP.correlations_from_samples(processed_samples.beamformed_samples,
                                            processed_samples.beamformed_samples,
                                            5e6 / 1500, details)
        if cupy_available:
            x = xp.asnumpy(processed_samples)
        else:
            x = processed_samples

        b = time.time()
        average_time[i] = (b - a) * 1000
        print(f'time: {average_time[i]} ms')

    tsum = np.average(average_time[2::])
    print(f'average time: {tsum} ms')

    # fft_and_plot(x[0][0], F_s)
    # print((b-a) * 1000)
    # fft_and_plot(x[2][0], F_s)


if __name__ == '__main__':
    # import os
    # print('PID: ', os.getpid())
    # time.sleep(60)
    # if cupy_available:
    #    xp.show_config()
    # Actually run the test.
    quick_test(500)

# Testing (500 avg)
# System: GPU RTX 2080, CPU i9-9940x 14 Core 3.30 GHz , RAM 32 GB
# |           Name           |    Trial    |  Time Saved  |  Speed Up  |
# | basic with cupy (GPU):   |  133.328 ms |    0.0 ms    |    1.00    |
# | basic with numpy (CPU):  |  776.517 ms | -643.189 ms  |    5.82    |
# | Accelerators cupy (GPU): |  105.813 ms |   27.515 ms  |    1.26    |
# | Fixed dtype cupy (GPU):  |   16.451 ms |  116.877 ms  |    8.10    |
# | Fix + Accel cupy (GPU):  |   15.203 ms |  118.125 ms  |    8.77    |
# | Fix2 + Accel cupy (GPU): |   13.621 ms |  119.707 ms  |    9.79    |
