"""
Copyright SuperDARN Canada 2020
Original Auth: Keith Kotyk
"""
import sys
import time
import threading
import numpy as np
import posix_ipc as ipc
from multiprocessing import shared_memory
import mmap
import math
import copy
import pickle
from functools import reduce

try:
    import cupy as cp
except:
    cupy_available = False
else:
    cupy_available = True

if cupy_available:
    import cupy as xp
else:
    import numpy as xp

if __debug__:
    from debug.borealis.utils.protobuf.rxsamplesmetadata_pb2 import RxSamplesMetadata
else:
    from release.borealis.utils.protobuf.rxsamplesmetadata_pb2 import RxSamplesMetadata

from utils.message_formats import ProcessedSequenceMessage, DebugDataStage, OutputDataset
import utils.options.signal_processing_options as spo
from utils import socket_operations as so
import utils.shared_macros as sm

pprint = sm.MODULE_PRINT("rx signal processing", "magenta")


def overlap_save_convolution(input_samples, filters, block_size):
    """
    Computes the linear convolution of input_samples with filters using the overlap-save method.
    This method uses circular convolution to do more efficient computations on smaller
    block sizes, then combines the results.
    See https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method for more details.

    :param      input_samples:  The input raw rf samples for each antenna.
    :type       input_samples:  ndarray [num_antennas, num_samples]
    :param      filters:        The filter(s).
    :type       filters:        ndarray [num_slices, num_taps]
    :param      block_size:     The number of samples to use for FFT.
    :type       block_size:     int
    """
    num_slices, num_taps = filters.shape

    # See https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method#Efficiency_considerations
    overlap = num_taps - 1
    step_size = block_size - overlap

    filters_fft = xp.fft.fft(filters, n=block_size, axis=-1)

    dims = input_samples.shape
    num_antennas = dims[-2]
    num_samps = dims[-1]
    num_blocks = int(np.ceil((num_samps - block_size + overlap) / step_size))
    if len(dims) > 2:
        # Enter this block for lowpass filtering, as we have already bandpass-filtered which adds
        # the extra 'slice' dimension. All lowpass filters are identical, so we can reduce the filter fft
        num_slices = dims[0]
        fft_dims = [num_slices, num_antennas, num_blocks, block_size]
        filters_fft = filters_fft[0, ...]
        # arg1:   [num_slices, num_antennas, num_blocks, block_size]
        # arg2:   [num_taps + num_samples - 1]
        # output: [num_slices, num_antennas, num_blocks, block_size]
        einsum_str = 'ijkl,l->ijkl'
        last_block_einsum = 'ijl,l->ijl'  # no num_blocks parameter for last block
    else:
        # This block is for bandpass filtering, so the input samples are all the same (wideband samples)
        fft_dims = [num_antennas, num_blocks, block_size]
        # arg1:   [num_antennas, num_blocks, block_size]
        # arg2:   [num_slices, block_size]
        # output: [num_slices, num_antennas, num_blocks, block_size]
        einsum_str = 'ijk,lk->lijk'
        last_block_einsum = 'ik,lk->lik'

    output_samples = xp.zeros((num_slices, num_antennas, num_samps), dtype=xp.complex64)

    # array to store input_samples in for each FFT
    x_k = xp.zeros(fft_dims, dtype=xp.complex64)

    # CUDA optimizes multiple FFTs, so we get the samples ready for each FFT in one array.
    for i in range(num_blocks):
        if i == 0:
            x_k[..., i, overlap:] = input_samples[..., 0:block_size - overlap]
        else:
            x_k[..., i, :] = input_samples[..., i * step_size - overlap:i * step_size - overlap + block_size]

    # Now, take FFT of the samples, multiply by FFT of filter, and take IFFT to get back in time domain
    samples_freq_domain = xp.fft.fft(x_k, axis=-1)
    filtered_samps_freq_domain = xp.einsum(einsum_str, samples_freq_domain, filters_fft)
    filtered_samps_time_domain = xp.fft.ifft(filtered_samps_freq_domain, axis=-1)

    # Grab the correct samples based on the overlap-save algorithm
    for i in range(num_blocks):
        output_samples[..., i * step_size:(i + 1) * step_size] = filtered_samps_time_domain[..., i, overlap:]

    # Now we have to calculate for the last few samples.
    # Last block will be smaller, so we zero-pad at the end.
    num_remaining = num_samps - (num_blocks * step_size - overlap)
    fft_dims[-1] = num_remaining + overlap - 1
    fft_dims.pop(-2)  # Get rid of num_blocks dimension - will be 1 for this last block
    x_k = xp.zeros(fft_dims, dtype=input_samples.dtype)
    x_k[..., 0:num_remaining] = input_samples[..., -num_remaining:]
    samples_freq_domain = xp.fft.fft(x_k)
    filters_freq_domain = xp.fft.fft(filters, n=num_remaining + overlap - 1)
    if len(dims) > 2:
        # This means that we are lowpass filtering, and all rows are identical so just take the first
        filters_freq_domain = filters_freq_domain[0, ...]
    filtered_freq_domain = xp.einsum(last_block_einsum, samples_freq_domain, filters_freq_domain)
    filtered_time_domain = xp.fft.ifft(filtered_freq_domain, axis=-1)
    output_samples[..., -num_remaining + overlap:] = filtered_time_domain[..., overlap:num_remaining]

    return output_samples


def bandpass_decimate(input_samples, bp_filters, mixing_freqs, dm_rate, rx_rate):
    """
    Apply a Frerking bandpass filter to the input samples. Several different frequencies can
    be centered on at simultaneously. The overlap-save method is used for computational efficiency.
    See https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method for details.

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
    bp_filters = xp.array(bp_filters)
    num_slices, num_taps = bp_filters.shape

    # See https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method#Efficiency_considerations for choice of 8192
    # filtered_samples = self.overlap_save_convolution(input_samples, np.flip(bp_filters, axis=-1), 8192)
    filtered_samples = overlap_save_convolution(input_samples, bp_filters, 8192)

    # To match with windowed_view() and convolution, take only samples where the convolution multiplication
    # wouldn't have required zero-padding of the input_samples.
    filtered_samples = filtered_samples[..., num_taps-1::dm_rate]

    # Multiply with phase correction, as per Frerking method.
    # See https://borealis.readthedocs.io/en/latest/frerking.html
    ph = xp.arange(filtered_samples.shape[-1], dtype=np.float64)[xp.newaxis, :]
    freqs = xp.array(mixing_freqs)[:, xp.newaxis]

    # [1, num_output_samples]
    # [num_slices, 1]
    ph = xp.fmod(ph * 2.0 * xp.pi * freqs/rx_rate * dm_rate, 2.0 * xp.pi)
    ph = xp.exp(1j * ph.astype(xp.float32))

    # [num_slices, num_antennas, num_output_samples]
    # [num_slices, 1, num_output_samples]
    corrected = filtered_samples * ph[:, xp.newaxis, :]

    return corrected


def lowpass_decimate(input_samples, lp_filter, dm_rate, block_size):
    """
    Apply a lowpass filter to the baseband input samples. Downsampling is done in parallel via a
    strided window view of the input samples.

    :param      input_samples:  Baseband input samples
    :type       input_samples:  ndarray [num_slices, num_antennas, num_samples]
    :param      lp_filter:      Lowpass filter taps
    :type       lp_filter:      ndarray [1, num_taps]
    :param      dm_rate:        The decimation rate of this stage.
    :type       dm_rate:        int

    """
    lp_filter = xp.array(lp_filter)

    filtered_samples = overlap_save_convolution(input_samples, lp_filter, block_size)

    # To match with windowed_view() and convolution, take only samples where the convolution multiplication
    # wouldn't have required zero-padding of the input_samples.
    filtered_samples = filtered_samples[..., lp_filter.shape[-1] - 1:input_samples.shape[-1]:dm_rate]

    return filtered_samples


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

        self.filter_outputs.append(
            bandpass_decimate(input_samples, self.filters[0], mixing_freqs, dm_rates[0], rx_rate))

        for i in range(1, len(self.filters)):
            # See https://en.wikipedia.org/wiki/Overlap%E2%80%93save_method#Efficiency_considerations
            if i == 1:
                block_size = 1024
            else:
                block_size = 128
            self.filter_outputs.append(
                lowpass_decimate(self.filter_outputs[i - 1], self.filters[i], dm_rates[i], block_size))

        # Create shared memory for antennas_iq data
        antennas_iq_samples = self.filter_outputs[-1]
        ant_shm = shared_memory.SharedMemory(create=True, size=antennas_iq_samples.nbytes)
        self.antennas_iq_samples = np.ndarray(antennas_iq_samples.shape, dtype=np.complex64, buffer=ant_shm.buf)

        # Move the antennas_iq samples to the CPU for beamforming
        if cupy_available:
            self.antennas_iq_samples[...] = xp.asnumpy(antennas_iq_samples)[...]
        else:
            self.antennas_iq_samples[...] = antennas_iq_samples[...]
        self.shared_mem['antennas_iq'] = ant_shm

        self.beamform_samples(self.antennas_iq_samples, beam_phases)

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

    def beamform_samples(self, filtered_samples, beam_phases):
        """
        Beamform the filtered samples for multiple beams simultaneously.

        :param      filtered_samples:  The filtered input samples.
        :type       filtered_samples:  ndarray [num_slices, num_antennas, num_samples]
        :param      beam_phases:       The beam phases used to phase each antenna's samples before
                                       combining.
        :type       beam_phases:       ndarray [num_slices, num_beams, num_antennas]

        """
        # [num_slices, num_beams, num_antennas]
        beam_phases = np.array(beam_phases)

        # [num_slices, num_beams, num_samples]
        final_shape = (filtered_samples.shape[0], beam_phases.shape[1], filtered_samples.shape[2])
        final_size = np.dtype(np.complex64).itemsize * reduce(lambda a, b: a * b, final_shape)
        bf_shm = shared_memory.SharedMemory(create=True, size=final_size)
        self.beamformed_samples = np.ndarray(final_shape, dtype=np.complex64, buffer=bf_shm.buf)
        self.beamformed_samples[...] = np.einsum('ijk,ilj->ilk', filtered_samples, beam_phases)

        self.shared_mem['bfiq'] = bf_shm

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


def fill_datawrite_message(processed_data, slice_details, data_outputs):
    """
    Fills the datawrite message with processed data.

    :param      processed_data:  The processed data message
    :type       processed_data:  ProcessedSequenceMessage
    :param      slice_details:   The details for each slice that was processed.
    :type       slice_details:   list
    :param      data_outputs:    The processed data outputs.
    :type       data_outputs:    dict
    """

    for sd in slice_details:
        output_dataset = OutputDataset(sd['slice_id'], sd['num_beams'], sd['num_range_gates'], sd['num_lags'])

        def add_array(ndarray):
            """
            Creates shared memory and stores ndarray in it.

            :param ndarray: numpy.ndarray
            :return name: String of the shared memory name.
            """
            if ndarray.size != 0:
                shm = shared_memory.SharedMemory(create=True, size=ndarray.nbytes)
                shared_array = np.ndarray(ndarray.shape, dtype=ndarray.dtype, buffer=shm.buf)
                shared_array[...] = ndarray[...]
                name = shm.name
                # This closes the current SharedMemory instance, but the memory isn't free until data_write unlinks it.
                shm.close()
                return name

        main_corrs = data_outputs['main_corrs'][sd['slice_num']]
        output_dataset.main_acf_shm = add_array(main_corrs)

        intf_available = True
        try:
            intf_corrs = data_outputs['intf_corrs'][sd['slice_num']]
            cross_corrs = data_outputs['cross_corrs'][sd['slice_num']]
        except KeyError as e:
            # No interferometer data
            intf_available = False

        if intf_available:
            output_dataset.intf_acf_shm = add_array(intf_corrs)
            output_dataset.xcf_shm = add_array(cross_corrs)

        processed_data.add_output_dataset(output_dataset)


def main():
    sig_options = spo.SignalProcessingOptions()

    sockets = so.create_sockets([sig_options.dsp_radctrl_identity,
                                 sig_options.dsp_driver_identity], sig_options.router_address)

    dsp_to_radar_control = sockets[0]
    dsp_to_driver = sockets[1]

    ringbuffer = None

    total_antennas = len(sig_options.main_antennas) + len(sig_options.intf_antennas)

    dm_rates = []
    dm_scheme_taps = []

    extra_samples = 0
    total_dm_rate = 0

    threads = []
    first_time = True
    while True:

        reply = so.recv_bytes(dsp_to_radar_control, sig_options.radctrl_dsp_identity, pprint)

        sqn_meta_message = pickle.loads(reply)

        rx_rate = np.float64(sqn_meta_message.rx_rate)
        output_sample_rate = np.float64(sqn_meta_message.output_sample_rate)
        first_rx_sample_off = sqn_meta_message.offset_to_first_rx_sample
        rx_center_freq = sqn_meta_message.rx_ctr_freq

        processed_data = ProcessedSequenceMessage()

        processed_data.sequence_num = sqn_meta_message.sequence_num
        processed_data.rx_sample_rate = rx_rate
        processed_data.output_sample_rate = output_sample_rate

        mixing_freqs = []
        main_beam_angles = []
        intf_beam_angles = []

        # Parse out details and force the data type so that Cupy can optimize with standardized
        # data types.
        slice_details = []
        for i, chan in enumerate(sqn_meta_message.rx_channels):
            detail = {}

            # This is the negative of what you would normally expect (i.e. -1 * offset of rxfreq from center freq)
            # because the filter taps do not get flipped when convolving. I.e. we do the cross-correlation instead of
            # convolution, to save some computational complexity from flipping the filter sequence.
            # It works out to the same result.
            mixing_freqs.append(rx_center_freq - chan.rx_freq)

            detail['slice_id'] = chan.slice_id
            detail['slice_num'] = i
            detail['first_range'] = np.float32(chan.first_range)
            detail['range_sep'] = np.float32(chan.range_sep)
            detail['tau_spacing'] = np.uint32(chan.tau_spacing)
            detail['num_range_gates'] = np.uint32(chan.num_ranges)
            detail['first_range_off'] = np.uint32(chan.first_range / chan.range_sep)
            lag_phase_offsets = []

            lags = []
            for lag in chan.lags:
                lags.append([lag.pulse_1, lag.pulse_2])
                lag_phase_offsets.append(lag.phase_offset_real + 1j * lag.phase_offset_imag)

            detail['lag_phase_offsets'] = np.array(lag_phase_offsets, dtype=np.complex64)

            detail['lags'] = np.array(lags, dtype=np.uint32)
            detail['num_lags'] = len(lags)

            main_beams = chan.beam_phases[:, :len(sig_options.main_antennas)]
            intf_beams = chan.beam_phases[:, len(sig_options.main_antennas):]

            detail['num_beams'] = main_beams.shape[0]

            slice_details.append(detail)
            main_beam_angles.append(main_beams)
            intf_beam_angles.append(intf_beams)

        # Different slices can have a different amount of beams used. Slices that use fewer beams
        # than the max number of beams are padded with zeros so that matrix calculations can be
        # used. The extra beams that are processed will be not be parsed for data writing.
        max_num_beams = max([x.shape[0] for x in main_beam_angles])

        def pad_beams(angles, ant_count):
            for x in angles:
                if x.shape[0] < max_num_beams:
                    temp = np.zeros_like((max_num_beams, ant_count), x.dtype)
                    temp[:x.shape[0], :] = x
                    x = temp    # Reassign to the new larger array with zero-padded beams

        pad_beams(main_beam_angles, len(sig_options.main_antennas))
        pad_beams(intf_beam_angles, len(sig_options.intf_antennas))

        main_beam_angles = np.array(main_beam_angles, dtype=np.complex64)
        intf_beam_angles = np.array(intf_beam_angles, dtype=np.complex64)
        mixing_freqs = np.array(mixing_freqs, dtype=np.float64)

        # Get meta from driver
        message = "Need data to process"
        so.send_data(dsp_to_driver, sig_options.driver_dsp_identity, message)
        reply = so.recv_bytes(dsp_to_driver, sig_options.driver_dsp_identity, pprint)

        rx_metadata = RxSamplesMetadata()
        rx_metadata.ParseFromString(reply)

        if sqn_meta_message.sequence_num != rx_metadata.sequence_num:
            pprint(sm.COLOR('red', "ERROR: Packets from driver and radctrl don't match"))
            err = "sqn_meta_message seq num {}, rx_metadata seq num {}".format(sqn_meta_message.sequence_num,
                                                                        rx_metadata.sequence_num)
            pprint(sm.COLOR('red', err))
            sys.exit(-1)

        # First time configuration
        if first_time:
            shm = ipc.SharedMemory(sig_options.ringbuffer_name)
            mapped_mem = mmap.mmap(shm.fd, shm.size)
            ringbuffer = np.frombuffer(mapped_mem, dtype=np.complex64).reshape(total_antennas, -1)

            if cupy_available:
                cp.cuda.runtime.hostRegister(ringbuffer.ctypes.data, ringbuffer.size, 0)

            dm_msg = "Decimation rates: "
            taps_msg = "Number of filter taps per stage: "
            for stage in sqn_meta_message.decimation_stages:
                dm_rates.append(stage.dm_rate)
                dm_scheme_taps.append(np.array(stage.filter_taps, dtype=np.complex64))

                dm_msg += str(stage.dm_rate) + " "
                taps_msg += str(len(stage.filter_taps)) + " "

            dm_rates = np.array(dm_rates, dtype=np.uint32)
            pprint(dm_msg)
            pprint(taps_msg)

            for dm, taps in zip(reversed(dm_rates), reversed(dm_scheme_taps)):
                extra_samples = (extra_samples * dm) + len(taps) // 2

            total_dm_rate = np.prod(dm_rates)

            first_time = False

        # Calculate where in the ringbuffer the samples are located.
        samples_needed = rx_metadata.numberofreceivesamples + 2 * extra_samples
        samples_needed = int(math.ceil(float(samples_needed) / float(total_dm_rate)) * total_dm_rate)

        sample_time_diff = rx_metadata.sequence_start_time - rx_metadata.initialization_time
        sample_in_time = (sample_time_diff * rx_rate) + first_rx_sample_off - extra_samples

        start_sample = int(math.fmod(sample_in_time, ringbuffer.shape[1]))
        end_sample = start_sample + samples_needed

        processed_data.initialization_time = rx_metadata.initialization_time
        processed_data.sequence_start_time = rx_metadata.sequence_start_time
        processed_data.gps_to_system_time_diff = rx_metadata.gps_to_system_time_diff
        processed_data.agc_status_bank_h = rx_metadata.agc_status_bank_h
        processed_data.lp_status_bank_h = rx_metadata.lp_status_bank_h
        processed_data.agc_status_bank_l = rx_metadata.agc_status_bank_l
        processed_data.lp_status_bank_l = rx_metadata.lp_status_bank_l
        processed_data.gps_locked = rx_metadata.gps_locked

        # This work is done in a thread
        def sequence_worker(**kwargs):
            sequence_num = kwargs['sequence_num']
            main_beam_angles = kwargs['main_beam_angles']
            intf_beam_angles = kwargs['intf_beam_angles']
            mixing_freqs = kwargs['mixing_freqs']
            slice_details = kwargs['slice_details']
            start_sample = kwargs['start_sample']
            end_sample = kwargs['end_sample']
            processed_data = kwargs['processed_data']

            pprint(sm.COLOR('green', "Processing #{}".format(sequence_num)))
            pprint("Mixing freqs for #{}: {}".format(sequence_num, mixing_freqs))
            pprint("Main beams shape for #{}: {}".format(sequence_num, main_beam_angles.shape))
            pprint("Intf beams shape for #{}: {}".format(sequence_num, intf_beam_angles.shape))
            if cupy_available:
                cp.cuda.runtime.setDevice(0)

            seq_begin_iden = sig_options.dspbegin_brian_identity + str(sequence_num)
            seq_end_iden = sig_options.dspend_brian_identity + str(sequence_num)
            dw_iden = sig_options.dsp_dw_identity + str(sequence_num)
            gpu_socks = so.create_sockets([seq_begin_iden, seq_end_iden, dw_iden],
                                          sig_options.router_address)

            dspbegin_to_brian = gpu_socks[0]
            dspend_to_brian = gpu_socks[1]
            dsp_to_dw = gpu_socks[2]

            start = time.time()

            indices = np.arange(start_sample, start_sample + samples_needed)

            # x.take makes a copy of the array. We want to avoid making a copy using Cupy so that
            # data is moved directly from the ring buffer to the GPU. Simple indexing creates a view
            # of existing data without making a copy.
            if cupy_available:
                if end_sample > ringbuffer.shape[1]:
                    piece1 = ringbuffer[:, start_sample:]
                    piece2 = ringbuffer[:, :end_sample - ringbuffer.shape[1]]

                    tmp1 = cp.array(piece1)
                    tmp2 = cp.array(piece2)

                    sequence_samples = cp.concatenate((tmp1, tmp2), axis=1)
                else:
                    sequence_samples = cp.array(ringbuffer[:, start_sample:end_sample])

            else:
                sequence_samples = ringbuffer.take(indices, axis=1, mode='wrap')

            copy_end = time.time()
            time_diff = (copy_end - start) * 1000
            pprint("Time to copy samples for #{}: {}ms".format(sequence_num, time_diff))
            reply_packet = {}
            reply_packet['sequence_num'] = sequence_num
            msg = pickle.dumps(reply_packet, protocol=pickle.HIGHEST_PROTOCOL)

            so.recv_bytes(dspbegin_to_brian, sig_options.brian_dspbegin_identity, pprint)
            so.send_bytes(dspbegin_to_brian, sig_options.brian_dspbegin_identity, msg)

            # Process main samples
            main_sequence_samples = sequence_samples[:len(sig_options.main_antennas), :]
            pprint("Main buffer shape: {}".format(main_sequence_samples.shape))
            processed_main_samples = DSP(main_sequence_samples, rx_rate, dm_rates, dm_scheme_taps, mixing_freqs,
                                         main_beam_angles)
            main_corrs = DSP.correlations_from_samples(processed_main_samples.beamformed_samples,
                                                       processed_main_samples.beamformed_samples,
                                                       output_sample_rate, slice_details)

            # If interferometer is used, process those samples too.
            if sig_options.intf_antenna_count > 0:
                intf_sequence_samples = sequence_samples[len(sig_options.main_antennas):, :]
                pprint("Intf buffer shape: {}".format(intf_sequence_samples.shape))
                processed_intf_samples = DSP(intf_sequence_samples, rx_rate, dm_rates,
                                             dm_scheme_taps, mixing_freqs, intf_beam_angles)

                intf_corrs = DSP.correlations_from_samples(processed_intf_samples.beamformed_samples,
                                                           processed_intf_samples.beamformed_samples,
                                                           output_sample_rate, slice_details)
                cross_corrs = DSP.correlations_from_samples(processed_intf_samples.beamformed_samples,
                                                            processed_main_samples.beamformed_samples,
                                                            output_sample_rate, slice_details)
            end = time.time()

            time_diff = (end - copy_end) * 1000
            reply_packet['kerneltime'] = time_diff
            msg = pickle.dumps(reply_packet, protocol=pickle.HIGHEST_PROTOCOL)

            pprint("Time to decimate, beamform and correlate for #{}: {}ms".format(sequence_num,
                                                                                   time_diff))

            time_diff = (end - start) * 1000
            pprint("Total time for #{}: {}ms".format(sequence_num, time_diff))

            so.recv_bytes(dspend_to_brian, sig_options.brian_dspend_identity, pprint)
            so.send_bytes(dspend_to_brian, sig_options.brian_dspend_identity, msg)

            # Extract outputs from processing into groups that will be put into message fields.
            start = time.time()
            data_outputs = {}

            def debug_data_in_shm(holder, data_array, array_name):
                """
                Adds an array of antennas data (filter outputs or antennas_iq) into a dictionary
                for later entry in a processed data message.

                :param holder: Dictionary to store the shared memory parameters.
                :param data_array: cp.ndarray or np.ndarray of the data.
                :param array_name: 'main' or 'intf'. String
                """
                shm = shared_memory.SharedMemory(create=True, size=data_array.nbytes)
                data = np.ndarray(data_array.shape, dtype=np.complex64, buffer=shm.buf)
                if cupy_available:
                    data[...] = cp.asnumpy(data_array)
                else:
                    data[...] = data_array

                if array_name == 'main':
                    holder.main_shm = shm.name
                elif array_name == 'intf':
                    holder.intf_shm = shm.name
                else:
                    raise RuntimeError("Error: unknown debug data array {}".format(array_name))

                holder.num_samps = data_array.shape[-1]
                shm.close()
            
            # Add the filter stage data if in debug mode
            if __debug__:
                for i, main_data in enumerate(processed_main_samples.filter_outputs[:-1]):
                    stage = DebugDataStage('stage_{}'.format(i))
                    debug_data_in_shm(stage, main_data, 'main')

                    if sig_options.intf_antenna_count > 0:
                        intf_data = processed_intf_samples.filter_outputs[i]
                        debug_data_in_shm(stage, intf_data, 'intf')

                    processed_data.add_debug_data(stage)

            # Add antennas_iq data
            stage = DebugDataStage()
            stage.stage_name = 'antennas'
            main_shm = processed_main_samples.shared_mem['antennas_iq']
            stage.main_shm = main_shm.name
            stage.num_samps = processed_main_samples.antennas_iq_samples.shape[-1]
            main_shm.close()
            if sig_options.intf_antenna_count > 0:
                intf_shm = processed_intf_samples.shared_mem['antennas_iq']
                stage.intf_shm = intf_shm.name
                intf_shm.close()
            processed_data.add_debug_data(stage)

            done_filling_debug = time.time()
            time_filling_debug = (done_filling_debug - start) * 1000
            pprint("Time to put antennas data in message for #{}: {}ms".format(sequence_num, time_filling_debug))

            # Add rawrf data
            if __debug__:
                # np.complex64 in bytes * num_antennas * num_samps
                rawrf_size = np.dtype(np.complex64).itemsize * ringbuffer.shape[0] * indices.shape[-1]
                rawrf_shm = shared_memory.SharedMemory(create=True, size=rawrf_size)
                rawrf_array = np.ndarray((ringbuffer.shape[0], indices.shape[-1]), dtype=np.complex64, buffer=rawrf_shm.buf)
                rawrf_array[...] = ringbuffer.take(indices, axis=1, mode='wrap')
                processed_data.rawrf_shm = rawrf_shm.name
                processed_data.rawrf_num_samps = indices.shape[-1]
                rawrf_shm.close()

                done_filling_rawrf = time.time()
                time_filling_rawrf = (done_filling_rawrf - done_filling_debug) * 1000
                pprint("Time to put rawrf in shared memory for #{}: {}ms".format(sequence_num, time_filling_rawrf))

            start_filling_bfiq_time = time.time()

            # Add bfiq and correlations data
            beamformed_m = processed_main_samples.beamformed_samples
            processed_data.bfiq_main_shm = processed_main_samples.shared_mem['bfiq'].name
            processed_data.max_num_beams = beamformed_m.shape[1]    # [num_slices, num_beams, num_samps]
            processed_data.num_samps = beamformed_m.shape[-1]
            processed_main_samples.shared_mem['bfiq'].close()

            data_outputs['main_corrs'] = main_corrs

            if sig_options.intf_antenna_count > 0:
                data_outputs['cross_corrs'] = cross_corrs
                data_outputs['intf_corrs'] = intf_corrs
                processed_data.bfiq_intf_shm = processed_intf_samples.shared_mem['bfiq'].name
                processed_intf_samples.shared_mem['bfiq'].close()

            # Fill message with the slice-specific fields
            fill_datawrite_message(processed_data, slice_details, data_outputs)

            sqn_message = pickle.dumps(processed_data, protocol=pickle.HIGHEST_PROTOCOL)

            end = time.time()
            time_for_bfiq_acf = (end - start_filling_bfiq_time) * 1000
            pprint("Time to add bfiq and acfs to processeddata message for #{}: {}ms".format(sequence_num, time_for_bfiq_acf))

            time_diff = (end - start) * 1000
            pprint("Time to serialize and send processed data for #{}: {}ms".format(sequence_num,
                                                                                    time_diff))
            so.send_bytes(dsp_to_dw, sig_options.dw_dsp_identity, sqn_message)

        args = {"sequence_num": copy.deepcopy(sqn_meta_message.sequence_num),
                "main_beam_angles": copy.deepcopy(main_beam_angles),
                "intf_beam_angles": copy.deepcopy(intf_beam_angles),
                "mixing_freqs": copy.deepcopy(mixing_freqs),
                "slice_details": copy.deepcopy(slice_details),
                "start_sample": copy.deepcopy(start_sample),
                "end_sample": copy.deepcopy(end_sample),
                "processed_data": copy.deepcopy(processed_data)}

        seq_thread = threading.Thread(target=sequence_worker, kwargs=args)
        seq_thread.daemon = True
        seq_thread.start()

        threads.append(seq_thread)

        if len(threads) > 1:
            thread = threads.pop(0)
            thread.join()


if __name__ == "__main__":
    main()
