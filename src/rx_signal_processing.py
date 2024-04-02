"""
    rx_signal_processing
    ~~~~~~~~~~~~~~~~~~~~~
    This process handles the digital signal processing side of Borealis

    :copyright: 2020 SuperDARN Canada
    :author: Keith Kotyk
"""
import os
import sys
import time
import threading
import numpy as np
from multiprocessing import shared_memory
import mmap
import math
import copy
import pickle
from functools import reduce

try:
    import cupy as xp
except ImportError:
    cupy_available = False
    import numpy as xp
else:
    cupy_available = True

# Import some additional modules here. This is avoided on import of this module to facilitate testing of the DSP class.
if __name__ == '__main__':
    import posix_ipc as ipc

    sys.path.append(os.environ['BOREALISPATH'])

    if __debug__:
        from build.debug.src.utils.protobuf.rxsamplesmetadata_pb2 import RxSamplesMetadata
    else:
        from build.release.src.utils.protobuf.rxsamplesmetadata_pb2 import RxSamplesMetadata

    from utils.message_formats import ProcessedSequenceMessage, DebugDataStage, OutputDataset


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
    return xp.lib.stride_tricks.as_strided(ndarray, shape=new_shape, strides=new_strides)


class DSP:
    """
    This class performs the DSP functions of Borealis. Filtering and downsampling are specified by lists of filter
    taps and downsampling factors, which must have the same length. The first filter stage is always a bandpass filter,
    specified by a list of mixing frequencies which can be selected simultaneously. All subsequent filter stages are
    lowpass.

    Beamforming can also be conducted on the filtered and downsampled data, requiring a list of complex antenna
    amplitudes for the beamforming operation. Multiple beams can be formed simultaneously.

    This class also supports extraction of lag profiles from multi-pulse sequences.
    """

    def __init__(self, rx_rate, filter_taps, mixing_freqs, dm_rates, shared_mem=True):
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
        """
        self.filters = None
        self.filter_outputs = []
        self.beamformed_samples = None
        self.antennas_iq_samples = None
        self.shared_mem = {}
        self.use_shared_mem = shared_mem
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
            self.apply_bandpass_decimate(input_samples, self.filters[0], self.mixing_freqs, self.dm_rates[0],
                                         self.rx_rate)
        )
        for i in range(1, len(self.filters)):
            self.filter_outputs.append(
                self.apply_lowpass_decimate(self.filter_outputs[i - 1], self.filters[i], self.dm_rates[i])
            )

    def move_filter_results(self):
        """
        Move the final results of filtering (antennas_iq data) to the CPU, optionally in SharedMemory if
        specified for this instance.
        """
        # Create an array on the CPU for antennas_iq data
        antennas_iq_samples = self.filter_outputs[-1]
        if self.use_shared_mem:
            ant_shm = shared_memory.SharedMemory(create=True, size=antennas_iq_samples.nbytes)
            buffer = ant_shm.buf
            self.shared_mem['antennas_iq'] = ant_shm
        else:
            buffer = None
        self.antennas_iq_samples = np.ndarray(antennas_iq_samples.shape, dtype=np.complex64, buffer=buffer)
        
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
        final_shape = (self.antennas_iq_samples.shape[0], beam_phases.shape[1], self.antennas_iq_samples.shape[2])
        final_size = np.dtype(np.complex64).itemsize * reduce(lambda a, b: a * b, final_shape)

        if self.use_shared_mem:
            bf_shm = shared_memory.SharedMemory(create=True, size=final_size)
            buffer = bf_shm.buf
            self.shared_mem['bfiq'] = bf_shm
        else:
            buffer = None
        self.beamformed_samples = np.ndarray(final_shape, dtype=np.complex64, buffer=buffer)

        # Apply beamforming
        self.beamformed_samples[...] = self.beamform_samples(self.antennas_iq_samples, beam_phases)

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
            bandpass[idx, :] = filter_taps[0] * np.exp(s * 2j * np.pi * (-1.0 * f) / rx_rate)
        filters.append(xp.array(bandpass, dtype=xp.complex64))

        for t in filter_taps[1:]:
            filters.append(xp.array(t[np.newaxis, :], dtype=xp.float32))

        return filters

    @staticmethod
    def apply_bandpass_decimate(input_samples, bp_filters, mixing_freqs, dm_rate, rx_rate):
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
        input_samples = windowed_view(input_samples, bp_filters.shape[-1], dm_rate)

        # [num_slices, num_taps]
        # [num_antennas, num_output_samples, num_taps]
        filtered = xp.einsum('ij,klj->ikl', bp_filters, input_samples)

        # Apply the phase correction for the Frerking method.
        ph = xp.arange(filtered.shape[-1], dtype=np.float32)[xp.newaxis, :]
        freqs = xp.array(mixing_freqs)[:, xp.newaxis]

        # [1, num_output_samples]
        # [num_slices, 1]
        # ph: [num_slices, num_output_samples]
        ph = xp.fmod(ph * 2.0 * xp.pi * freqs / rx_rate * dm_rate, 2.0 * xp.pi)
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
        input_samples = windowed_view(input_samples, lp_filter.shape[-1], dm_rate)

        # [1, num_taps]
        # [num_slices, num_antennas, num_output_samples, num_taps]
        # filtered: [num_slices, num_antennas, num_output_samples]
        filtered = xp.einsum('ij,klmj->klm', lp_filter, input_samples)

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
        return np.einsum('ijk,ilj->ilk', filtered_samples, beam_phases)

    @staticmethod
    def correlations_from_samples(beamformed_samples_1, beamformed_samples_2, output_sample_rate, slice_index_details):
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
            if slice_info['lags'].size == 0:
                values.append(np.array([]))
                continue

            range_off = np.arange(slice_info['num_range_gates'], dtype=np.int32) + slice_info['first_range_off']
            tau_in_samples = slice_info['tau_spacing'] * 1e-6 * output_sample_rate
            lag_pulses_as_samples = np.array(slice_info['lags'], np.int32) * np.int32(tau_in_samples)

            # [num_range_gates, 1, 1]
            # [1, num_lags, 2]
            samples_for_all_range_lags = (range_off[..., np.newaxis, np.newaxis] +
                                          lag_pulses_as_samples[np.newaxis, :, :])

            # [num_range_gates, num_lags, 2]
            row = samples_for_all_range_lags[..., 1].astype(np.int32)

            # [num_range_gates, num_lags, 2]
            col = samples_for_all_range_lags[..., 0].astype(np.int32)

            values_for_slice = np.empty((beamformed_samples_1.shape[1], row.shape[0], row.shape[1]), dtype=np.complex64)

            for lag in range(row.shape[1]):
                values_for_slice[:, :, lag] = np.einsum('ij,ij->ji',
                                                        beamformed_samples_1[s, :, row[:, lag]],
                                                        beamformed_samples_2[s, :, col[:, lag]].conj())

            # [num_beams, num_range_gates, num_lags]
            values_for_slice = np.einsum('ijk,k->ijk', values_for_slice, slice_info['lag_phase_offsets'])

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

            :param      ndarray: array to be created
            :type       ndarray: numpy.ndarray
            :returns:   The shared memory name.
            :rtype:     str
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
    options = Options()

    sockets = so.create_sockets([options.dsp_to_radctrl_identity,
                                 options.dsp_to_driver_identity], options.router_address)

    dsp_to_radar_control = sockets[0]
    dsp_to_driver = sockets[1]

    ringbuffer = None

    total_antennas = len(options.main_antennas) + len(options.intf_antennas)

    dm_rates = []
    dm_scheme_taps = []

    extra_samples = 0
    total_dm_rate = 0

    threads = []
    first_time = True
    while True:

        reply = so.recv_bytes(dsp_to_radar_control, options.radctrl_to_dsp_identity, log)

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

            mixing_freqs.append(chan.rx_freq - rx_center_freq)

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

            main_beams = chan.beam_phases[:, :len(options.main_antennas)]
            intf_beams = chan.beam_phases[:, len(options.main_antennas):]

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

        pad_beams(main_beam_angles, len(options.main_antennas))
        pad_beams(intf_beam_angles, len(options.intf_antennas))

        main_beam_angles = np.array(main_beam_angles, dtype=np.complex64)
        intf_beam_angles = np.array(intf_beam_angles, dtype=np.complex64)
        mixing_freqs = np.array(mixing_freqs, dtype=np.float64)

        # Get meta from driver
        message = "Need data to process"
        so.send_data(dsp_to_driver, options.driver_to_dsp_identity, message)
        reply = so.recv_bytes(dsp_to_driver, options.driver_to_dsp_identity, log)

        rx_metadata = RxSamplesMetadata()
        rx_metadata.ParseFromString(reply)

        if sqn_meta_message.sequence_num != rx_metadata.sequence_num:
            log.error("driver packets != radctrl packets",
                      sqn_meta_sqn_num=sqn_meta_message.sequence_num,
                      rx_meta_sqn_num=rx_metadata.sequence_num)
            sys.exit(-1)

        # First time configuration
        if first_time:
            shm = ipc.SharedMemory(options.ringbuffer_name)
            mapped_mem = mmap.mmap(shm.fd, shm.size)
            ringbuffer = np.frombuffer(mapped_mem, dtype=np.complex64).reshape(total_antennas, -1)

            if cupy_available:
                xp.cuda.runtime.hostRegister(ringbuffer.ctypes.data, ringbuffer.size, 0)

            taps_per_stage = []
            for stage in sqn_meta_message.decimation_stages:
                dm_rates.append(stage.dm_rate)
                dm_scheme_taps.append(np.array(stage.filter_taps, dtype=np.complex64))
                taps_per_stage.append(len(stage.filter_taps))
            log.info("stage decimation and filter taps",
                     decimation_rates=dm_rates,
                     filter_taps_per_stage=taps_per_stage)

            dm_rates = np.array(dm_rates, dtype=np.uint32)

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

            if cupy_available:
                xp.cuda.runtime.setDevice(0)

            seq_begin_iden = options.dspbegin_to_brian_identity + str(sequence_num)
            seq_end_iden = options.dspend_to_brian_identity + str(sequence_num)
            dw_iden = options.dsp_to_dw_identity + str(sequence_num)
            gpu_socks = so.create_sockets([seq_begin_iden, seq_end_iden, dw_iden],
                                          options.router_address)

            dspbegin_to_brian = gpu_socks[0]
            dspend_to_brian = gpu_socks[1]
            dsp_to_dw = gpu_socks[2]

            # Generate a timer dict for a uniform log
            log_dict = {"time_units": "ms"}
            start_timer = time.perf_counter()

            # Copy samples from ring buffer
            indices = np.arange(start_sample, start_sample + samples_needed)
            # x.take makes a copy of the array. We want to avoid making a copy using Cupy so that
            # data is moved directly from the ring buffer to the GPU. Simple indexing creates a view
            # of existing data without making a copy.
            if cupy_available:
                if end_sample > ringbuffer.shape[1]:
                    piece1 = ringbuffer[:, start_sample:]
                    piece2 = ringbuffer[:, :end_sample - ringbuffer.shape[1]]
                    tmp1 = xp.array(piece1)
                    tmp2 = xp.array(piece2)
                    sequence_samples = xp.concatenate((tmp1, tmp2), axis=1)
                else:
                    sequence_samples = xp.array(ringbuffer[:, start_sample:end_sample])
            else:
                sequence_samples = ringbuffer.take(indices, axis=1, mode='wrap')
            log_dict["copy_samples_from_ringbuffer_time"] = (time.perf_counter() - start_timer) * 1e3

            # Tell brian DSP is about to begin
            mark_timer = time.perf_counter()
            reply_packet = {"sequence_num": sequence_num}
            msg = pickle.dumps(reply_packet, protocol=pickle.HIGHEST_PROTOCOL)
            so.recv_bytes(dspbegin_to_brian, options.brian_to_dspbegin_identity, log)
            so.send_bytes(dspbegin_to_brian, options.brian_to_dspbegin_identity, msg)
            log_dict["dsp_begin_msg_time"] = (time.perf_counter() - mark_timer) * 1e3

            # Process main samples
            mark_timer = time.perf_counter()
            main_sequence_samples = sequence_samples[:len(options.main_antennas), :]
            main_sequence_samples_shape = main_sequence_samples.shape
            main_processor = DSP(rx_rate, dm_scheme_taps, mixing_freqs, dm_rates)
            main_processor.apply_filters(main_sequence_samples)
            main_processor.move_filter_results()
            main_processor.beamform(main_beam_angles)
            main_corrs = DSP.correlations_from_samples(main_processor.beamformed_samples,
                                                       main_processor.beamformed_samples,
                                                       output_sample_rate, slice_details)
            log_dict["main_dsp_time"] = (time.perf_counter() - mark_timer) * 1e3

            # Process intf samples if intf exists
            mark_timer = time.perf_counter()
            intf_sequence_samples_shape = None
            if options.intf_antenna_count > 0:
                intf_sequence_samples = sequence_samples[len(options.main_antennas):, :]
                intf_sequence_samples_shape = intf_sequence_samples.shape
                intf_processor = DSP(rx_rate, dm_scheme_taps, mixing_freqs, dm_rates)
                intf_processor.apply_filters(intf_sequence_samples)
                intf_processor.move_filter_results()
                intf_processor.beamform(intf_beam_angles)
                intf_corrs = DSP.correlations_from_samples(intf_processor.beamformed_samples,
                                                           intf_processor.beamformed_samples,
                                                           output_sample_rate, slice_details)
                cross_corrs = DSP.correlations_from_samples(intf_processor.beamformed_samples,
                                                            main_processor.beamformed_samples,
                                                            output_sample_rate, slice_details)
            log_dict["intf_dsp_time"] = (time.perf_counter() - mark_timer) * 1e3

            # Tell brian DSP how long it took
            mark_timer = time.perf_counter()
            reply_packet["kerneltime"] = log_dict["main_dsp_time"] + log_dict["intf_dsp_time"]
            msg = pickle.dumps(reply_packet, protocol=pickle.HIGHEST_PROTOCOL)
            so.recv_bytes(dspend_to_brian, options.brian_to_dspend_identity, log)
            so.send_bytes(dspend_to_brian, options.brian_to_dspend_identity, msg)
            log_dict["dsp_end_msg_time"] = (time.perf_counter() - mark_timer) * 1e3

            log_dict["total_sequence_process_time"] = (time.perf_counter() - start_timer) * 1e3
            log.info("processing sequence",
                     sequence_num=sequence_num,
                     mixing_freqs=mixing_freqs,
                     mixing_freqs_units='Hz',
                     main_beam_angles=main_beam_angles.shape,
                     intf_beam_angles=main_beam_angles.shape,
                     main_buffer_shape=main_sequence_samples_shape,
                     intf_buffer_shape=intf_sequence_samples_shape,
                     **log_dict)

            # Generate a new timer dict for a uniform log
            log_dict = {"time_units": "ms"}
            start_timer = time.perf_counter()

            # Extract outputs from processing into groups that will be put into message fields.
            data_outputs = {}

            def debug_data_in_shm(holder, data_array, array_name):
                """
                Adds an array of antennas data (filter outputs or antennas_iq) into a dictionary
                for later entry in a processed data message.

                :param  holder:     Dictionary to store the shared memory parameters.
                :type   holder:     dict
                :param  data_array: array to hold the data
                :type   data_array: cp.ndarray or np.ndarray
                :param  array_name: 'main' or 'intf'
                :type   array_name: str
                """

                shm = shared_memory.SharedMemory(create=True, size=data_array.nbytes)
                data = np.ndarray(data_array.shape, dtype=np.complex64, buffer=shm.buf)
                if cupy_available:
                    data[...] = xp.asnumpy(data_array)
                else:
                    data[...] = data_array
                try:
                    assert array_name in ["main", "intf"]
                    if array_name == 'main':
                        holder.main_shm = shm.name
                    elif array_name == 'intf':
                        holder.intf_shm = shm.name
                except Exception as e:
                    log.error(f"unknown array name {array_name} not in [main, intf]", error=e)
                    log.exception(f"unknown array name {array_name} [main, intf]", error=e)
                    sys.exit(1)

                holder.num_samps = data_array.shape[-1]
                shm.close()

            # Add the filter stage data if in debug mode
            if __debug__:
                for i, main_data in enumerate(main_processor.filter_outputs[:-1]):
                    stage = DebugDataStage(f'stage_{i}')
                    debug_data_in_shm(stage, main_data, 'main')

                    if options.intf_antenna_count > 0:
                        intf_data = intf_processor.filter_outputs[i]
                        debug_data_in_shm(stage, intf_data, 'intf')

                    processed_data.add_debug_data(stage)

            # Add antennas_iq data
            stage = DebugDataStage()
            stage.stage_name = 'antennas'
            main_shm = main_processor.shared_mem['antennas_iq']
            stage.main_shm = main_shm.name
            stage.num_samps = main_processor.antennas_iq_samples.shape[-1]
            main_shm.close()
            if options.intf_antenna_count > 0:
                intf_shm = intf_processor.shared_mem['antennas_iq']
                stage.intf_shm = intf_shm.name
                intf_shm.close()
            processed_data.add_debug_data(stage)
            log_dict["add_antiq_to_stage_time"] = (time.perf_counter() - start_timer) * 1e3
            mark_timer = time.perf_counter()

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
                log_dict["put_rawrf_in_shm_time"] = (time.perf_counter() - mark_timer) * 1e3

            # Add bfiq and correlations data
            mark_timer = time.perf_counter()
            beamformed_m = main_processor.beamformed_samples
            processed_data.bfiq_main_shm = main_processor.shared_mem['bfiq'].name
            processed_data.max_num_beams = beamformed_m.shape[1]    # [num_slices, num_beams, num_samps]
            processed_data.num_samps = beamformed_m.shape[-1]
            main_processor.shared_mem['bfiq'].close()

            data_outputs['main_corrs'] = main_corrs

            if options.intf_antenna_count > 0:
                data_outputs['cross_corrs'] = cross_corrs
                data_outputs['intf_corrs'] = intf_corrs
                processed_data.bfiq_intf_shm = intf_processor.shared_mem['bfiq'].name
                intf_processor.shared_mem['bfiq'].close()

            # Fill message with the slice-specific fields
            fill_datawrite_message(processed_data, slice_details, data_outputs)
            sqn_message = pickle.dumps(processed_data, protocol=pickle.HIGHEST_PROTOCOL)
            log_dict["add_bfiq_and_acfs_to_stage_time"] = (time.perf_counter() - mark_timer) * 1e3

            so.send_bytes(dsp_to_dw, options.dw_to_dsp_identity, sqn_message)

            log_dict["total_serialize_send_time"] = (time.perf_counter() - start_timer) * 1e3
            log.info("processing sequence",
                     sequence_num=sequence_num,
                     **log_dict)

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
    from utils.options import Options
    from utils import socket_operations as so
    from utils import log_config

    log = log_config.log()
    log.info(f"RX_SIGNAL_PROCESSING BOOTED")
    try:
        main()
        log.info(f"RX_SIGNAL_PROCESSING EXITED")
    except Exception as main_exception:
        log.critical("RX_SIGNAL_PROCESSING CRASHED", error=main_exception)
        log.exception("RX_SIGNAL_PROCESSING CRASHED", exception=main_exception)
