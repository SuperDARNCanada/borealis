"""
Copyright SuperDARN Canada 2020
Original Auth: Keith Kotyk
"""
import sys
import os
import mmap
import time
import threading
import numpy as np
import posix_ipc as ipc
from multiprocessing import shared_memory
import mmap
import zmq
import dsp
import math
import copy

try:
    import cupy as cp
except:
    cupy_available = False
else:
    cupy_available = True

borealis_path = os.environ['BOREALISPATH']
if not borealis_path:
    raise ValueError("BOREALISPATH env variable not set")

if __debug__:
    sys.path.append(borealis_path + '/build/debug/utils/protobuf')
else:
    sys.path.append(borealis_path + '/build/release/utils/protobuf')

import rxsamplesmetadata_pb2
import sigprocpacket_pb2
import processeddata_pb2

sys.path.append(borealis_path + '/utils/')
import signal_processing_options.signal_processing_options as spo
from zmq_borealis_helpers import socket_operations as so
import shared_macros.shared_macros as sm

pprint = sm.MODULE_PRINT("rx signal processing", "magenta")


def ndarray_in_shr_mem(ndarray):
    """
    This function opens a new shared memory section and copys a ndarray into it.

    :param      ndarray:  The ndarray
    :type       ndarray:  ndarray

    :returns:   Dict holding the name and new shr mem array.
    :rtype:     dict
    """
    new_shm = ipc.SharedMemory(name=None, flags=(ipc.O_CREAT | ipc.O_EXCL), size=ndarray.nbytes)
    mapfile = mmap.mmap(new_shm.fd, new_shm.size)

    shr_arr = np.frombuffer(mapfile, dtype=ndarray.dtype).reshape(ndarray.shape)
    shr_arr[:] = ndarray

    new_shm.close_fd()
    return {'name': new_shm.name, 'data': shr_arr}


def fill_datawrite_proto(processed_data, slice_details, data_outputs):
    """
    Fills the datawrite protobuf with processed data.

    :param      processed_data:  The processed data protobuf
    :type       processed_data:  protobuf
    :param      slice_details:   The details for each slice that was processed.
    :type       slice_details:   list
    :param      data_outputs:    The processed data outputs.
    :type       data_outputs:    dict
    """

    for sd in slice_details:
        output_data_set = processed_data.outputdataset.add()

        output_data_set.slice_id = sd['slice_id']
        output_data_set.num_beams = sd['num_beams']
        output_data_set.num_ranges = sd['num_range_gates']
        output_data_set.num_lags = sd['num_lags']

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
        output_data_set.mainacf = add_array(main_corrs)

        try:
            intf_corrs = data_outputs['intf_corrs'][sd['slice_num']]
            output_data_set.intacf = add_array(intf_corrs)

            cross_corrs = data_outputs['cross_corrs'][sd['slice_num']]
            output_data_set.xcf = add_array(cross_corrs)
        except:
            # No interferometer data
            pass


def main():
    sig_options = spo.SignalProcessingOptions()

    sockets = so.create_sockets([sig_options.dsp_radctrl_identity,
                                 sig_options.dsp_driver_identity,
                                 sig_options.dsp_exphan_identity,
                                 sig_options.dsp_dw_identity], sig_options.router_address)

    dsp_to_radar_control = sockets[0]
    dsp_to_driver = sockets[1]
    dsp_to_experiment_handler = sockets[2]
    dsp_to_dw = sockets[3]

    ringbuffer = None

    total_antennas = sig_options.main_antenna_count + sig_options.intf_antenna_count

    dm_rates = []
    dm_scheme_taps = []

    extra_samples = 0
    total_dm_rate = 0

    threads = []
    first_time = True
    while True:

        reply = so.recv_bytes(dsp_to_radar_control, sig_options.radctrl_dsp_identity, pprint)

        sp_packet = sigprocpacket_pb2.SigProcPacket()
        sp_packet.ParseFromString(reply)

        rx_rate = np.float64(sp_packet.rxrate)
        output_sample_rate = np.float64(sp_packet.output_sample_rate)
        first_rx_sample_off = sp_packet.offset_to_first_rx_sample
        rx_center_freq = sp_packet.rxctrfreq

        processed_data = processeddata_pb2.ProcessedData()

        processed_data.sequence_num = sp_packet.sequence_num
        processed_data.rx_sample_rate = rx_rate
        processed_data.output_sample_rate = output_sample_rate

        mixing_freqs = []
        main_beam_angles = []
        intf_beam_angles = []

        # Parse out details and force the data type so that Cupy can optimize with standardized
        # data types.
        slice_details = []
        for i, chan in enumerate(sp_packet.rxchannel):
            detail = {}

            # This is the negative of what you would normally expect (i.e. -1 * offset of rxfreq from center freq)
            # because the filter taps do not get flipped when convolving. I.e. we do the cross-correlation instead of
            # convolution, to save some computational complexity from flipping the filter sequence.
            # It works out to the same result.
            mixing_freqs.append(rx_center_freq - chan.rxfreq)

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

            detail['lag_phase_offsets'] = np.array(lag_phase_offsets, dtype=np.complex128)

            detail['lags'] = np.array(lags, dtype=np.uint32)
            detail['num_lags'] = len(lags)

            main_beams = []
            intf_beams = []
            for bd in chan.beam_directions:
                main_beam = []
                intf_beam = []

                for j, phase in enumerate(bd.phase):
                    p = phase.real_phase + 1j * phase.imag_phase

                    if j < sig_options.main_antenna_count:
                        main_beam.append(p)
                    else:
                        intf_beam.append(p)

                main_beams.append(main_beam)
                intf_beams.append(intf_beam)

            detail['num_beams'] = len(main_beams)

            slice_details.append(detail)
            main_beam_angles.append(main_beams)
            intf_beam_angles.append(intf_beams)

        # Different slices can have a different amount of beams used. Slices that use fewer beams
        # than the max number of beams are padded with zeros so that matrix calculations can be
        # used. The extra beams that are processed will be not be parsed for data writing.
        max_num_beams = max([len(x) for x in main_beam_angles])

        def pad_beams(angles, ant_count):
            for x in angles:
                if len(x) < max_num_beams:
                    beam_pad = [0.0j] * ant_count
                    for i in range(max_num_beams - len(x)):
                        x.append(beam_pad)

        pad_beams(main_beam_angles, sig_options.main_antenna_count)
        pad_beams(intf_beam_angles, sig_options.intf_antenna_count)

        main_beam_angles = np.array(main_beam_angles, dtype=np.complex64)
        intf_beam_angles = np.array(intf_beam_angles, dtype=np.complex64)
        mixing_freqs = np.array(mixing_freqs, dtype=np.float64)

        # Get meta from driver
        message = "Need data to process"
        so.send_data(dsp_to_driver, sig_options.driver_dsp_identity, message)
        reply = so.recv_bytes(dsp_to_driver, sig_options.driver_dsp_identity, pprint)

        rx_metadata = rxsamplesmetadata_pb2.RxSamplesMetadata()
        rx_metadata.ParseFromString(reply)

        if sp_packet.sequence_num != rx_metadata.sequence_num:
            pprint(sm.COLOR('red', "ERROR: Packets from driver and radctrl don't match"))
            err = "sp_packet seq num {}, rx_metadata seq num {}".format(sp_packet.sequence_num,
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
            for stage in sp_packet.decimation_stages:
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
            reply_packet = sigprocpacket_pb2.SigProcPacket()
            reply_packet.sequence_num = sequence_num
            msg = reply_packet.SerializeToString()

            request = so.recv_bytes(dspbegin_to_brian, sig_options.brian_dspbegin_identity, pprint)
            so.send_bytes(dspbegin_to_brian, sig_options.brian_dspbegin_identity, msg)

            # Process main samples
            main_sequence_samples = sequence_samples[:sig_options.main_antenna_count, :]
            pprint("Main buffer shape: {}".format(main_sequence_samples.shape))
            processed_main_samples = dsp.DSP(main_sequence_samples, rx_rate, dm_rates,
                                             dm_scheme_taps, mixing_freqs, main_beam_angles)
            main_corrs = dsp.DSP.correlations_from_samples(processed_main_samples.beamformed_samples,
                                                           processed_main_samples.beamformed_samples,
                                                           output_sample_rate,
                                                           slice_details)

            # If interferometer is used, process those samples too.
            if sig_options.intf_antenna_count > 0:
                intf_sequence_samples = sequence_samples[sig_options.main_antenna_count:, :]
                pprint("Intf buffer shape: {}".format(intf_sequence_samples.shape))
                processed_intf_samples = dsp.DSP(intf_sequence_samples, rx_rate, dm_rates,
                                                 dm_scheme_taps, mixing_freqs, intf_beam_angles)

                intf_corrs = dsp.DSP.correlations_from_samples(processed_intf_samples.beamformed_samples,
                                                               processed_intf_samples.beamformed_samples,
                                                               output_sample_rate,
                                                               slice_details)
                cross_corrs = dsp.DSP.correlations_from_samples(processed_intf_samples.beamformed_samples,
                                                                processed_main_samples.beamformed_samples,
                                                                output_sample_rate,
                                                                slice_details)
            end = time.time()

            time_diff = (end - copy_end) * 1000
            reply_packet.kerneltime = time_diff
            msg = reply_packet.SerializeToString()

            pprint("Time to decimate, beamform and correlate for #{}: {}ms".format(sequence_num,
                                                                                   time_diff))

            time_diff = (end - start) * 1000
            pprint("Total time for #{}: {}ms".format(sequence_num, time_diff))

            request = so.recv_bytes(dspend_to_brian, sig_options.brian_dspend_identity, pprint)
            so.send_bytes(dspend_to_brian, sig_options.brian_dspend_identity, msg)

            # Extract outputs from processing into groups that will be put into proto fields.
            start = time.time()
            data_outputs = {}
            stages = []

            def debug_data_in_shm(holder, data_array, array_name):
                """
                Adds an array of antennas data (filter outputs or antennas_iq) into a dictionary
                for later entry in a processed data packet.

                :param holder: Dictionary to store the shared memory parameters.
                :param data_array: cp.ndarray or np.ndarray of the data.
                :param array_name: 'main' or 'intf'. String
                """
                shm = shared_memory.SharedMemory(create=True, size=x.nbytes)
                data = np.ndarray(x.shape, dtype=np.complex64, buffer=shm.buf)
                if cupy_available:
                    data[...] = cp.asnumpy(data_array)
                else:
                    data[...] = data_array
                holder['{}_shm'.format(array_name)] = shm.name
                holder['num_samps'] = x.shape[-1]
                shm.close()
            
            # Add the filter stage data if in debug mode
            if __debug__:
                for i, main_data in enumerate(processed_main_samples[:-1]):
                    stage = {}
                    stage['stage_name'] = 'stage_{}'.format(i)
                    debug_data_in_shm(stage, main_data, 'main')

                    if sig_options.intf_antenna_count > 0:
                        intf_data = processed_intf_samples[i]
                        debug_data_in_shm(stage, intf_data, 'intf')
                    stages.append(stage)

            # Add antennas_iq data
            stage = {}
            stage['stage_name'] = 'antennas'
            main_shm = processed_main_samples.shared_mem['antennas_iq']
            stage['main_shm'] = main_shm.name
            stage['num_samps'] = processed_main_samples.antennas_iq_samples.shape[-1]
            main_shm.close()
            if sig_options.intf_antenna_count > 0:
                intf_shm = processed_intf_samples.shared_mem['antennas_iq']
                stage['intf_shm'] = intf_shm.name
                intf_shm.close()
            stages.append(stage)

            # Put all filter stage and antennas data in the protobuf
            for stage in stages:
                debug_data = processed_data.debug_data.add()
                debug_data.stagename = stage['stage_name']
                debug_data.main_shm = stage['main_shm']
                if 'intf_shm' in stage.keys():
                    debug_data.intf_shm = stage['intf_shm']
                debug_data.num_samps = stage['num_samps']

            done_filling_debug = time.time()
            time_filling_debug = (done_filling_debug - start) * 1000
            pprint("Time to put antennas data in protobuf for #{}: {}ms".format(sequence_num, time_filling_debug))

            # Add rawrf data
            if __debug__:
                # np.complex64 in bytes * num_antennas * num_samps
                rawrf_size = np.dtype(np.complex64).itemsize * ringbuffer.shape[0] * indices.shape[-1]
                rawrf_shm = shared_memory.SharedMemory(create=True, size=rawrf_size)
                rawrf_array = np.ndarray((ringbuffer.shape[0], indices.shape[-1]), dtype=np.complex64, buffer=rawrf_shm.buf)
                rawrf_array[...] = ringbuffer.take(indices, axis=1, mode='wrap')
                processed_data.rf_samples_location = rawrf_shm.name
                processed_data.rawrf_num_samps = indices.shape[-1]
                rawrf_shm.close()

            done_filling_rawrf = time.time()
            time_filling_rawrf = (done_filling_rawrf - done_filling_debug) * 1000
            #pprint("Time to put rawrf in shared memory for #{}: {}ms".format(sequence_num, time_filling_rawrf))

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

            # Fill protobuf with the slice-specific fields
            fill_datawrite_proto(processed_data, slice_details, data_outputs)

            message = processed_data.SerializeToString()

            end = time.time()
            time_for_bfiq_acf = (end - done_filling_rawrf) * 1000
            pprint("Time to add bfiq and acfs to protofobuf for #{}: {}ms".format(sequence_num, time_for_bfiq_acf))

            time_diff = (end - start) * 1000
            pprint("Time to serialize and send processed data for #{}: {}ms".format(sequence_num,
                                                                                    time_diff))
            so.send_bytes(dsp_to_dw, sig_options.dw_dsp_identity, message)

        args = {"sequence_num": copy.deepcopy(sp_packet.sequence_num),
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
