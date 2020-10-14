import sys
import os
import mmap
import time
import threading
import numpy as np
import posix_ipc as ipc
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
    new_shm = ipc.SharedMemory(name=None, flags=(ipc.O_CREAT|ipc.O_EXCL), size=ndarray.nbytes)
    mapfile = mmap.mmap(new_shm.fd, new_shm.size)

    shr_arr = np.frombuffer(mapfile, dtype=ndarray.dtype).reshape(ndarray.shape)
    shr_arr[:] = ndarray

    return {'name' : new_shm.name, 'data' : shr_arr}

def create_datawrite_proto(processed_data, slice_details, data_outputs):

    for sd in slice_details:
        output_data_set = processed_data.outputdataset.add()

        output_data_set.slice_id = sd['slice_id']
        output_data_set.num_beams = sd['num_beams']
        output_data_set.num_lags = sd['num_lags']

        def add_array(ndarray, proto):
            for x in np.nditer(ndarray, order='C'):
                o = proto.add()
                o.real = x.real
                o.imag = x.imag


        main_corrs = data_outputs['main_corrs'][sd['slice_num']]
        add_array(main_corrs, output_data_set.mainacf)

        try:
            intf_corrs = data_outputs['intf_corrs'][sd['slice_num']]
            add_array(intf_corrs, output_data_set.intacf)

            cross_corrs = data_outputs['cross_corrs'][sd['slice_num']]
            add_array(cross_corrs, output_data_set.xcf)
        except:
            # No interferometer data
            pass


        for i in range(sd['num_beams']):
            beam = output_data_set.beamformedsamples.add()
            beam.beamnum = i

            main_samps = data_outputs['beamformed_m'][sd['slice_num']][i]
            add_array(main_samps, beam.mainsamples)

            try:
                intf_samps = data_outputs['beamformed_i'][sd['slice_num']][i]
                add_array(main_samps, beam.mainsamples)
            except:
                # No interferometer data
                pass

        def add_debug_data(stage, name):
            debug_data = output_data_set.debugsamples.add()
            debug_data.stagename = name

            all_ant_samps = stage[sd['slice_num']]
            for j in range(all_ant_samps.shape[0]):
                ant = debug_data.antennadata.add()
                add_array(all_ant_samps[j], ant.antennasamples)

        for i, stage in enumerate(data_outputs['debug_outputs'][:-1]):
            add_debug_data(stage, "stage_" + str(i))

        stage = data_outputs['debug_outputs'][-1]
        add_debug_data(stage, "antennas")

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
        first_rx_sample_off = np.uint32(sp_packet.offset_to_first_rx_sample * rx_rate)

        processed_data = processeddata_pb2.ProcessedData()

        processed_data.sequence_num = sp_packet.sequence_num
        processed_data.rx_sample_rate = rx_rate
        processed_data.output_sample_rate = output_sample_rate

        mixing_freqs = []
        main_beam_angles = []
        intf_beam_angles = []

        slice_details = []
        for i,chan in enumerate(sp_packet.rxchannel):
            detail = {}

            mixing_freqs.append(chan.rxfreq)

            detail['slice_id'] = chan.slice_id
            detail['slice_num'] = i
            detail['first_range'] = np.float32(chan.first_range)
            detail['range_sep'] = np.float32(chan.range_sep)
            detail['tau_spacing'] = np.uint32(chan.tau_spacing)
            detail['num_range_gates'] = np.uint32(chan.num_ranges)
            detail['first_range_off'] = np.uint32(chan.first_range / chan.range_sep)

            lags = []
            for lag in chan.lags:
                lags.append([lag.pulse_1,lag.pulse_2])

            detail['lags'] = np.array(lags, dtype=np.uint32)
            detail['num_lags'] = len(lags)

            main_beams = []
            intf_beams = []
            for bd in chan.beam_directions:
                main_beam = []
                intf_beam = []

                for j,phase in enumerate(bd.phase):
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



        message = "Need data to process"
        so.send_data(dsp_to_driver, sig_options.driver_dsp_identity, message)
        reply = so.recv_bytes(dsp_to_driver, sig_options.driver_dsp_identity, pprint)

        rx_metadata = rxsamplesmetadata_pb2.RxSamplesMetadata()
        rx_metadata.ParseFromString(reply)

        if sp_packet.sequence_num != rx_metadata.sequence_num:
            pprint(sm.COLOR('red',"ERROR: Packets from driver and radctrl don't match"))
            err = "sp_packet seq num {}, rx_metadata seq num {}".format(sp_packet.sequence_num,
                                                                    rx_metadata.sequence_num)
            pprint(sm.COLOR('red', err))
            sys.exit(-1)

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

            for dm,taps in zip(reversed(dm_rates), reversed(dm_scheme_taps)):
                extra_samples = (extra_samples * dm) + len(taps)/2

            total_dm_rate = np.prod(dm_rates)

            first_time = False

        samples_needed = rx_metadata.numberofreceivesamples + 2 * extra_samples
        samples_needed = int(math.ceil(float(samples_needed)/float(total_dm_rate)) * total_dm_rate)

        sample_time_diff = rx_metadata.sequence_start_time - rx_metadata.initialization_time
        sample_in_time = (sample_time_diff * rx_rate) + first_rx_sample_off - extra_samples

        start_sample = int(math.fmod(sample_in_time, ringbuffer.shape[1]))
        end_sample = start_sample + samples_needed

        processed_data.initialization_time = rx_metadata.initialization_time
        processed_data.sequence_start_time = rx_metadata.sequence_start_time

        def sequence_worker(**kwargs):
            sequence_num = kwargs['sequence_num']
            main_beam_angles = kwargs['main_beam_angles']
            intf_beam_angles = kwargs['intf_beam_angles']
            mixing_freqs = kwargs['mixing_freqs']
            slice_details = kwargs['slice_details']
            start_sample = kwargs['start_sample']
            end_sample = kwargs['end_sample']
            processed_data = kwargs['processed_data']


            pprint(sm.COLOR('green',"Processing #{}".format(sequence_num)))
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

            if cupy_available:
                if end_sample > ringbuffer.shape[1]:
                    piece1 = ringbuffer[:,start_sample:]
                    piece2 = ringbuffer[:,:end_sample-ringbuffer.shape[1]]

                    tmp1 = cp.array(piece1)
                    tmp2 = cp.array(piece2)

                    sequence_samples = cp.concatenate((tmp1,tmp2), axis=1)
                else:
                    sequence_samples = cp.array(ringbuffer[:,start_sample:end_sample])

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

            main_sequence_samples = sequence_samples[:sig_options.main_antenna_count,:]
            pprint("Main buffer shape: {}".format(main_sequence_samples.shape))
            processed_main_samples = dsp.DSP(main_sequence_samples, rx_rate, dm_rates,
                                                dm_scheme_taps, mixing_freqs, main_beam_angles)
            main_corrs = dsp.DSP.correlations_from_samples(processed_main_samples.beamformed_samples,
                                    processed_main_samples.beamformed_samples, output_sample_rate,
                                    slice_details)

            if sig_options.intf_antenna_count > 0:
                intf_sequence_samples = sequence_samples[sig_options.main_antenna_count:,:]
                pprint("Intf buffer shape: {}".format(intf_sequence_samples.shape))
                processed_intf_samples = dsp.DSP(intf_sequence_samples, rx_rate, dm_rates,
                                                    dm_scheme_taps, mixing_freqs, intf_beam_angles)

                intf_corrs = dsp.DSP.correlations_from_samples(processed_intf_samples.beamformed_samples,
                                processed_intf_samples.beamformed_samples, output_sample_rate,
                                slice_details)
                cross_corrs = dsp.DSP.correlations_from_samples(processed_main_samples.beamformed_samples,
                                processed_intf_samples.beamformed_samples, output_sample_rate,
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

            start = time.time()
            data_outputs = {}
            if __debug__:
                if cupy_available:
                    filter_outputs_m = [cp.asnumpy(x) for x in processed_main_samples.filter_outputs]

                    if sig_options.intf_antenna_count > 0:
                        filter_outputs_i = [cp.asnumpy(x) for x in processed_intf_samples.filter_outputs]
                else:
                    filter_outputs_m = processed_main_samples.filter_outputs

                    if sig_options.intf_antenna_count > 0:
                        filter_outputs_i = processed_intf_samples.filter_outputs


                if sig_options.intf_antenna_count > 0:
                    filter_outputs = [np.concatenate((x,y)) for x,y in zip(filter_outputs_m,
                                                                                filter_outputs_i)]
                else:
                    filter_outputs = filter_outputs_m

            else:
                if cupy_available:
                    filter_outputs_m = [cp.asnumpy(processed_main_samples.filter_outputs[-1])]

                    if sig_options.intf_antenna_count > 0:
                        filter_outputs_i = [cp.asnumpy(processed_intf_samples.filter_outputs[-1])]
                else:
                    filter_outputs_m = processed_main_samples.filter_outputs[-1]

                    if sig_options.intf_antenna_count > 0:
                        filter_outputs_i = processed_intf_samples.filter_outputs[-1]


                if sig_options.intf_antenna_count > 0:
                    filter_outputs = [np.concatenate((filter_outputs_m,filter_outputs_i))]
                else:
                    filter_outputs = filter_outputs_m

            data_outputs['debug_outputs'] = filter_outputs
            debug_rf = ndarray_in_shr_mem(ringbuffer.take(indices, axis=1, mode='wrap'))
            processed_data.rf_samples_location = debug_rf['name']
            if cupy_available:
                beamformed_m = cp.asnumpy(processed_main_samples.beamformed_samples)

                if sig_options.intf_antenna_count > 0:
                    beamformed_i = cp.asnumpy(processed_intf_samples.beamformed_samples)
            else:
                beamformed_m = processed_main_samples.beamformed_samples

                if sig_options.intf_antenna_count > 0:
                    beamformed_i = processed_intf_samples.beamformed_samples

            data_outputs['beamformed_m'] = beamformed_m
            data_outputs['beamformed_i'] = beamformed_i
            data_outputs['main_corrs'] = main_corrs

            if sig_options.intf_antenna_count > 0:
                data_outputs['cross_corrs'] = cross_corrs
                data_outputs['intf_corrs'] = intf_corrs

            create_datawrite_proto(processed_data, slice_details, data_outputs)

            message = processed_data.SerializeToString()

            end = time.time()
            time_diff = (end-start) * 1000
            pprint("Time to serialize and send processed data for #{}: {}ms".format(sequence_num,
                                                                                    time_diff))
            so.send_bytes(dsp_to_dw, sig_options.dw_dsp_identity, message)




        args = {"sequence_num" : copy.deepcopy(sp_packet.sequence_num),
                "main_beam_angles" : copy.deepcopy(main_beam_angles),
                "intf_beam_angles" : copy.deepcopy(intf_beam_angles),
                "mixing_freqs" : copy.deepcopy(mixing_freqs),
                "slice_details" : copy.deepcopy(slice_details),
                "start_sample" : copy.deepcopy(start_sample),
                "end_sample" : copy.deepcopy(end_sample),
                "processed_data" : copy.deepcopy(processed_data)}

        seq_thread = threading.Thread(target=sequence_worker, kwargs=args)
        seq_thread.daemon = True
        seq_thread.start()

        threads.append(seq_thread)

        if len(threads) > 1:
            thread = threads.pop(0)
            thread.join()

if __name__ == "__main__":
    main()
















