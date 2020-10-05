import sys
import os
import mmap
import time
import threading
import numpy as np
import posix_ipc as ipc
import zmq
import dsp
import math

borealis_path = os.environ['BOREALISPATH']
if not borealis_path:
    raise ValueError("BOREALISPATH env variable not set")

if __debug__:
    sys.path.append(borealis_path + '/build/debug/utils/protobuf')
else:
    sys.path.append(borealis_path + '/build/release/utils/protobuf')

import rxsamplesmetadata_pb2
import sigprocpacket_pb2

sys.path.append(borealis_path + '/utils/')
import signal_processing_options.signal_processing_options as spo
from zmq_borealis_helpers import socket_operations as so
import shared_macros.shared_macros as sm

pprint = sm.MODULE_PRINT("rx signal processing", "magenta")

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

    first_time = True
    while True:

        reply = so.recv_bytes(dsp_to_radar_control, sig_options.radctrl_dsp_identity, pprint)

        sp_packet = sigprocpacket_pb2.SigProcPacket()
        sp_packet.ParseFromString(reply)

        seq_begin_iden = sig_options.dspbegin_brian_identity + str(sp_packet.sequence_num)
        seq_end_iden = sig_options.dspend_brian_identity + str(sp_packet.sequence_num)
        gpu_socks = so.create_sockets([seq_begin_iden, seq_end_iden], sig_options.router_address)

        dspbegin_to_brian = gpu_socks[0]
        dspend_to_brian = gpu_socks[1]

        rx_rate = np.float64(sp_packet.rxrate)
        output_sample_rate = np.float64(sp_packet.output_sample_rate)
        first_rx_sample_off = np.uint32(sp_packet.offset_to_first_rx_sample * rx_rate)

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

        pad_beams(main_beams, sig_options.main_antenna_count)
        pad_beams(intf_beams, sig_options.intf_antenna_count)
        
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
        else:
            pprint("Processing sequence number: {}".format(sp_packet.sequence_num))

        pprint("Main beams shape: {}".format(main_beam_angles.shape))
        pprint("Intf beams shape: {}".format(intf_beam_angles.shape))

        if first_time:
            shm = ipc.SharedMemory(sig_options.ringbuffer_name)
            mapped_mem = mmap.mmap(shm.fd, shm.size)
            ringbuffer = np.frombuffer(mapped_mem, dtype=np.complex64).reshape(total_antennas, -1)

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
        indices = np.arange(start_sample, start_sample + samples_needed)


        reply_packet = sigprocpacket_pb2.SigProcPacket()
        reply_packet.sequence_num = sp_packet.sequence_num
        msg = reply_packet.SerializeToString()

        request = so.recv_bytes(dspbegin_to_brian, sig_options.brian_dspbegin_identity, pprint)
        so.send_bytes(dspbegin_to_brian, sig_options.brian_dspbegin_identity, msg)

        start = time.time()

        main_buffer = ringbuffer[:sig_options.main_antenna_count,:]
        main_sequence_samples = main_buffer.take(indices, axis=1, mode='wrap')
        pprint("Main buffer shape: {}".format(main_sequence_samples.shape))
        processed_main_samples = dsp.DSP(main_sequence_samples, rx_rate, dm_rates, dm_scheme_taps,
                                    mixing_freqs, main_beam_angles)
        main_corrs = dsp.DSP.correlations_from_samples(processed_main_samples.beamformed_samples,
                            processed_main_samples.beamformed_samples, output_sample_rate,
                            slice_details)

        if sig_options.intf_antenna_count > 0:
            intf_buffer = ringbuffer[sig_options.main_antenna_count:,:]
            intf_sequence_samples = intf_buffer.take(indices, axis=1, mode='wrap')
            pprint("Intf buffer shape: {}".format(intf_sequence_samples.shape))
            processed_intf_samples = dsp.DSP(intf_sequence_samples, rx_rate, dm_rates, dm_scheme_taps,
                                        mixing_freqs, intf_beam_angles)

            intf_corrs = dsp.DSP.correlations_from_samples(processed_intf_samples.beamformed_samples,
                            processed_intf_samples.beamformed_samples, output_sample_rate,
                            slice_details)
            cross_corrs = dsp.DSP.correlations_from_samples(processed_main_samples.beamformed_samples,
                            processed_intf_samples.beamformed_samples, output_sample_rate,
                            slice_details)

        end = time.time()

        time_diff = (end - start) * 1000
        reply_packet.kerneltime = time_diff
        msg = reply_packet.SerializeToString()

        pprint("Time to decimate, beamform and correlate: {}ms".format(time_diff))

        request = so.recv_bytes(dspend_to_brian, sig_options.brian_dspend_identity, pprint)
        so.send_bytes(dspend_to_brian, sig_options.brian_dspend_identity, msg)

if __name__ == "__main__":
    main()
















