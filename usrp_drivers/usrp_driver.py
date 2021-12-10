"""
Copyright SuperDARN Canada 2021

This file was adapted into Python in December 2021 by Remington Rohel, based
off of the file usrp_driver.cpp written previously.
"""
import time

import numpy as np
import uhd
import os
import sys
from datetime import datetime

import usrp
from utils.zmq_borealis_helpers import socket_operations as so
from utils.driver_options.driveroptions import DriverOptions
import utils.shared_macros.shared_macros as sm

# TODO: Make shared_memory module in Python
# import utils.shared_memory.shared_memory as shm

borealis_path = os.environ['BOREALISPATH']
if not borealis_path:
    raise ValueError("BOREALISPATH env variable not set")

if __debug__:
    sys.path.append(borealis_path + '/build/debug/utils/protobuf')
else:
    sys.path.append(borealis_path + '/build/release/utils/protobuf')

import rxsamplesmetadata_pb2
import driverpacket_pb2

tx_print = sm.MODULE_PRINT("Transmit", "blue")
rx_print = sm.MODULE_PRINT("Receive", "yellow")

# Delay needed for before any set_time_commands will work.
set_time_command_delay = 5e-3   # seconds
tuning_delay = 300e-3           # seconds

# Module clocks: one for box_time (from the N200s, supplied by Octoclock-G)
# as well as one for the operating system time (by NTP). Updated upon recv of RX packet.
box_time = uhd.types.TimeSpec()
# system_time = ???

def make_tx_samples(driver_packet: DriverPacket, driver_options: DriverOptions):
    """Makes a set of vectors of the samples for each TX channel from the driver packet.

    Values in a protobuffer have no contiguous underlying storage so values need to be parsed into a vector.

    :param  driver_packet:  A received driver packet from radar_control.
    :type   driver_packet:  DriverPacket
    :param  driver_options: The parsed config options needed by the driver.
    :type   driver_options: DriverOptions
    """
    samples = np.empty(driver_packet.channel_samples_size(), dtype=complex)

    for channel in range(driver_packet.channel_samples_size()):
        # Get the number of real samples in this particular channel (_size() is from protobuf)
        num_samps = driver_packet.channel_samples(channel).real_size()
        v = np.empty(num_samps, dtype=complex)

        # Type for smp? protobuf object, containing repeated double real and double imag
        smp = driver_packet.channel_samples(channel)
        for smp_num in range(num_samps):
            v[smp_num] = np.complex(smp.real(smp_num), smp.imag(smp_num))

        samples[channel] = v

        for s in samples:
            if s.size != samples[0].size:
                # TODO: Handle this error. Sample buffers are of different lengths.
                pass

        return samples

def transmit(usrp_d: usrp.USRP, driver_options: DriverOptions):
    tx_print("Enter transmit thread")

    identities = [driver_options.driver_to_radctrl_identity,
                  driver_options.driver_to_dsp_identity,
                  driver_options.driver_to_brian_identity]

    sockets = so.create_sockets(identities, driver_options.router_address)
    driver_to_radctrl = sockets[0]
    driver_to_dsp = sockets[1]
    driver_to_brian = sockets[2]

    receive_channels = driver_options.receive_channels

    samples_set = False

    rx_rate = usrp_d.get_rx_rate(receive_channels[0])

    # driverpacket::DriverPacket
    # driver_packet;
    #
    # zmq::socket_t
    # start_trigger(driver_c, ZMQ_PAIR);
    # ERR_CHK_ZMQ(start_trigger.connect("inproc://thread"))

    tx_channels = driver_options.transmit_channels
    tx_stream = usrp_d.get_usrp_tx_stream

    # std::vector < std::vector < std::vector < std::complex < float >> >> pulses;
    # std::vector < std::vector < std::complex < float >> > last_pulse_sent;

    tx_center_freq = usrp_d.get_tx_center_freq(tx_channels[0])
    tx_center_freq = usrp_d.get_rx_center_freq(receive_channels[0])

    sqn_num = 0
    expected_sqn_num = 0
    #
    # uint32_t num_recv_samples;
    #
    # size_t ringbuffer_size;
    #
    # uhd::time_spec_t sequence_start_time;
    # uhd::time_spec_t initialization_time;

    agc_signal_read_delay = driver_options.agc_signal_read_delay * 1e-6

    # auto clocks = borealis_clocks
    # auto system_since_epoch = std::chrono::duration<double>(clocks.system_time.time_since_epoch());
    # auto gps_to_system_time_diff = system_since_epoch.count() - clocks.box_time.get_real_secs();
    #
    # zmq::message_t request;
    #
    # start_trigger.recv(&request);
    #   memcpy(&ringbuffer_size, static_cast<size_t*>(request.data()), request.size());
    #
    #   start_trigger.recv(&request);
    #   memcpy(&initialization_time, static_cast<uhd::time_spec_t*>(request.data()), request.size());

    # This loop accepts pulse by pulse from the radar_control. It parses the samples, configures the
    # USRP, sets up the timing, and then sends samples/timing to the USRPs.

    while True:
        more_pulses = True
#     std::vector<double> time_to_send_samples;
        agc_status_bank_h = 0b0
        lp_status_bank_h = 0b0
        agc_status_bank_l = 0b0
        lp_status_bank_l = 0b0
        while more_pulses:
            pulse_data = so.recv_data(driver_to_radctrl, driver_options.radctrl_to_driver_identity)
#
            # Here we accept our driver_packet from the radar_control. We use that info in order to
            # configure the USRP devices based on experiment requirements.
            if __debug__:
                tx_setup_start_time = time.monotonic_ns()
            driver_packet = driverpacket_pb2.DriverPacket()
            if not driver_packet.ParseFromString(pulse_data):
                # TODO: Handle error
                pass

            sqn_num = np.uint32(driver_packet.sequence_num)
            seq_time = driver_packet.seqtime
            if sqn_num != expected_sqn_num:
                if __debug__:
                    tx_print("SEQUENCE NUMBER MISMATCH: SQN {} EXPECTED: ".format(expected_sqn_num))
                # TODO: Handle error

            if __debug__:
                tx_print("Burst flags: SOB {} EOB {}".format(driver_packet.sob, driver_packet.eob))

            set_ctr_freq_start = time.monotonic_ns()

            # If there is new center frequency data, set TX center frequency for each USRP TX channel.
            if tx_center_freq != driver_packet.txcenterfreq:
                if driver_packet.txcenterfreq > 0.0 and driver_packet.sob:
                    if __debug__:
                        tx_print("Setting tx center freq to {}".format(driver_packet.txcenterfreq))
                    tx_center_freq = usrp_d.set_tx_center_freq(driver_packet.txcenterfreq, tx_channels,
                                                               uhd.types.TimeSpec(tuning_delay))


            # rxcenterfreq() will return 0 if it hasn't changed, so check for changes here
            if rx_center_freq != driver_packet.rxcenterfreq:
                if driver_packet.rxcenterfreq > 0.0 and driver_packet.sob:
                    if __debug__:
                        tx_print("Setting rx center freq to {}".format(driver_packet.rxcenterfreq))
                    rx_center_freq = usrp_d.set_rx_center_freq(driver_packet.rxcenterfreq, receive_channels,
                                                               uhd.types.TimeSpec(tuning_delay))

            set_ctr_freq_end = time.monotonic_ns()
            set_ctr_freq_duration = set_ctr_freq_end - set_ctr_freq_start
            tx_print("Center Frequency set time: {} us".format(set_ctr_freq_duration/1e3))

            sample_unpack_start = time.monotonic_ns()

            if driver_packet.sob:
                pulses.clear()
            # Parse new samples from driver packet if they exist.
            if driver_packet.channel_samples.size > 0:
                # ~700 us to unpack 4x1600 samples (with C++ driver)
                last_pulse_sent = make_tx_samples(driver_packet, driver_options)
                samples_set = True
            pulses.push_back(last_pulse_sent)

            sample_unpack_end = time.monotonic_ns()
            sample_unpack_duration = sample_unpack_end - sample_unpack_start
            tx_print("Sample unpack time: {} us".format(sample_unpack_duration/1e3))

            tx_setup_end_time = time.monotonic_ns()
            tx_setup_duration = tx_setup_end_time - tx_setup_start_time

            tx_print("Total setup time: {} us".format(tx_setup_duration/1e3))

            time_to_send_samples.push_back(driver_packet.timetosendsamples)

            if driver_packet.sob:
                num_recv_samples = driver_packet.numberofreceivesamples

            if driver_packet.eob:
                more_pulses = False

        if not samples_set:
            # TODO: Throw error
            continue

        # If grabbing start of vector using samples[i] it doesn't work (samples are firked)
        # You need to grab the ptr to the vector using samples[a][b].data(). See tx_waveforms
        # for how to do this properly. Also see uhd::tx_streamer::send(...) in the uhd docs
        # see 'const buffs_type &'' argument to the send function, the description should read
        # 'Typedef for a pointer to a single, or a collection of pointers to send buffers'.
        pulse_ptrs = np.empty(pulses.size)
        for i in range(pulses.size):
            ptrs = np.empty(pulses[i].size)
            for j in range(pulses[i].size):
                ptrs[j] = pulses[i][j].data()
            pulse_ptrs[i] = ptrs

        # Getting usrp box time to find out when to send samples. box_time continuously being updated.
        delay = uhd.types.TimeSpec(set_time_command_delay)
        time_now = box_time
        sequence_start_time = time_now + delay

        seqn_sampling_time = num_recv_samples / rx_rate

        full_usrp_start = time.monotonic_ns()

#     TIMEIT_IF_TRUE_OR_DEBUG(false, COLOR_BLUE("TRANSMIT") << " full usrp time stuff ",
#       [&]() {
#
#         // Here we are time-aligning our time_zero to the start of a sample. Do this by recalculating
#         // time_zero using the calculated value of start_sample.
#         // TODO: Account for offset btw TX/RX (seems to change with sampling rate at least)
#
#         auto time_diff = sequence_start_time - initialization_time;
#         double future_start_sample = std::floor(time_diff.get_real_secs() * rx_rate);
#         auto time_from_initialization = uhd::time_spec_t((future_start_sample/rx_rate));
#
#         sequence_start_time = initialization_time + time_from_initialization;
#
#         TIMEIT_IF_TRUE_OR_DEBUG(false ,COLOR_BLUE("TRANSMIT") << " time to send all samples to USRP: ",
#           [&]() {
#             for (uint32_t i=0; i<pulses.size(); i++){
#               auto md = TXMetadata();
#               md.set_has_time_spec(true);
#               auto time = sequence_start_time + uhd::time_spec_t(time_to_send_samples[i]/1.0e6);
#               md.set_time_spec(time);
#               //The USRP tx_metadata start_of_burst and end_of_burst describe start and end of the pulse
#               //samples.
#               md.set_start_of_burst(true);
#               md.set_end_of_burst(false);
#
#
#               //This will loop until all samples are sent to the usrp. Send will block until all samples sent
#               //or timed out(too many samples to send within timeout period). Send has a default timing of
#               //0.1 seconds.
#               auto samples_per_pulse = pulses[i][0].size();
#
#               TIMEIT_IF_TRUE_OR_DEBUG(false, COLOR_BLUE("TRANSMIT") << " time to send pulse " << i <<
#                                       " to USRP: ",
#                 [&]() {
#                   uint64_t total_samps_sent = 0;
#                   while (total_samps_sent < samples_per_pulse) {
#                     auto num_samps_to_send = samples_per_pulse - total_samps_sent;
#
#
#                     auto num_samps_sent = tx_stream->send(pulse_ptrs[i],
#                                                           num_samps_to_send, md.get_md()); //TODO(keith): Determine timeout properties.
#                     DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " Samples sent " << num_samps_sent);
#
#                     total_samps_sent += num_samps_sent;
#                     md.set_start_of_burst(false);
#                     md.set_has_time_spec(false);
#
#                   }
#                   md.set_end_of_burst(true);
#                   tx_stream->send("", 0, md.get_md());
#
#                 }() //pulse lambda
#               ); //pulse timeit macro
#             }
#
#             // Read AGC and Low Power signals, bitwise OR to catch any time the signals are active
#             // during this sequence for each USRP individually
#             usrp_d.clear_command_time();
#             auto read_time = sequence_start_time + (seqtime * 1e-6) + agc_signal_read_delay;
#             usrp_d.set_command_time(read_time);
#             agc_status_bank_h = agc_status_bank_h | usrp_d.get_agc_status_bank_h();
#             lp_status_bank_h = lp_status_bank_h | usrp_d.get_lp_status_bank_h();
#             agc_status_bank_l = agc_status_bank_l | usrp_d.get_agc_status_bank_l();
#             lp_status_bank_l = lp_status_bank_l | usrp_d.get_lp_status_bank_l();
#             usrp_d.clear_command_time();
#
#             for (uint32_t i=0; i<pulses.size(); i++) {
#               uhd::async_metadata_t async_md;
#               std::vector<size_t> acks(tx_channels.size(),0);
#               std::vector<size_t> lates(tx_channels.size(),0);
#               size_t channel_acks = 0;
#               size_t channel_lates = 0;
#               //loop through all messages for the ACK packets (may have underflow messages in queue)
#               while (channel_acks < tx_channels.size() and tx_stream->recv_async_msg(async_md))
#               {
#                   if (async_md.event_code == uhd::async_metadata_t::EVENT_CODE_BURST_ACK)
#                   {
#                     channel_acks++;
#                     acks[async_md.channel]++;
#                   }
#
#                   if (async_md.event_code == uhd::async_metadata_t::EVENT_CODE_TIME_ERROR)
#                   {
#
#                     channel_lates++;
#                     lates[async_md.channel]++;
#
#                   }
#               }
#
#               for(uint32_t j=0; j<lates.size(); j++) {
#                 DEBUG_MSG(COLOR_BLUE("TRANSMIT") << ": channel " << j <<
#                               " got " << lates[j] << " lates for pulse " << i);
#               }
#
#               DEBUG_MSG(COLOR_BLUE("TRANSMIT") << ": Sequence " << sqn_num <<" Got "
#                             << channel_acks << " acks out of " << tx_channels.size()
#                             << " channels for pulse " << i);
#               DEBUG_MSG(COLOR_BLUE("TRANSMIT") << ": Sequence " << sqn_num << " Got "
#                             << channel_lates << " lates out of " << tx_channels.size()
#                             << " channels for pulse " << i);
#               }
#           }() //all pulses lambda
#         ); //all pulses timeit macro
#       }() // full usrp function lambda
#     ); // full usrp function timeit macro
#
#     rxsamplesmetadata::RxSamplesMetadata samples_metadata;
#
#     clocks = borealis_clocks;
#     system_since_epoch = std::chrono::duration<double>(clocks.system_time.time_since_epoch());
#     // get_real_secs() may lose precision of the fractional seconds, but it's close enough
#     gps_to_system_time_diff = system_since_epoch.count() - clocks.box_time.get_real_secs();
#
#     samples_metadata.set_gps_locked(usrp_d.gps_locked());
#     samples_metadata.set_gps_to_system_time_diff(gps_to_system_time_diff);
#
#     if (!usrp_d.gps_locked()) {
#       RUNTIME_MSG("GPS UNLOCKED! time diff: " << COLOR_RED(gps_to_system_time_diff*1000.0) << "ms");
#     }
#
#     auto end_time = borealis_clocks.box_time;
#     auto sleep_time = uhd::time_spec_t(seqn_sampling_time) - (end_time-sequence_start_time) + delay;
#     // sleep_time is how much longer we need to wait in tx thread before the end of the sampling time
#
#     DEBUG_MSG(COLOR_BLUE("TRANSMIT") << ": Sleep time " << sleep_time.get_real_secs() * 1e6
#                   << " us");
#
#     if(sleep_time.get_real_secs() > 0.0) {
#       auto duration = std::chrono::duration<double>(sleep_time.get_real_secs());
#       std::this_thread::sleep_for(duration);
#     }
#
#
#     samples_metadata.set_rx_rate(rx_rate);
#     samples_metadata.set_initialization_time(initialization_time.get_real_secs());
#     samples_metadata.set_sequence_start_time(sequence_start_time.get_real_secs());
#     samples_metadata.set_ringbuffer_size(ringbuffer_size);
#     samples_metadata.set_numberofreceivesamples(num_recv_samples);
#     samples_metadata.set_sequence_num(sqn_num);
#     auto actual_finish = borealis_clocks.box_time;
#     samples_metadata.set_sequence_time((actual_finish - time_now).get_real_secs());
#
#     samples_metadata.set_agc_status_bank_h(agc_status_bank_h);
#     samples_metadata.set_lp_status_bank_h(lp_status_bank_h);
#     samples_metadata.set_agc_status_bank_l(agc_status_bank_l);
#     samples_metadata.set_lp_status_bank_l(lp_status_bank_l);
#
#     std::string samples_metadata_str;
#     samples_metadata.SerializeToString(&samples_metadata_str);
#
#     // Here we wait for a request from dsp for the samples metadata, then send it, bro!
#     // https://www.youtube.com/watch?v=WIrWyr3HgXI
#     auto request = RECV_REQUEST(driver_to_dsp, driver_options.get_dsp_to_driver_identity());
#     SEND_REPLY(driver_to_dsp, driver_options.get_dsp_to_driver_identity(), samples_metadata_str);
#
#     // Here we wait for a request from brian for the samples metadata, then send it
#     request = RECV_REQUEST(driver_to_brian, driver_options.get_brian_to_driver_identity());
#     SEND_REPLY(driver_to_brian, driver_options.get_brian_to_driver_identity(), samples_metadata_str);
#
#     expected_sqn_num++;
#     DEBUG_MSG(std::endl << std::endl);
#   } // while(1)