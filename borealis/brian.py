#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# brian.py
# 2018-01-30
# Communicate with all processes to administrate the borealis software

import sys
import os
import time
from datetime import datetime
import threading
import argparse
import zmq
import pickle

from borealis.utils import socket_operations as so
from borealis.utils import shared_macros as sm

if __debug__:
    from build.debug.borealis.utils.protobuf import rxsamplesmetadata_pb2
else:
    from build.release.borealis.utils.protobuf import rxsamplesmetadata_pb2

TIME_PROFILE = True

brian_print = sm.MODULE_PRINT("sequence timing", "red")

def router(opts):
    """The router is responsible for moving traffic between modules by routing traffic using
    named sockets.

    Args:
        opts (ExperimentOptions): Options parsed from config.
    """
    context = zmq.Context().instance()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.ROUTER_MANDATORY, 1)
    router.bind(opts.router_address)

    sys.stdout.write("Starting router!\n")
    frames_to_send = []
    while True:
        events = router.poll(timeout=1)
        if events:
            dd = router.recv_multipart()

            sender, receiver, empty, data = dd
            if __debug__:
                output = "Router input/// Sender -> {}: Receiver -> {}\n"
                output = output.format(sender, receiver)
                sys.stdout.write(output)
            frames_received = [receiver,sender,empty,data]
            frames_to_send.append(frames_received)

            if __debug__:
                output = "Router output/// Receiver -> {}: Sender -> {}\n"
                output = output.format(receiver, sender)
                sys.stdout.write(output)
        non_sent = []
        for frames in frames_to_send:
            try:
                router.send_multipart(frames)
            except zmq.ZMQError as e:
                if __debug__:
                    output = "Unable to send frame Receiver -> {}: Sender -> {}\n"
                    output = output.format(frames[0], frames[1])
                    sys.stdout.write(output)
                non_sent.append(frames)
        frames_to_send = non_sent


def sequence_timing(opts):
    """Thread function for sequence timing

    This function controls the flow of data between brian's sequence timing and other parts of the
    radar system. This function serves to check whether the sequence timing is working as expected
    and to rate control the system to make sure the processing can handle data rates.
    :param context: zmq context, if None, then this method will get one
    :type context: zmq context, optional
    """

    ids = [opts.brian_to_radctrl_identity,
           opts.brian_to_driver_identity,
           opts.brian_to_dspbegin_identity,
           opts.brian_to_dspend_identity]

    sockets_list = so.create_sockets(ids, opts.router_address)

    brian_to_radar_control = sockets_list[0]
    brian_to_driver = sockets_list[1]
    brian_to_dsp_begin = sockets_list[2]
    brian_to_dsp_end = sockets_list[3]

    sequence_poller = zmq.Poller()
    sequence_poller.register(brian_to_radar_control, zmq.POLLIN)
    sequence_poller.register(brian_to_dsp_begin, zmq.POLLIN)
    sequence_poller.register(brian_to_dsp_end, zmq.POLLIN)
    sequence_poller.register(brian_to_driver, zmq.POLLIN)

    context = zmq.Context().instance()

    start_new_sock = context.socket(zmq.PAIR)
    start_new_sock.bind("inproc://start_new")

    def start_new():
        """ This function serves to rate control the system. If processing is faster than the
        sequence time than the speed of the driver is the limiting factor. If processing takes
        longer than sequence time, than the dsp unit limits the speed of the system.
        """
        start_new = context.socket(zmq.PAIR)
        start_new.connect("inproc://start_new")

        want_to_start = False
        good_to_start = True
        dsp_finish_counter = 2

        # starting a new sequence and keeping the system correctly pipelined is dependent on 3
        # conditions. We trigger a 'want_to_start' when the samples have been collected from the
        # driver and dsp is ready to do its job. This signals that the driver is capable of
        # collecting new data. 'good_to_start' is triggered once the samples have been copied to
        # the GPU and the filtering begins. 'extra_good_to_start' is needed to make sure the
        # system can keep up with the demand if the gpu is working hard. Without this flag its
        # possible to overload the gpu and crash the system with overallocation of memory. This
        # is set once the filtering is complete.
        #
        # The last flag is actually a counter because on the first run it is 2 sequences
        # behind the current sequence and then after that its only 1 sequence behind. The dsp
        # is always processing the work while a new sequence is being collected.
        if TIME_PROFILE:
            time_now = datetime.utcnow()
        while True:
            if want_to_start and good_to_start and dsp_finish_counter:
                #Acknowledge new sequence can begin to Radar Control by requesting new sequence
                #metadata
                if __debug__:
                    brian_print("Requesting metadata from Radar control")

                so.send_request(brian_to_radar_control, opts.radctrl_to_brian_identity,
                                "Requesting metadata")
                want_to_start = good_to_start = False
                dsp_finish_counter -= 1

            message = start_new.recv_string()
            if message == "want_to_start":
                if TIME_PROFILE:
                    brian_print('Driver ready: {}'.format(datetime.utcnow() - time_now))
                    time_now = datetime.utcnow()
                want_to_start = True

            if message == "good_to_start":
                if TIME_PROFILE:
                    brian_print('Copied to GPU: {}'.format(datetime.utcnow() - time_now))
                    time_now = datetime.utcnow()
                good_to_start = True

            if message == "extra_good_to_start":
                if TIME_PROFILE:
                    brian_print('DSP finished w/ data: {}'.format(datetime.utcnow() - time_now))
                    time_now = datetime.utcnow()
                dsp_finish_counter = 1;

    thread = threading.Thread(target=start_new)
    thread.daemon = True
    thread.start()

    time.sleep(1)

    last_processing_time = 0

    first_time = True
    late_counter = 0
    while True:

        if first_time:
            #Request new sequence metadata
            if __debug__:
                brian_print("Requesting metadata from Radar control")
            so.send_request(brian_to_radar_control, opts.radctrl_to_brian_identity,
                            "Requesting metadata")
            first_time = False

        socks = dict(sequence_poller.poll())

        if brian_to_driver in socks and socks[brian_to_driver] == zmq.POLLIN:

            #Receive metadata of completed sequence from driver such as timing
            reply = so.recv_obj(brian_to_driver, opts.driver_to_brian_identity, brian_print)
            meta = rxsamplesmetadata_pb2.RxSamplesMetadata()
            meta.ParseFromString(reply)

            if __debug__:
                reply_output = "Driver sent -> time {} ms, sqnum {}"
                reply_output = reply_output.format(meta.sequence_time*1e3, meta.sequence_num)
                brian_print(reply_output)

            #Requesting acknowledgement of work begins from DSP
            if __debug__:
                brian_print("Requesting work begins from DSP")
            iden = opts.dspbegin_to_brian_identity + str(meta.sequence_num)
            so.send_request(brian_to_dsp_begin, iden, "Requesting work begins")

            start_new_sock.send_string("want_to_start")

        if brian_to_radar_control in socks and socks[brian_to_radar_control] == zmq.POLLIN:

            #Get new sequence metadata from radar control
            reply = so.recv_obj(brian_to_radar_control, opts.radctrl_to_brian_identity, brian_print)

            sigp = pickle.loads(reply)

            if __debug__:
                reply_output = "Radar control sent -> sequence {} time {} ms"
                reply_output = reply_output.format(sigp.sequence_num, sigp.sequence_time)
                brian_print(reply_output)

            #Request acknowledgement of sequence from driver
            if __debug__:
                brian_print("Requesting ack from driver")
            so.send_request(brian_to_driver, opts.driver_to_brian_identity, "Requesting ack")

        if brian_to_dsp_begin in socks and socks[brian_to_dsp_begin] == zmq.POLLIN:

            #Get acknowledgement that work began in processing.
            reply = so.recv_bytes_from_any_iden(brian_to_dsp_begin)

            sig_p = pickle.loads(reply)

            if __debug__:
                reply_output = "Dsp began -> sqnum {}".format(sig_p['sequence_num'])
                brian_print(reply_output)

            #Requesting acknowledgement of work ends from DSP

            if __debug__:
                brian_print("Requesting work end from DSP")
            iden = opts.dspend_to_brian_identity + str(sig_p['sequence_num'])
            so.send_request(brian_to_dsp_end, iden, "Requesting work ends")

            #acknowledge we want to start something new.
            start_new_sock.send_string("good_to_start")


        if brian_to_dsp_end in socks and socks[brian_to_dsp_end] == zmq.POLLIN:

            #Receive ack that work finished on previous sequence.
            reply = so.recv_bytes_from_any_iden(brian_to_dsp_end)

            sig_p = pickle.loads(reply)

            if __debug__:
                reply_output = "Dsp sent -> time {}, sqnum {}"
                reply_output = reply_output.format(sig_p['kerneltime'], sig_p['sequence_num'])
                brian_print(reply_output)

            if sig_p['sequence_num'] != 0:
                if sig_p['kerneltime'] > last_processing_time:
                    late_counter += 1
                else:
                    late_counter = 0
            last_processing_time = sig_p['kerneltime']

            if __debug__:
                brian_print("Late counter {}".format(late_counter))

            #acknowledge that we are good and able to start something new.
            start_new_sock.send_string("extra_good_to_start")

def main():
    parser = argparse.ArgumentParser()
    help_msg = 'Run only the router. Do not run any of the other threads or functions.'
    parser.add_argument('--router-only', action='store_true', help=help_msg)
    args = parser.parse_args()

    opts = options.ExperimentOptions()
    threads = []
    threads.append(threading.Thread(target=router, args=(opts,)))

    if not args.router_only:
        threads.append(threading.Thread(target=sequence_timing, args=(opts,)))

    for thread in threads:
        thread.daemon = True
        thread.start()

    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()

