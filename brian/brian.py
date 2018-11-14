#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# brian.py
# 2018-01-30
# Communicate with all processes to administrate the borealis software

import sys
import os
import zmq
import time
from datetime import datetime, timedelta
import threading

sys.path.append(os.environ["BOREALISPATH"])

if __debug__:
	sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')  # TODO need to get this from scons environment, 'release' may be 'debug'
else:
	sys.path.append(os.environ["BOREALISPATH"] + '/build/release/utils/protobuf')
import driverpacket_pb2
import sigprocpacket_pb2
import rxsamplesmetadata_pb2
import processeddata_pb2
sys.path.append(os.environ["BOREALISPATH"] + '/utils/experiment_options')
import experimentoptions as options

sys.path.append(os.environ["BOREALISPATH"] + '/utils/zmq_borealis_helpers')
import socket_operations as so

# REVIEW TODO: Need to fill out docstring here
def router(opts):  # REVIEW #26 The function name and the name of the router socket are the same 'router'
    context = zmq.Context().instance()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.ROUTER_MANDATORY, 1)
    router.bind(opts.router_address)

    sys.stdout.write("Starting router!\n")
    while True:
        dd = router.recv_multipart()
        sender, receiver, empty, data = dd  # REVIEW This line's effects were not obvious to me - I assumed only data would be set to anything
        output = "Router input/// Sender -> {}: Receiver -> {}: empty: Data -> {}\n".format(*dd)
        sys.stdout.write(output)  # REVIEW Should this be a debug output?
        frames = [receiver, sender, empty, data]
        output = "Router output/// Receiver -> {}: Sender -> {}: empty: Data -> {}\n".format(*frames)
        sys.stdout.write(output)  # REVIEW Should this be a debug output?
        sent = False
        while not sent:
            try:
                router.send_multipart(frames)
                sent = True
            except zmq.ZMQError as e:
                sys.stdout.write("Trying to send \n")
                time.sleep(0.5)


def sequence_timing(opts):  # REVIEW Why does the docstring say this simulates flow of data? It looks like it actually passes data between blocks, Also, the docstring params list is wrong
    """Thread function for sequence timing

    This function simulates the flow of data between brian's sequence timing and other parts of the
    radar system. This function serves to check whether the sequence timing is working as expected
    and to rate control the system to make sure the processing can handle data rates.
    :param context: zmq context, if None, then this method will get one
    :type context: zmq context, optional
    """


    #ids = [BRIAN_RADCTRL_IDEN, BRIAN_DRIVER_IDEN, BRIAN_DSPBEGIN_IDEN, BRIAN_DSPEND_IDEN]

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

    def printing(msg):
        SEQUENCE_TIMING = "\033[31m" + "SEQUENCE TIMING: " + "\033[0m"
        sys.stdout.write(SEQUENCE_TIMING + msg + "\n")

    context = zmq.Context().instance()

    start_new_sock = context.socket(zmq.PAIR)
    start_new_sock.bind("inproc://start_new")

    def start_new():  # REVIEW #26 The function name and the name of the pair socket are the same 'start_new'
        """ This function serves to rate control the system. If processing is faster than the
        sequence time then the speed of the driver is the limiting factor. If processing takes
        longer than sequence time, then the dsp unit limits the speed of the system.
        """
        start_new = context.socket(zmq.PAIR)
        start_new.connect("inproc://start_new")  # REVIEW #29 magic string

        want_to_start = False
        good_to_start = True
        while True:

            if want_to_start and good_to_start:
                # Acknowledge new sequence can begin to Radar Control by requesting new sequence
                # metadata
                printing("Requesting metadata from Radar control")
                so.send_request(brian_to_radar_control, opts.radctrl_to_brian_identity, "Requesting metadata")  # REVIEW #29 magic string
                want_to_start = good_to_start = False

            message = start_new.recv()
            if message == "want_to_start":  # REVIEW #29 magic string
                want_to_start = True

            if message == "good_to_start":  # REVIEW #29 magic string
                good_to_start = True

    thread = threading.Thread(target=start_new)
    thread.daemon = True
    thread.start()

    time.sleep(1) # REVIEW #23 Why wait?

    pulse_seq_times = {}
    driver_times = {}
    processing_times = {}

    first_time = True
    processing_done = True  # REVIEW #42 this boolean is not used
    late_counter = 0
    while True:

        if first_time:
            # Request new sequence metadata
            printing("Requesting metadata from Radar control")
            so.send_request(brian_to_radar_control, opts.radctrl_to_brian_identity, "Requesting metadata") # REVIEW #29 magic string
            first_time = False

        socks = dict(sequence_poller.poll())
        if brian_to_radar_control in socks and socks[brian_to_radar_control] == zmq.POLLIN:

            # Get new sequence metadata from radar control
            reply = so.recv_reply(brian_to_radar_control, opts.radctrl_to_brian_identity, printing)

            sigp = sigprocpacket_pb2.SigProcPacket()
            sigp.ParseFromString(reply)
            reply_output = "Radar control sent -> sequence {} time {}".format(sigp.sequence_num,
                                                                              sigp.sequence_time)
            printing(reply_output)

            pulse_seq_times[sigp.sequence_num] = sigp.sequence_time

            # Request acknowledgement of sequence from driver
            printing("Requesting ack from driver")
            so.send_request(brian_to_driver, opts.driver_to_brian_identity, "Requesting ack") # REVIEW #29 magic string

        if brian_to_driver in socks and socks[brian_to_driver] == zmq.POLLIN:

            # Receive metadata of completed sequence from driver such as timing
            reply = so.recv_reply(brian_to_driver, opts.driver_to_brian_identity, printing)
            meta = rxsamplesmetadata_pb2.RxSamplesMetadata()
            meta.ParseFromString(reply)
            reply_output = "Driver sent -> time {}, sqnum {}".format(meta.sequence_time,
                                                                     meta.sequence_num)
            printing(reply_output)

            driver_times[meta.sequence_num] = meta.sequence_time

            # Requesting acknowledgement of work begins from DSP
            printing("Requesting work begins from DSP")
            so.send_request(brian_to_dsp_begin, opts.dspbegin_to_brian_identity, "Requesting work begins") # REVIEW #29 magic string

        if brian_to_dsp_begin in socks and socks[brian_to_dsp_begin] == zmq.POLLIN:

            # Get acknowledgement that work began in processing.
            reply = so.recv_reply(brian_to_dsp_begin, opts.dspbegin_to_brian_identity, printing)
            reply_output = "Dsp sent -> {}".format(reply)
            printing(reply_output)

            # Requesting acknowledgement of work ends from DSP
            printing("Requesting work end from DSP")
            so.send_request(brian_to_dsp_end, opts.dspend_to_brian_identity, "Requesting work ends") # REVIEW #29 magic string

            # acknowledge we want to start something new
            start_new_sock.send("want_to_start") # REVIEW #29 magic string

        if brian_to_dsp_end in socks and socks[brian_to_dsp_end] == zmq.POLLIN:

            # Receive ack that work finished on previous sequence.
            reply = so.recv_reply(brian_to_dsp_end, opts.dspend_to_brian_identity, printing)

            proc_d = processeddata_pb2.ProcessedData()
            proc_d.ParseFromString(reply)
            reply_output = "Dsp sent -> time {}, sqnum {}".format(proc_d.processing_time,
                                                                  proc_d.sequence_num)
            printing(reply_output)

            processing_times[proc_d.sequence_num] = proc_d.processing_time
            if proc_d.sequence_num != 0:  # REVIEW Maybe instead of waiting for good to start, we just keep an eye on late counter and keep it below a certain threshold?
                if proc_d.processing_time > processing_times[proc_d.sequence_num-1]:
                    late_counter += 1
                else:
                    late_counter = 0
            printing("Late counter {}".format(late_counter))  # REVIEW TODO: Something about late counter if it keeps increasing beyond a certain point?

            # acknowledge that we are good and able to start something new
            start_new_sock.send("good_to_start")  # REVIEW #29 magic string

if __name__ == "__main__":

    opts = options.ExperimentOptions()
    threads = []
    threads.append(threading.Thread(target=router,args=(opts,)))
    threads.append(threading.Thread(target=sequence_timing,args=(opts,)))

    for thread in threads:
        thread.daemon = True
        thread.start()

    while True: # REVIEW Does this need to be here? does it just take up another thread?
        time.sleep(1)
