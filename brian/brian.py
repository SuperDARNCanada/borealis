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

def router(opts):
    context = zmq.Context().instance()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.ROUTER_MANDATORY, 1)
    router.bind(opts.router_address)

    sys.stdout.write("Starting router!\n")
    while True:
        dd = router.recv_multipart()
        #sys.stdout.write(dd)
        sender, receiver, empty, data = dd
        output = "Router input/// Sender -> {}: Receiver -> {}\n".format(sender, receiver) #: empty: {} Data -> {}\n".format(*dd)
        sys.stdout.write(output)
        frames = [receiver,sender,empty,data]
        output = "Router output/// Receiver -> {}: Sender -> {}\n".format(receiver, sender) #: empty: {} Data -> {}\n".format(*frames)
        sys.stdout.write(output)
        sent = False
        while not sent:
            try:
                router.send_multipart(frames)
                sent = True
            except zmq.ZMQError as e:
                sys.stdout.write("Trying to send \n")
                time.sleep(0.5)

late_counter = 0
def sequence_timing(opts):
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
        while True:

            if want_to_start and good_to_start and dsp_finish_counter:
                #Acknowledge new sequence can begin to Radar Control by requesting new sequence
                #metadata
                printing("Requesting metadata from Radar control")
                so.send_request(brian_to_radar_control, opts.radctrl_to_brian_identity, "Requesting metadata")
                want_to_start = good_to_start = False
                dsp_finish_counter -= 1

            message = start_new.recv_string()
            #print("message: {}".format(message))
            if message == "want_to_start":
                want_to_start = True

            if message == "good_to_start":
                good_to_start = True

            if message == "extra_good_to_start":
                dsp_finish_counter = 1;

    thread = threading.Thread(target=start_new)
    thread.daemon = True
    thread.start()

    time.sleep(1)

    pulse_seq_times = {}
    driver_times = {}
    processing_times = {}

    first_time = True
    processing_done = True

    while True:

        if first_time:
            #Request new sequence metadata
            printing("Requesting metadata from Radar control")
            so.send_request(brian_to_radar_control, opts.radctrl_to_brian_identity, "Requesting metadata")
            first_time = False

        socks = dict(sequence_poller.poll())

        if brian_to_driver in socks and socks[brian_to_driver] == zmq.POLLIN:

            #Receive metadata of completed sequence from driver such as timing
            reply = so.recv_obj(brian_to_driver, opts.driver_to_brian_identity, printing)
            meta = rxsamplesmetadata_pb2.RxSamplesMetadata()
            meta.ParseFromString(reply)
            reply_output = "Driver sent -> time {} ms, sqnum {}".format(meta.sequence_time*1e3, meta.sequence_num)
            printing(reply_output)

            driver_times[meta.sequence_num] = meta.sequence_time

            #Requesting acknowledgement of work begins from DSP
            printing("Requesting work begins from DSP")
            so.send_request(brian_to_dsp_begin, opts.dspbegin_to_brian_identity, "Requesting work begins")
            
            #acknowledge we want to start something new
            start_new_sock.send_string("want_to_start")
            
        if brian_to_radar_control in socks and socks[brian_to_radar_control] == zmq.POLLIN:

            #Get new sequence metadata from radar control
            reply = so.recv_obj(brian_to_radar_control, opts.radctrl_to_brian_identity, printing)

            sigp = sigprocpacket_pb2.SigProcPacket()
            sigp.ParseFromString(reply)
            reply_output = "Radar control sent -> sequence {} time {} ms".format(sigp.sequence_num,
                                                                              sigp.sequence_time)
            printing(reply_output)

            pulse_seq_times[sigp.sequence_num] = sigp.sequence_time

            #Request acknowledgement of sequence from driver
            printing("Requesting ack from driver")
            so.send_request(brian_to_driver, opts.driver_to_brian_identity, "Requesting ack")

        if brian_to_dsp_begin in socks and socks[brian_to_dsp_begin] == zmq.POLLIN:

            #def dspb_f():
                #Get acknowledgement that work began in processing.
            reply = so.recv_obj(brian_to_dsp_begin, opts.dspbegin_to_brian_identity, printing)
            sig_p = sigprocpacket_pb2.SigProcPacket()
            sig_p.ParseFromString(reply)
            reply_output = "Dsp began -> sqnum {}".format(sig_p.sequence_num)
            printing(reply_output)

            #Requesting acknowledgement of work ends from DSP
            printing("Requesting work end from DSP")
            so.send_request(brian_to_dsp_end, opts.dspend_to_brian_identity, "Requesting work ends")
            
            #acknowledge that we are good and able to start something new
            start_new_sock.send_string("good_to_start")

            # dspb_t = threading.Thread(target=dspb_f)
            # dspb_t.daemon = True
            # dspb_t.start()


        if brian_to_dsp_end in socks and socks[brian_to_dsp_end] == zmq.POLLIN:
            #def dspe_f():
            global late_counter
            #Receive ack that work finished on previous sequence.
            reply = so.recv_obj(brian_to_dsp_end, opts.dspend_to_brian_identity, printing)

            sig_p = sigprocpacket_pb2.SigProcPacket()
            sig_p.ParseFromString(reply)
            reply_output = "Dsp sent -> time {}, sqnum {}".format(sig_p.kerneltime, sig_p.sequence_num)
            printing(reply_output)

            processing_times[sig_p.sequence_num] = sig_p.kerneltime
            if sig_p.sequence_num != 0:
                if sig_p.kerneltime > processing_times[sig_p.sequence_num-1]:
                    late_counter +=1
                else:
                    late_counter = 0
            printing("Late counter {}".format(late_counter))

            #acknowledge that we are good and able to start something new
            start_new_sock.send_string("extra_good_to_start")

            # dspe_t = threading.Thread(target=dspe_f)
            # dspe_t.daemon = True
            # dspe_t.start()
if __name__ == "__main__":

    opts = options.ExperimentOptions()
    threads = []
    threads.append(threading.Thread(target=router,args=(opts,)))
    threads.append(threading.Thread(target=sequence_timing,args=(opts,)))

    for thread in threads:
        thread.daemon = True
        thread.start()

    while True:
        time.sleep(1)
