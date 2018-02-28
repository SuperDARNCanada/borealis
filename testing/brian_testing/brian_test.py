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

sys.path.append(os.environ["BOREALISPATH"])

if __debug__:  # TODO need to get build flavour from scons environment, 'release' may be 'debug'
        sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')
else:
        sys.path.append(os.environ["BOREALISPATH"] + '/build/release/utils/protobuf')

# import driverpacket_pb2
# import sigprocpacket_pb2
# TODO: Socket options to look at: IDENTITY, AFFINITY, LINGER
# TODO: USE send_multipart, with human-readable identities, then the router knows how to handle the
# response to the request, see chapter 3, figures 29/30

RADCTRL_EXPHAN_IDEN = b"RADCTRL_EXPHAN_IDEN"
RADCTRL_DSP_IDEN = b"RADCTRL_DSP_IDEN"
RADCTRL_DRIVER_IDEN = b"RADCTRL_DRIVER_IDEN"
RADCTRL_BRIAN_IDEN = b"RADCTRL_BRIAN_IDEN"

DRIVER_RADCTRL_IDEN = b"DRIVER_RADCTRL_IDEN"
DRIVER_DSP_IDEN = b"DRIVER_DSP_IDEN"
DRIVER_BRIAN_IDEN = b"DRIVER_BRIAN_IDEN"

EXPHAN_RADCTRL_IDEN = b"EXPHAN_RADCTRL_IDEN"
EXPHAN_DSP_IDEN = b"EXPHAN_DSP_IDEN"

DSP_RADCTRL_IDEN = b"DSP_RADCTRL_IDEN"
DSP_DRIVER_IDEN = b"DSP_DRIVER_IDEN"
DSP_EXPHAN_IDEN = b"DSP_EXPHAN_IDEN"
DSP_DW_IDEN = b"DSP_DW_IDEN"
DSP_BRIAN_IDEN = b"DSP_BRIAN_IDEN"

DW_DSP_IDEN = b"DW_DSP_IDEN"

BRIAN_RADCTRL_IDEN = b"BRIAN_RADCTRL_IDEN"
BRIAN_DRIVER_IDEN = b"BRIAN_DRIVER_IDEN"
BRIAN_DSP_IDEN = b"BRIAN_DSP_IDEN"


ROUTER_ADDRESS="tcp://127.0.0.1:7878"

def create_sockets(identities, router_addr=ROUTER_ADDRESS):
    """Gives a unique identity to a socket and then connects to the router

    :param socket: DEALER socket to get identity
    :type socket: ZMQ DEALER socket
    :param identity: Unique id for the socket
    :type identity: String
    :param router_addr: Address for router, defaults to ROUTER_ADDRESS
    :type router_addr: String, optional
    """

    context = zmq.Context().instance()
    num_sockets = len(identities)
    sockets = [context.socket(zmq.DEALER) for _ in range(num_sockets)]
    for sk, iden in zip(sockets, identities):
        sk.setsockopt(zmq.IDENTITY, iden)
        sk.connect(router_addr)

    return sockets


def recv_data(socket, sender_iden, pprint):
    recv_identity, empty, data = socket.recv_multipart()
    if recv_identity != sender_iden:
        err_msg = "Expected identity {}, received from identity {}."
        err_msg = err_msg.format(sender_iden, recv_identity)
        pprint(err_msg)
        return None
    else:
        return data

recv_reply = recv_request = recv_pulse = recv_data

def send_data(socket, recv_iden, msg):
    frames = [recv_iden, b"", b"{}".format(msg)]
    socket.send_multipart(frames)

send_reply = send_request = send_pulse = send_data

def radar_control(context=None):
    """
    Thread for radar_control sockets testing
    :param context: zmq context, if None, then this method will get one
    :return:
    """
    # Request socket to experiment handler (requests an experiment)
    # Request socket to brian (ack data)
    # Reply socket to dsp (for signal processing metadata)
    # Reply socket to brian (for signal processing metadata)
    # Push socket to driver (for pulses)

    ids = [RADCTRL_EXPHAN_IDEN, RADCTRL_DSP_IDEN, RADCTRL_DRIVER_IDEN, RADCTRL_BRIAN_IDEN]
    sockets_list = create_sockets(ids)

    radar_control_to_exp_handler = sockets_list[0]
    radar_control_to_dsp = sockets_list[1]
    radar_control_to_driver = sockets_list[2]
    radar_control_to_brian = sockets_list[3]

    def printing(msg):
        RADAR_CONTROL = "\033[33m" + "RADAR_CONTROL: " + "\033[0m"
        sys.stdout.write(RADAR_CONTROL + msg + "\n")

    time.sleep(1)
    while True:
        #time.sleep(1)
        #radar_control sends a request for an experiment to experiment_handler
        printing("Requesting experiment")
        send_request(radar_control_to_exp_handler, EXPHAN_RADCTRL_IDEN,
            "Requesting Experiment")

        # radar_control receives new experiment
        reply = recv_reply(radar_control_to_exp_handler, EXPHAN_RADCTRL_IDEN, printing)
        reply_output = "Experiment handler sent -> {}".format(reply)
        printing(reply_output)

        #Brian requests sequence metadata for timeouts
        request = recv_request(radar_control_to_brian, BRIAN_RADCTRL_IDEN, printing)
        request_output = "Brian requested -> {}".format(request)
        printing(request_output)

        send_reply(radar_control_to_brian, BRIAN_RADCTRL_IDEN, "Giving sequence metadata")

        #Radar control receives request for metadata from DSP
        request = recv_request(radar_control_to_dsp, DSP_RADCTRL_IDEN, printing)
        request_output = "DSP requested -> {}".format(request)
        printing(request_output)

        send_reply(radar_control_to_dsp, DSP_RADCTRL_IDEN, "Giving sequence metadata")


        # sending pulses to driver
        printing("Sending pulses")
        num_pulses = range(8)
        for i in num_pulses:
            if i == num_pulses[0]:
                pulse = "sob_pulse"
            elif i == num_pulses[-1]:
                pulse = "eob_pulse"
            else:
                pulse = str(i)

            send_pulse(radar_control_to_driver, DRIVER_RADCTRL_IDEN, pulse)

        # Request ack to start new sequence
        send_request(radar_control_to_brian, BRIAN_RADCTRL_IDEN, "Requesting ack")

        data = recv_data(radar_control_to_brian, BRIAN_RADCTRL_IDEN, printing)
        data_output = "Brian sent -> {}".format(data)
        printing(data_output)

def experiment_handler(context=None):
    """
    Thread for experiment_handler sockets testing
    :param context: zmq context, if None, then this method will get one
    :return:
    """
    # Reply socket to radar_control (replying with experiment)
    # Request socket to dsp (processed samples)

    ids = [EXPHAN_RADCTRL_IDEN, EXPHAN_DSP_IDEN]
    sockets_list = create_sockets(ids)

    exp_handler_to_radar_control = sockets_list[0]
    exp_handler_to_dsp = sockets_list[1]

    def printing(msg):
        EXPERIMENT_HANDLER = "\033[34m" + "EXPERIMENT HANDLER: " + "\033[0m"
        sys.stdout.write(EXPERIMENT_HANDLER + msg + "\n")

    time.sleep(1)
    while True:
        # experiment_handler replies with an experiment to radar_control
        request = recv_request(exp_handler_to_radar_control, RADCTRL_EXPHAN_IDEN, printing)
        request_msg = "Radar control made request -> {}.".format(request)
        printing(request_msg)

        # sending experiment back to radar control
        printing("Sending experiment")
        send_reply(exp_handler_to_radar_control, RADCTRL_EXPHAN_IDEN, "Giving experiment")

        # Recv complete processed data from DSP
        send_request(exp_handler_to_dsp, DSP_EXPHAN_IDEN, "Need completed data")

        data = recv_data(exp_handler_to_dsp, DSP_EXPHAN_IDEN, printing)
        data_output = "Dsp sent -> {}".format(data)
        printing(data_output)

        # time.sleep(1)

def driver(context=None):
    """
    Thread for driver sockets testing
    :param context: zmq context, if None, then this method will get one
    :return:
    """
    # Reply socket to brian (acks/timing info)
    # Reply socket to dsp (acks/timing info)
    # Pull socket from radar_control (pulses)

    ids = [DRIVER_DSP_IDEN, DRIVER_RADCTRL_IDEN, DRIVER_BRIAN_IDEN]
    sockets_list = create_sockets(ids)

    driver_to_dsp = sockets_list[0]
    driver_to_radar_control = sockets_list[1]
    driver_to_brian = sockets_list[2]

    def printing(msg):
        DRIVER = "\033[34m" + "DRIVER: " + "\033[0m"
        sys.stdout.write(DRIVER + msg + "\n")

    time.sleep(1)
    while True:

        #getting pulses from radar control
        while True:
            pulse = recv_pulse(driver_to_radar_control, RADCTRL_DRIVER_IDEN, printing)
            printing("Received pulse {}".format(pulse))
            if pulse == "eob_pulse":
                break

        time.sleep(1)

        #sending sequence data to dsp
        request = recv_request(driver_to_dsp, DSP_DRIVER_IDEN, printing)
        request_output = "Dsp sent -> {}".format(request)
        printing(request_output)

        send_reply(driver_to_dsp, DSP_DRIVER_IDEN, "Completed sequence data")

        #sending collected data to brian
        request = recv_request(driver_to_brian, BRIAN_DRIVER_IDEN, printing)
        request_output = "Brian sent -> {}".format(request)
        printing(request_output)

        send_reply(driver_to_brian, BRIAN_DRIVER_IDEN, "Completed sequence data")



def dsp(context=None):
    """
    Thread for dsp sockets testing
    :param context: zmq context, if None, then this method will get one
    :return:
    """

    # Sockets to radar_control, driver and data_write
    # Request socket to radar_control (signal processing metadata)
    # Request socket to driver (acks/timing info)
    # Reply socket to brian (acks/work beginning/ending)
    # Reply socket to experiment handler (processed samples)
    # Reply socket to data_write (processed samples)

    ids = [DSP_RADCTRL_IDEN, DSP_DRIVER_IDEN, DSP_EXPHAN_IDEN, DSP_DW_IDEN, DSP_BRIAN_IDEN]
    sockets_list = create_sockets(ids)

    dsp_to_radar_control = sockets_list[0]
    dsp_to_driver = sockets_list[1]
    dsp_to_experiment_handler = sockets_list[2]
    dsp_to_data_write = sockets_list[3]
    dsp_to_brian = sockets_list[4]


    def printing(msg):
        DSP = "\033[35m" + "DSP: " + "\033[0m"
        sys.stdout.write(DSP + msg + "\n")

    time.sleep(1)
    while True:

        printing("Requesting metadata from radar control")
        # DSP makes a request for new sequence processing metadata to radar control
        send_request(dsp_to_radar_control, RADCTRL_DSP_IDEN,"Need metadata")

        # Radar control sends back metadata
        reply = recv_reply(dsp_to_radar_control, RADCTRL_DSP_IDEN, printing)
        reply_output = "Radar control sent -> {}".format(reply)
        printing(reply_output)


        # request data from driver
        send_request(dsp_to_driver, DRIVER_DSP_IDEN, "Need data to process")

        reply = recv_reply(dsp_to_driver, DRIVER_DSP_IDEN, printing)
        reply_output = "Driver sent -> {}".format(reply)
        printing(reply_output)

        # Copy samples to device
        time.sleep(0.1)

        # acknowledge start of work
        request = recv_request(dsp_to_brian, BRIAN_DSP_IDEN, printing)
        request_output = "Brian sent -> {}".format(request)
        printing(request_output)
        send_data(dsp_to_brian, BRIAN_DSP_IDEN, "Ack start of work")

        # doing work!
        time.sleep(1)

        # acknowledge end of work
        request = recv_request(dsp_to_brian, BRIAN_DSP_IDEN, printing)
        request_output = "Brian sent -> {}".format(request)
        printing(request_output)
        send_data(dsp_to_brian, BRIAN_DSP_IDEN, "Ack end of work")

        # send data to experiment handler
        request = recv_request(dsp_to_experiment_handler, EXPHAN_DSP_IDEN, printing)
        request_output = "Experiment handler sent -> {}".format(request)
        printing(request_output)

        send_data(dsp_to_experiment_handler, EXPHAN_DSP_IDEN, "All the datas")

        # send data to data write
        request = recv_request(dsp_to_data_write, DW_DSP_IDEN, printing)
        request_output = "Data write sent -> {}".format(request)
        printing(request_output)

        send_data(dsp_to_data_write, DW_DSP_IDEN, "All the datas")




def data_write(context=None):
    """
    Thread for data_write sockets testing
    :param context: zmq context, if None, then this method will get one
    :return:
    """
    # Request socket to dsp (processed samples)

    ids = [DW_DSP_IDEN]
    sockets_list = create_sockets(ids)

    data_write_to_dsp = sockets_list[0]

    def printing(msg):
        DATA_WRITE = "\033[32m" + "DATA WRITE: " + "\033[0m"
        sys.stdout.write(DATA_WRITE + msg + "\n")

    time.sleep(1)
    while True:
        # Request processed data
        send_request(data_write_to_dsp, DSP_DW_IDEN, "Requesting processed data")

        data = recv_data(data_write_to_dsp, DSP_DW_IDEN, printing)
        data_output = "Dsp sent -> {}".format(data)
        printing(data_output)


def sequence_timing():

    ids = [BRIAN_RADCTRL_IDEN, BRIAN_DRIVER_IDEN, BRIAN_DSP_IDEN]
    sockets_list = create_sockets(ids)

    brian_to_radar_control = sockets_list[0]
    brian_to_driver = sockets_list[1]
    brian_to_dsp = sockets_list[2]

    def printing(msg):
        SEQUENCE_TIMING = "\033[31m" + "SEQUENCE TIMING: " + "\033[0m"
        sys.stdout.write(SEQUENCE_TIMING + msg + "\n")

    time.sleep(1)

    while True:

        #Request new sequence metadata
        printing("Requesting metadata from Radar control")
        send_request(brian_to_radar_control, RADCTRL_BRIAN_IDEN, "Requesting metadata")

        reply = recv_reply(brian_to_radar_control, RADCTRL_BRIAN_IDEN, printing)
        reply_output = "Radar control sent -> {}".format(reply)
        printing(reply_output)

        #Request acknowledgement of sequence from driver
        printing("Requesting ack from driver")
        send_request(brian_to_driver, DRIVER_BRIAN_IDEN, "Requesting ack")

        reply = recv_reply(brian_to_driver, DRIVER_BRIAN_IDEN, printing)
        reply_output = "Driver sent -> {}".format(reply)
        printing(reply_output)

        #Requesting acknowledgement of work begins from DSP
        printing("Requesting work begins from DSP")
        send_request(brian_to_dsp, DSP_BRIAN_IDEN, "Requesting work begins")

        reply = recv_reply(brian_to_dsp, DSP_BRIAN_IDEN, printing)
        reply_output = "Dsp sent -> {}".format(reply)
        printing(reply_output)

        #Requesting acknowledgement of work ends from DSP
        printing("Requesting work end from DSP")
        send_request(brian_to_dsp, DSP_BRIAN_IDEN, "Requesting work ends")

        reply = recv_reply(brian_to_dsp, DSP_BRIAN_IDEN, printing)
        reply_output = "Dsp sent -> {}".format(reply)
        printing(reply_output)


        #Acknowledge new sequence can begin to Radar Control
        request = recv_request(brian_to_radar_control, RADCTRL_BRIAN_IDEN, printing)
        request_output = "Radar control sent -> {}".format(request)
        printing(request_output)

        send_reply(brian_to_radar_control, RADCTRL_BRIAN_IDEN, "Begin new sequence")



def router():
    context = zmq.Context().instance()
    router = context.socket(zmq.ROUTER)
    router.bind(ROUTER_ADDRESS)

    sys.stdout.write("Starting router!\n")
    while True:
        dd = router.recv_multipart()
        #sys.stdout.write(dd)
        sender, receiver, empty, data = dd
        output = "Router input/// Sender -> {}: Receiver -> {}: empty: Data -> {}\n".format(*dd)
        #sys.stdout.write(output)
        frames = [receiver,sender,empty,data]
        output = "Router output/// Receiver -> {}: Sender -> {}: empty: Data -> {}\n".format(*frames)
        #sys.stdout.write(output)
        router.send_multipart(frames)


if __name__ == "__main__":
    import threading

    sys.stdout.write("BRIAN_TESTER: Main\n")

    threads = []

    threads.append(threading.Thread(target=router))
    time.sleep(0.5)
    threads.append(threading.Thread(target=radar_control))
    threads.append(threading.Thread(target=experiment_handler))
    threads.append(threading.Thread(target=sequence_timing))
    threads.append(threading.Thread(target=driver))
    threads.append(threading.Thread(target=dsp))
    threads.append(threading.Thread(target=data_write))


    for thread in threads:
        thread.daemon = True
        thread.start()

    sys.stdout.write("BRIAN_TESTER: Threads all set up\n")
    while True:
        time.sleep(1)
