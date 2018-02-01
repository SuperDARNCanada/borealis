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

#import driverpacket_pb2
#import sigprocpacket_pb2v
# TODO: Socket options to look at: IDENTITY, AFFINITY, LINGER
# TODO: USE send_multipart, with human-readable identities, then the router knows how to handle the
# response to the request, see chapter 3, figures 29/30

RADAR_CONTROL_IDENTITY=b"radar_control"
DRIVER_IDENTITY=b"driver"
EXPERIMENT_HANDLER_IDENTITY=b"experiment_handler"
DSP_IDENTITY=b"dsp"
DATA_WRITE_IDENTITY=b"data_write"

DEALER_ADDRESS="tcp://127.0.0.1:6969"
ROUTER_ADDRESS="tcp://127.0.0.1:7878"


def make_zmq_connection(socket_list, req_address=ROUTER_ADDRESS, rep_address=DEALER_ADDRESS,
                        push_address=ROUTER_ADDRESS, pull_address=DEALER_ADDRESS):
    """
    Make a zmq socket connection for each socket passed in via the argument socket_list
    :param socket_list: Python list of zmq sockets
    :param req_address: String representing address to connect request sockets to
    :param rep_address: String representing address to connect reply sockets to
    :param push_address: String representing address to connect push sockets to
    :param pull_address: String representing address to connect pull sockets to
    :return: 
    """
    for socket in socket_list:
        if socket.socket_type == zmq.PUSH:
            socket.connect(push_address)
        elif socket.socket_type == zmq.PULL:
            socket.connect(pull_address)
        elif socket.socket_type == zmq.REQ:
            socket.connect(req_address)
        elif socket.socket_type == zmq.REP:
            socket.connect(rep_address)


def set_zmq_identities(socket_list, identity=None):
    """
    
    :param socket_list: 
    :param identity:
    :return: 
    """
    for socket in socket_list:
        # if socket.socket_type == zmq.PULL or socket.socket_type == zmq.REP:
        socket.setsockopt(zmq.IDENTITY, identity)


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
    context = context or zmq.Context.instance()
    sockets_list = []
    sockets_list.append(context.socket(zmq.REQ))
    sockets_list.append(context.socket(zmq.REQ))
    sockets_list.append(context.socket(zmq.REP))
    sockets_list.append(context.socket(zmq.REP))
    sockets_list.append(context.socket(zmq.PUSH))

    set_zmq_identities(sockets_list, RADAR_CONTROL_IDENTITY)
    #make_zmq_connection(sockets_list)

    radar_control_to_exp_handler = sockets_list[0]
    radar_control_to_brian = sockets_list[1]
    radar_control_to_dsp = sockets_list[2]
    radar_control_to_brian = sockets_list[3]
    radar_control_to_driver = sockets_list[4]

    radar_control_to_exp_handler.connect(DEALER_ADDRESS)

    while True:
        time.sleep(1.0)
        print("RADAR_CONTROL")
        # radar_control sends a request for an experiment to experiment_handler
        radar_control_to_exp_handler.send_multipart([EXPERIMENT_HANDLER_IDENTITY,b"", b"Requesting Experiment",])
        recv_identity, empty, reply = radar_control_to_exp_handler.recv_multipart()
        if recv_identity != RADAR_CONTROL_IDENTITY:
            print("RADAR_CONTROL: Identity {} not {}".format(recv_identity, RADAR_CONTROL_IDENTITY))
        else:
            print("RADAR_CONTROL: reply: {}".format(reply))


def experiment_handler(context=None):
    """
    Thread for experiment_handler sockets testing
    :param context: zmq context, if None, then this method will get one
    :return: 
    """
    # Reply socket to radar_control (replying with experiment)
    # Request socket to dsp (processed samples)
    context = context or zmq.Context.instance()
    sockets_list = []
    sockets_list.append(context.socket(zmq.REP))
    sockets_list.append(context.socket(zmq.REQ))

    set_zmq_identities(sockets_list, EXPERIMENT_HANDLER_IDENTITY)
    #make_zmq_connection(sockets_list)

    exp_handler_to_radar_control = sockets_list[0]
    exp_handler_to_dsp = sockets_list[1]

    exp_handler_to_radar_control.bind(DEALER_ADDRESS)

    while True:
        time.sleep(2.0)
        print("EXPERIMENT_HANDLER")
        # experiment_handler replies with an experiment to radar_control
        recv_identity, empty, request = exp_handler_to_radar_control.recv_multipart()
        if recv_identity != EXPERIMENT_HANDLER_IDENTITY:
            print("EXPERIMENT_HANDLER: Identity {} not {}".format(recv_identity, EXPERIMENT_HANDLER_IDENTITY))
        else:
            print("EXPERIMENT_HANDLER: request: {}".format(request))
        exp_handler_to_radar_control.send_multipart([RADAR_CONTROL_IDENTITY, b"", b"Giving experiment",])


def driver(context=None):
    """
    Thread for driver sockets testing
    :param context: zmq context, if None, then this method will get one
    :return: 
    """
    # Reply socket to brian (acks/timing info)
    # Reply socket to dsp (acks/timing info)
    # Pull socket from radar_control (pulses)
    context = context or zmq.Context.instance()
    sockets_list = []
    sockets_list.append(context.socket(zmq.REP))
    sockets_list.append(context.socket(zmq.REP))
    sockets_list.append(context.socket(zmq.PULL))

    set_zmq_identities(sockets_list, DRIVER_IDENTITY)
    make_zmq_connection(sockets_list)

    driver_to_brian = sockets_list[0]
    driver_to_dsp = sockets_list[1]
    driver_to_radar_control = sockets_list[2]

    while True:
        time.sleep(1.0)
        print("DRIVER")


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
    context = context or zmq.Context.instance()
    sockets_list = []
    sockets_list.append(context.socket(zmq.REQ))
    sockets_list.append(context.socket(zmq.REQ))
    sockets_list.append(context.socket(zmq.REP))
    sockets_list.append(context.socket(zmq.REP))
    sockets_list.append(context.socket(zmq.REP))

    set_zmq_identities(sockets_list, DSP_IDENTITY)
    make_zmq_connection(sockets_list)

    dsp_to_radar_control = sockets_list[0]
    dsp_to_driver = sockets_list[1]
    dsp_to_brian = sockets_list[2]
    dsp_to_experiment_handler = sockets_list[3]
    dsp_to_data_write = sockets_list[4]

    while True:
        time.sleep(1.0)
        print("DSP")


def data_write(context=None):
    """
    Thread for data_write sockets testing
    :param context: zmq context, if None, then this method will get one
    :return: 
    """
    # Request socket to dsp (processed samples)
    context = context or zmq.Context.instance()
    sockets_list = []
    sockets_list.append(context.socket(zmq.REQ))

    set_zmq_identities(sockets_list, DATA_WRITE_IDENTITY)
    make_zmq_connection(sockets_list)

    data_write_to_dsp = sockets_list[0]

    while True:
        time.sleep(1.0)
        print("DATA_WRITE")

if __name__ == "__main__":
    import threading

    print("BRIAN_TESTER: Main")

    context = zmq.Context().instance()
    #dealer = context.socket(zmq.DEALER)
    #dealer.bind(DEALER_ADDRESS)
    #router = context.socket(zmq.ROUTER)
    #router.bind(ROUTER_ADDRESS)

    threads = []

    threads.append(threading.Thread(group=None, target=radar_control, args=(context,)))
    threads.append(threading.Thread(group=None, target=experiment_handler, args=(context,)))
    #threads.append(threading.Thread(group=None, target=driver, args=(context,)))
    #threads.append(threading.Thread(group=None, target=dsp, args=(context,)))
    #threads.append(threading.Thread(group=None, target=data_write, args=(context,)))

    for thread in threads:
        thread.daemon = True
        thread.start()

    #zmq.proxy(router, dealer)
    print("BRIAN_TESTER: Threads all set up")
    while True:
        time.sleep(1)
