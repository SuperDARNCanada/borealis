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

RADCTRL_EXP_IDEN = b"RADCTRL_EXP_IDEN"
RADCTRL_DSP_IDEN = b"RADCTRL_DSP_IDEN"
RADCTRL_DRIVER_IDEN = b"RADCTRL_DRIVER_IDEN"

DRIVER_RADCTRL_IDEN = b"DRIVER_RADCTRL_IDEN"
DRIVER_DSP_IDEN = b"DRIVER_DSP_IDEN"

EXPHAN_RADCTRL_IDEN = b"EXPHAN_RADCTRL_IDEN"
EXPHAN_DSP_IDEN = b"EXPHAN_DSP_IDEN"

DSP_RADCTRL_IDEN = b"DSP_RADCTRL_IDEN"
DSP_DRIVER_IDEN = b"DSP_DRIVER_IDEN"
DSP_EXPHAN_IDEN = b"DSP_EXPHAN_IDEN"

DW_DSP_IDEN = b"DW_DSP_IDEN"
DSP_DW_IDEN = b"DSP_DW_IDEN"

ROUTER_ADDRESS="tcp://127.0.0.1:7878"
DEALER_ADDRESS = "tcp://127.0.0:6969"

def make_zmq_connection(socket_list,req_address=ROUTER_ADDRESS, rep_address=DEALER_ADDRESS,
                        push_address=ROUTER_ADDRESS, pull_address=ROUTER_ADDRESS):
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
        elif socket.socket_type == zmq.DEALER:
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

# def receive_request(socket, sender_iden, recv_iden, pprint):
#     recv_identity, empty, request = socket.recv_multipart()
#     if recv_identity != sender_iden:
#         err_msg = "Expected identity {}, received from identity {}."
#         err_msg = err_msg.format(sender_iden, recv_identity)
#         pprint(err_msg)
#         return None
#     else:
#         return request


def send_data(socket, recv_iden, msg):
    frames = [recv_iden, b"", b"{}".format(msg)]
    socket.send_multipart(frames)

send_reply = send_request = send_pulse = send_data
# def send_reply(socket, recv_iden, msg):
#     frames = [recv_iden, b"", b"{}".format(msg)]
#     socket.send_multipart(frames)


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
    #
    context = context or zmq.Context.instance()
    sockets_list = []
    sockets_list.append(context.socket(zmq.DEALER))
    # # sockets_list.append(context.socket(zmq.REQ))
    sockets_list.append(context.socket(zmq.DEALER))
    # # sockets_list.append(context.socket(zmq.REP))
    sockets_list.append(context.socket(zmq.DEALER))


    radar_control_to_exp_handler = sockets_list[0]
    radar_control_to_exp_handler.setsockopt(zmq.IDENTITY,RADCTRL_EXP_IDEN)
    # radar_control_to_brian = sockets_list[1]
    radar_control_to_dsp = sockets_list[1]
    radar_control_to_dsp.setsockopt(zmq.IDENTITY,RADCTRL_DSP_IDEN)
    # radar_control_to_brian = sockets_list[3]
    radar_control_to_driver = sockets_list[2]
    radar_control_to_driver.setsockopt(zmq.IDENTITY, RADCTRL_DRIVER_IDEN)
    make_zmq_connection(sockets_list)

    def printing(msg):
        RADAR_CONTROL = "\033[33m" + "RADAR_CONTROL: " + "\033[0m"
        print(RADAR_CONTROL + msg)

    # def rad_recv_reply(socket, sender_iden):
    #     return receive_reply(socket, sender_iden, RADAR_CONTROL_IDENTITY, printing)

    # def rad_recv_request(socket, sender_iden):
    #     return receive_reply(socket, sender_iden, RADAR_CONTROL_IDENTITY, printing)

    while True:
        #radar_control sends a request for an experiment to experiment_handler
        printing("Requesting experiment")
        send_request(radar_control_to_exp_handler, EXPHAN_RADCTRL_IDEN,
            "Requesting Experiment")

        # radar_control receives new experiment
        reply = recv_reply(radar_control_to_exp_handler, EXPHAN_RADCTRL_IDEN, printing)
        reply_output = "Experiment handler sent -> {}".format(reply)
        printing(reply_output)

        #Radar control receives request for metadata from DSP
        request = recv_request(radar_control_to_dsp, DSP_RADCTRL_IDEN, printing)
        request_output = "DSP requested -> {}".format(request)
        printing(request_output)

        send_reply(radar_control_to_dsp, DSP_RADCTRL_IDEN, "Giving metadata")

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


        # Get ack from driver
        data = recv_data(radar_control_to_driver, DRIVER_RADCTRL_IDEN, printing)
        data_output = "Driver sent -> {}".format(data)
        printing(data_output)

        # Get copy ack from dsp
        data = recv_data(radar_control_to_dsp, DSP_RADCTRL_IDEN, printing)
        data_output = "Dsp sent -> {}".format(data)
        printing(data_output)

        # Get completed processing ack from dsp
        data = recv_data(radar_control_to_dsp, DSP_RADCTRL_IDEN, printing)
        data_output = "Dsp sent -> {}".format(data)
        printing(data_output)

        #time.sleep(1)


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
    sockets_list.append(context.socket(zmq.DEALER))
    sockets_list.append(context.socket(zmq.DEALER))


    exp_handler_to_radar_control = sockets_list[0]
    exp_handler_to_radar_control.setsockopt(zmq.IDENTITY, EXPHAN_RADCTRL_IDEN)

    exp_handler_to_dsp = sockets_list[1]
    exp_handler_to_dsp.setsockopt(zmq.IDENTITY, EXPHAN_DSP_IDEN)


    make_zmq_connection(sockets_list)
    def printing(msg):
        EXPERIMENT_HANDLER = "\033[34m" + "EXPERIMENT HANDLER: " + "\033[0m"
        print(EXPERIMENT_HANDLER + msg)

    # def exp_recv_request(socket, sender_iden):
    #     return receive_request(socket, sender_iden, EXPERIMENT_HANDLER_IDENTITY, printing)

    while True:
        # experiment_handler replies with an experiment to radar_control
        request = recv_request(exp_handler_to_radar_control, RADCTRL_EXP_IDEN, printing)
        output_msg = "Radar control made request -> {}.".format(request)
        printing(output_msg)

        # sending experiment back to radar control
        printing("Sending experiment")
        send_reply(exp_handler_to_radar_control, RADCTRL_EXP_IDEN, "Giving experiment")

        # Recv complete processed data from DSP
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
    context = context or zmq.Context.instance()
    sockets_list = []
    # sockets_list.append(context.socket(zmq.REP))
    sockets_list.append(context.socket(zmq.DEALER))
    sockets_list.append(context.socket(zmq.DEALER))



    # driver_to_brian = sockets_list[0]
    driver_to_dsp = sockets_list[0]
    driver_to_dsp.setsockopt(zmq.IDENTITY, DRIVER_DSP_IDEN)

    driver_to_radar_control = sockets_list[1]
    driver_to_radar_control.setsockopt(zmq.IDENTITY, DRIVER_RADCTRL_IDEN)

    make_zmq_connection(sockets_list)

    def printing(msg):
        DRIVER = "\033[34m" + "DRIVER: " + "\033[0m"
        print(DRIVER + msg)

    while True:

        #getting pulses from radar control
        while True:
            pulse = recv_pulse(driver_to_radar_control, RADCTRL_DRIVER_IDEN, printing)
            printing("Received pulse {}".format(pulse))
            if pulse == "eob_pulse":
                break


        time.sleep(1)

        #sending collected data to dsp
        printing("Sending completed data to dsp")
        send_data(driver_to_dsp, DSP_DRIVER_IDEN, "Completed sequence data")

        # send ack to radar control
        printing("Sending ack to radar control")
        send_data(driver_to_radar_control, RADCTRL_DRIVER_IDEN, "Ack completed sequence")








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
    #
    context = context or zmq.Context.instance()
    sockets_list = []
    sockets_list.append(context.socket(zmq.DEALER))
    sockets_list.append(context.socket(zmq.DEALER))
    sockets_list.append(context.socket(zmq.DEALER))
    sockets_list.append(context.socket(zmq.DEALER))
    # sockets_list.append(context.socket(zmq.REP))

    #set_zmq_identities(sockets_list, DSP_IDENTITY)


    dsp_to_radar_control = sockets_list[0]
    dsp_to_radar_control.setsockopt(zmq.IDENTITY, DSP_RADCTRL_IDEN)
    dsp_to_driver = sockets_list[1]
    dsp_to_driver.setsockopt(zmq.IDENTITY, DSP_DRIVER_IDEN)
    # dsp_to_brian = sockets_list[2]
    dsp_to_experiment_handler = sockets_list[2]
    dsp_to_experiment_handler.setsockopt(zmq.IDENTITY, DSP_EXPHAN_IDEN)
    dsp_to_data_write = sockets_list[3]
    dsp_to_data_write.setsockopt(zmq.IDENTITY, DSP_DW_IDEN)

    make_zmq_connection(sockets_list)

    def printing(msg):
        DSP = "\033[35m" + "DSP: " + "\033[0m"
        print(DSP + msg)

    def dsp_recv_reply(socket,sender_iden):
        return receive_reply(socket, sender_iden, DSP_IDENTITY, printing)

    while True:

        printing("Requesting metadata from radar control")
        # DSP makes a request for new sequence processing metadata to radar control
        send_request(dsp_to_radar_control, RADCTRL_DSP_IDEN,"Need metadata")

        # Radar control sends back metadata
        reply = recv_reply(dsp_to_radar_control, RADCTRL_DSP_IDEN, printing)
        reply_output = "Radar control sent -> {}".format(reply)
        printing(reply_output)


        data = recv_data(dsp_to_driver, DRIVER_DSP_IDEN, printing)
        data_output = "Driver sent -> {}".format(data)
        printing(data_output)

        # Copy samples to device
        time.sleep(0.1)

        # acknowledge copy and ready to begin collecting new samples
        printing("Sending copy ack to radar control")
        send_data(dsp_to_radar_control, RADCTRL_DSP_IDEN, "Ack copy samples")

        # doing work!
        time.sleep(1)
        printing("Sending completed processing ack")
        send_data(dsp_to_radar_control, RADCTRL_DSP_IDEN, "Ack completed processing")

        # send data to experiment handler
        printing("Sending data to experiment handler")
        send_data(dsp_to_experiment_handler, EXPHAN_DSP_IDEN, "All the datas")

        # send data to data write
        printing("Sending data to data write")
        send_data(dsp_to_data_write, DW_DSP_IDEN, "All the datas")




def data_write(context=None):
    """
    Thread for data_write sockets testing
    :param context: zmq context, if None, then this method will get one
    :return:
    """
    # Request socket to dsp (processed samples)
    context = context or zmq.Context.instance()
    sockets_list = []
    sockets_list.append(context.socket(zmq.DEALER))

    #set_zmq_identities(sockets_list, DATA_WRITE_IDENTITY)

    data_write_to_dsp = sockets_list[0]
    data_write_to_dsp.setsockopt(zmq.IDENTITY, DW_DSP_IDEN)
    make_zmq_connection(sockets_list)

    def printing(msg):
        DATA_WRITE = "\033[32m" + "DATA WRITE: " + "\033[0m"
        print(DATA_WRITE + msg)

    while True:
        # Receive processed data
        data = recv_data(data_write_to_dsp, DSP_DW_IDEN, printing)
        data_output = "Dsp sent -> {}".format(data)
        printing(data_output)








if __name__ == "__main__":
    import threading

    print("BRIAN_TESTER: Main")

    context = zmq.Context().instance()
    router = context.socket(zmq.ROUTER)
    router.bind(ROUTER_ADDRESS)

    threads = []

    threads.append(threading.Thread(target=radar_control))
    threads.append(threading.Thread(target=experiment_handler))
    threads.append(threading.Thread(target=driver))
    threads.append(threading.Thread(target=dsp))
    threads.append(threading.Thread(target=data_write))
    #threads.append(threading.Thread(target=monitor, args=(context,)))

    for thread in threads:
        thread.daemon = True
        thread.start()


    print("BRIAN_TESTER: Threads all set up")
    # zmq.proxy(router, dealer,capture)
    # while True:
    #     time.sleep(1)
    time.sleep(0.1)
    while True:
        dd = router.recv_multipart()
        #print(dd)
        sender, receiver, empty, data = dd
        #print(sender, receiver, empty, data)
        frames = [receiver,sender,empty,data]
        #print(frames)
        router.send_multipart(frames)