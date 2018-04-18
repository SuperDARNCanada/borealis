#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#

import zmq
import os
import sys

sys.path.append(os.environ['BOREALISPATH'])

from experiments import one_box_test_experiment
from radar_status.radar_status import RadarStatus
from utils.experiment_options.experimentoptions import ExperimentOptions
from utils.zmq_borealis_helpers import socket_operations
import threading


def printing(msg):
    EXPERIMENT_HANDLER = "\033[34m" + "EXPERIMENT HANDLER: " + "\033[0m"
    sys.stdout.write(EXPERIMENT_HANDLER + msg + "\n")


def experiment_handler(semaphore):
    
    # setup two sockets - one to get ACF data and
    # another to talk to runradar.
    options = ExperimentOptions()
    ids = [options.exphan_to_radctrl_identity, options.exphan_to_dsp_identity]
    sockets_list = socket_operations.create_sockets(ids)

    exp_handler_to_radar_control = sockets_list[0]
    exp_handler_to_dsp = sockets_list[1]

    # TODO: change line to be scheduled
    exp = one_box_test_experiment.OneBox()
    change_flag = False

    def update_experiment():
        # Recv complete processed data from DSP
        socket_operations.send_request(exp_handler_to_dsp,
                                       options.dsp_to_exphan_identity,
                                       "Need completed data")

        data = socket_operations.recv_data(exp_handler_to_dsp,
                                           options.dsp_to_exphan_identity, printing)

        # TODO merge with marci-docs branch and use inspect library to find update or not.
        some_data = None  # TODO get the data from data socket and pass to update

        semaphore.acquire()
        change_flag = exp.update(some_data)
        if change_flag:
            exp.build_scans()
        semaphore.release()

        if __debug__:
            data_output = "Dsp sent -> {}".format(data)
            printing(data_output)

    thread = threading.Thread(target=update_experiment)
    thread.daemon = True
    thread.start()

    while True:

        # WAIT until radar_control is ready to receive a changed experiment
        message = socket_operations.recv_request(exp_handler_to_radar_control,
                                                 options.radctrl_to_exphan_identity,
                                                 printing)
        if __debug__:
            request_msg = "Radar control made request -> {}.".format(message)
            printing(request_msg)

        semaphore.acquire()
        if message == 'EXPNEEDED':
            printing("Sending new experiment from beginning")
            # starting anew
            exp.build_scans()
            try:
                socket_operations.send_reply(exp_handler_to_radar_control,
                                             options.radctrl_to_exphan_identity, exp)
            except zmq.ZMQError: # the queue was full - radarcontrol not receiving.
                pass  #TODO handle this. Shutdown and restart all modules.

        elif message.status == 'NOERROR':
            # no errors
            if change_flag:
                try:
                    socket_operations.send_reply(exp_handler_to_radar_control,
                                                 options.radctrl_to_exphan_identity, exp)
                except zmq.ZMQError:  # the queue was full - radarcontrol not receiving.
                    pass  # TODO handle this. Shutdown and restart all modules.
            else:
                try:
                    socket_operations.send_reply(exp_handler_to_radar_control,
                                                 options.radctrl_to_exphan_identity, None)
                except zmq.ZMQError:  # the queue was full - radarcontrol not receiving.
                    pass  # TODO handle this. Shutdown and restart all modules.


        # elif message.status == 'WARNING':
        #     #TODO: log the warning
        #     if change_flag:
        #         ctrl_socket.send_pyobj(prog)
        #     else:
        #         ctrl_socket.send_pyobj(None)
        # elif message.status == 'EXITERROR':
        #     #TODO: log the error
        #     #TODO: determine what to do here, may want to revert experiment back to original (could reload to original by calling new instance)
        #     if change_flag:
        #         ctrl_socket.send_pyobj(prog)
        #     else:
        #         ctrl_socket.send_pyobj(None)

        semaphore.release()



if __name__ == "__main__":

    semaphore = threading.Semaphore()
    experiment_handler(semaphore)
