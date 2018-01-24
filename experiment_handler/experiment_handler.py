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


def setup_data_socket(addr, context):
    """
    To setup the socket for communication with the 
    signal processing block. 
    :return: 
    """
    cpsocket = context.socket(zmq.PAIR)
    try:
        cpsocket.connect(addr)
    except:
        pass  # TODO
    return cpsocket


def setup_control_socket(addr, context):
    """
    to send data to receive code.
    :return: 
    """
    cpsocket = context.socket(zmq.PAIR)
    try:
        cpsocket.bind(addr)
    except:
        pass
        # TODO
    return cpsocket


def experiment_handler():
    
    # setup two sockets - one to get ACF data and
    # another to talk to runradar.
    options = ExperimentOptions()
    context = zmq.Context()
    data_socket = setup_data_socket(options.data_to_experiment_address, context)
    ctrl_socket = setup_control_socket(options.experiment_handler_to_radar_control_address, context)



    change_flag = False
    while True:

        # WAIT until runradar is ready to receive a changed prog.
        message = ctrl_socket.recv_pyobj()
        if isinstance(message, RadarStatus):
            if message.status == 'EXPNEEDED':
                print("received READY message {} so starting new experiment from beginning".format(message.status))
                # starting anew
                # TODO: change line to be scheduled
                prog = one_box_test_experiment.OneBox()
                prog.build_scans()
                try:
                    ctrl_socket.send_pyobj(prog, flags=zmq.NOBLOCK)
                except zmq.ZMQError: # the queue was full - radarcontrol not receiving.
                    pass  #TODO handle this. Shutdown and restart all modules.
            elif message.status == 'NOERROR':
                # no errors 
                if change_flag:
                    ctrl_socket.send_pyobj(prog)
                else:
                    ctrl_socket.send_pyobj(None)
            elif message.status == 'WARNING':
                #TODO: log the warning
                if change_flag:
                    ctrl_socket.send_pyobj(prog)
                else:
                    ctrl_socket.send_pyobj(None)
            elif message.status == 'EXITERROR':
                #TODO: log the error
                #TODO: determine what to do here, may want to revert experiment back to original (could reload to original by calling new instance)
                if change_flag:
                    ctrl_socket.send_pyobj(prog)
                else:
                    ctrl_socket.send_pyobj(None)

        some_data = None  # TODO get the data from data socket and pass to update
        change_flag = prog.update(some_data)
        if change_flag:
            prog.build_scans()

if __name__ == "__main__":
    experiment_handler()
