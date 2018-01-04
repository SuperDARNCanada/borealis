#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#

import zmq

# TODO: dynamic import
# NOTE: Have to edit PYTHONPATH='..../placeholderOS/' in the environment for this to work.
from experiments import normalscan
from radar_status.radar_status import RadarStatus
# importlib.import_module('normalscan')
from utils.experiment_options.experimentoptions import ExperimentOptions


# importlib.import_module('radar_status') TODO


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
    data_socket = setup_data_socket(options.data_to_experiment_socket, context)
    ctrl_socket = setup_control_socket(options.experiment_handler_to_radar_control_address, context)


    change_flag = False
    while True:

        # WAIT until runradar is ready to receive a changed prog.
        message = ctrl_socket.recv_pyobj()
        if isinstance(message, RadarStatus):
            if message.status == 'EXPNEEDED':
                print("received READY {} and starting program as new".format(message.status))
                # starting anew
                # TODO: change line to be scheduled
                prog = normalscan.Normalscan()
                prog.build_scans() # REVIEW #30 we should talk about where best to put this call REPLY: Ok - has to be after the experiment is made or changed, so in experiment_handler is really the only place I think would make sense
                ctrl_socket.send_pyobj(prog) # REVIEW #0 Can block if ctrl_socket not valid, can specify NOBLOCK, not sure if this is useful REPLY: ? Not sure what you mean here of it being invalid and how we would handle that at this point?
            elif message.status == 'NOERROR':
                # no errors 
                if change_flag:
                    ctrl_socket.send_pyobj(prog)
                else:
                    ctrl_socket.send_pyobj(None) # REVIEW #1 Does the control socket expect a response from every message? Is this send_pyobj necessary? REPLY: Yes, may have modified the experiment every time so it needs to know if there is a new one.
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
