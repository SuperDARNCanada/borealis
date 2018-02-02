#!/usr/bin/python

"""
    experiment_handler
    ~~~~~~~~~~~~~~~~~~
    This program runs a given experiment. It will use the experiment's build_scans method to 
    create the iterable ScanClassBase objects that will be used by the radar_control block, 
    then it will pass the experiment to the radar_control block to run. 

    It will be passed some data to use in its update method at the end of every integration time. 
    This has yet to be implemented but will allow experiment_prototype to modify themselves based on 
    received data as feedback.
    
    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""

import zmq
import os
import sys
import argparse
import inspect
import importlib

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from radar_status.radar_status import RadarStatus
from utils.experiment_options.experimentoptions import ExperimentOptions
from experiment_prototype.experiment_exception import ExperimentException


def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
     
    :returns: the usage message
    """

    usage_message = """experiment_handler.py [-h] experiment_module
    
    Pass the module containing the experiment to the experiment handler as the a required 
    argument. It will search for the module in the BOREALISPATH/experiment_prototype package. It will 
    retrieve the class from within the module (your experiment). 
    
    It will use the experiment's build_scans method to create the iterable ScanClassBase objects
    that will be used by the radar_control block, then it will pass the experiment to the 
    radar_control block to run. 

    It will be passed some data to use in its .update() method at the end of every integration time. 
    This has yet to be implemented but will allow experiment_prototype to modify themselves based on 
    received data as feedback. This is not a necessary method for all experiment_prototype and if there is 
    no update method experiment updates will not occur."""

    return usage_message


def retrieve_experiment():
    """
    Retrieve the experiment class from the provided module given as an argument.

    :raise ExperimentException: if the experiment module provided as an argument does not contain
     a single class that inherits from ExperimentPrototype class.    
    :returns: Experiment, the experiment class, inherited from ExperimentPrototype.
    """

    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("experiment_module", help="The name of the module in the experiment_prototype "
                                                  "package that contains your Experiment class, "
                                                  "e.g. normalscan")
    args = parser.parse_args()

    if __debug__:
        print("Running the experiment: " + args.experiment_module)
    experiment = args.experiment_module
    experiment_mod = importlib.import_module("." + experiment, package="experiment_prototype")

    experiment_classes = {}
    for class_name, obj in inspect.getmembers(experiment_mod, inspect.isclass):
        experiment_classes[class_name] = obj

    # need to have one ExperimentPrototype and one user-specified class.
    try:
        del experiment_classes['ExperimentPrototype']
    except KeyError:
        errmsg = "Your experiment is not built from parent class ExperimentPrototype - exiting"
        raise ExperimentException(errmsg)

    num_classes = 0
    for experiment_name, experiment_obj in experiment_classes.items():
        num_classes += 1
        if num_classes != 1:
            errmsg = "You have more than one experiment class in your experiment file - exiting"
            raise ExperimentException(errmsg)
        Experiment = experiment_obj  # this is the experiment class that we need to run.

    try:
        return Experiment
    except NameError:
        errmsg = "Cannot find the experiment inside your module. Please make sure there is a " \
                 "class that inherits from ExperimentPrototype in your module."
        raise ExperimentException(errmsg)

    # TODO inspect the Experiment class to determine if the update method exists!


def setup_data_socket(addr, context):
    """
    To setup the socket for communication with the datawrite process. 
    
    :returns: the socket to the datawrite process over which data will be passed.
    """

    data_socket = context.socket(zmq.PAIR)
    try:
        data_socket.connect(addr)
    except:
        pass  # TODO
    return data_socket


def setup_control_socket(addr, context):
    """
    To send the experiment to the radar_control process for running the radar.
    
    :returns: the socket to the radar_control process.
    """

    control_socket = context.socket(zmq.PAIR)
    try:
        control_socket.bind(addr)
    except:
        pass
        # TODO
    return control_socket


def experiment_handler():
    """
    Run the experiment. This is the main process when this program is called.
    
    This process runs the experiment from the module that was passed in as an argument.  It 
    currently does not exit unless killed. It may be updated in the future to exit if provided 
    with an error flag.
    
    This process begins with setup of sockets and retrieving the experiment class from the module. 
    It then waits for a message of type RadarStatus to come in from the radar_control block. If 
    the status is 'EXPNEEDED', meaning an experiment is needed, experiment_handler will build the
    scan iterable objects (of class ScanClassBase) and will pass them to radar_control. Other 
    statuses will be implemented in the future.
    
    In the future, the update method will be implemented where the experiment can be modified by
    the incoming data.
    """
    
    # setup two sockets - one to get ACF data and
    # another to talk to runradar.
    options = ExperimentOptions()
    context = zmq.Context()
    data_socket = setup_data_socket(options.data_to_experiment_address, context)
    ctrl_socket = setup_control_socket(options.experiment_handler_to_radar_control_address, context)

    Experiment = retrieve_experiment()

    change_flag = False
    while True:
        # WAIT until runradar is ready to receive a changed prog.
        message = ctrl_socket.recv_pyobj()
        if isinstance(message, RadarStatus):
            if message.status == 'EXPNEEDED':
                print("received READY message {} so starting new experiment from "
                      "beginning".format(message.status))
                # starting anew
                # TODO: change line to be scheduled
                prog = Experiment()
                if __debug__:
                    print(prog)
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

if __name__ == '__main__':
    experiment_handler()
