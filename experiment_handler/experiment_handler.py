#!/usr/bin/env python3

"""
    experiment_handler process
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    This program runs a given experiment. It will use the experiment's build_scans method to 
    create the iterable ScanClassBase objects that will be used by the radar_control block, 
    then it will pass the experiment to the radar_control block to run. 

    It will be passed some data to use in its update method at the end of every integration time. 
    This has yet to be implemented but will allow experiment_prototype to modify itself 
    based on received data as feedback.
    
    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""

import zmq
import os
import sys
import argparse
import inspect
import importlib
import threading
import pickle
import json

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from utils.experiment_options.experimentoptions import ExperimentOptions
from utils.zmq_borealis_helpers import socket_operations
from experiment_prototype.experiment_exception import ExperimentException


def printing(msg):
    EXPERIMENT_HANDLER = "\033[34m" + "EXPERIMENT HANDLER: " + "\033[0m"
    sys.stdout.write(EXPERIMENT_HANDLER + msg + "\n")


def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
     
    :returns: the usage message
    """

    usage_message = """ experiment_handler.py [-h] experiment_module
    
    Pass the module containing the experiment to the experiment handler as a required 
    argument. It will search for the module in the BOREALISPATH/experiment_prototype 
    package. It will retrieve the class from within the module (your experiment). 
    
    It will use the experiment's build_scans method to create the iterable ScanClassBase 
    objects that will be used by the radar_control block, then it will pass the 
    experiment to the radar_control block to run. 

    It will be passed some data to use in its .update() method at the end of every 
    integration time. This has yet to be implemented but will allow experiments to 
    modify themselves based on received data as feedback. This is not a necessary method 
    for all experiments and if there is no update method experiment updates will not 
    occur."""

    return usage_message


def experiment_parser():
    """
    Creates the parser to retrieve the experiment module.
    
    :returns: parser, the argument parser for the experiment_handler. 
    """

    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("experiment_module", help="The name of the module in the experiment_prototype "
                                                  "package that contains your Experiment class, "
                                                  "e.g. normalscan")

    return parser


def retrieve_experiment():
    """
    Retrieve the experiment class from the provided module given as an argument.

    :raise ExperimentException: if the experiment module provided as an argument does not contain
     a single class that inherits from ExperimentPrototype class.    
    :returns: Experiment, the experiment class, inherited from ExperimentPrototype.
    """

    parser = experiment_parser()
    args = parser.parse_args()

    if __debug__:
        print("Running the experiment: " + args.experiment_module)
    experiment = args.experiment_module
    experiment_mod = importlib.import_module("experiments." + experiment)

    experiment_classes = {}
    for class_name, obj in inspect.getmembers(experiment_mod, inspect.isclass):
        experiment_classes[class_name] = obj

    # need to have one ExperimentPrototype and one user-specified class.
    try:
        experiment_proto_class = experiment_classes['ExperimentPrototype']
        del experiment_classes['ExperimentPrototype']
    except KeyError:
        errmsg = "Your experiment is not built from parent class ExperimentPrototype - exiting"
        raise ExperimentException(errmsg)

    list_experiments = []
    for class_name, class_obj in experiment_classes.items():
        if experiment_proto_class in inspect.getmro(class_obj):  # an experiment
            # must inherit from ExperimentPrototype
            # other utility classes might be in the file but we will ignore them.
            list_experiments.append(class_obj)

    if len(list_experiments) != 1:
        errmsg = "You have zero or more than one experiment class in your experiment " \
                 "file - exiting"
        raise ExperimentException(errmsg)

    Experiment = list_experiments[0]  # this is the experiment class that we need to run.

    try:
        return Experiment
    except NameError:
        errmsg = "Cannot find the experiment inside your module. Please make sure there is a " \
                 "class that inherits from ExperimentPrototype in your module."
        raise ExperimentException(errmsg)


def experiment_handler(semaphore):
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

    options = ExperimentOptions()
    ids = [options.exphan_to_radctrl_identity, options.exphan_to_dsp_identity]
    sockets_list = socket_operations.create_sockets(ids, options.router_address)

    exp_handler_to_radar_control = sockets_list[0]
    exp_handler_to_dsp = sockets_list[1]

    Experiment = retrieve_experiment()
    experiment_update = False
    for method_name, obj in inspect.getmembers(Experiment, inspect.ismethod):
        if method_name == 'update':
            experiment_update = True
    if __debug__:
        print("Experiment has update method: " + str(experiment_update))

    exp = Experiment()
    change_flag = False

    def update_experiment():
        # Recv complete processed data from DSP
        socket_operations.send_request(exp_handler_to_dsp,
                                       options.dsp_to_exphan_identity,
                                       "Need completed data")

        data = socket_operations.recv_data(exp_handler_to_dsp,
                                           options.dsp_to_exphan_identity, printing)

        some_data = None  # TODO get the data from data socket and pass to update

        semaphore.acquire()
        change_flag = exp.update(some_data)
        if change_flag:
            exp.build_scans()
            print "REBUILDING EXPERIMENT BECAUSE change_flag = TRUE!!!"
        semaphore.release()

        if __debug__:
            data_output = "Dsp sent -> {}".format(data)
            printing(data_output)

    if experiment_update:
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
            serialized_exp = pickle.dumps(exp, protocol=pickle.HIGHEST_PROTOCOL)
            try:
                socket_operations.send_reply(exp_handler_to_radar_control,
                                             options.radctrl_to_exphan_identity,
                                             serialized_exp)
            except zmq.ZMQError: # the queue was full - radarcontrol not receiving.
                pass  #TODO handle this. Shutdown and restart all modules.

        elif message == 'NOERROR':
            # no errors
            if change_flag:
                serialized_exp = pickle.dumps(exp, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                serialized_exp = pickle.dumps(None, protocol=pickle.HIGHEST_PROTOCOL)

            try:
                socket_operations.send_reply(exp_handler_to_radar_control,
                                             options.radctrl_to_exphan_identity, serialized_exp)
            except zmq.ZMQError:  # the queue was full - radarcontrol not receiving.
                pass  # TODO handle this. Shutdown and restart all modules.

        # TODO: handle errors with revert back to original experiment. requires another
        # message
        semaphore.release()


if __name__ == "__main__":

    semaphore = threading.Semaphore()
    experiment_handler(semaphore)

