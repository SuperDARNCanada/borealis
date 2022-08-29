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

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from utils.experiment_options.experimentoptions import ExperimentOptions
from utils.zmq_borealis_helpers import socket_operations
from experiment_prototype.experiment_exception import ExperimentException
from experiment_prototype.experiment_prototype import ExperimentPrototype


def printing(msg):
    EXPERIMENT_HANDLER = "\033[34m" + "EXPERIMENT HANDLER: " + "\033[0m"
    sys.stdout.write(EXPERIMENT_HANDLER + msg + "\n")


def usage_msg():
    """
    Return the usage message for this process.

    This is used if a -h flag or invalid arguments are provided.

    :returns: the usage message
    """

    usage_message = """ experiment_handler.py [-h] experiment_module scheduling_mode_type

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
    parser.add_argument("scheduling_mode_type", help="The type of scheduling time for this experiment "
                                                     "run, e.g. common, special, or discretionary.")
    parser.add_argument("--kwargs_string", default='', 
                        help="String of keyword arguments for the experiment.")

    return parser


def retrieve_experiment(experiment_module_name):
    """
    Retrieve the experiment class from the provided module given as an argument.

    :param experiment_module_name: The name of the experiment module to run
     from the Borealis project's experiments directory.

    :raise ExperimentException: if the experiment module provided as an argument does not contain
     a single class that inherits from ExperimentPrototype class.
    :returns: Experiment, the experiment class, inherited from ExperimentPrototype.
    """

    if __debug__:
        printing("Running the experiment: " + experiment_module_name)
    experiment_mod = importlib.import_module("experiments." + experiment_module_name)

    # find the class or classes *defined* in this module.
    # returns list of class name and object
    experiment_classes = [(m[0], m[1]) for m in inspect.getmembers(
                          experiment_mod, inspect.isclass) if
                          m[1].__module__ == experiment_mod.__name__]

    # remove any classes that do not have ExperimentPrototype as parent.
    for (class_name, class_obj) in experiment_classes:
        if ExperimentPrototype not in inspect.getmro(class_obj):
            # an experiment must inherit from ExperimentPrototype
            # other utility classes might be in the file but we will ignore them.
            experiment_classes.remove((class_name, class_obj))

    # experiment_classes should now only have classes *defined* in the module,
    # that have ExperimentPrototype as parent.
    if len(experiment_classes) == 0:
        errmsg = "No experiment classes are present that are built from"\
                 " parent class ExperimentPrototype - exiting"
        raise ExperimentException(errmsg)
    if len(experiment_classes) > 1:
        errmsg = "You have more than one experiment class in your " \
                 "experiment file - exiting"
        raise ExperimentException(errmsg)

    # this is the experiment class that we need to run.
    Experiment = experiment_classes[0][1]

    printing('Retrieving experiment: {} from module {}'.format(
             experiment_classes[0][0], experiment_mod))

    return Experiment


def send_experiment(exp_handler_to_radar_control, iden, serialized_exp):
    """
    Send the experiment to radar_control module.

    :param exp_handler_to_radar_control: socket to send the experiment on
    :param iden: ZMQ identity
    :param serialized_exp: Either a pickled experiment or a None.
    """
    try:
        socket_operations.send_exp(exp_handler_to_radar_control, iden, serialized_exp)
    except zmq.ZMQError:  # the queue was full - radarcontrol not receiving.
        pass  # TODO handle this. Shutdown and restart all modules.


def experiment_handler(semaphore, args):
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

    experiment_name = args.experiment_module
    scheduling_mode_type = args.scheduling_mode_type

    Experiment = retrieve_experiment(experiment_name)
    experiment_update = False
    for method_name, obj in inspect.getmembers(Experiment, inspect.isfunction):
        if method_name == 'update':
            experiment_update = True

    if __debug__:
        if experiment_update:
            printing("Experiment has an updated method.")

    if args.kwargs_string:
        # parse kwargs and pass to experiment
        kwargs = {}
        kwargs_list = args.kwargs_string.split(',')
        for element in kwargs_list:
            kwarg = element.split('=')
            kwargs[kwarg[0]] = kwarg[1]
        exp = Experiment(**kwargs)
    else:
        exp = Experiment()

    exp._set_scheduling_mode(scheduling_mode_type)
    change_flag = True

    def update_experiment():
        # Recv complete processed data from DSP or datawrite? TODO
        #socket_operations.send_request(exp_handler_to_dsp,
        #                               options.dsp_to_exphan_identity,
        #                               "Need completed data")

        #data = socket_operations.recv_data(exp_handler_to_dsp,
        #                             options.dsp_to_exphan_identity, printing)

        some_data = None  # TODO get the data from data socket and pass to update

        semaphore.acquire()
        change_flag = exp.update(some_data)
        if change_flag:
            if __debug__:
                printing("Building an updated experiment.")
            exp.build_scans()
            printing("Experiment {exp} with CPID {cp} successfully updated"
                     .format(exp=exp.__class__.__name__, cp=exp.cpid))
        semaphore.release()

    update_thread = threading.Thread(target=update_experiment)

    while True:

        if not change_flag:
            serialized_exp = pickle.dumps(None, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            exp.build_scans()
            printing("Successful experiment {exp} built with CPID {cp}".format(
                     exp=exp.__class__.__name__, cp=exp.cpid))
            serialized_exp = pickle.dumps(exp, protocol=pickle.HIGHEST_PROTOCOL)
            # use the newest, fastest protocol (currently version 4 in python 3.4+)
            change_flag = False

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
            send_experiment(exp_handler_to_radar_control,
                            options.radctrl_to_exphan_identity, serialized_exp)

        elif message == 'NOERROR':
            # no errors
            send_experiment(exp_handler_to_radar_control,
                            options.radctrl_to_exphan_identity, serialized_exp)

        # TODO: handle errors with revert back to original experiment. requires another
        # message
        semaphore.release()

        if experiment_update:
            # check if a thread is already running !!!
            if not update_thread.isAlive():
                if __debug__:
                    printing("Updating experiment")
                update_thread = threading.Thread(target=update_experiment)
                update_thread.daemon = True
                update_thread.start()


def main(sys_args):
    semaphore = threading.Semaphore()
    parser = experiment_parser()
    args = parser.parse_args(args=sys_args)
    experiment_handler(semaphore, args)


if __name__ == "__main__":
    main(sys.argv[1:])
