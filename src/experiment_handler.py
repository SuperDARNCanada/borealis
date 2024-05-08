#!/usr/bin/env python3

"""
    experiment_handler process
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    This program runs a given experiment. It will use the experiment's build_scans method to
    create the iterable InterfaceClassBase objects that will be used by the radar_control block,
    then it will pass the experiment to the radar_control block to run.

    It will be passed some data to use in its update method at the end of every integration time.
    This has yet to be implemented but will allow experiment_prototype to modify itself
    based on received data as feedback.

    :copyright: 2018 SuperDARN Canada
    :author: Marci Detwiller
"""
import argparse
import importlib
import inspect
from pathlib import Path
import pickle
import sys
import threading

import structlog
import zmq

from utils.options import Options
from utils import socket_operations
from experiment_prototype.experiment_exception import ExperimentException
from experiment_prototype.experiment_prototype import ExperimentPrototype


def usage_msg():
    """
    Return the usage message for this process.

    This is used if a -h flag or invalid arguments are provided.

    :returns:   the usage message
    :rtype:     str
    """

    usage_message = """ experiment_handler.py [-h] experiment_module scheduling_mode_type

    Pass the module containing the experiment to the experiment handler as a required
    argument. It will search for the module in the BOREALISPATH/experiment_prototype
    package. It will retrieve the class from within the module (your experiment).

    It will use the experiment's build_scans method to create the iterable InterfaceClassBase
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

    :returns:   parser, the argument parser for the experiment_handler.
    :rtype:     argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument(
        "experiment_module",
        help="The name of the module in the experiment_prototype package that contains "
        "your Experiment class, e.g. normalscan",
    )
    parser.add_argument(
        "scheduling_mode_type",
        help="The type of scheduling time for this experiment run, e.g. common, "
        "special, or discretionary.",
    )
    parser.add_argument(
        "--embargo",
        action="store_true",
        help="Embargo the file (makes the CPID negative)",
    )
    parser.add_argument(
        "--kwargs",
        nargs="+",
        default="",
        help="Keyword arguments for the experiment. Each must be formatted as kw=val",
    )

    return parser


def retrieve_experiment(experiment_module_name):
    """
    Retrieve the experiment class from the provided module given as an argument.

    :param  experiment_module_name:     The name of the experiment module to run from the Borealis
                                        project's experiments directory.
    :type   experiment_module_name:     str

    :raise  ExperimentException:    if the experiment module provided as an argument does not
                                    contain a single class that inherits from ExperimentPrototype
                                    class.

    :returns:   The found experiment that inherits from ExperimentPrototype
    :rtype:     ExperimentPrototype
    """

    log.debug("loading experiment", experiment_module_name=experiment_module_name)
    experiment_mod = importlib.import_module(
        "borealis_experiments." + experiment_module_name
    )

    # find the class or classes *defined* in this module.
    # returns list of class name and object
    experiment_classes = [
        (m[0], m[1])
        for m in inspect.getmembers(experiment_mod, inspect.isclass)
        if m[1].__module__ == experiment_mod.__name__
    ]

    # remove any classes that do not have ExperimentPrototype as parent.
    for class_name, class_obj in experiment_classes:
        if ExperimentPrototype not in inspect.getmro(class_obj):
            # an experiment must inherit from ExperimentPrototype
            # other utility classes might be in the file but we will ignore them.
            experiment_classes.remove((class_name, class_obj))

    # experiment_classes should now only have classes *defined* in the module, that have
    # ExperimentPrototype as parent.
    if len(experiment_classes) == 0:
        errmsg = (
            "No experiment classes are present that are built from"
            " parent class ExperimentPrototype - exiting"
        )
        raise ExperimentException(errmsg)
    if len(experiment_classes) > 1:
        errmsg = (
            "You have more than one experiment class in your "
            "experiment file - exiting"
        )
        raise ExperimentException(errmsg)

    # this is the experiment class that we need to run.
    experiment = experiment_classes[0][1]

    log.verbose(
        "retrieving experiment from module",
        experiment_class=experiment_classes[0][0],
        experiment_module=experiment_mod,
    )

    return experiment


def send_experiment(exp_handler_to_radar_control, iden, serialized_exp):
    """
    Send the experiment to radar_control module.

    :param  exp_handler_to_radar_control:   socket to send the experiment on
    :type   exp_handler_to_radar_control:   ZMQ socket
    :param  iden:                           ZMQ identity
    :type   iden:                           str
    :param  serialized_exp:                 Either a pickled experiment or a None.
    :type   serialized_exp:                 bytes
    """

    try:
        socket_operations.send_exp(exp_handler_to_radar_control, iden, serialized_exp)
    except zmq.ZMQError as e:  # the queue was full - radar_control not receiving.
        log.warning("zmq queue full not receiving", error=e)
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
    scan iterable objects (of class InterfaceClassBase) and will pass them to radar_control. Other
    statuses will be implemented in the future.

    In the future, the update method will be implemented where the experiment can be modified by
    the incoming data.

    :param  semaphore:  Semaphore to protect operations on the experiment
    :type   semaphore:  threading.Semaphore
    :param  args:       Command line provided arguments
    :type   args:       argparse.Namespace
    """

    options = Options()
    ids = [options.exphan_to_radctrl_identity, options.exphan_to_dsp_identity]
    sockets_list = socket_operations.create_sockets(ids, options.router_address)

    exp_handler_to_radar_control = sockets_list[0]
    exp_handler_to_dsp = sockets_list[1]

    experiment_name = args.experiment_module
    scheduling_mode_type = args.scheduling_mode_type

    experiment_class = retrieve_experiment(experiment_name)
    experiment_update = False
    for method_name, obj in inspect.getmembers(experiment_class, inspect.isfunction):
        if method_name == "update":
            experiment_update = True
            log.debug(
                "experiment contains an updated method",
                experiment_name=experiment_class.experiment_name,
            )

    if args.kwargs:
        # parse kwargs and pass to experiment
        kwargs = {}
        for element in args.kwargs:
            kwarg = element.split("=")
            kwargs[kwarg[0]] = kwarg[1]
        exp = experiment_class(**kwargs)
    else:
        exp = experiment_class()

    exp._set_scheduling_mode(scheduling_mode_type)
    exp._embargo_files(args.embargo)
    change_flag = True

    def update_experiment():
        # Recv complete processed data from DSP or datawrite? TODO
        # socket_operations.send_request(exp_handler_to_dsp,
        #                               options.dsp_to_exphan_identity,
        #                               "Need completed data")

        # data = socket_operations.recv_data(exp_handler_to_dsp,
        #                             options.dsp_to_exphan_identity, log)

        some_data = None  # TODO get the data from data socket and pass to update

        semaphore.acquire()
        change_flag = exp.update(some_data)
        if change_flag:
            log.debug("building an updated experiment")
            exp.build_scans()
            log.info(
                "experiment successfully updated",
                experiment_name=exp.__class__.__name__,
                cpid=exp.cpid,
            )
        semaphore.release()

    update_thread = threading.Thread(target=update_experiment)

    while True:
        if not change_flag:
            serialized_exp = pickle.dumps(None, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            exp.build_scans()
            log.info(
                "experiment successfully built",
                experiment_name=exp.__class__.__name__,
                cpid=exp.cpid,
            )
            serialized_exp = pickle.dumps(exp, protocol=pickle.HIGHEST_PROTOCOL)
            # Use the newest, fastest protocol (currently version 4 in python 3.4+)
            change_flag = False

        # Wait until radar_control is ready to receive a changed experiment
        message = socket_operations.recv_request(
            exp_handler_to_radar_control, options.radctrl_to_exphan_identity, log
        )

        log.debug("radar control made a request", request=message)

        semaphore.acquire()
        if message in ["EXPNEEDED", "NOERROR"]:
            if message == "EXPNEEDED":
                log.info("sending new experiment", message=message)
            # Starting anew if EXPNEEDED, otherwise sending None
            send_experiment(
                exp_handler_to_radar_control,
                options.radctrl_to_exphan_identity,
                serialized_exp,
            )

        # TODO: handle errors with revert back to original experiment. requires another message
        semaphore.release()

        if experiment_update:
            # Check if a thread is already running !!!
            if not update_thread.isAlive():
                log.debug("updating experiment thread")
                update_thread = threading.Thread(target=update_experiment)
                update_thread.daemon = True
                update_thread.start()


def main(sys_args):
    semaphore = threading.Semaphore()
    parser = experiment_parser()
    args = parser.parse_args(args=sys_args)
    experiment_handler(semaphore, args)


if __name__ == "__main__":
    from utils import log_config

    log = log_config.log()
    log.info(f"EXPERIMENT_HANDLER BOOTED")
    try:
        main(sys.argv[1:])
        log.info(f"EXPERIMENT_HANDLER EXITED")
    except Exception as main_exception:
        log.critical("EXPERIMENT_HANDLER CRASHED", error=main_exception)
        log.exception("EXPERIMENT_HANDLER CRASHED", exception=main_exception)

else:
    caller = Path(inspect.stack()[-1].filename)
    module_name = caller.name.split(".")[0]
    log = structlog.getLogger(module_name)
