#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# Marci Detwiller
#

import zmq
import sys
import importlib

sys.path.append('../radar_control')
sys.path.append('../experiments')
# TODO: dynamic import ??

from experiments import normalscan
# importlib.import_module('normalscan')

from radar_control import radar_status
# importlib.import_module('radar_status') TODO


def setup_data_socket():
    """
    To setup the socket for communication with the 
    signal processing block. 
    :return: 
    """
    context=zmq.Context() # REVIEW #33 Apparently it's best to just use one zmq context in the entire application - http://stackoverflow.com/questions/32280271/zeromq-same-context-for-multiple-sockets. So maybe have a global context or set it up in the main function and pass it to these functions.
    cpsocket=context.socket(zmq.PAIR) # REPLY OK TODO
    try:
        cpsocket.connect("ipc:///tmp/feeds/4") # REVIEW #29 Use config file for these ipc feed locations REPLY OK
    except:
        pass  # TODO
    return cpsocket


def setup_control_socket():
    """
    to send data to receive code.
    :return: 
    """
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    try:
        cpsocket.bind("ipc:///tmp/feeds/5")
    except:
        pass
        # TODO
    return cpsocket


def main():
    
    # setup two sockets - one to get ACF data and
    # another to talk to runradar.
    data_socket = setup_data_socket()
    ctrl_socket = setup_control_socket()
    
#    print "Number of Scan types: %d" % (len(prog.scan_objects)) # REVIEW #33 is this commented code necessary or can you remove it?
#    print "Number of AveragingPeriods in Scan #1: %d" % (len(prog.scan_objects[0].aveperiods)) #NOTE: this is currently not taking beam direction into account.
#    print "Number of Sequences in Scan #1, Averaging Period #1: %d" % (len(prog.scan_objects[0].aveperiods[0].integrations))
#    print "Number of Pulse Types in Scan #1, Averaging Period #1, Sequence #1: %d" % (len(prog.scan_objects[0].aveperiods[0].integrations[0].cpos))

    change_flag = False
    while True:

        # WAIT until runradar is ready to receive a changed prog.
        message = ctrl_socket.recv_pyobj()
        if isinstance(message, radar_status.RadarStatus): # REVIEW #6 TODO we need to talk about the design of this loop probably. not sure what each of the cases mean just by looking. there's code duplication
            if message.status == 'CPNEEDED':
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

        some_data = None  # REVIEW #6 TODO get the data from data socket and pass to update
        change_flag = prog.update(some_data)
        if change_flag:
            prog.build_scans()

main()  # REVIEW #39 python - use if __name__ == "__main__" TODO
