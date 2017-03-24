#!/usr/bin/python

import zmq
import sys
import importlib

sys.path.append('../radar_control')
sys.path.append('../experiments')
# TODO: dynamic import ??

import normalscan
#importlib.import_module('normalscan')

import radar_status
#importlib.import_module('radar_status')

def setup_data_socket(): #to send data to receive code.
    """
    To setup the socket for communication with the 
    signal processing block. 
    """
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.connect("ipc:///tmp/feeds/4")
    return cpsocket


def setup_control_socket(): #to send data to receive code.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.bind("ipc:///tmp/feeds/5")
    return cpsocket


def main():
    
    # setup two sockets - one to get ACF data and
    # another to talk to runradar.
    data_socket=setup_data_socket()
    ctrl_socket=setup_control_socket()
    
#    print "Number of Scan types: %d" % (len(prog.scan_objects))
#    print "Number of AveragingPeriods in Scan #1: %d" % (len(prog.scan_objects[0].aveperiods)) #NOTE: this is currently not taking beam direction into account.
#    print "Number of Sequences in Scan #1, Averaging Period #1: %d" % (len(prog.scan_objects[0].aveperiods[0].integrations))
#    print "Number of Pulse Types in Scan #1, Averaging Period #1, Sequence #1: %d" % (len(prog.scan_objects[0].aveperiods[0].integrations[0].cpos))

    change_flag = False
    while True:

        # WAIT until runradar is ready to receive a changed prog.
        message=ctrl_socket.recv_pyobj() 
        if isinstance(message, radar_status.RadarStatus):
            if message.status == 0:
                print("received READY {} and starting program as new".format(message.status))
                # starting anew
                # TODO: change line to be scheduled
                prog=normalscan.Normalscan()

                prog.build_Scans()
                ctrl_socket.send_pyobj(prog) 
            elif message.status == 1:
                # no errors 
                if change_flag == True:
                    ctrl_socket.send_pyobj(prog)
                else:
                    ctrl_socket.send_pyobj(None)
            elif message.status == 2:
                #TODO: log the warning
                if change_flag == True:
                    ctrl_socket.send_pyobj(prog)
                else:
                    ctrl_socket.send_pyobj(None)
            elif message.status == 3:
                #TODO: log the error
                #TODO: determine what to do here, may want to revert experiment back to original (could reload to original by calling new instance)
                if change_flag == True:
                    ctrl_socket.send_pyobj(prog)
                else:
                    ctrl_socket.send_pyobj(None)
                


        some_data = None
        change_flag = prog.update(some_data)
        if change_flag == True:
            prog.build_Scans()
        

    #    if json.loads(message)=="UPDATE":
#            print "Time to update"
#            if updateflag==False:
#                cpsocket.send(json.dumps("NO"))
#            elif updateflag==True:
#                cpsocket.send(json.dumps("YES"))
#                message=cpsocket.recv()
#                if json.loads(message)=="READY":
#                    #need to send a dictionary here or use other serialization.
#                    cpsocket.send(json.dumps(prog))


main()
    
