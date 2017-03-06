#!/usr/bin/python

import zmq
import sys
from myexperiment import setup_my_experiment,change_my_experiment

sys.path.append('../radarcontrol')
import radar_status

def setup_data_socket(): #to send data to receive code.
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
    prog=setup_my_experiment()
    prog.build_Scans()
    
    # SEND the first setup prog to runradar to run. 
    message=ctrl_socket.recv_pyobj() 
    print "received message"
    print message
    ctrl_socket.send_pyobj(prog)

    print "Number of Scan types: %d" % (len(prog.scan_objects))
    print "Number of AveragingPeriods in Scan #1: %d" % (len(prog.scan_objects[0].aveperiods)) #NOTE: this is currently not taking beam direction into account.
    print "Number of Sequences in Scan #1, Averaging Period #1: %d" % (len(prog.scan_objects[0].aveperiods[0].integrations))
    print "Number of Pulse Types in Scan #1, Averaging Period #1, Sequence #1: %d" % (len(prog.scan_objects[0].aveperiods[0].integrations[0].cpos))


    while True:

        some_data = None
        prog, change_flag = change_my_experiment(prog, some_data)
        if change_flag == True:
            prog.build_Scans()
        
        # WAIT until runradar is ready to receive a changed prog.
        message=ctrl_socket.recv_pyobj() 
        if isinstance(message, radar_status.RadarStatus):
            print "received READY {}".format(message.status)
        if change_flag == True:
            ctrl_socket.send_pyobj(prog)
        else:
            ctrl_socket.send_pyobj(None)

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
    
