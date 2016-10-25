#!/usr/bin/python

# write an experiment that creates a new control program.

from controlprog import controlprog
import zmq
import json

def experiment():
    prog=controlprog(1,150)
    prog.cpo_list[0].freq=9811
    prog.cpo_list[0].sequence=[0,5,8,12,18,30,32,33]
    prog.cpo_list[0].channels=[0,3,5,7,8,9]


    # change your control program here. Use selfcheck(myctrlprog) and print myctrlprog() to see what can be changed

    #prog.interfacing is a dictionary of how each cpo interacts with the other cpo's - default "NONE" in all possible spots
    # This must be modified here, example below.
            # NOTE keys are as such: (0,1), (0,2), (1,2), NEVER includes (2,0) etc.
            # The only interface options are:
            # if_types=frozenset(['NONE', 'SCAN', 'INTTIME', 'INTEGRATION', 'SAME_SEQ', 'MULTI_SEQ'])

            #INTERFACING TYPES:
            # NONE : Only the default, must be changed.
            # SCAN : Scan by scan interfacing. cpo #1 will scan first followed by cpo #2 and subsequent cpo's.
            # INTTIME : nave by nave interfacing (full integration time of one sequence, then the next). Time/number of sequences dependent on intt and intn in cp_object.
                    # Effectively simultaneous scan interfacing, interleaving each integration time in the scans. cpo #1 first inttime or beam direction
                    # will run followed by cpo #2's first inttime,etc. if cpo #1's len(scan) is greater than cpo #2's, cpo #2's last
                    # integration will run and then all the rest of cpo #1's will continue until the full scan is over.
                    # cpo 1 and 2 must have the same scan boundary, if any boundary. All other may differ.
            # INTEGRATION : integration by integration interfacing (one sequence of one cp_object, then the next). 
                    # cpo #1 and cpo #2 must have same intt and intn. Integrations will switch between one and the other until time is up/nave is reached.
            # PULSE : Simultaneous sequence interfacing, pulse by pulse creates a single sequence. cpo A and B might have different frequencies (stereo) 
                    # and/or may have different pulse length, mpinc, sequence, but must have the same
                    # integration time. They must also have same len(scan), although they may use different directions in scan. They must 
                    # have the same scan boundary if any. A time offset between the pulses starting may be set (seq_timer in cp_object).
                    # cpo A and B will have integrations that run at the same time. 

    # example
#    prog.interface[0,1]="INTEGRATION"
#    prog.interface[0,2]="PULSE"
#    prog.interface[0,3]="INTEGRATION"
#    prog.interface[3,4]="SCAN"
#    prog.interface[4,5]="PULSE"
#    prog.interface[1,2]="PULSE"
#    prog.interface[1,3]="INTEGRATION"
#    prog.interface[2,3]="INTEGRATION"
#    prog.interface[0,4]="SCAN"
#    prog.interface[1,4]="SCAN"
#    prog.interface[2,4]="SCAN"
#    prog.interface[0,5]="SCAN"
#    prog.interface[1,5]="SCAN"
#    prog.interface[2,5]="SCAN"
#    prog.interface[3,5]="SCAN"

    return prog
    


def build_RCP():
    """Build the experiment into Scans, AveragingPeriods, Sequences, and Pulses and send this data
    over to radarctrl to create pulses and computation block to receive data"""

        

    # Build experiment written by researcher.
    prog=experiment()
    # get wavetables and load them in their the cp_objects.
    prog.get_wavetables()
    # Build scans
    prog.build_Scans()

    print "Number of Scan types: %d" % (len(prog.scan_objects)) 
    print "Number of AveragingPeriods in Scan #1: %d" % (len(prog.scan_objects[0].aveperiods)) #NOTE: this is currently not taking beam direction into account.
    print "Number of Sequences in Scan #1, Averaging Period #1: %d" % (len(prog.scan_objects[0].aveperiods[0].integrations))
    print "Number of Pulse Types in Scan #1, Averaging Period #1, Sequence #1: %d" % (len(prog.scan_objects[0].aveperiods[0].integrations[0].cpos))

    #transfer these classes to dictionaries to send via JSON to


    #updateflag=True
    #context=zmq.Context()
    #cpsocket=context.socket(zmq.PAIR)
    #cpsocket.bind("tcp://10.65.0.25:33555")

    #while(True):
    #    message=cpsocket.recv() 
    #    print "received message"	
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
	
    return prog


build_RCP()
