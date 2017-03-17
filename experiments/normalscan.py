#!/usr/bin/python

# write an experiment that creates a new control program.
import sys
from experiment_prototype import ExperimentPrototype
import zmq
import json


class Normalscan(ExperimentPrototype):
    
    def __init__(self):
        super(Normalscan,self).__init__(1,150) #number of cpo_list dictionaries to interface, 'control program' ID  
        self.cpo_list[0].update({      
            "txchannels":   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            "rxchannels":   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            "sequence":     [0,14,22,24,27,31,42,43],
            "pulse_shift":  [0,0,0,0,0,0,0,0],
            "mpinc":        1500, # us
            "pulse_len":    300, # us
            "nrang":        75, # range gates
            "frang":        90, # first range gate, in km
            "intt":         3000, # duration of an integration, in ms
            "intn":         21, # number of averages if intt is None.
            "beamdir":      [-26.25,-22.75,-19.25,-15.75,-12.25,-8.75,
                -5.25,-1.75,1.75,5.25,8.75,12.25,15.75,19.25,22.75,26.25],
            "scan":         [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],
            "scanboundf":   1, 
            "scanbound":    60000,
            "txfreq":       12300,
            "rxfreq":       12300,
            "clrfrqf":      1, 
            "clrfrqrange":  [12200,12500],
            "xcf":          1,
            "acfint":       1, 
            "wavetype":     "SINE",
            "seqtimer":     0
        })


#def change_my_experiment(prog, acfdata):
#    """ Use this function to change your experiment based on
#    ACF data retrieved from the signal processing block"""
#    change_flag = False
#    return prog, change_flag
#
#def setup_my_experiment():
#    prog=ControlProg(3,150)
#    prog.cpo_list[0].freq=9811
#    prog.cpo_list[0].sequence=[0,5,8,12,18,30,32,33]
#    prog.cpo_list[0].txchannels=[0]
#    prog.cpo_list[1].freq=13588
#    prog.cpo_list[1].sequence=[0,5,8,12]
#    prog.cpo_list[1].txchannels=[0]
#    prog.cpo_list[2].freq=13006
#    prog.cpo_list[2].sequence=[0,5,8,12]
#    prog.cpo_list[2].txchannels=[0]
#    # change your control program here. Use selfcheck(myctrlprog) and print myctrlprog() to see what can be changed
#
#    #prog.interfacing is a dictionary of how each cpo interacts with the other cpo's - default "NONE" in all possible spots
#    # This must be modified here, example below.
#            # NOTE keys are as such: (0,1), (0,2), (1,2), NEVER includes (2,0) etc.
#            # The only interface options are:
#            # if_types=frozenset(['NONE', 'SCAN', 'INTTIME', 'INTEGRATION', 'SAME_SEQ', 'MULTI_SEQ'])
#
#            #INTERFACING TYPES:
#            # NONE : Only the default, must be changed.
#            # SCAN : Scan by scan interfacing. cpo #1 will scan first followed by cpo #2 and subsequent cpo's.
#            # INTTIME : nave by nave interfacing (full integration time of one sequence, then the next). Time/number of sequences dependent on intt and intn in cp_object.
#                    # Effectively simultaneous scan interfacing, interleaving each integration time in the scans. cpo #1 first inttime or beam direction
#                    # will run followed by cpo #2's first inttime,etc. if cpo #1's len(scan) is greater than cpo #2's, cpo #2's last
#                    # integration will run and then all the rest of cpo #1's will continue until the full scan is over.
#                    # cpo 1 and 2 must have the same scan boundary, if any boundary. All other may differ.
#            # INTEGRATION : integration by integration interfacing (one sequence of one cp_object, then the next). 
#                    # cpo #1 and cpo #2 must have same intt and intn. Integrations will switch between one and the other until time is up/nave is reached.
#            # PULSE : Simultaneous sequence interfacing, pulse by pulse creates a single sequence. cpo A and B might have different frequencies (stereo) 
#                    # and/or may have different pulse length, mpinc, sequence, but must have the same
#                    # integration time. They must also have same len(scan), although they may use different directions in scan. They must 
#                    # have the same scan boundary if any. A time offset between the pulses starting may be set (seq_timer in cp_object).
#                    # cpo A and B will have integrations that run at the same time. 
#
#    # example
#    prog.interface[0,1]="PULSE"
#    prog.interface[0,2]="PULSE"
##    prog.interface[0,3]="INTEGRATION"
##    prog.interface[3,4]="SCAN"
##    prog.interface[4,5]="PULSE"
#    prog.interface[1,2]="PULSE"
##    prog.interface[1,3]="INTEGRATION"
##    prog.interface[2,3]="INTEGRATION"
##    prog.interface[0,4]="SCAN"
##    prog.interface[1,4]="SCAN"
##    prog.interface[2,4]="SCAN"
##    prog.interface[0,5]="SCAN"
##    prog.interface[1,5]="SCAN"
##    prog.interface[2,5]="SCAN"
##    prog.interface[3,5]="SCAN"
#
#    return prog
