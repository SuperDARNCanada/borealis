#!/usr/bin/python

# write an experiment that creates a new control program.

from controlprog import controlprog
import zmq

def experiment():
	prog=controlprog(1,150)
	prog.cpo_list[0].freq=9811
	prog.cpo_list[0].sequence=[0,5,8,12]
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
                # SAME_SEQ : Simultaneous same-sequence interfacing, using same pulse sequence. cpo A and B might have different frequencies (stereo) 
                        # and/or may use different antennas. They might also have different pulse length but must have same mpinc, sequence, 
                        # integration time. They must also have same len(scan), although they may use different directions in scan. They must 
                        # have the same scan boundary if any. A time offset between the pulses starting may be set, with max value of mpinc.
                        # cpo A and B will have integrations that run at the same time. 
                # MULTI_SEQ : Simultaneous multi-sequence interfacing. This is more difficult and reduces receive time significantly. Researcher will
			# need to ensure that the multi-sequence is really what they want. cpo A and cpo B will run simultaneously. Both
                        # first pulses will start at the same moment or they may have a time offset so B pulse sequence begins later than A
                        # pulse sequence in each integration.

	# example
	# prog.interface[0,1]="SCAN"

	updateflag=False

	while(True):
		context=zmq.Context()
		cpsocket=context.socket(zmq.PAIR)
		cpsocket.bind("tcp://10.65.0.25:33044")

		message=cpsocket.recv()
		print "received message"	
		if json.loads(message)=="UPDATE":
			print "Time to update"
			if updateflag==False:
				cpsocket.send(json.dumps("NO"))
			elif updateflag==True:
				cpsocket.send(json.dumps("YES"))
				message=cpsocket.recv()
				if json.loads(message)=="READY":
					cpsocket.send(json.dumps(prog))
					
	
 	return prog



experiment()	
