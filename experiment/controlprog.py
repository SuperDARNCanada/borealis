#!/usr/bin/python

# A new radar control program.

from cp_object import cp_object, interfacing, if_type
import numpy as np

class controlprog():
	def __init__(self, cponum, cpid):
		# your radar control program will be a list of cp_objects that will be combined in radarctrl.py
		self.cpo_id=cpid # need a unique id for each new cp.
		# how many cp objects would you like in your RCP?
		self.cpo_num=cponum # default 1
		cpo_list=[]
		for num in range(self.cpo_num):
			cpo_list.append(cp_object())
			#cpo_list[num].cp_comp=self.cpo_num # cp_comp is the number of cp_objects in this RCP, don't need this in cp_object though.
			cpo_list[num].cpid[1]=num # second number in cpid array is the ID of this cp_object in the overall RCP (first, 
			#second, third, up to cp_comp).
			cpo_list[num].cpid[0]=self.cpo_id
		self.cpo_list=cpo_list
		# change your control program in your experiment. Use selfcheck(myctrlprog) and print myctrlprog() to see what can be changed
		
		self.interface=interfacing(self.cpo_num) #dictionary of how each cpo interacts with the other cpo's - default "NONE" in all possible spots, must be modified in your experiment.
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
		
		# must set all possible interfacing variables. Is this straightforward? Is there a better way to do this?

	def __call__(self):
		print 'CPID [cpo_id]: {}'.format(self.cpo_id)
		print 'Num of CP Objects [cpo_num]: {}'.format(self.cpo_num)
		for i in range(self.cpo_num):
			print '\n'
			print 'CP Object : {}'.format(i)
			print self.cpo_list[i]()
		print '\n'
		print 'Interfacing [interface]: {}'.format(self.interface)
		return None

