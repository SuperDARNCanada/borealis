#!/usr/bin/python

# A new radar control program.

from cp_object import cp_object
import numpy as np

class controlprog(num,cpid):
	def __init__:
		self.cpo_num=num # default should be 1
		self.cpid=cpid # default should be 150 for normalscan.
	# your radar control program will be a list of cp_objects that will be combined in radarctrl.py
	cpo_id=150 # default control program id for normalscan - need a unique id for each new cp.
	# how many cp objects would you like in your RCP?
	cpo_list=[]
	for num in range(cpo_num):
		cpo_list.append(cp_object())
		cpo_list[num].cp_comp=cpo_num # cp_comp is the number of cp_objects in this RCP
		cpo_list[num].cpid[1]=num # second number in cpid array is the ID of this cp_object in the overall RCP (first, 
		#second, third, up to cp_comp).
		cpo_list[num].cpid[0]=cpo_id
	# space to change defaults of RCP. If unsure what to change try display(myctrlprog) and remember to selfcheck(myctrlprog)
	# cpo_list[0].freq=13000
	# cpo_list[0].sequence=[0,2,5,7,8,15,24,29]
	# ETC
	
	# next step is interfacing components. 
	interfacing=np.empty([cpo_num,cpo_num],dtype=int) #matrix of how each cpo interacts with the other cpo's.
	for a in range(cpo_num):
		for b in range(cpo_num):
			interfacing[a,b]=0 # no interfacing required between cpo and itself; but interfacing required for the rest.
	#INTERFACING TYPES:
	#0=no interfacing required, only allowed between cpo and itself.
	#1=scan by scan interfacing. cpo #1 will scan first followed by cpo #2 and subsequent cpo's.
	#2=simultaneous scan interfacing, interleaving each integration in the scans. cpo #1 first integration or beam direction
		# will run followed by cpo #2's first integration,etc. if cpo #1's len(scan) is greater than cpo #2's, cpo #2's last
		# integration will run and then all the rest of cpo #1's will continue until the full scan is over.
		# cpo 1 and 2 must have the same scan boundary, if any boundary. All other may differ.
	#3=simultaneous same-sequence interfacing, using same pulse sequence. cpo A and B might have different frequencies (stereo) 
		# and/or may use different antennas. They might also have different pulse length but must have same mpinc, sequence, 
		# integration time. They must also have same len(scan), although they may use different directions in scan. They must 
		# have the same scan boundary if any. A time offset between the pulses starting may be set, with max value of mpinc.
		# cpo A and B will have integrations that run at the same time. 
	#4=simultaneous multi-sequence interfacing. This is more difficult and reduces receive time significantly. Researcher will 
		# need to ensure that the multi-sequence is really what they want. cpo A and cpo B will run simultaneously. Both
		# first pulses will start at the same moment or they may have a time offset so B pulse sequence begins later than A
		# pulse sequence in each integration.

	#interfacing[2,1]=1  
	#interfacing[1,2]=interfacing[2,1]

	# must set all possible interfacing variables. Is this straightforward? Is there a better way to do this?
