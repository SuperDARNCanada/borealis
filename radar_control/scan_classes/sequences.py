#!/usr/bin/python

""" Scans are made up of AveragingPeriods, these are typically a 3sec time of 
the same pulse sequence pointing in one direction.  AveragingPeriods are made 
up of Sequences, typically the same sequence run ave. 21 times after a clear 
frequency search.  Sequences are made up of pulse_time lists, which give 
timing, CPObject, and pulsenumber. CPObject provides channels, pulseshift (if
any), freq, pulse length, beamdir, and wavetype. 
"""

import sys
import operator 

#class RadarPulse(): # REVIEW #33 Can remove this commented code
#
#    def __init__(self, cp_object, phase): #pulsen, **kwargs): 
#        'Pulse data, incl timing and channel data but not including phasing data'
#        # pulse metadata will be send alongside separate phasing data
#        # to avoid send 
#        #self.SOB=kwargs.get('SOB',False)
#        #self.EOB=kwargs.get('EOB',False)
#        self.cpoid=cp_object.cpid[1]
#        #self.pulsen=pulsen
#        self.channels=cp_object.channels
#        self.phase=phase # in degrees
#        #self.pulse_shift=cp_object.pulse_shift[pulsen]
#        self.freq=cp_object.freq
#        self.pullen=cp_object.pullen
#        #self.iwave_table=cp_object.iwave_table
#        #self.qwave_table=cp_object.qwave_table
#        #self.phshifts=
#        self.wavetype=cp_object.wavetype
#      #  try:
#      #      self.timing=cp_object.seqtimer
#      #          +cp_object.mpinc*cp_object.sequence[pulsen]
#      #  except IndexError:
#      #      errmsg="Invalid Pulse Number %d for this CP_Object" % (pulsen)
#      #      sys.exit(errmsg)


class Sequence(): # REVIEW #41 We think this should be refactored - there are member variables scattered throughout, it's all one function (init) there are no getters/setters. It's quite long and difficult to debug. There are a few generalized pythonic ways to do some of these things that would reduce code length and complexity. Possibly while loops can be changed from current method with boolean flag to while(True) with breaks when you currently set the boolean to False.
# REVIEW #41 aveperiod appears to be an object of type AveragingPeriod, which logically contains Sequences, this creates too much coupling between the two object types - it looks like Sequence only needs a couple pieces of information (rxrate, cpos) these could be passed in individually instead of the entire Aveperiod object.
    def __init__(self, aveperiod, seqn_keys): # REVIEW #1 We're not sure what seqn_keys is from the name and most of the code. It appears to be an index into the cpos.
        # TODO: pass clear frequencies to pass to pulses
        #args=locals()

        # Get the CPObjects in this sequence.
        self.keys=seqn_keys
        self.cpos={}
        for i in seqn_keys:
            self.cpos[i]=aveperiod.cpos[i] # REVIEW #41 Should use 'getters' and 'setters' when accessing member variables of a class instead of using the dot notation (example - aveperiod.get_cpos(i))
        # All interfacing at this point is PULSE.
        
        # TODO: use number of frequencies to determine power output of
        #   each frequency (1/2, 1/3)
        
        pulse_remaining=True
        index=0
        pulse_time=[] 
        # List of lists formatted [timing, cpo, pulse_n],...
        while (pulse_remaining): # REVIEW #39 We think this can be replaced by for(cpoid in #seqn_keys):
        #for pulse_index, pulse in enumerate(self.cpos[cpoid]['sequence']):
         #       inter_pulse_time = pulse*mpinc
          #      pulse_time.append([inter_pulse_time, cpoid, pulse_index)

#pulse_time.sort()
#self.pulse_time = pulse_time
            cpos_done=0
            for cpoid in seqn_keys:
                try:
                    timing=(self.cpos[cpoid]['sequence'][index]
                        * self.cpos[cpoid]['mpinc'])
                    pulsen=index
                    pulse_time.append([timing,cpoid,pulsen])
                except IndexError:     
                    cpos_done=cpos_done+1
            if cpos_done==len(seqn_keys):
                pulse_remaining=False
            index=index+1
        pulse_time.sort() 
        # Will sort by timing first and then by cpo if timing =.
        self.pulse_time=pulse_time

        # NOTE: beamdirs are set up in Scan.

        # Set up the combined pulse list
        self.combined_pulse_list=[]
        fpulse_index=0
        while (fpulse_index<len(self.pulse_time)):
            pulse=self.pulse_time[fpulse_index][:] # REVIEW #28 We think that pulse would be well suited to being a dictionary
            # Pulse is just a list of repeat, timing (us), cpoid, pulse number # REVIEW #1 pulse is actually (timing, cpoid, pulse number)

            # Determine if we are combining samples based on timing of
            #   pulses
            combine_pulses=True
            lpulse_index=fpulse_index
            pulse_list=[]        
            pulse_list.append(False) 
            # Will change repeat value later if True.
            pulse_list.append(pulse[0])
            # Append the start time.
            pulse[0]=0
            pulse_list.append(pulse)
            if fpulse_index!=len(self.pulse_time)-1:
                # If not the last index
                while (combine_pulses):
                    if (self.pulse_time[lpulse_index+1][0] <= pulse_list[1]
                            + self.cpos[pulse[1]]['pulse_len'] + 125):
                        # 125 us is for two TR/RX times.
                        lpulse_index=lpulse_index+1
                        next_pulse=self.pulse_time[lpulse_index][:]
                        next_pulse[0]=next_pulse[0]-pulse_list[1]
                        pulse_list.append(next_pulse)
                        if (lpulse_index == len(self.pulse_time) - 1):
                            combine_pulses=False
                    else:
                        combine_pulses=False
            self.combined_pulse_list.append(pulse_list)
            fpulse_index=lpulse_index+1
            # Jump ahead depending how many pulses we've combined.
            # Combined pulse list is a list of lists of pulses, 
            #   combined as to how they are sent as samples to 
            #   driver.

        # Find repeats now and replace that spot in list with just
        # [True,timing]
        for pulse_index in range(0,len(self.combined_pulse_list)): # REVIEW #33 We can get rid of the 'if pulse-index==0' code by starting pulse_index from 1 (range(1,len...)) and keeping the comment
            if pulse_index==0:
                continue 
                # First pulse, can't be a repeat.
            pulse=self.combined_pulse_list[pulse_index]
            last_pulse_index = pulse_index - 1
            isarepeat = True
            while (isarepeat):
                if self.combined_pulse_list[last_pulse_index][0]==True:
                    # Last pulse was a repeat, compare to previous
                    last_pulse_index=last_pulse_index - 1
                else:
                    isarepeat=False
            last_pulse=self.combined_pulse_list[last_pulse_index]
            if len(pulse)!=len(last_pulse):
                continue
                # Not a repeat, different number of CPObjects 
                #   combined in these pulses.
            for pulse_part_index in range(2,len(pulse)):
                #0th index in pulse is the repeat value
                #1st index in pulse is the start time
                if (pulse[pulse_part_index][0] != 
                    last_pulse[pulse_part_index][0]):
                    # Timing off the start of the pulse is the not same. 
                    break
                pulse_cpoid=pulse[pulse_part_index][1]
                last_pulse_cpoid=last_pulse[pulse_part_index][1]
                if pulse_cpoid != last_pulse_cpoid:
                    # CPOs aren't the same or in the same order
                    break
                pulse_n=pulse[pulse_part_index][2]
                last_pulse_n=last_pulse[pulse_part_index][2]
                if (self.cpos[pulse_cpoid]['pulse_shift'][pulse_n] != 
                        self.cpos[last_pulse_cpoid]['pulse_shift'][last_pulse_n]):
                    # Pulse shift for the pulse number is not the same
                    break
                # Nothing wrong with above pulse_part, check the next.
            else: # no break in above for loop
                pulse_start=pulse[1]
                self.combined_pulse_list[pulse_index]=[True,pulse_start] 
                # Replace the pulse list values with only the timing
                #   and the repeat value.
                # Non repeats will look like [False,pulse_time[i],
                #   pulse_time[i+1]...]
        
        # FIND THE max length of the last pulse 
        last_pulse_len = 0
        last_pulse_type = -1
        for i in range(len(self.combined_pulse_list)): # REVIEW #39 Might be a pythonic way to find max value in list (lambda functions?)
            if self.combined_pulse_list[last_pulse_type][0]==True:
               last_pulse_type -= 1
            else:
                break 
        for element in self.combined_pulse_list[last_pulse_type]:
            if type(element) == list: # REVIEW #28 Won't it always be at index 2 or higher? dictionary might help...
                # get cpo
                if self.cpos[element[1]]['pulse_len'] > last_pulse_len:
                    last_pulse_len = self.cpos[element[1]]['pulse_len']
        self.last_pulse_len = last_pulse_len
        
        # FIND the max scope sync time
        # 19 is the sample delay below; how do we calculate this? # REVIEW #6 TODO find out the reason for these magic numbers
        self.ssdelay=((self.cpos[seqn_keys[0]]['nrang']+19+10)# REVIEW #33 You can get rid of this line and just set ssdelay to 0, as you did above when finding the max last pulse len
                        * self.cpos[seqn_keys[0]]['pulse_len'])
        for cpoid in seqn_keys:
            newdelay=(self.cpos[cpoid]['nrang']+29)*self.cpos[cpoid]['pulse_len'] 
            if newdelay>self.ssdelay:
                self.ssdelay=newdelay
        # The delay is long enough for any CPO pulse length and nrang
        #    to be accounted for.

        # FIND the sequence time
        # Time before the first pulse is 70 us when RX and TR set up
        #    for the first pulse. The timing to the last pulse is
        #    added, as well as its pulse length and the ssdelay.n 
        self.seqtime=(70 + self.combined_pulse_list[-1][1] # REVIEW #29 Hard coded number here for RX/TR setup, consider using value from config file
            + self.last_pulse_len) # REVIEW #0 If we calculate the sequence time and include the 70us at the beginning, then we may be sampling for an extra 70us when we don't really need to. Just check to be consisten with the driver

        # FIND the total scope sync time and number of samples to receive.
        self.sstime=self.seqtime + self.ssdelay
        self.numberofreceivesamples=int(aveperiod.rxrate 
            * self.sstime*1e-6)
        print self.combined_pulse_list

