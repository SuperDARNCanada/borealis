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

#class RadarPulse():
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


class Sequence():

    def __init__(self, aveperiod, seqn_keys): 
        # TODO: pass clear frequencies to pass to pulses
        #args=locals()

        # Get the CPObjects in this sequence.
        self.keys=seqn_keys
        self.cpos={}
        for i in seqn_keys:
            self.cpos[i]=aveperiod.cpos[i]
        # All interfacing at this point is PULSE.
        
        # TODO: use number of frequencies to determine power output of
        #   each frequency (1/2, 1/3)
        
        pulse_remaining=True
        index=0
        pulse_time=[] 
        # List of lists formatted [timing, cpo, pulse_n],...
        while (pulse_remaining):
            cpos_done=0
            for cpoid in seqn_keys:
                try:
                    timing=(self.cpos[cpoid].sequence[index]
                        * self.cpos[cpoid].mpinc)
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

        # 19 is the sample delay below; how do we calculate this?
        self.ssdelay=((self.cpos[seqn_keys[0]].nrang+19+10)
                        * self.cpos[seqn_keys[0]].pullen)
        for cpoid in seqn_keys:
            newdelay=(self.cpos[cpoid].nrang+29)*self.cpos[cpoid].pullen
            if newdelay>self.ssdelay:
                self.ssdelay=newdelay
        # The delay is long enough for any CPO pulse length and nrang
        #    to be accounted for.
        # Time before the first pulse is 70 us when RX and TR set up
        #    for the first pulse. The timing to the last pulse is
        #    added, as well as its pulse length and the ssdelay.
        self.sstime=(70+self.pulse_time[-1][0]
            + self.cpos[pulse_time[-1][1]].pullen+self.ssdelay)
        # TODO: this could be wrong, may want to calculate after 
        #    pulses are combined. Pulse length may be longer on 2nd
        #    last pulse ...
        # NOTE: beamdirs are set up in Scan.

        self.numberofreceivesamples=int(aveperiod.rxrate 
            * self.sstime*1e-6)
        # Set up the combined pulse list
        self.combined_pulse_list=[]
        fpulse_index=0
        while (fpulse_index<len(self.pulse_time)):
            pulse=self.pulse_time[fpulse_index][:]
            # Pulse is just a list of timing (us), cpoid, pulse number

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
                            + self.cpos[pulse[1]].pullen + 125):
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
        for pulse_index in range(0,len(self.combined_pulse_list)):
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
                if (self.cpos[pulse_cpoid].pulse_shift[pulse_n] != 
                        self.cpos[last_pulse_cpoid].pulse_shift[last_pulse_n]):
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


class AveragingPeriod():
    
    """ Made up of multiple pulse sequences (integrations) for one 
    integration time.
    """

    def __init__(self, scan, ave_keys): 
        #make a list of the cpos in this AveragingPeriod.
        self.rxrate=scan.rxrate
        self.keys=ave_keys
        self.cpos={}
        for i in ave_keys:
            self.cpos[i]=scan.cpos[i]
        
        # Create a smaller interfacing dictionary for this to pass on.
        # INTT or SCAN types have been already removed, so this dict
        #   will only include the interfacing between CP objects in 
        #   this AveragingPeriod, so only include INTEGRATION and PULSE
        interface_keys=[]
        for m in range(len(ave_keys)):
            for n in range(m+1,len(ave_keys)):
                interface_keys.append([ave_keys[m],ave_keys[n]])
        self.interface={}
        for p,q in interface_keys:
            self.interface[p,q]=scan.interface[p,q]

        # Metadata for this AveragingPeriod: clear frequency search, 
        #   integration time, number of averages goal
        self.clrfrqf=[] 
        # List of cpos in this Averaging Period that have a clrfrq 
        #   search requirement.
        self.clrfrqrange=[] 
        # List of ranges needing to be searched.
        for cpo in self.keys:
            if self.cpos[cpo].clrfrqf==1:
                self.clrfrqf.append(cpo)
                if self.cpos[cpo].clrfrqrange not in self.clrfrqrange:
                    self.clrfrqrange.append(self.cpos[cpo].clrfrqrange)

        self.intt=self.cpos[self.keys[0]].intt
        for cpo in self.keys:
            if self.cpos[cpo].intt!=self.intt:
                errmsg="CPO %d and %d are INTTIME mixed and do not have the \
                    same Averaging Period duration intt" % (self.keys[0], 
                    self.keys[cpo])
                sys.exit(errmsg)
        self.intn=self.cpos[self.keys[0]].intn
        for cpo in self.keys:
            if self.cpos[cpo].intn!=self.intn:
                errmsg="CPO %d and %d are INTTIME mixed and do not have the \
                    same NAVE goal intn" % (self.keys[0], self.keys[cpo])
                sys.exit(errmsg)

        # NOTE: Do not need beam information inside the AveragingPeriod
        # will change this when iterating through the aveperiods in a scan.
        
        # Determine how this averaging period is made by separating out
        #   the INTEGRATION mixed.
        self.cpo_integrations=self.get_integrations()
        self.integrations=[]
        for integration_cpo_list in self.cpo_integrations:
            self.integrations.append(Sequence(self,integration_cpo_list)) 
      
  
    def get_integrations(self):
        integ_combos=[]

        # Remove INTEGRATION combos as we are trying to separate those.
        for num1,num2 in self.interface.keys():
            if self.interface[num1, num2]=="PULSE":
                integ_combos.append([num1,num2]) 

        integ_combos=sorted(integ_combos)
        #if [2,4] and [1,4], then also must be [1,2] in the combos list
        i=0
        while (i<len(integ_combos)):
            k=0
            while (k<len(integ_combos[i])):
                j=i+1
                while (j<len(integ_combos)):
                    if integ_combos[i][k]==integ_combos[j][0]:
                        add_n=integ_combos[j][1]
                        integ_combos[i].append(add_n)
                        # Combine the indices if there are 3+ CPObjects
                        #   combining in same seq.
                        for m in range(0,len(integ_combos[i])-1): 
                            # Try all values in seq_combos[i] except 
                            #   the last value, which is = to add_n.
                            try:
                                integ_combos.remove([integ_combos[i][m],add_n]) 
                                # seq_combos[j][1] is the known last 
                                #   value in seq_combos[i]
                            except ValueError:
                                errmsg="Interfacing not Valid: CPO %d and CPO \
                                    %d are combined in-integration period and \
                                    do not interface the same with CPO %d" % (
                                    integ_combos[i][m], integ_combos[i][k], 
                                    add_n)
                                sys.exit(errmsg)
                        j=j-1 
                        # this means that the former scan_combos[j]
                        #   has been deleted and there are new values
                        #   at index j, so decrement before 
                        #   incrementing.
                    j=j+1
                k=k+1
            i=i+1
        # Now combos is a list of lists, where a cpobject occurs only
        #   once in the nested list.
        for i in range(len(self.keys)): 
            found=False
            for k in range(len(integ_combos)):
                for j in range(len(integ_combos[k])):
                    if self.keys[i]==integ_combos[k][j]:
                        found=True
                        break
                if found==False:
                    continue
                break
            else: # no break
                integ_combos.append([self.keys[i]]) 
                # append the cpo on its own, is not scan combined.
        integ_combos=sorted(integ_combos)
        return integ_combos


class Scan():
    """ Made up of AveragingPeriods at defined beam directions.
    """

    def __init__(self, controlprog, scan_keys):
        self.rxrate=controlprog.config[u'rx_sample_rate']
        self.keys=scan_keys
        self.cpos={}
        for i in scan_keys: 
            self.cpos[i]=controlprog.cpo_list[i]
        
        # Create smaller interfacing dictionary for this scan to pass.
        # This dictionary will only include the cpo's in this scan,
        #   therefore it will not include any SCAN interfacing.
        interface_keys=[]
        for m in range(len(scan_keys)):
            for n in range(m+1, len(scan_keys)):
                interface_keys.append([scan_keys[m],scan_keys[n]])
        self.interface={}
        for p,q in interface_keys:
            self.interface[p,q]=controlprog.interface[p,q]
      
        # scan metadata - must be the same between all cpo's combined
        # in scan.  Metadata includes:
        self.scanboundf=self.cpos[self.keys[0]].scanboundf
        for cpo in self.keys:        
            if self.cpos[i].scanboundf!=self.scanboundf:
                errmsg="Scan Boundary Flag not the Same Between CPO's %d and \
                    %d combined in Scan" % (self.keys[0], cpo)
                sys.exit(errmsg)
        if self.scanboundf==1:
            self.scanbound=self.cpos[self.keys[0]].scanbound
            for cpo in self.keys:        
                if self.cpos[i].scanbound!=self.scanbound:
                    errmsg="Scan Boundary not the Same Between CPO's %d and %d \
                         combined in Scan" % (self.keys[0], cpo)
                    sys.exit(errmsg)

        # NOTE: for now we assume that when INTTIME combined, the 
        #   AveragingPeriods of the various Cp_objects in the scan are
        #   just interleaved 1 then the other.
        
        # Create a dictionary of beam directions for cpo # 
        self.beamdir={}
        self.scan_beams={}
        for cpo in self.keys:
            self.beamdir[cpo]=self.cpos[cpo].beamdir
            self.scan_beams[cpo]=self.cpos[cpo].scan

        # Determine how many averaging periods to make by separating 
        #   out the INTTIME mixed.
        self.cpo_inttimes=self.get_inttimes()
        
        # However we need to make sure the length of the scan for this
        #   AveragingPeriod is the same for all CPO's.
        for cpos in self.get_inttimes():
            for cpo in cpos:
                if len(self.scan_beams[cpo])!=len(self.scan_beams[cpos[0]]):
                    errmsg="CPO %d and %d are mixed within the AveragingPeriod \
                         but do not have the same number of AveragingPeriods \
                        in their scan" % (self.keys[0], cpo)

        self.aveperiods=[]
        for aveperiod_cpo_list in self.cpo_inttimes:
            # Each component is an inttime, we should create 
            # AveragingPeriods and pass the cpo's in that period.
            self.aveperiods.append(AveragingPeriod(self,aveperiod_cpo_list))
        
        #order of the Averaging Periods - will be in cpo # order.
        #self.aveperiods=sorted(self.aveperiods, key=operator.attrgetter('timing'))

    def get_inttimes(self):
        intt_combos=[]

        for num1,num2 in self.interface.keys():
            if (self.interface[num1, num2]=="PULSE" or 
                    self.interface[num1, num2]=="INTEGRATION"): 
                intt_combos.append([num1,num2]) 
        # Save only the keys that are combinations within inttime.

        intt_combos=sorted(intt_combos)
        #if [2,4] and [1,4], then also must be [1,2] in the combos list
        i=0
        while (i<len(intt_combos)):
            k=0
            while (k<len(intt_combos[i])):
                j=i+1
                while (j<len(intt_combos)):
                    if intt_combos[i][k]==intt_combos[j][0]:
                        add_n=intt_combos[j][1]
                        intt_combos[i].append(add_n) 
                        # Combine the indices if there are 3+ CPObjects
                        #   combining in same seq.
                        for m in range(0,len(intt_combos[i])-1): 
                        # Try all values in seq_combos[i] except the 
                        #   last value, which is = to add_n.
                            try:
                                intt_combos.remove([intt_combos[i][m],add_n]) 
                                # seq_combos[j][1] is the known last 
                                #   value in seq_combos[i]
                            except ValueError:
                                errmsg='Interfacing not Valid: CPO %d and CPO \
                                    %d are combined in-scan and do not \
                                    interface the same with CPO %d' % (
                                    intt_combos[i][m], intt_combos[i][k], 
                                    add_n)
                                sys.exit(errmsg)
                        j=j-1 
                        # This means that the former scan_combos[j] has 
                        #   been deleted and there are new values at 
                        #   index j, so decrement before 
                        #   incrementing.
                    j=j+1
                k=k+1
            i=i+1
        # Now scan_combos is a list of lists, where a cpobject occurs 
        #   only once in the nested list.
        for i in range(len(self.keys)):
            found=False
            for k in range(len(intt_combos)):
                for j in range(len(intt_combos[k])):
                    if self.keys[i]==intt_combos[k][j]:
                        found=True
                        break
                if found==False:
                    continue
                break
            else: # no break
                intt_combos.append([self.keys[i]]) 
                # Append the cpo on its own, is not scan combined.
        intt_combos=sorted(intt_combos) 
        return intt_combos

