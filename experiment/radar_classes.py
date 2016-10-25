#!/usr/bin/python

# create pulse class, sequence class,
# seqrun class, scan class.


# Scans are made up of seqruns, these are typically a 3second time of the same pulse sequence pointing in one direction.
# Seqruns are made up of sequences, typically the same sequence run ave. 21 times after a clear frequency search.
# Sequences are made up of pulses, spaced according to a certain ptab
# Pulses are defined by pulsenumber, channels, pulseshift (if any), freq, pulse length, beamdir, and wavetype. 

import sys
import operator # easy sorting of list of class instances


class RadarPulse():

    def __init__(self, cp_object, pulsen, **kwargs): 
        'Pulse data, incl timing and channel data but not including phasing data'
        # pulse metadata will be send alongside separate phasing data
        # to avoid send 
        self.SOB=kwargs.get('SOB',False)
        self.EOB=kwargs.get('EOB',False)
        self.cpoid=cp_object.cpid[1]
        self.pulsen=pulsen
        self.channels=cp_object.channels
        self.pulse_shift=cp_object.pulse_shift[pulsen]
        self.freq=cp_object.freq
        self.pullen=cp_object.pullen
        #self.iwave_table=cp_object.iwave_table
        #self.qwave_table=cp_object.qwave_table
        #self.phshifts=
        #self.beamdir=cp_object.beamdir #not needed, pulse metadata need only include array of phaseshift for channels, need not know its beamdir.
        self.wavetype=cp_object.wavetype
        try:
            self.timing=cp_object.seqtimer+cp_object.mpinc*cp_object.sequence[pulsen]
        except IndexError:
            errmsg="Invalid Pulse Number %d for this CP_Object" % (pulsen)
            sys.exit(errmsg)

class Sequence():

    def __init__(self, aveperiod, seqn_keys): 
        #args=locals()
        #make a list of the cpos in this sequence.
        self.keys=seqn_keys
        self.cpos={}
        for i in seqn_keys:
            self.cpos[i]=aveperiod.cpos[i]
        #no need to make an interfacing dictionary - all interfacing at this point is PULSE.

        pulses=[]
        #beamdir={} # dictionary because there are multiple cpos and they might have different beam directions.
        if len(seqn_keys)==1 and len(self.cpos[seqn_keys[0]].sequence)==1:
            only_pulse=True
        else:
            only_pulse=False
            first_pulse_time=self.cpos[seqn_keys[0]].sequence[0]*self.cpos[seqn_keys[0]].mpinc+self.cpos[seqn_keys[0]].seqtimer
            first_cpoid=0
            last_pulse_time=self.cpos[seqn_keys[0]].sequence[-1]*self.cpos[seqn_keys[0]].mpinc+self.cpos[seqn_keys[0]].seqtimer
            last_cpoid=0
            for cpoid in self.cpos.keys():
                cpo_pulse_one_time=self.cpos[cpoid].sequence[0]*self.cpos[cpoid].mpinc+self.cpos[cpoid].seqtimer
                if cpo_pulse_one_time<first_pulse_time:
                    first_pulse_time=cpo_pulse_one_time
                    first_cpoid=cpoid
                cpo_last_pulse_time=self.cpos[cpoid].sequence[-1]*self.cpos[cpoid].mpinc+self.cpos[cpoid].seqtimer
                if cpo_last_pulse_time>last_pulse_time:
                    last_pulse_time=cpo_last_pulse_time
                    last_cpoid=cpoid

        for cpoid in seqn_keys:
            #beamdir[cpo]=self.cpos[cpo].beamdir[
            for i in range(len(self.cpos[cpoid].sequence)):
                if only_pulse==True:
                    pulses.append(RadarPulse(self.cpos[cpoid],i,SOB=True,EOB=True))
                elif cpoid==first_cpoid and i==0: 
                    pulses.append(RadarPulse(self.cpos[cpoid],i,S0B=True))
                elif cpoid==last_cpoid and i==len(self.cpos[cpoid].sequence)-1:
                    pulses.append(RadarPulse(self.cpos[cpoid],i,EOB=False))
                else:
                    pulses.append(RadarPulse(self.cpos[cpoid],i)) 
        # need to sort self.pulses because likely not in correct order with multiple cpos.
        self.pulses=sorted(pulses, key=operator.attrgetter('timing'))
        # 19 is the sample delay below; how do we calculate this?
        self.ssdelay=(self.cpos[seqn_keys[0]].nrang+19+10)*self.cpos[seqn_keys[0]].pullen
        for cpoid in seqn_keys:
            newdelay=(self.cpos[cpoid].nrang+29)*self.cpos[cpoid].pullen
            if newdelay>self.ssdelay:
                self.ssdelay=newdelay
        # get beamdir of the pulses
        

class AveragingPeriod():
    """Made up of multiple pulse sequences (integrations) for one integration time"""
    def __init__(self, scan, ave_keys): 
        #make a list of the cpos in this AveragingPeriod.
        self.keys=ave_keys
        self.cpos={}
        for i in ave_keys:
            self.cpos[i]=scan.cpos[i]
        
        #create a smaller interfacing dictioary for this to pass on.
        #this interfacing dictionary will not have INTT or SCAN types as those have been already removed.
        #this dictionary will only include the interfacing between CP objects in this AveragingPeriod,
        #therefore it will only include INTEGRATION and PULSE.
        interface_keys=[]
        for m in range(len(ave_keys)):
            for n in range(m+1,len(ave_keys)):
                interface_keys.append([ave_keys[m],ave_keys[n]])
        self.interface={}
        for p,q in interface_keys:
            self.interface[p,q]=scan.interface[p,q]

        # metadata for this AveragingPeriod: clear frequency search, integration time, number of averages goal,
        self.clrfrqf=[] #list of cpos in this Averaging Period that have a clrfrq search requirement.
        self.clrfrqrange=[] #list of ranges needed to be searched.
        for cpo in self.keys:
            if self.cpos[cpo].clrfrqf==1:
                self.clrfrqf.append(cpo)
                if self.cpos[cpo].clrfrqrange not in self.clrfrqrange:
                    self.clrfrqrange.append(self.cpos[cpo].clrfrqrange)

        self.intt=self.cpos[self.keys[0]].intt
        for cpo in self.keys:
            if self.cpos[cpo].intt!=self.intt:
                errmsg="CPO %d and %d are INTTIME mixed and do not have the same Averaging Period duration intt" % (self.keys[0], self.keys[cpo])
                sys.exit(errmsg)
        self.intn=self.cpos[self.keys[0]].intn
        for cpo in self.keys:
            if self.cpos[cpo].intn!=self.intn:
                errmsg="CPO %d and %d are INTTIME mixed and do not have the same NAVE goal intn" % (self.keys[0], self.keys[cpo])
                sys.exit(errmsg)

        # do not need beam information inside the AveragingPeriod, change this when iterating through the aveperiods in a scan.
        #self.scan_beams={} # could be different beam numbers and beam directions for the various CPO's in this AveragingPeriod.
        
        #determine how this averaging period is made by separating out the INTEGRATION mixed.
        self.cpo_integrations=self.get_integrations()
        self.integrations=[]
        for integration_cpo_list in self.cpo_integrations:
            self.integrations.append(Sequence(self,integration_cpo_list)) 
        
    def get_integrations(self):
        integ_combos=[]

        for num1,num2 in self.interface.keys():
            if self.interface[num1, num2]=="PULSE": # remove integration combos as those are what we are trying to separate.
                integ_combos.append([num1,num2]) # save only the keys that are combinations within inttime.

        integ_combos=sorted(integ_combos)
        #if [2,4] and [1,4], then also must be [1,2] in the scan_combos list!
        i=0
        while (i<len(integ_combos)):
            #print "i: ", i,        seq_combos[i]
            k=0
            while (k<len(integ_combos[i])):
                j=i+1
                while (j<len(integ_combos)):
                    #print "j: ", j, seq_combos[j]
                    if integ_combos[i][k]==integ_combos[j][0]:
                        add_n=integ_combos[j][1]
                        #print "Adding ", add_n, " to sequence combo ", seq_combos[i]
                        integ_combos[i].append(add_n) #combine the indices if there are 3+ cp_objects combining in same seq.
                        for m in range(0,len(integ_combos[i])-1): # try all values in seq_combos[i] except the last value, which is = to add_n.
                            try:
                                #print "Removing sequence combo ", seq_combos[i][m], add_n
                                integ_combos.remove([integ_combos[i][m],add_n]) # seq_combos[j][1] is the known last value in seq_combos[i]
                            except ValueError:
                                # if there is an error because that index does not exist!
                                errmsg='Interfacing not Valid: CPO %d and CPO %d are combined in-integration period and do not interface the same with CPO %d' % (integ_combos[i][m], integ_combos[i][k], add_n)
                                sys.exit(errmsg)
                        j=j-1 # this means that the former scan_combos[j] has been deleted and there are new values at index j, so decrement before incrementing in for.
                    j=j+1 # increment
                k=k+1
            i=i+1
        # now scan_combos is a list of lists, where a cp object occurs in only once in the nested list.
        for i in range(len(self.keys)): #only the cpo's in this scan.
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
                integ_combos.append([self.keys[i]]) # append the cpo on its own, is not scan combined.

        integ_combos=sorted(integ_combos) # should sort correctly?
        return integ_combos


class Scan():
    """Made up of AveragingPeriods at defined beam directions"""
    def __init__(self, controlprog, scan_keys): # *cpos... these objects are INTTIME, INTEGRATION, or PULSE mixed.
        # passing the list of cp_objects involved in this scan and the controlprog with all info for those cp objects.
        # NEED to determine how interfaced within this scan.
        self.keys=scan_keys
        # make a list of the cpos in this scan so we don't have to pass the whole controlprog.
        self.cpos={}
        for i in scan_keys: #scan keys is a list of ints that define what cp_objects are in this scan.
            self.cpos[i]=controlprog.cpo_list[i] # list of cp_objects in this scan.
        #self.cpos[1]()
        #create smaller interfacing dictionary for this scan to pass on.
        #this dictionary will only include the cpo's in this scan,
        # therefore it will not include any SCAN interfacing.
        interface_keys=[]
        for m in range(len(scan_keys)):
            for n in range(m+1, len(scan_keys)):
                interface_keys.append([scan_keys[m],scan_keys[n]])
        self.interface={}
        for p,q in interface_keys:
            self.interface[p,q]=controlprog.interface[p,q]
      
        # scan metadata - must be the same between all cpo's combined in scan.
        # Metadata includes:
        self.scanboundf=self.cpos[self.keys[0]].scanboundf
        for cpo in self.keys:        
            if self.cpos[i].scanboundf!=self.scanboundf:
                errmsg="Scan Boundary Flag not the Same Between CPO's %d and %d combined in Scan" % (self.keys[0], cpo)
                sys.exit(errmsg)
        if self.scanboundf==1:
            self.scanbound=self.cpos[self.keys[0]].scanbound
            for cpo in self.keys:        
                if self.cpos[i].scanbound!=self.scanbound:
                    errmsg="Scan Boundary not the Same Between CPO's %d and %d combined in Scan" % (self.keys[0], cpo)
                    sys.exit(errmsg)

        #NOTE: for now we assume that when INTTIME combined, the Averaging Periods of the various Cp_objects in the scan are
        #interleaved 1 then the other.
        
        # create a dictionary of beam directions for cpo # 
        self.beamdir={}
        self.scan_beams={}
        for cpo in self.keys:
            self.beamdir[cpo]=self.cpos[cpo].beamdir
            self.scan_beams[cpo]=self.cpos[cpo].scan

        # determine how many averaging periods to make by separating out the INTTIME mixed.
        self.cpo_inttimes=self.get_inttimes()
        
        # However we need to make sure the length of the scan for this AveragingPeriod is the same for all CPO's.
        for cpos in self.get_inttimes():
            for cpo in cpos:
                if len(self.scan_beams[cpo])!=len(self.scan_beams[cpos[0]]):
                    errmsg="CPO %d and %d are mixed within the Averaging Period but do not have the same number of AveragingPeriods in their scan" % (self.keys[0], cpo)

        self.aveperiods=[]
        for aveperiod_cpo_list in self.cpo_inttimes:
            # each component is an inttime, we should create AveragingPeriods and pass the cpo's in that period.
            self.aveperiods.append(AveragingPeriod(self,aveperiod_cpo_list)) # append an instance of AveragingPeriod giving it the scan for the cp_objects 
                #and the list of cp_objects included in that 
        
        #order of the Averaging Periods - will be in cpo # order.
        #self.aveperiods=sorted(self.aveperiods, key=operator.attrgetter('timing'))


    def get_inttimes(self):
        #interface_keys passed are keys for all objects within a single scan
        intt_combos=[]

        for num1,num2 in self.interface.keys():
            if self.interface[num1, num2]=="PULSE" or self.interface[num1, num2]=="INTEGRATION": # remove intt combos as those are what we are trying to separate.
                intt_combos.append([num1,num2]) # save only the keys that are combinations within inttime.

        intt_combos=sorted(intt_combos)
        #if [2,4] and [1,4], then also must be [1,2] in the scan_combos list!
        i=0
        while (i<len(intt_combos)):
            #print "i: ", i,        seq_combos[i]
            k=0
            while (k<len(intt_combos[i])):
                j=i+1
                while (j<len(intt_combos)):
                    #print "j: ", j, seq_combos[j]
                    if intt_combos[i][k]==intt_combos[j][0]:
                        add_n=intt_combos[j][1]
                        #print "Adding ", add_n, " to sequence combo ", seq_combos[i]
                        intt_combos[i].append(add_n) #combine the indices if there are 3+ cp_objects combining in same seq.
                        for m in range(0,len(intt_combos[i])-1): # try all values in seq_combos[i] except the last value, which is = to add_n.
                            try:
                                #print "Removing sequence combo ", seq_combos[i][m], add_n
                                intt_combos.remove([intt_combos[i][m],add_n]) # seq_combos[j][1] is the known last value in seq_combos[i]
                            except ValueError:
                                # if there is an error because that index does not exist!
                                errmsg='Interfacing not Valid: CPO %d and CPO %d are combined in-scan and do not interface the same with CPO %d' % (intt_combos[i][m], intt_combos[i][k], add_n)
                                sys.exit(errmsg)
                        j=j-1 # this means that the former scan_combos[j] has been deleted and there are new values at index j, so decrement before incrementing in for.
                    j=j+1 # increment
                k=k+1
            i=i+1
        # now scan_combos is a list of lists, where a cp object occurs in only once in the nested list.
        for i in range(len(self.keys)): #only the cpo's in this scan.
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
                intt_combos.append([self.keys[i]]) # append the cpo on its own, is not scan combined.

        intt_combos=sorted(intt_combos) # should sort correctly?
        return intt_combos

