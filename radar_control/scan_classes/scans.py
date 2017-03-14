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

import averaging_periods
import sequences


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

