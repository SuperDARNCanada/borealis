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

import sequences

class AveragingPeriod():
    
    """ Made up of multiple pulse sequences (integrations) for one 
    integration time.
    """

    def __init__(self, scan, ave_keys): 
        #make a list of the cpos in this AveragingPeriod.
        self.rxrate=scan.rxrate
        self.keys=ave_keys # REVIEW #26 Why is this called a key
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

        # TODO: SET UP CLEAR FREQUENCY SEARCH CAPABILITY
        for cpo in self.keys:
            if self.cpos[cpo]['clrfrqf'] == 1:
                self.clrfrqf.append(cpo)
                if self.cpos[cpo]['clrfrqrange'] not in self.clrfrqrange:
                    self.clrfrqrange.append(self.cpos[cpo]['clrfrqrange'])

        self.intt=self.cpos[self.keys[0]]['intt']
        for cpo in self.keys:
            if self.cpos[cpo]['intt'] != self.intt:
                errmsg="CPO %d and %d are INTTIME mixed and do not have the \
                    same Averaging Period duration intt" % (self.keys[0], 
                    self.keys[cpo])
                sys.exit(errmsg)
        self.intn=self.cpos[self.keys[0]]['intn']
        for cpo in self.keys:
            if self.cpos[cpo]['intn'] != self.intn:
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
            self.integrations.append(sequences.Sequence(self,integration_cpo_list)) 
      
  
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

