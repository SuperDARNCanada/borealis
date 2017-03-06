#!/usr/bin/python

# A new radar control program.
import sys
import math

import numpy as np
import json

import radar_classes
from cpobject import CPObject, interfacing, if_type

class ControlProg():
    """Class combining control program objects, defining how they 
    interface and some overall metadata
    """

    def __init__(self, cponum, cpid):    

        self.cpo_id=cpid 
        # Unique ID for each new cp.
        self.cpo_num=cponum 
        # Number of CPObjects in this program.
        cpo_list=[]
        for num in range(self.cpo_num):
            cpo_list.append(CPObject())
            cpo_list[num].cpid[1]=num 
            # Second number in cpid array is the ID of this cp_object
            #   in the controlprog.
            cpo_list[num].cpid[0]=self.cpo_id
        self.cpo_list=cpo_list

        # Next some metadata that you can change, with defaults.
        # TODO: make default none and have a function that will 
        #   calculate something appropriate if left blank.
        self.txctrfreq=12000 # in kHz.
        self.txrate=12000000 # sampling rate, samples per sec
        self.rxctrfreq=12000 # in kHz. 
        # rx sampling rate is set in config.
        self.xcf=1 
        # Get cross-correlation data in processing block.
        self.acfint=1 
        # Determine lag-zero interferometer power in fitacf.

        self.interface=interfacing(self.cpo_num) 
        # Dictionary of how each cpo interacts with the other cpos.
        # Default is "NONE" for all, must be modified in experiment.
        # NOTE keys are as such: (0,1), (0,2), (1,2), NEVER 
        # includes (2,0) etc. The only interface options are:
        # if_types=frozenset(['NONE', 'SCAN', 'INTTIME', 'INTEGRATION',
        # 'SAME_SEQ', 'MULTI_SEQ'])

        """ 
        INTERFACING TYPES:
        
        NONE : Only the default, must be changed.
        SCAN : Scan by scan interfacing. cpo #1 will scan first 
            followed by cpo #2 and subsequent cpo's.
        INTTIME : nave by nave interfacing (full integration time of
             one sequence, then the next). Time/number of sequences 
            dependent on intt and intn in cp_object. Effectively 
            simultaneous scan interfacing, interleaving each 
            integration time in the scans. cpo #1 first inttime or 
            beam direction will run followed by cpo #2's first inttime,
            etc. if cpo #1's len(scan) is greater than cpo #2's, cpo 
            #2's last integration will run and then all the rest of cpo
            #1's will continue until the full scan is over. CPObject 1
            and 2 must have the same scan boundary, if any boundary. 
            All other may differ.
        INTEGRATION : integration by integration interfacing (one 
            sequence of one cp_object, then the next). CPObject #1 and
            CPO #2 must have same intt and intn. Integrations will 
            switch between one and the other until time is up/nave is
            reached.
        PULSE : Simultaneous sequence interfacing, pulse by pulse 
            creates a single sequence. CPO A and B might have different
            frequencies (stereo) and/or may have different pulse 
            length, mpinc, sequence, but must have the same integration
            time. They must also have same len(scan), although they may
            use different directions in scan. They must have the same 
            scan boundary if any. A time offset between the pulses 
            starting may be set (seq_timer in cp_object). CPObject A 
            and B will have integrations that run at the same time. 
        """

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

    def build_Scans(self):
        """Will run after a controlprogram instance is set up and
        modified
        """

        with open('../config.ini') as config_data:
            self.config=json.load(config_data)

        # Check interfacing
        self.check_objects()
        self.check_interfacing()
        # Find separate scans.
        self.cpo_scans=self.get_scans() 
        # Returns list of scan lists. Each scan list is a list of the
        #   cpo numbers for that scan.
        self.scan_objects=[]
        for scan_cpo_list in self.cpo_scans:
            self.scan_objects.append(radar_classes.Scan(self, scan_cpo_list)) 
        # Append a Scan instance, passing this controlprog, list of cpo
        #   numbers to include in scan.
    
    def check_objects(self):
        if (self.cpo_num<1):
            errmsg="Error: No objects in control program"
            sys.exit(errmsg)
        for cpo in range(self.cpo_num):
            selferrs=self.cpo_list[cpo].selfcheck()
            if (not selferrs): 
                # If returned error dictionary is empty
                continue
            errmsg="Self Check Errors Occurred with Object Number : {} \nSelf \
                Check Errors are : {}".format(cpo, selferrs)
            sys.exit(errmsg)
        else: # no break
            print "No Self Check Errors. Continuing..."
        return None

    def check_interfacing(self):
        # Check that the keys in the interface are not NONE and are 
        #   valid.
        for key in self.interface.keys():
            if self.interface[key]=="NONE":
                errmsg='Interface keys are still default, must set key \
                    {}'.format(key)
                sys.exit(errmsg)

        for num1,num2 in self.interface.keys(): 
            if ((num1>=self.cpo_num) or (num2>=self.cpo_num) or (num1<0) 
                    or (num2<0)):
                errmsg='Interfacing key ({}, {}) is not necessary and not \
                    valid'.format(num1, num2)
                sys.exit(errmsg)
            if self.interface[num1, num2] not in if_type():
                errmsg='Interfacing Not Valid Type between CPO {} and CPO \
                    {}'.format(num1, num2)
                sys.exit(errmsg)
        return None

    def get_scans(self):
        """Take my own interfacing and get info on how many scans and 
            which cpos make which scans
        """
        scan_combos=[]
    
        for num1,num2 in self.interface.keys():
            if (self.interface[num1, num2]=="PULSE" or 
                    self.interface[num1, num2]=="INT_TIME" or 
                    self.interface[num1, num2]=="INTEGRATION"):
                scan_combos.append([num1,num2])
            # Save the keys that are scan combos.

        scan_combos=sorted(scan_combos)
        #if [2,4] and [1,4], then also must be [1,2] in the scan_combos
        i=0
        while (i<len(scan_combos)):
            k=0
            while (k<len(scan_combos[i])):
                j=i+1
                while (j<len(scan_combos)):
                    if scan_combos[i][k]==scan_combos[j][0]:
                        add_n=scan_combos[j][1]
                        scan_combos[i].append(add_n)   
                        # Combine the indices if there are 3+ CPObjects
                        #   combining in same seq.
                        for m in range(0,len(scan_combos[i])-1): 
                            # Try all values in seq_combos[i] except 
                            #    the last value, which is = to add_n.
                            try:
                                scan_combos.remove([scan_combos[i][m],add_n]) 
                                # seq_combos[j][1] is the known last 
                                #   value in seq_combos[i]
                            except ValueError:
                                errmsg='Interfacing not Valid: CPO {} and CPO \
                                    {} are combined in-scan and do not \
                                    interface the same with CPO {}'.format(
                                    scan_combos[i][m], scan_combos[i][k], add_n
                                    )
                                sys.exit(errmsg)
                        j=j-1 
                        # This means that the former scan_combos[j] has
                        #   been deleted and there are new values at 
                        #   index j, so decrement before 
                        #   incrementing in for.
                    j=j+1 
                k=k+1
            i=i+1
        # now scan_combos is a list of lists, where a cp object occurs
        #   only once in the nested list.
        for cpo in range(self.cpo_num):
            found=False
            for i in range(len(scan_combos)):
                for j in range(len(scan_combos[i])):
                    if cpo==scan_combos[i][j]:
                        found=True
                        break
                if found==False:
                    continue
                break
            else: # no break
                scan_combos.append([cpo]) 
            # Append the cpo on its own, is not scan combined.

        scan_combos=sorted(scan_combos)
        return scan_combos
                           
#    def get_integrations(self, seq_combos):
#        """Determine how the sequence combos are interfaced before building"""
#        # seq_combos is a list of lists that defines how many different types of sequences there are in this control program.
#        interface_integrations=[]
#        integ # will create a dictionary using the sequence combos as values for keys (number of different integrations)
#        for num1,num2 in self.interface.keys():
#            if self.interface[num1, num2]=="INTEGRATION"
#                interface_integrations.append([num1,num2]) # save keys of integration combos.
#        interface_integrations=sorted(interface_integrations)
#        for num1,num2 in interface_integrations:
#            for i in range(len(seq_combos)):
#                for k in seq_combos[i]:
#                    if k==num1:
#                        # we have found an interface integration between pulse sequences.
                        
#    def get_wavetables(self):
#        #NOTE: will there be any other wavetypes.
#        self.iwave_table=[]
#        self.qwave_table=[]
#
#        for cpo in self.cpo_list:
#            if cpo.wavetype=="SINE":
#                wave_table_len=8192
#                for i in range(0, wave_table_len):
#                    cpo.iwave_table.append(math.cos(i*2*math.pi/wave_table_len))
#                    cpo.qwave_table.append(math.sin(i*2*math.pi/wave_table_len))
#
#            else:
#                errmsg="Wavetype %s not defined" % (cpo.wavetype)
#                sys.exit(errmsg)
#
#        #return iwave_table, qwave_table
 

#    def get_seq_combos(self): 
#        """TAKE MY OWN INTERFACING SPECS AND COMBINE INTO WORKABLE 
#        ARRAY - NOT USED
#        """
#        # check the interfacing, and combine sequences first.
#        seq_combos=[]
#    
#        for num1,num2 in self.interface.keys():
#            if self.interface[num1, num2]=="PULSE": # Changed the interface options or self.interface[num1, num2]=="MULTI_SEQ":
#                seq_combos.append([num1,num2]) # save the keys that are sequence combinations.
#
#        seq_combos=sorted(seq_combos)
#        #print seq_combos
#        #if [2,4] and [1,4], then also must be [1,2] in the seq_combos list!
#        i=0
#        while (i<len(seq_combos)):
#        #for i in range(0,(len(seq_combos)-1)):     
#            #print "i: ", i,        seq_combos[i]
#            k=0
#            #for k in range(len(seq_combos[i])):
#            while (k<len(seq_combos[i])):
#                j=i+1
#                while (j<len(seq_combos)):
#                #for j in range(i+1,len(seq_combos)):
#                    #print "j: ", j, seq_combos[j]
#                    if seq_combos[i][k]==seq_combos[j][0]:
#                        add_n=seq_combos[j][1]
#                        #print "Adding ", add_n, " to sequence combo ", seq_combos[i]
#                        seq_combos[i].append(add_n) #combine the indices if there are 3+ cp_objects combining in same seq.
#                        #if seq_combos[i][1]>seq_combos[j][1]:
#                            #seq_combos.remove([seq_combos[j][1],seq_combos[i][1]])
#                        #else:
#                            #print seq_combos[i][1], seq_combos[j][1]
#                        for m in range(0,len(seq_combos[i])-1): # try all values in seq_combos[i] except the last value, which is = to add_n.
#                            try:
#                                #print "Removing sequence combo ", seq_combos[i][m], add_n
#                                seq_combos.remove([seq_combos[i][m],add_n]) # seq_combos[j][1] is the known last value in seq_combos[i]
#                            except ValueError:
#                                # if there is an error because that index does not exist!
#                                errmsg='Interfacing not Valid: CPO %d and CPO %d are a sequence combination and do not interface the same with CPO %d' % (seq_combos[i][m], seq_combos[i][k], add_n)
#                                sys.exit(errmsg)
#                                #print seq_combos
#                        j=j-1 # this means that the former seq_combos[j] has been deleted and there are new values at index j, so decrement before incrementing in for.
#                    j=j+1 # increment
#                k=k+1
#            i=i+1
#
#        # now seq_combos is a list of lists, where a cp object occurs in only once in the nested list.
#        for cpo in range(self.cpo_num):
#            found=False
#            for i in range(len(seq_combos)):
#                for j in range(len(seq_combos[i])):
#                    if cpo==seq_combos[i][j]:
#                        found=True
#                        break
#                if found==False:
#                    continue
#                break
#            else: # no break
#                seq_combos.append([cpo]) # append the cpo on its own, is not sequence combined.
#
#        seq_combos=sorted(seq_combos)
#        return seq_combos
# 
