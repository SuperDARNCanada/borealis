#!/usr/bin/python

# A new radar control program.
import sys
from cp_object import cp_object, interfacing, if_type
import numpy as np
import radar_classes

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
        
        #self.build_Scans() #NOTE: this could happen if cp_objects are
            # passed on initialization to control program, then all
            # scans,etc. down to pulse metadata could be set up at that
            # time 



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
        # check interfacing
        self.check_interfacing()
        # find separate scans.
        self.cpo_scans=self.get_scans() # list of scans. Each scan is a list of the cpo number for that scan.
        self.scan_objects=[]
        for scan_cpo_list in self.cpo_scans:
            # each i is a new scan to build
            # we will do the separation of interfacing within the Scan class.
            self.scan_objects.append(radar_classes.Scan(self, scan_cpo_list)) # append a Scan instance, passing this controlprog, list of cpo numbers to include in scan.
                #inttimes=get_inttimes(scan_keys,)
                #cpo_scans[i]=inttimes 
                #for j in cpo_scans[
    
    def check_interfacing(self):
        # check that the keys in the interface are not NONE and are valid.
        for key in self.interface.keys():
            if self.interface[key]=="NONE":
                errmsg='Interface keys are still default, must set key {}'.format(key)
                sys.exit(errmsg)

        for num1,num2 in self.interface.keys(): # all keys in dictionary
            if (num1>=self.cpo_num) or (num2>=self.cpo_num) or (num1<0) or (num2<0):
                errmsg='Interfacing key (%d, %d) is not necessary and not valid' % (num1, num2)
                sys.exit(errmsg)
            if self.interface[num1, num2] not in if_type():
                errmsg='Interfacing Not Valid Type between CPO %d and CPO %d' % (num1, num2)
                sys.exit(errmsg)
        
        return None

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
                        
    def get_scans(self):
        """Take my own interfacing and get info on how many scans and which
            cpos make which scans"""
        scan_combos=[]
    
        for num1,num2 in self.interface.keys():
            if self.interface[num1, num2]=="PULSE" or self.interface[num1, num2]=="INT_TIME" or self.interface[num1, num2]=="INTEGRATION":
                scan_combos.append([num1,num2]) # save the keys that are scan combinations.

        scan_combos=sorted(scan_combos)
        #if [2,4] and [1,4], then also must be [1,2] in the scan_combos list!
        i=0
        while (i<len(scan_combos)):
            #print "i: ", i,        seq_combos[i]
            k=0
            while (k<len(scan_combos[i])):
                j=i+1
                while (j<len(scan_combos)):
                    #print "j: ", j, seq_combos[j]
                    if scan_combos[i][k]==scan_combos[j][0]:
                        add_n=scan_combos[j][1]
                        #print "Adding ", add_n, " to sequence combo ", seq_combos[i]
                        scan_combos[i].append(add_n) #combine the indices if there are 3+ cp_objects combining in same seq.
                        for m in range(0,len(scan_combos[i])-1): # try all values in seq_combos[i] except the last value, which is = to add_n.
                            try:
                                #print "Removing sequence combo ", seq_combos[i][m], add_n
                                scan_combos.remove([scan_combos[i][m],add_n]) # seq_combos[j][1] is the known last value in seq_combos[i]
                            except ValueError:
                                # if there is an error because that index does not exist!
                                errmsg='Interfacing not Valid: CPO %d and CPO %d are combined in-scan and do not interface the same with CPO %d' % (scan_combos[i][m], scan_combos[i][k], add_n)
                                sys.exit(errmsg)
                        j=j-1 # this means that the former scan_combos[j] has been deleted and there are new values at index j, so decrement before incrementing in for.
                    j=j+1 # increment
                k=k+1
            i=i+1
        # now scan_combos is a list of lists, where a cp object occurs in only once in the nested list.
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
                scan_combos.append([cpo]) # append the cpo on its own, is not scan combined.

        scan_combos=sorted(scan_combos) # should sort correctly?
        return scan_combos
                            

    def get_seq_combos(self): 
        'TAKE MY OWN INTERFACING SPECS AND COMBINE INTO WORKABLE ARRAY'
        # check the interfacing, and combine sequences first.
        seq_combos=[]
    
        for num1,num2 in self.interface.keys():
            if self.interface[num1, num2]=="PULSE": # Changed the interface options or self.interface[num1, num2]=="MULTI_SEQ":
                seq_combos.append([num1,num2]) # save the keys that are sequence combinations.

        seq_combos=sorted(seq_combos)
        #print seq_combos
        #if [2,4] and [1,4], then also must be [1,2] in the seq_combos list!
        i=0
        while (i<len(seq_combos)):
        #for i in range(0,(len(seq_combos)-1)):     
            #print "i: ", i,        seq_combos[i]
            k=0
            #for k in range(len(seq_combos[i])):
            while (k<len(seq_combos[i])):
                j=i+1
                while (j<len(seq_combos)):
                #for j in range(i+1,len(seq_combos)):
                    #print "j: ", j, seq_combos[j]
                    if seq_combos[i][k]==seq_combos[j][0]:
                        add_n=seq_combos[j][1]
                        #print "Adding ", add_n, " to sequence combo ", seq_combos[i]
                        seq_combos[i].append(add_n) #combine the indices if there are 3+ cp_objects combining in same seq.
                        #if seq_combos[i][1]>seq_combos[j][1]:
                            #seq_combos.remove([seq_combos[j][1],seq_combos[i][1]])
                        #else:
                            #print seq_combos[i][1], seq_combos[j][1]
                        for m in range(0,len(seq_combos[i])-1): # try all values in seq_combos[i] except the last value, which is = to add_n.
                            try:
                                #print "Removing sequence combo ", seq_combos[i][m], add_n
                                seq_combos.remove([seq_combos[i][m],add_n]) # seq_combos[j][1] is the known last value in seq_combos[i]
                            except ValueError:
                                # if there is an error because that index does not exist!
                                errmsg='Interfacing not Valid: CPO %d and CPO %d are a sequence combination and do not interface the same with CPO %d' % (seq_combos[i][m], seq_combos[i][k], add_n)
                                sys.exit(errmsg)
                                #print seq_combos
                        j=j-1 # this means that the former seq_combos[j] has been deleted and there are new values at index j, so decrement before incrementing in for.
                    j=j+1 # increment
                k=k+1
            i=i+1

        # now seq_combos is a list of lists, where a cp object occurs in only once in the nested list.
        for cpo in range(self.cpo_num):
            found=False
            for i in range(len(seq_combos)):
                for j in range(len(seq_combos[i])):
                    if cpo==seq_combos[i][j]:
                        found=True
                        break
                if found==False:
                    continue
                break
            else: # no break
                seq_combos.append([cpo]) # append the cpo on its own, is not sequence combined.

        seq_combos=sorted(seq_combos)
        return seq_combos
 
