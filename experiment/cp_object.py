#!/usr/bin/python

# Definition of the cp_object class
# All cp_objects will create one RCP that will be combined in a "make_RCP" and then implemented in "radarctrl".

import os
import sys

def if_type():
    return frozenset(['SCAN', 'INTTIME', 'INTEGRATION', 'PULSE'])

def interfacing(cpo_num):
    if_list=[]
    for i in range(cpo_num):
        for j in range(i+1,cpo_num):
            if_list.append((i,j))
    if_keys=tuple(if_list) # keys defining interfacing between the cp_objects	
    if_dict={}
    for num1,num2 in if_keys:
        if_dict[num1, num2]="NONE"

    return if_dict	

# cp_object may be better utilized as a dictionary - create tuple of keys once we know what all keys will be.

class cp_object:
    'Class to define transmit specifications of a certain frequency, beam, and pulse sequence'
    def __init__(self):
        # instance variables and defaults
        self.cpid=[150,0] # two numbers: overall RCP ID and the 1st component in that RCP
        #self.cp_comp=1 # total number of components in RCP
        self.channels=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] #what antennas do you want to transmit on
        self.sequence=[0,14,22,24,27,31,42,43] #sequence, to be multiplied by tau, default normalscan
        self.pulse_shift=[0,0,0,0,0,0,0,0] # pulse phase shifts between pulses (degrees) - orig for Ashton to do some rm of self-clutter
        self.mpinc=1500 # period length of pulse (us)
        self.pullen=300 # pulse length, (duty cycle * mpinc) (us)
        self.intt=3000 # duration of a single integration, in ms
        self.intn=21 # max number of averages (aka full pulse sequences transmitted) in an integration time intt (default 3s)
        self.beamdir=[-26.25,-22.75,-19.25,-15.75,-12.25,-8.75,-5.25,-1.75,1.75,5.25,8.75,12.25,15.75,19.25,22.75,26.25] # array of angle off boresight, array length equal to intn; positive is E of N (clockwise)
        # add something here for manual phase shifting between sequences/scans (so that relative phases can be set without beam direction specification)
        self.scan=[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0] # list of beam number in order of transmission in scan, ie. scan[14]=1 
                # corresponds to beamdir[1] being the 15th integration in the scan (second last).
        self.scanboundf=1 # scan boundary exists?
        self.scanbound=60000 # max length of scan past the minute/hour if boundary exists, ms
        self.freq=12300 # in kHz
        self.clrfrqf=1 # flag for clear frequency search
        self.clrfrqrange=300 # clear frequency range if searching
        # receiving params
        self.xcf=1 # flag for cross-correlation
        self.acfint=1 # flag for getting lag-zero power of interferometer
        self.wavetype='SINE'
        self.seqtimer=0 # in us. Sequence starts at 0s into start of integration. Useful for when there are multiple cp_objects, and
                # you want to run them simultaneously (not scan by scan)
        # to add: scan boundary start time? if specific minutes, etc.
        # objects. types: interleaving (one scan of cp_object[0] then one scan of cp_object[1], etc.); 
        # simultaneous (using different channels or same). If same-channel simultaneous could be stereo with 
        # same pulse sequence or multi-sequence?
        self.iwave_table=[]
        self.qwave_table=[]
        #these are reloaded once the wavetype is set.


    def __call__(self):
        'Print values in this CP component.'
        # instance variables and defaults
        print 'CPID [cpid]: {}'.format(self.cpid) # id of control program
        print 'Channels/Antennas to Use [channels]: {}'.format(self.channels) #what antennas do you want to transmit on
        print 'Number of Pulses : {}'.format(len(self.sequence)) #how many pulses in sequence
        print 'Pulse Sequence [sequence]: {}'.format(self.sequence) #sequence, to be multiplied by tau, default normalscan
        print 'Multi-Pulse Increment (us) [mpinc]: {}'.format(self.mpinc) # period length of pulse
        print 'Pulse Length (us) [pullen]: {}'.format(self.pullen) # pulse length, (duty cycle * mpinc)
        print 'Integration Time (ms) [intt]: {}'.format(self.intt) # duration of a single integration, in ms
        print 'Number of Integrations (nave max) [intn]: {}'.format(self.intn) # number of integrations in a scan
        print 'Beam Directions off Boresight (Degree) [beamdir]: {}'.format(self.beamdir) # array of angle off boresight, array length equal to intn; positive is E of N (clockwise)
        print 'Scan Beam Directions [scan]: {}'.format(self.scan) # list of beam number in order of transmission in scan.
        print 'Scan Boundary Exists [boundf]: {}'.format(self.scanboundf) # scan boundary exists?
        if (self.scanboundf==1) :
                print 'Scan Boundary (ms) [scanbound]: {}'.format(self.scanbound) # max length of scan past the minute/hour if boundary exists, ms
        print 'Transmit Frequency [freq]: {}'.format(self.freq) # in kHz
        print 'Clear Frequency Search Flag [clrfrqf]: {}'.format(self.clrfrqf) # flag for clear frequency search
        if (self.clrfrqf==1):
                print 'Clear Frequency Range [clrfrqrange]: {}'.format(self.clrfrqrange) # clear frequency range if searching
        print 'Wave Type [wavetype]: {}'.format(self.wavetype)
        print 'Pulse Sequence Timer [seqtimer]: {}'.format(self.seqtimer)
        # receiving params
        print 'XCF flag [xcf]: {}'.format(self.xcf) # flag for cross-correlation
        print 'Lag-Zero INT Power Flag [acfint]: {}'.format(self.acfint) # flag for getting lag-zero power of interferometer
        return None

    def selfcheck(self):
        'Do a quick self-test to ensure values in this CP component make sense.'	
        #self.cpid=[150,0] 
        error_count=0
        error_dict={}
        if self.cp_comp<=self.cpid[1]:
            error_dict[error_count]='CP Object identifier cpid[1] is greater than total cp components cp_comp'
            error_count=error_count+1

        #if self.cpid[0] is not unique

        # check none greater than 15, no duplicates
        if len(self.channels)>16:
            error_dict[error_count]='CP Object Has Too Many Channels'
            error_count=error_count+1	
        for i in range(len(self.channels)):
            if self.channels[i]>= 16:
                error_dict[error_count]='CP Object Specifies Channel Numbers Over 16'
                error_count=error_count+1
            for j in range(i+1, len(self.channels)):
                if self.channels[i]==self.channels[j]:
                    error_dict[error_count]='CP Object Has Duplicate Channels'
                    error_count=error_count+1	
                                        
        # check sequence increasing
        if len(self.sequence)!=1:
            for i in range(1, len(self.sequence)):
                if self.sequence[i]<=self.sequence[i-1]:
                    error_dict[error_count]='CP Object Sequence Not Increasing'
                    error_count=error_count+1	
                        
        #check pullen and mpinc make sense (values in us)
        if self.pullen > self.mpinc:
            error_dict['error_count']='CP Pulse Length Greater than MPINC'
            error_count=error_count+1	
        if self.pullen < 0.01:	
            error_dict[error_count]='CP Pulse Length Too Small'
            error_count=error_count+1	

        # check intn and intt make sense given mpinc, and pulse sequence.		
        seq_len=self.mpinc*(self.sequence[-1]+1)

        self.intt=3000 # duration of the direction, in ms
        self.intn=21 # max number of averages (aka full pulse sequences transmitted) in an integration time intt (default 3s)

        # check no duplicates in beam directions
        for i in range(len(self.beamdir)):
            for j in range(i+1,len(self.beamdir)):
                if self.beamdir[i]==self.beamdir[j]:
                    error_dict[error_count]='CP Beam Direction Has Duplicates'
                    error_count=error_count+1
                if self.beamdir[i]>self.beamdir[j]:
                    error_dict[error_count]='CP Beam Directions Not in Order Clockwise (E of N is positive)'
                    error_count=error_count+1	

        if (not self.scan): # if empty
            error_dict[error_count]='Scan Empty'
            error_count=error_count+1

        # check beam numbers in scan exist
        for i in self.scan:
            if i>=len(self.beamdir):
                error_dict[error_count]='CP Scan Beam Direction DNE, not Enough Beams in beamdir'
                error_count=error_count+1	

        # check scan boundary not less than minimum required scan time.
        if self.boundf==1:
            if self.scanbound<(len(self.scan)*self.intt):
                error_dict[error_count]='CP Scan Too Long for ScanBoundary'
                error_count=error_count+1	

        #self.freq=12300 # in kHz
        #self.clrfrqf=1 # flag for clear frequency search
        #self.clrfrqrange=300 # clear frequency range if searching
        # receiving params
        #self.xcf=1 # flag for cross-correlation
        #self.acfint=1 # flag for getting lag-zero power of interferometer
        #self.wavetype='SINE'
        #self.seqtimer=0 
        # cpid is a unique value?

        return error_dict


