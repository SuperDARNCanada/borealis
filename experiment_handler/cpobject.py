#!/usr/bin/python

# Definition of the cp_object class
# All CPObjects will create one ControlProg and then implemented in
# runradar.

import os
import sys

def if_type():
    return frozenset(['SCAN', 'INTTIME', 'INTEGRATION', 'PULSE'])

def interfacing(cpo_num):
    if_list=[]
    for i in range(cpo_num):
        for j in range(i+1,cpo_num):
            if_list.append((i,j))
    if_keys=tuple(if_list) 
    if_dict={}
    for num1,num2 in if_keys:
        if_dict[num1, num2]="NONE"

    return if_dict	

# TODO: CPObject may be better utilized as a dictionary; create tuple 
#   of keys once we know what all keys will be.

class CPObject:
    """Class to define transmit specifications of a certain frequency, 
    beam, and pulse sequence.
    """

    def __init__(self):
        self.cpid=[150,0] 
        # Two numbers: overall RCP ID and the 1st CPObject in that RCP
        self.txchannels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15] 
        self.rxchannels = self.txchannels # initilization
        # What antennas do you want to transmit on
        self.sequence=[0,14,22,24,27,31,42,43] 
        # Sequence, to be multiplied by tau, default normalscan
        self.pulse_shift=[0,0,0,0,0,0,0,0] 
        # pulse phase shifts between pulses (degrees) - orig for Ashton
        #    to do some rm of self-clutter
        self.mpinc=1500 
        # Period length of pulse (us)
        self.pullen=300 
        # Pulse length, (duty cycle * mpinc) (us)
        self.nrang=75 
        # Number of range gates
        self.frang=90 
        # First range gate begins at this number (in km)
        self.intt=3000 
        # Duration of a single integration, in ms
        self.intn=21 
        # Max number of averages (aka full pulse sequences transmitted)
        #   in an integration time intt (default 3s)
        self.beamdir=[-26.25,-22.75,-19.25,-15.75,-12.25,-8.75,-5.25,-1.75,
            1.75,5.25,8.75,12.25,15.75,19.25,22.75,26.25] 
        # Array of angle off boresight, array length equal to number of beams; 
        #   positive is E of N (clockwise)

        # TODO: Potentially add something here for manual phase shifts
        #   between sequences/scans (so that relative phases can be set
        #   without beam direction specification)
        self.scan=[15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0] 
        # List of beam number in order of transmission in scan, ie. 
        #   scan[14]=1 corresponds to beamdir[1] being the 15th 
        #   integration in the scan (second last).
        self.scanboundf=1 
        # 1 if scan boundary exists
        self.scanbound=60000 
        # Max length of scan past the minute/hour if bound exists, ms
        self.txfreq=12300 
        self.rxfreq = self.txfreq
        # Frequency in kHz
        self.clrfrqf=1 
        # flag for clear frequency search - 1 if required
        self.clrfrqrange=[12200,12500] 
        # Clear frequency range if searching

        # Receiving parameters
        self.xcf=1 
        # Flag for cross-correlation
        self.acfint=1 
        # Flag for getting lag-zero power of interferometer
        self.wavetype='SINE'
        self.seqtimer=0 
        # Timing in us. Sequence starts at 0s into start of 
        #   integration. Useful for when there are multiple CPObjects,
        #   and you want to run them simultaneously (not scan by scan)

        # TODO: add scan boundary start time? if specific minutes, etc.

    def __call__(self):
        """Print values in this CP component."""
        print 'CPID [cpid]: {}'.format(self.cpid) 
        print 'Channels/Antennas to Transmit [txchannels]: {}'.format(self.txchannels) 
        print 'Channels/Antennas to Receive [rxchannels]: {}'.format(self.rxchannels) 
        print 'Number of Pulses : {}'.format(len(self.sequence)) 
        print 'Pulse Sequence [sequence]: {}'.format(self.sequence) 
        print 'Multi-Pulse Increment (us) [mpinc]: {}'.format(self.mpinc)
        print 'Pulse Length (us) [pullen]: {}'.format(self.pullen)
        print 'Number of Range Gates: {}'.format(self.nrang) 
        print 'First Range Gate (km): {}'.format(self.frang)
        print 'Integration Time (ms) [intt]: {}'.format(self.intt) 
        print 'Number of Integrations (nave max) [intn]: {}'.format(self.intn) 
        print 'Beam Directions off Boresight (Degree) [beamdir]: {}'.format(
            self.beamdir) 
        print 'Scan Beam Directions [scan]: {}'.format(self.scan) 
        print 'Scan Boundary Exists [boundf]: {}'.format(self.scanboundf)
        if (self.scanboundf==1) :
            print 'Scan Boundary (ms) [scanbound]: {}'.format(
                self.scanbound)
        print 'Transmit Frequency [freq] in kHz: {}'.format(self.txfreq)
        print 'Receive Frequency [freq] in kHz: {}'.format(self.rxfreq)
        print 'Clear Frequency Search Flag [clrfrqf]: {}'.format(self.clrfrqf)
        if (self.clrfrqf==1):
            print 'Clear Frequency Range [clrfrqrange]: {}'.format(
                self.clrfrqrange)
        print 'Wave Type [wavetype]: {}'.format(self.wavetype)
        print 'Pulse Sequence Timer [seqtimer]: {}'.format(self.seqtimer)
        print 'XCF flag [xcf]: {}'.format(self.xcf)
        print 'Lag-Zero INT Power Flag [acfint]: {}'.format(self.acfint)
        return None

    def selfcheck(self):
        """Do a quick self-test to ensure values in this CP component
         make sense.
        """
        error_count=0
        error_dict={}
        #if self.cpid[0] is not unique

        # check none greater than 15, no duplicates
        if len(self.txchannels)>16 or len(self.rxchannels)>16:
            error_dict[error_count]="CP Object Has Too Many Channels"
            error_count=error_count+1	
        for i in range(len(self.txchannels)):
            if self.txchannels[i]>= 16 or self.rxchannels[i]>=16:
                error_dict[error_count]="CP Object Specifies Channel \
                    Numbers Over 16"
                error_count=error_count+1
            for j in range(i+1, len(self.txchannels)):
                if self.txchannels[i]==self.txchannels[j] or self.rxchannels[i]==self.rxchannels[j]:
                    error_dict[error_count]="CP Object Has Duplicate Channels"
                    error_count=error_count+1	
                                        
        # check sequence increasing
        if len(self.sequence)!=1:
            for i in range(1, len(self.sequence)):
                if self.sequence[i]<=self.sequence[i-1]:
                    error_dict[error_count]="CP Object Sequence Not Increasing"
                    error_count=error_count+1	
                        
        #check pullen and mpinc make sense (values in us)
        if self.pullen > self.mpinc:
            error_dict['error_count']="CP Pulse Length Greater than MPINC"
            error_count=error_count+1	
        if self.pullen < 0.01:	
            error_dict[error_count]="CP Pulse Length Too Small"
            error_count=error_count+1	

        # check intn and intt make sense given mpinc, and pulse sequence.		
        seq_len=self.mpinc*(self.sequence[-1]+1)

# TODO: Check these
#        self.intt=3000 # duration of the direction, in ms
#        self.intn=21 # max number of averages (aka full pulse sequences transmitted) in an integration time intt (default 3s)

        # check no duplicates in beam directions
        for i in range(len(self.beamdir)):
            for j in range(i+1,len(self.beamdir)):
                if self.beamdir[i]==self.beamdir[j]:
                    error_dict[error_count]="CP Beam Direction Has Duplicates"
                    error_count=error_count+1
                if self.beamdir[i]>self.beamdir[j]:
                    error_dict[error_count]="CP Beam Directions Not in Order \
                        Clockwise (E of N is positive)"
                    error_count=error_count+1	

        if (not self.scan): # if empty
            error_dict[error_count]="Scan Empty"
            error_count=error_count+1

        # check beam numbers in scan exist
        for i in self.scan:
            if i>=len(self.beamdir):
                error_dict[error_count]="CP Scan Beam Direction DNE, not \
                    Enough Beams in beamdir"
                error_count=error_count+1	

        # check scan boundary not less than minimum required scan time.
        if self.scanboundf==1:
            if self.scanbound<(len(self.scan)*self.intt):
                error_dict[error_count]="CP Scan Too Long for ScanBoundary"
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


