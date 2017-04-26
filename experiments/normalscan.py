#!/usr/bin/python

# write an experiment that creates a new control program.
from experiment_prototype import ExperimentPrototype


class Normalscan(ExperimentPrototype):
    
    def __init__(self):
        super(Normalscan,self).__init__(1,150) #number of cpo_list dictionaries to interface, 'control program' ID  

        # If you created this experiment with x number of cpo_list dictionaries, update cpo_list[0] through cpo_list[x-1]
        self.cpo_list[0].update({      
            "txchannels":   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            "rxchannels":   [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15],
            "sequence":     [0,14,22,24,27,31,42,43],
            "pulse_shift":  [0,0,0,0,0,0,0,0],
            "mpinc":        1500, # us
            "pulse_len":    300, # us
            "nrang":        75, # range gates
            "frang":        90, # first range gate, in km
            "intt":         3000, # duration of an integration, in ms
            "intn":         21, # number of averages if intt is None.
            "beamdir":      [-26.25,-22.75,-19.25,-15.75,-12.25,-8.75,
                -5.25,-1.75,1.75,5.25,8.75,12.25,15.75,19.25,22.75,26.25],
            "scan":         [15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0],
            "scanboundf":   1, 
            "scanbound":    60000,
            "txfreq":       12300,
            "rxfreq":       12300,
            "clrfrqf":      1, 
            "clrfrqrange":  [12200,12500],
            "xcf":          1,
            "acfint":       1, 
            "wavetype":     "SINE",
            "seqtimer":     0
        })

        # Other things you can change if you wish.
        #self.txctrfreq = 12000
        #self.txrate = 12000000 
        #self.rxctrfreq = 12000 
        #self.xcf = 1
        #self.acfint = 1

        
        # Update the following interface dictionary if you have more than one cpo_list dictionary.
        
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

         # NOTE keys are as such: (0,1), (0,2), (1,2), NEVER includes (2,0) etc.
#        self.interface.update({
#            (0,1) : 'PULSE'
#        }) 

    def update(self, acfdata):
        """
        Use this function to change your experiment based on ACF data
        retrieved from the rx_signal_processing block

        :param acfdata ??? TBD
        """ # TODO update with how acfdata will be passed in
        change_flag = False
        return change_flag


