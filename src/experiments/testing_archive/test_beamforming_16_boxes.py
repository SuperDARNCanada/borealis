#!/usr/bin/python

import os
import sys

sys.path.append(os.environ['BOREALISPATH'])
# write an experiment that creates a new control program.
from experiment_prototype.experiment_prototype import ExperimentPrototype


class OneBox(ExperimentPrototype):

    def __init__(self):
        cpid = 100000000
        super(OneBox, self).__init__(cpid)

        pulse_sequence = [0, 14, 22, 24, 27, 31, 42, 43]
        #pulse_sequence = [0,3,15,41,66,95,97,106,142,152,220,221,225,242,295,330,338,354,382,388,402,415,486,504,523,546,553]
        self.add_slice({  # slice_id = 0, there is only one slice.
            "tx_antennas": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "rx_main_antennas": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "rx_int_antennas": [0, 1, 2, 3],
            "pulse_sequence":pulse_sequence,#[0, 14, 22, 24, 27, 31, 42, 43],
            "pulse_shift": [0] * len(pulse_sequence),
            "mpinc": 1500,  # us
            "pulse_len": 300,  # us
            "nrang": 75,  # range gates
            "frang": 180,  # first range gate, in km
            "intt": 3000,  # duration of an integration, in ms
            "intn": 21,  # number of averages if intt is None.
            "beam_angle": [-0.0], # [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75,
                          # -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75, 26.25],
            "rx_beam_order": [0],
            "tx_beam_order": [0],
            #"scanboundflag": True,  # there is a scan boundary
            #"scanbound": 60000,  # ms
            "freq": 13332,
            #"clrfrqflag": True,  # search for clear frequency before transmitting
            #"clrfrqrange": [13200, 13500],  # frequency range for clear frequency search, kHz
            # including a clrfrqrange overrides freq and freq so these are no longer necessary
            # as they will be set by the frequency chosen from the range.
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        })

        # USED THE FOLLOWING FOR TESTING SECOND SLICE

        # self.add_slice({  # slice_id = 0, there is only one slice.
        #     "txantennas": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #     "rx_main_antennas": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #     "rx_int_antennas": [0, 1, 2, 3],
        #     "pulse_sequence": [0, 14, 22, 24, 27, 31, 42, 43],
        #     "pulse_shift": [0, 0, 0, 0, 0, 0, 0, 0],
        #     "mpinc": 1500,  # us
        #     "pulse_len": 300,  # us
        #     "nrang": 75,  # range gates
        #     "frang": 180,  # first range gate, in km
        #     "intt": 3000,  # duration of an integration, in ms
        #     "intn": 21,  # number of averages if intt is None.
        #     "beam_angle": [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75,
        #                    -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75, 26.25],
        #     "beam_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        #     "scanboundflag": True,  # there is a scan boundary
        #     "scanbound": 60000,  # ms
        #     "clrfrqflag": True,  # search for clear frequency before transmitting
        #     "clrfrqrange": [13100, 13200],  # frequency range for clear frequency search, kHz
        #     # including a clrfrqrange overrides freq so these are no longer necessary
        #     # as they will be set by the frequency chosen from the range.
        #     "xcf": True,  # cross-correlation processing
        #     "acfint": True,  # interferometer acfs
        # }, interfacing_dict={0: 'PULSE'})
        # Other things you can change if you wish. You may want to discuss with us about it beforehand.
        # These apply to the experiment and all slices as a whole.
        #self.txctrfreq = 12000 # kHz, oscillator mixer frequency on the USRP for TX
        # self.txrate = 12000000 # Hz, sample rate fed to DAC
        #self.rxctrfreq = 12000 # kHz, mixer frequency on the USRP for RX

        print(self.rxctrfreq)

        """ 
        INTERFACING TYPES:
        
        NONE : Only the default, must be changed.
        SCAN : Scan by scan interfacing. Experiment slice 1 will scan first  followed by slice 2 and subsequent slices.
        INTTIME : integration time interfacing (full integration time of one sequence, then the next). Time/number of 
            sequences dependent on intt and intn in the slice. Effectively simultaneous scan interfacing, interleaving 
            each integration time in the scans. Slice 1 first inttime or beam direction will run followed by slice 
            2's first inttime, etc. If slice 1's len(beam_order) is greater than slice 2's, slice 2's last 
            integration will run and then all the rest of slice 1's will continue until the full scan is over. 
            Experiment slice 1 and 2 must have the same scan boundary, if any boundary. 
        INTEGRATION : pulse sequence or integration interfacing (one sequence of one slice, then the next). Experiment 
            slice 1 and 2 must have same intt and intn. Integrations will switch between one and the other slice until 
            time is up or the required number of averages is reached.
        PULSE : Simultaneous sequence interfacing, pulse by pulse creates a single sequence. Experiment Slice 1 and 2 
            might have different frequencies and/or may have different pulse length, mpinc, sequence. They must also 
            have same len(scan), although they may use different directions in scan. They must have the same scan 
            boundary if any. A time offset between the pulses starting may be set (seq_timer in the slice). Slice 1 
            and 2 will have integrations that run at the same time. 
        """

        # Update the following interface dictionary if you have more than one slice dictionary in your slice_list.
        # The keys in the interface dictionary correspond to the slice_ids of the slices in your slice_list.

        # NOTE keys are as such: (0,1), (0,2), (1,2), NEVER includes (2,0) etc.

    #        self.interface.update({
    #            (0,1) : 'PULSE'
    #        })

#    def update(self, acfdata):
        """
        Use this function to change your experiment based on ACF data retrieved from the rx_signal_processing block. 
        This function is called after every integration period so that your experiment can be changed to adjust to 
        existing conditions. Talk to us if you have something specific in mind that you're not sure if you can 
        implement here. 

        :param acfdata ??? TBD
        :rtype boolean
        :return change_flag, indicating whether the experiment has changed or not. True = change has occurred.
        """  # TODO update with how acfdata will be passed in

        # TODO : docs about what can and cannot be changed. Warning about changing centre frequencies.

#        change_flag = False
#        return change_flag
