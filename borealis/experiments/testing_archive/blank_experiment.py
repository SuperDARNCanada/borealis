#!/usr/bin/python

# write an experiment that creates a new control program.
from experiment_prototype.experiment_prototype import ExperimentPrototype

class Blank(ExperimentPrototype):

    def __init__(self):
        cpid = 00000
        super(Blank, self).__init__(cpid)

        # self.add_slice({  # slice_id = 0, the first slice
        #     "txantennas": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #     "rx_main_antennas": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #     "rx_int_antennas": [0, 1, 2, 3],
        #     "pulse_sequence": [0, 14, 22, 24, 27, 31, 42, 43],
        #     "pulse_shift": [0, 0, 0, 0, 0, 0, 0, 0],
        #     "mpinc": 1500,  # us
        #     "pulse_len": 300,  # us
        #     "nrang": 75,  # range gates
        #     "frang": 180,  # first range gate, in km
        #     "intt": 3500,  # duration of an integration, in ms
        #     "beam_angle": [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75,
        #                    -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75,
        #                    26.25],
        #     "beam_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        #     "scanboundflag": True,  # there is a scan boundary
        #     "scanbound": 60000,  # ms
        #     "clrfrqflag": True,  # search for clear frequency before transmitting
        #     "clrfrqrange": [12200, 12500],  # frequency range for clear frequency search,
        #     # kHz including a clrfrqrange overrides freq so these are no
        #     # longer necessary as they will be set by the frequency chosen from the range.
        #     "xcf": True,  # cross-correlation processing
        #     "acfint": True,  # interferometer acfs
        # })

        # self.add_slice({  # slice_id = 1
        #     "txantennas": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #     "rx_main_antennas": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        #     "rx_int_antennas": [0, 1, 2, 3],
        #     "pulse_sequence": [0, 14, 22, 24, 27, 31, 42, 43],
        #     "pulse_shift": [0, 0, 0, 0, 0, 0, 0, 0],
        #     "mpinc": 1500,  # us
        #     "pulse_len": 300,  # us
        #     "nrang": 75,  # range gates
        #     "frang": 90,  # first range gate, in km
        #     "intt": 3500,  # duration of an integration, in ms
        #     "beam_angle": [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75,
        #                    -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75, 26.25],
        #     "beam_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
        #     "scanboundflag": True,  # there is a scan boundary
        #     "scanbound": 60000,  # ms
        #     "clrfrqflag": True,  # search for clear frequency before transmitting
        #     "clrfrqrange": [10200, 10500],  # range for clear frequency search, kHz
        #     "xcf": True,  # cross-correlation processing
        #     "acfint": True,  # interferometer acfs
        # }, interfacing_dict={0: 'SCAN'})

        # Other things you can change if you wish. You may want to discuss with us about
        # it beforehand.
        # These apply to the experiment and all slices as a whole.
        # self.txctrfreq = 12000 # kHz, oscillator mixer frequency on the USRP for TX
        # self.txrate = 12000000 # Hz, sample rate fed to DAC
        # self.rxctrfreq = 12000 # kHz, mixer frequency on the USRP for RX

        # Update the following interface dictionary if you have more than one slice
        # dictionary in your slice_list. The keys in the interface dictionary
        # correspond to the slice_ids of the slices in your slice_list.
        # Take a look at the documentation for the frozenset interface_types in
        # experiment_prototype to understand the types of interfacing (PULSE, INTEGRATION,
        # INTTIME, or SCAN).

        # NOTE keys are as such: (0,1), (0,2), (1,2), NEVER includes (2,0) etc.

        # self.interface.update({
        #     (0, 1): 'SCAN'  # Full scan of one slice, then full scan of the next.
        # })

    def update(self, acfdata):
        """
        Check if we want to update the experiment after an integration time's data is 
        available.
        
        Use this function to change your experiment based on ACF data retrieved from the 
        rx_signal_processing block. This function is called after every integration period
        so that your experiment can be changed to adjust to existing conditions. Talk to 
        us if you have something specific in mind that you're not sure if you can 
        implement here. 

        :param acfdata: ??? TBD
        :returns change_flag: indicating whether the experiment has changed or not. 
         True = change has occurred.
        """  # TODO update with how acfdata will be passed in

        change_flag = False
        return change_flag
