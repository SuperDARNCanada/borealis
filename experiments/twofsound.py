#!/usr/bin/python

# write an experiment that creates a new control program.
import os
import sys

BOREALISPATH = os.environ['BOREALISPATH']
#sys.path.append(BOREALISPATH + "/experiment_prototype")

#import test
from experiment_prototype.experiment_prototype import ExperimentPrototype

class Twofsound(ExperimentPrototype):

    def __init__(self):
        cpid = 3503
        super(Twofsound, self).__init__(cpid)

        tx_ant = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        rx_main_ant = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        rx_int_ant = [0]
        self.add_slice({  # slice_id = 0, the first slice
            "tx_antennas": tx_ant,
            "rx_main_antennas": rx_main_ant,
            "rx_int_antennas": rx_int_ant,
            "pulse_sequence": [0, 14, 22, 24, 27, 31, 42, 43],
            "pulse_shift": [0, 0, 0, 0, 0, 0, 0, 0],
            "mpinc": 1500,  # us
            "pulse_len": 300,  # us
            "nrang": 75,  # range gates
            "frang": 180,  # first range gate, in km
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75,
                           -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75,
                           26.25],
            "beam_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            "scanboundflag": True,  # there is a scan boundary
            "scanbound": 60000,  # ms
            #"clrfrqflag": True,  # search for clear frequency before transmitting
            #"clrfrqrange": [13100, 13400],  # frequency range for clear frequency search,
            "txfreq" : 13100,
            # kHz including a clrfrqrange overrides rxfreq and txfreq so these are no
            # longer necessary as they will be set by the frequency chosen from the
            # range.
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        })

        self.add_slice({  # slice_id = 1
            "tx_antennas": tx_ant,
            "rx_main_antennas": rx_main_ant,
            "rx_int_antennas": rx_int_ant,
            "pulse_sequence": [0, 14, 22, 24, 27, 31, 42, 43],
            "pulse_shift": [0, 0, 0, 0, 0, 0, 0, 0],
            "mpinc": 1500,  # us
            "pulse_len": 300,  # us
            "nrang": 75,  # range gates
            "frang": 90,  # first range gate, in km
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75,
                           -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75,
                           26.25],
            "beam_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            "scanboundflag": True,  # there is a scan boundary
            "scanbound": 60000,  # ms
            #"clrfrqflag": True,  # search for clear frequency before transmitting
            #"clrfrqrange": [10200, 10500],  # range for clear frequency search, kHz
            "txfreq": 14500,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }, interfacing_dict={0: 'SCAN'})

        # Other things you can change if you wish. You may want to discuss with us about
        # it beforehand.
        # These apply to the experiment and all slices as a whole.
        # self.txctrfreq = 12000 # kHz, oscillator mixer frequency on the USRP for TX
        # self.txrate = 12000000 # Hz, sample rate fed to DAC
        # self.rxctrfreq = 12000 # kHz, mixer frequency on the USRP for RX

        # Update the following interface dictionary if you have more than one slice
        # dictionary in your slice_list and you did not specify the interfacing when
        # adding the slice. The keys in the interface dictionary correspond to the
        # slice_ids of the slices in your slice_list.
        # Take a look at the documentation for the frozenset interface_types in
        # experiment_prototype to understand the types of interfacing (PULSE,
        # INTEGRATION, INTTIME, or SCAN).

        # NOTE keys are as such: (0,1), (0,2), (1,2), NEVER includes (2,0) etc.

        # self.interface.update({
        #     (0, 1): 'SCAN'  # Full scan of one slice, then full scan of the next.
        # })
