#!/usr/bin/python

# write an experiment that creates a new control program.
import os
import sys

BOREALISPATH = os.environ['BOREALISPATH']
#sys.path.append(BOREALISPATH + "/experiment_prototype")

#import test
from experiment_prototype.experiment_prototype import ExperimentPrototype
from experiment_prototype.decimation_scheme.decimation_scheme import DecimationStage, DecimationScheme
from experiments.test_decimation_schemes import *

class Twofsound(ExperimentPrototype):

    def __init__(self):
        cpid = 3503
        rxrate = 5.0e6
        output_rx_rate = 10.0e3/3

        tx_ant = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        rx_main_ant = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        rx_int_ant = [0, 1, 2, 3]
        pulse_sequence = [0, 9, 12, 20, 22, 26, 27] #[0, 14, 22, 24, 27, 31, 42, 43]
        tau_spacing = 2400 # 1500 # us
        slice_1 = {  # slice_id = 0, the first slice
            "tx_antennas": tx_ant,
            "rx_main_antennas": rx_main_ant,
            "rx_int_antennas": rx_int_ant,
            "pulse_sequence": pulse_sequence,
            "pulse_shift": [0, 0, 0, 0, 0, 0, 0, 0],
            "mpinc": tau_spacing,
            "pulse_len": 300,  # us
            "nrang": 75,  # range gates
            "frang": 180,  # first range gate, in km
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75,
                           -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75,
                           26.25],
            "rx_beam_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            "tx_beam_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            #"scanboundflag": True,  # there is a scan boundary
            #"scanbound": 60000,  # ms
            #"clrfrqflag": True,  # search for clear frequency before transmitting
            #"clrfrqrange": [13100, 13400],  # frequency range for clear frequency search,
            "freq" : 10500,
            # kHz including a clrfrqrange overrides freq so these are no
            # longer necessary as they will be set by the frequency chosen from the
            # range.
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }

        slice_2 = {  # slice_id = 1
            "tx_antennas": tx_ant,
            "rx_main_antennas": rx_main_ant,
            "rx_int_antennas": rx_int_ant,
            "pulse_sequence": pulse_sequence,
            "pulse_shift": [0, 0, 0, 0, 0, 0, 0, 0],
            "mpinc": tau_spacing,
            "pulse_len": 300,  # us
            "nrang": 75,  # range gates
            "frang": 90,  # first range gate, in km
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": [-26.25, -22.75, -19.25, -15.75, -12.25, -8.75,
                           -5.25, -1.75, 1.75, 5.25, 8.75, 12.25, 15.75, 19.25, 22.75,
                           26.25],
            "rx_beam_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            "tx_beam_order": [15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            #"scanboundflag": True,  # there is a scan boundary
            #"scanbound": 60000,  # ms
            #"clrfrqflag": True,  # search for clear frequency before transmitting
            #"clrfrqrange": [10200, 10500],  # range for clear frequency search, kHz
            "freq": 13000,
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }

        list_of_slices = [slice_1, slice_2]
        sum_of_freq = 0
        for slice in list_of_slices:
            sum_of_freq += slice['freq']# kHz, oscillator mixer frequency on the USRP for TX
        rxctrfreq = txctrfreq = int(sum_of_freq/len(list_of_slices))
        
        
        super(Twofsound, self).__init__(cpid, output_rx_rate=output_rx_rate, rx_bandwidth=rxrate,
                txctrfreq=txctrfreq, rxctrfreq=rxctrfreq, 
                decimation_scheme=create_test_scheme_9(),
                comment_string='Twofsound classic scan-by-scan')

        print(self.txctrfreq)

        self.add_slice(slice_1)

        self.add_slice(slice_2, interfacing_dict={0: 'SCAN'})

        # Other things you can change if you wish. You may want to discuss with us about
        # it beforehand.
        # These apply to the experiment and all slices as a whole.


        # self.txrate = 12000000 # Hz, sample rate fed to DAC

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
