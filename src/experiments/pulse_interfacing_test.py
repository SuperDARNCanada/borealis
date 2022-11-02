#!/usr/bin/python

"""
Copyright SuperDARN Canada 2022
Author: Remington Rohel

This experiment is to test the various features of PULSE interfacing, and
to verify the correct output waveforms are generated.
"""

import sys
import os
import copy

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


class PulseInterfacingTest(ExperimentPrototype):

    def __init__(self):
        cpid = 3384
        super().__init__(cpid, comment_string='CONCURRENT interface testing')

        beams_to_use = [0]  # Camp on one beam (boresight)
        num_ranges = scf.STD_NUM_RANGES

        slice_0 = {  # slice_id = 0
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": [0.0],
            "rx_beam_order": beams_to_use,
            "tx_beam_order": beams_to_use,
            "freq" : 10500    # kHz
        }

        slice_1 = copy.deepcopy(slice_0)
        slice_2 = copy.deepcopy(slice_0)
        slice_3 = copy.deepcopy(slice_0)

        # Offset by 150us from slice_0
        slice_1['seqoffset'] = scf.PULSE_LEN_45KM // 2
        slice_1['tx_antennas'] = [0, 2]
        slice_1['freq'] = 13000  # kHz

        # Offset by 300us from slice_0 - no overlap with slice_0
        slice_2['seqoffset'] = scf.PULSE_LEN_45KM
        slice_2['tx_antennas'] = [1]
        slice_2['freq'] = 12500  # kHz

        slice_3['freq'] = 12000  # kHz

        # slice_1 and slice_2 don't share transmit antennas, so they shouldn't
        # have their power divided.

        self.add_slice(slice_0)
        self.add_slice(slice_1, interfacing_dict={0: 'CONCURRENT'})
        self.add_slice(slice_2, interfacing_dict={1: 'CONCURRENT'})
        self.add_slice(slice_3, interfacing_dict={0: 'CONCURRENT'})

