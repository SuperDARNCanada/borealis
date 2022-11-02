#!/usr/bin/python

# Keith Kotyk
# Copyright SuperDARN Canada 2020

import os
import sys
import copy
import numpy as np

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


def phase_encode(beam_iter, sequence_num, num_pulses, num_samples):
    return np.array([ 125.73471064,   60.71636783,  120.78349373,   84.34937441,
        135.91385006, -160.56231581,  129.70333278,  -61.5067707 ])


class BorealisPaper(ExperimentPrototype):

    def __init__(self):
        cpid = 10101

        default_slice = {
            "pulse_sequence": scf.SEQUENCE_8P,
            "tau_spacing": scf.TAU_SPACING_8P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_8P,
            "beam_angle": [1.75],
            "rx_beam_order": [0],
            "tx_beam_order": [0],
            "freq" : 13100,
            "acf" : True
        }

        slice2 = copy.deepcopy(default_slice)
        slice2['pulse_phase_offset'] = phase_encode

        super(BorealisPaper, self).__init__(cpid, comment_string="Phase encoding test for borealis paper")

        self.add_slice(default_slice)

        self.add_slice(slice2, interfacing_dict={0: 'SCAN'})

