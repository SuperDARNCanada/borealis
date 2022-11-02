#!/usr/bin/python

# Updated 4 July 2019
# Incoherent Multiple Pulse Sequence Testing
#
# Ashton Reimer

# Updated 23 March 2020


# write an experiment that creates a new control program.
import os
import sys
import copy
import numpy as np

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


def phase_encode(beam_iter, sequence_num, num_pulses):
    return np.random.uniform(-180.0, 180, num_pulses)

class ImptTest(ExperimentPrototype):

    def __init__(self):
        cpid = 3313

        default_slice = {  # slice_id = 0, the first slice
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
        }

        impt_slice = copy.deepcopy(default_slice)
        impt_slice['pulse_phase_offset'] = phase_encode

        super(ImptTest, self).__init__(cpid, comment_string="Reimer IMPT Experiment")

        self.add_slice(default_slice)

        self.add_slice(impt_slice, interfacing_dict={0: 'SCAN'})

