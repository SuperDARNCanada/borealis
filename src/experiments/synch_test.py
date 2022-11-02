#!/usr/bin/python

import os
import sys

# write an experiment that creates a new control program.
from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


class SynchTest(ExperimentPrototype):
    # with 7 PULSE sequence
    def __init__(self):
        cpid = 3583

        super().__init__(cpid)

        self.add_slice({  # slice_id = 0, there is only one slice.
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": [0.0],    # boresite
            "beam_order": [0]*16, 
            "scanbound": [3.55*i for i in range(16)],
            "txfreq" : scf.COMMON_MODE_FREQ_1, #kHz
            "acf": False,
            "xcf": False,  # cross-correlation processing
            "acfint": False,  # interferometer acfs
        })
