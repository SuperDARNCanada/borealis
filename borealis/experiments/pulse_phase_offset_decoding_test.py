#!/usr/bin/python

import os
import sys
import numpy as np

# write an experiment that creates a new control program.
from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


def phase_encode(beam_iter, sequence_num, num_pulses, num_samples):
    return np.random.uniform(-180.0, 180.0, num_pulses)


class PulsePhaseOffsetDecodingTest(ExperimentPrototype):
    # with 7 PULSE sequence
    def __init__(self):
        cpid = 10001000

        super().__init__(cpid, comment_string="Testing Pulse Phase Offset removal in ACF Generation")

        self.add_slice({  # slice_id = 0, there is only one slice.
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": [0.0],
            "beam_order": [0],
            "txfreq" : scf.COMMON_MODE_FREQ_1, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
            "pulse_phase_offset": phase_encode
        })
