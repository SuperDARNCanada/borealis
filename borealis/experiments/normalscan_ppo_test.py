#!/usr/bin/python

# write an experiment that creates a new control program.

import sys
import os
import numpy as np

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


def phase_encode(beam_iter, sequence_num, num_pulses, num_samples):
    return np.arange(num_pulses) * 0. #+ sequence_num * np.arange(num_pulses)


class Normalscan_PPO_Test(ExperimentPrototype):

    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int

        """
        cpid = 100000051
        super().__init__(cpid)

        if scf.IS_FORWARD_RADAR:
            beams_to_use = scf.STD_16_FORWARD_BEAM_ORDER
        else:
            beams_to_use = scf.STD_16_REVERSE_BEAM_ORDER

        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES
        if scf.opts.site_id in ["sas", "pgr"]:
            num_ranges = scf.STD_NUM_RANGES

        # default frequency set here
        freq = scf.COMMON_MODE_FREQ_1
        
        if kwargs:
            if 'freq' in kwargs.keys():
                freq = kwargs['freq']
        
        self.printing('Frequency set to {}'.format(freq))

        self.add_slice({  # slice_id = 0, there is only one slice.
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            #"scanbound": scf.easy_scanbound(scf.INTT_7P, beams_to_use), #1 min scan
            "txfreq" : freq, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
            "pulse_phase_offset": phase_encode
        })

