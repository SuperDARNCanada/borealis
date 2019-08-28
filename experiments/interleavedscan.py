#!/usr/bin/python3

#Copyright SuperDARN Canada 2019

import os
import sys
import copy

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf

class InterleavedScan(ExperimentPrototype):
    """Notes on InterleavedScan purpose here TODO"""
    def __init__(self):
        cpid = 191

        forward_beams = [0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15]
        reverse_beams = [15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0]

        if scf.IS_FORWARD_RADAR:
            beams_to_use = forward_beams
        else:
            beams_to_use = reverse_beams


        slice_1 = {  # slice_id = 0, the first slice
            "pulse_sequence": scf.SEQUENCE_8P,
            "tau_spacing": scf.TAU_SPACING_8P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            "txfreq" : 10500, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
            "lag_table": scf.STD_8P_LAG_TABLE, # lag table needed for 8P since not all lags used.
        }
        super(InterleavedScan, self).__init__(cpid)

        self.add_slice(slice_1)