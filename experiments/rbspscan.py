#!/usr/bin/python3

#Copyright SuperDARN Canada 2019

import os
import sys
import copy

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf

class RBSPScan(ExperimentPrototype):
    """notes on RBSPScan purpose here TODO"""
    def __init__(self,):
        cpid = 200

        forward_beams = [0, "westbm", 1, "meridonalbm", 2, "eastbm", 3, "westbm", 4, "meridonalbm",
                         5, "eastbm", 6, "westbm", 7, "meridonalbm", 8, "eastbm", 9, "westbm", 10,
                         "meridonalbm", 11, "eastbm", 12, "westbm", 13, "meridonalbm", 14, "eastbm",
                         15]
        reverse_beams = [15, "eastbm", 14, "meridonalbm", 13, "westbm", 12, "westbm", 11,
                        "meridonalbm", 10, "westbm", 9, "eastbm", 8, "meridonalbm", 7, "westbm", 6,
                        "eastbm", 5, "meridonalbm", 4, "westbm", 3, "eastbm", 2, "meridonalbm", 1,
                        "westbm", 0]

        if scf.IS_FORWARD_RADAR:
            beams_to_use = forward_beams
        else:
            beams_to_use = reverse_beams

        if scf.opts.site_id in ["sas"]:
            westbm = 2
            meridonalbm = 3
            eastbm = 5
        if scf.opts.site_id in ["pgr"]:
            westbm = 12
            meridonalbm = 13
            eastbm = 15
        if scf.opts.site_id in ["inv", "rkn", "cly"]:
            westbm = 6
            meridonalbm = 7
            eastbm = 9

        if scf.opts.site_id in ["sas", "pgr", "cly"]:
            freq = 10500
        if scf.opts.site_id in ["rkn"]:
            freq = 12200
        if scf.opts.site_id in ["inv"]:
            freq = 12100

        beams_to_use = [westbm if bm == "westbm" else bm for bm in beams_to_use]
        beams_to_use = [meridonalbm if bm == "meridonalbm" else bm for bm in beams_to_use]
        beams_to_use = [eastbm if bm == "eastbm" else bm for bm in beams_to_use]

        slice_1 = {  # slice_id = 0, the first slice
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 3750,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            "txfreq" : freq, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }
        super(RBSPScan, self).__init__(cpid)

        self.add_slice(slice_1)