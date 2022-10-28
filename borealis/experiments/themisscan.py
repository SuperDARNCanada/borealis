#!/usr/bin/python3

#Copyright SuperDARN Canada 2019

import os
import sys
import copy

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


class ThemisScan(ExperimentPrototype):
    """notes on ThemisScan purpose here TODO"""
    def __init__(self,):
        cpid = 3300

        forward_beams = [0, "camp", 1, "camp", 2,  "camp", 3, "camp", 4, "camp", 5, "camp", 6,
                         "camp", 7, "camp", 8, "camp", 9, "camp", 10, "camp", 11, "camp", 12,
                         "camp", 13, "camp", 14, "camp", 15, "camp", "camp", "camp", "camp", "camp",
                         "camp", "camp"]
        reverse_beams = [15, "camp", 14, "camp", 13,  "camp", 12, "camp", 11, "camp", 10, "camp", 9,
                         "camp", 8, "camp", 7, "camp", 6, "camp", 5, "camp", 4, "camp", 3, "camp",
                         2, "camp", 1, "camp", 0, "camp", "camp", "camp", "camp", "camp", "camp",
                         "camp"]

        if scf.IS_FORWARD_RADAR:
            beams_to_use = forward_beams
        else:
            beams_to_use = reverse_beams

        if scf.opts.site_id in ["sas", "inv", "cly"]:
            camp = 6
        if scf.opts.site_id in ["pgr"]:
            camp = 12
        if scf.opts.site_id in ["rkn"]:
            camp = 7

        if scf.opts.site_id in ["sas", "pgr", "cly"]:
            freq = 10500
        if scf.opts.site_id in ["rkn"]:
            freq = 12200
        if scf.opts.site_id in ["inv"]:
            freq = 12100

        beams_to_use = [camp if bm == "camp" else bm for bm in beams_to_use]

        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES
        if scf.opts.site_id in ["sas", "pgr",]:
            num_ranges = scf.STD_NUM_RANGES

        slice_1 = {  # slice_id = 0, the first slice
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 2600,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": beams_to_use,
            "tx_beam_order": beams_to_use,
            "scanbound" : [i * 3 for i in range(len(beams_to_use))],
            "freq" : freq, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }
        super(ThemisScan, self).__init__(cpid)

        self.add_slice(slice_1)
