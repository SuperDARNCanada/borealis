#!/usr/bin/python
"""
Tauscan experiment

Copywrite SuperDARN 2020

Keith Kotyk

"""
import os
import sys
import copy

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf

class Tauscan(ExperimentPrototype):
    """docstring for Tauscan"""
    def __init__(self):
        cpid = 503

        if scf.IS_FORWARD_RADAR:
            beams_to_use = scf.STD_16_FORWARD_BEAM_ORDER
        else:
            beams_to_use = scf.STD_16_REVERSE_BEAM_ORDER
h
        if scf.opts.site_id in ["sas", "pgr", "cly"]:
            freq = 10500,
        if scf.opts.site_id in ["rkn"]:
            freq = 10200
        if scf.opts.site_id in ["inv"]:
            freq = 10300

        slice_1 = {
            "pulse_sequence": [0, 10, 13, 14, 19, 21, 31, 33, 38, 39, 42],
            "tau_spacing": 3000,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 5600,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            "scanbound" : [i * 5.6 for i in range(len(beams_to_use))],
            "txfreq" : 10500, #kHz
        }

        super(Tauscan, self).__init__()

