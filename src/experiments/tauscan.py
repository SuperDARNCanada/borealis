#!/usr/bin/python
"""
Tauscan experiment

Copywrite SuperDARN 2020

Keith Kotyk

"""
import os
import sys

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


class Tauscan(ExperimentPrototype):
    """A 11-pulse sequence that consists of a single pulse pulse followed by a back to back 5-pulse
    Farley sequence. The analysis produces a 12-pulse ACF with no missing lags."""
    def __init__(self):
        cpid = 503

        super(Tauscan, self).__init__(cpid, comment_string=Tauscan.__doc__)

        if scf.IS_FORWARD_RADAR:
            beams_to_use = scf.STD_16_FORWARD_BEAM_ORDER
        else:
            beams_to_use = scf.STD_16_REVERSE_BEAM_ORDER

        if scf.opts.site_id == "sas":
            freq = 13200
        elif scf.opts.site_id == "pgr":
            freq = 13100
        else:
            freq = 13500

        slice_1 = {
            "pulse_sequence": [0, 10, 13, 14, 19, 21, 31, 33, 38, 39, 42],
            "tau_spacing": 3000,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": 100,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 7000,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": beams_to_use,
            "tx_beam_order": beams_to_use,
            "scanbound" : [i * 7.0 for i in range(len(beams_to_use))],
            "freq" : freq, #kHz
            "acf" : True,
            "xcf" : True,
            "acfint" : True,
            "comment" : Tauscan.__doc__
        }

        self.add_slice(slice_1)

