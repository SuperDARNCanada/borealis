#!/usr/bin/python

# A one-off experiment for a collaboration with HAARP.
# Run beams 2, 3, 4, 5, 6 at Clyde. Beam 4 range gate 72 overlaps with Gakona, AK

import sys
import os

import experiments.superdarn_common_fields as scf
from experiment_prototype import ExperimentPrototype


class HAARPScan(ExperimentPrototype):

    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int

        """
        cpid = 3530
        super(HAARPScan, self).__init__(cpid)

        if scf.IS_FORWARD_RADAR:
            beams_to_use = [2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 2]
        else:
            beams_to_use = [6, 5, 4, 3, 2, 6, 5, 4, 3, 2, 6, 5, 4, 3, 2, 6]

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
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            "scanbound": [i * 3.5 for i in range(len(beams_to_use))],  # 1 min scan
            "txfreq": freq,  # kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        })
