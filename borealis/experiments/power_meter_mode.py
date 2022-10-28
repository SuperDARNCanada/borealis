#!/usr/bin/python

import sys
import os

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


class PowerMeterMode(ExperimentPrototype):

    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int

        """
        cpid = 3580
        super(PowerMeterMode, self).__init__(cpid)

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
            "pulse_sequence": [0],
            "tau_spacing": 300,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": 1,
            "first_range": 0,
            "intt": 4000,  # duration of an integration, in ms
            "beam_angle": [0.0],
            "beam_order": [0],
            #"scanbound": [i * 3.5 for i in range(len(beams_to_use))], #1 min scan
            "txfreq" : freq, #kHz
            "acf": False,
            "xcf": False,  # cross-correlation processing
            "acfint": False,  # interferometer acfs
        })

