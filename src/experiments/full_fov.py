#!/usr/bin/python

# Copyright SuperDARN Canada 2022
# The mode transmits with a pre-calculated phase progression across
# the array which illuminates the full FOV, and receives on all antennas.
# The first pulse in each sequence starts on the 0.1 second boundaries,
# to enable bistatic listening on other radars.

import sys
import os

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


class FullFOV(ExperimentPrototype):
    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int

        """
        cpid = 3800
        super().__init__(cpid)

        num_ranges = scf.STD_NUM_RANGES
        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES

        # default frequency set here
        freq = scf.COMMON_MODE_FREQ_1

        if kwargs:
            if 'freq' in kwargs.keys():
                freq = kwargs['freq']

        self.printing('Frequency set to {}'.format(freq))

        num_antennas = scf.opts.main_antenna_count

        self.add_slice({  # slice_id = 0, there is only one slice.
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": [[i for i in range(num_antennas)]],
            "tx_beam_order": [0],   # only one pattern
            "tx_antenna_pattern": scf.easy_widebeam,
            "freq": freq,  # kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
            "align_sequences": True     # align start of sequence to tenths of a second
        })

