#!/usr/bin/python

# Copyright SuperDARN Canada 2022
# This mode transmits two frequencies simultaneously across the entire FOV.
# The mode transmits with a pre-calculated phase progression across
# half the array for each frequency, so each antenna only transmits a
# single frequency, and receives on all antennas.
# The first pulse in each sequence starts on the 0.1 second boundaries,
# to enable bistatic listening on other radars.

import sys
import os
import copy
import numpy as np

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


class FullFOV3Freq(ExperimentPrototype):
    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int

        """
        cpid = 3716
        super().__init__(cpid)

        num_ranges = scf.STD_NUM_RANGES
        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES

        tx_freq_1 = scf.COMMON_MODE_FREQ_1
        tx_freq_2 = scf.COMMON_MODE_FREQ_2

        if kwargs:
            if 'freq1' in kwargs.keys():
                tx_freq_1 = int(kwargs['freq1'])

            if 'freq2' in kwargs.keys():
                tx_freq_2 = int(kwargs['freq2'])

        num_antennas = scf.opts.main_antenna_count

        # This slice uses the left half of the array to transmit on one frequency.
        slice_0 = {
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 3500, #scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": [[i for i in range(num_antennas)]],
            "tx_beam_order": [0],  # only one pattern
            "tx_antenna_pattern": scf.easy_widebeam,
            "tx_antennas": [i for i in range(num_antennas // 2)],
            "freq": tx_freq_1,  # kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
            "align_sequences": True  # align start of sequence to tenths of a second
        }

        # Transmit on the second frequency on the right half of the array
        slice_1 = copy.deepcopy(slice_0)
        slice_1['freq'] = tx_freq_2
        slice_1['tx_antennas'] = [i + (num_antennas // 2) for i in range(num_antennas // 2)]

        slice_2 = copy.deepcopy(slice_0)
        slice_2['freq'] = 10400
        slice_2['tx_antennas'] = [i for i in range(num_antennas)]

        self.add_slice(slice_0)
        self.add_slice(slice_1, interfacing_dict={0: 'CONCURRENT'})
        self.add_slice(slice_2, interfacing_dict={0: 'CONCURRENT'})
