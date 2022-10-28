#!/usr/bin/python
"""
Copyright SuperDARN Canada 2022
This mode is a comparison between the transmission characteristics of full_fov.py and normalscan.py,
running on one frequency but interleaving the two transmissions each averaging period.
The first pulse in each sequence starts on the 0.1 second boundaries, to enable bistatic listening on other radars.
"""

import sys
import os
import copy
import numpy as np

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


def boresight(frequency_khz, tx_antennas, antenna_spacing_m):
    """tx_antenna_pattern function for boresight transmission."""
    num_antennas = scf.opts.main_antenna_count
    pattern = np.zeros((1, num_antennas), dtype=np.complex64)
    pattern[0, tx_antennas] = 1.0 + 0.0j
    return pattern


class FullFOVComparison(ExperimentPrototype):
    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int, kHz

        """
        cpid = 3812
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

        slice_0 = {  # slice_id = 0
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": [[i for i in range(len(scf.STD_16_BEAM_ANGLE))]],
            "tx_beam_order": [0],  # only one pattern
            "tx_antenna_pattern": scf.easy_widebeam,
            "freq": freq,  # kHz
            "align_sequences": True,  # align start of sequence to tenths of a second
            "scanbound": scf.easy_scanbound(scf.INTT_7P, scf.STD_16_BEAM_ANGLE),
            "wait_for_first_scanbound": False
        }

        slice_1 = copy.deepcopy(slice_0)
        slice_1['tx_antenna_pattern'] = boresight

        self.add_slice(slice_0)
        self.add_slice(slice_1, interfacing_dict={0: 'AVEPERIOD'})
