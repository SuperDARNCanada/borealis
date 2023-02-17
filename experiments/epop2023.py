#!/usr/bin/python
"""
Copyright SuperDARN Canada 2023
This mode was made for collaboration with the RRI instrument on the CASSIOPE satellite
by request of Dr. Kuldeep Pandey in February 2023.
"""

import sys
import os
import numpy as np

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


def boresight(frequency_khz, tx_antennas, antenna_spacing_m):
    """tx_antenna_pattern function for boresight transmission."""
    num_antennas = scf.opts.main_antenna_count
    pattern = np.zeros((1, num_antennas), dtype=np.complex64)
    pattern[0, tx_antennas] = 1.0 + 0.0j
    return pattern


class Epop2023(ExperimentPrototype):
    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int, kHz

        """
        cpid = 3813
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
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": [0.0],    # boresight only
            "rx_beam_order": [0],   # boresight only
            "tx_beam_order": [0],   # only one pattern
            "tx_antenna_pattern": boresight,
            "freq": freq,  # kHz
            "align_sequences": True,  # align start of sequence to tenths of a second
        }

        self.add_slice(slice_0)
