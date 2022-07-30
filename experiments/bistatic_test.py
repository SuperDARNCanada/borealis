#!/usr/bin/python

# Copyright SuperDARN Canada 2022
# The mode transmits with a pre-calculated phase progression across
# the array which illuminates the full FOV, and receives on all antennas.
# The first pulse in each sequence starts on the 0.1 second boundaries,
# to enable bistatic listening on other radars.
# This mode also chooses a frequency from another radar to listen in on,
# also across the entire FOV simultaneously.
import copy
import sys
import os

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype


class FullFOV(ExperimentPrototype):
    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int

        """
        cpid = 3715
        super().__init__(cpid)

        num_ranges = scf.STD_NUM_RANGES
        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES

        # default frequency set here
        if scf.opts.site_id in ['inv', 'cly']:
            freq = scf.COMMON_MODE_FREQ_2
        else:
            freq = scf.COMMON_MODE_FREQ_1

        if kwargs:
            if 'freq' in kwargs.keys():
                freq = kwargs['freq']

        self.printing('Frequency set to {}'.format(freq))

        num_antennas = scf.opts.main_antenna_count

        slice_0 = {
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
        }

        slice_1 = copy.deepcopy(slice_0)

        # Remove these to create an rx-only slice
        slice_1.pop('tx_antenna_pattern')
        slice_1.pop('tx_beam_order')

        if scf.opts.site_id == 'rkn':
            freq1 = 12200       # INV freq 2
        elif scf.opts.site_id == 'pgr':
            freq1 = 12500       # CLY freq 2
        else:
            freq1 = 10900       # RKN freq 1
        slice_1['freq'] = freq1

        self.add_slice(slice_0)
        self.add_slice(slice_1, interfacing_dict={0: 'CONCURRENT'})
