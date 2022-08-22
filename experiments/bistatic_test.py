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


class BistaticTest(ExperimentPrototype):
    """
    This experiment has different behaviour depending on the site that 
    is operating it. SAS, INV, and CLY operate normally (i.e. monostatically),
    while RKN and PGR 'listen in' on CLY, therefore operating as separate
    bistatic systems with CLY. All sites run a widebeam mode that 
    receives (and transmits for some sites) the entire FOV simultaneously.
    """
    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int, kHz

        """
        cpid = 4000
        super().__init__(cpid)

        num_ranges = scf.STD_NUM_RANGES
        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES

        # default frequency set here
        if scf.opts.site_id in ['cly', 'inv', 'sas']:
            freq = scf.COMMON_MODE_FREQ_1
        else:   # RKN and PGR listen to CLY
            freq = 10700    # CLY freq 1

        num_antennas = scf.opts.main_antenna_count

        slice_0 = {
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": [[i for i in range(len(scf.STD_16_BEAM_ANGLE)]],
            "freq": freq,  # kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
            "align_sequences": True     # align start of sequence to tenths of a second
        }

        if scf.opts.site_id in ['cly', 'inv', 'sas']:   # RKN and PGR listen-only sites
            slice_0['tx_antenna_pattern'] = scf.easy_widebeam
            slice_0['tx_beam_order'] = [0]  # Only one pattern

        self.add_slice(slice_0)

