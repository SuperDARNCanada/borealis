#!/usr/bin/python
"""
Copyright SuperDARN Canada 2022
The mode transmits with a pre-calculated phase progression across the array which illuminates the full FOV,
and receives on all antennas. The first pulse in each sequence starts on the 0.1 second boundaries, to enable bistatic
listening on other radars. This mode uses 15-km range gates for high spatial resolution.
"""

import sys
import os

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype
from experiment_prototype.decimation_scheme.decimation_scheme import \
    DecimationScheme, DecimationStage, create_firwin_filter_by_attenuation


def create_15km_scheme():
    """
    Frankenstein script by Devin Huyghebaert for 15 km range gates with
    a special ICEBEAR collab mode. Copied from IB_collab_mode.py experiment.

    Built off of the default scheme used for 45 km with minor changes.

    :return DecimationScheme: a decimation scheme for use in experiment.
    """

    rates = [5.0e6, 500.0e3, 100.0e3, 50.0e3]  # last stage 50.0e3/3->50.0e3
    dm_rates = [10, 5, 2, 5]  # third stage 6->2
    transition_widths = [150.0e3, 40.0e3, 15.0e3, 1.0e3]  # did not change
    # bandwidth is double cutoffs.  Did not change
    cutoffs = [20.0e3, 10.0e3, 10.0e3, 5.0e3]
    ripple_dbs = [150.0, 80.0, 35.0, 8.0]  # changed last stage 9->8
    scaling_factors = [10.0, 100.0, 100.0, 100.0]  # did not change
    all_stages = []

    for stage in range(0, len(rates)):
        filter_taps = list(
            scaling_factors[stage] * create_firwin_filter_by_attenuation(
                rates[stage], transition_widths[stage], cutoffs[stage],
                ripple_dbs[stage]))
        all_stages.append(DecimationStage(stage, rates[stage],
                          dm_rates[stage], filter_taps))

    # changed from 10e3/3->10e3
    return (DecimationScheme(rates[0], rates[-1]/dm_rates[-1],
                             stages=all_stages))


class FullFOV15Km(ExperimentPrototype):
    def __init__(self, **kwargs):
        """
        kwargs:

        freq: int, kHz

        """
        cpid = 3801
        decimation_scheme = create_15km_scheme()

        super().__init__(cpid, output_rx_rate=decimation_scheme.output_sample_rate, decimation_scheme=decimation_scheme,
                         comment_string='Full FOV 15km Resolution Experiment')

        num_ranges = scf.STD_NUM_RANGES * 3     # Each range is a third of the usual size, want same spatial extent

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
            "pulse_len": scf.PULSE_LEN_15KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_7P,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": [[i for i in range(num_antennas)]],
            "tx_beam_order": [0],   # only one pattern
            "tx_antenna_pattern": scf.easy_widebeam,
            "freq": freq,  # kHz
        })

