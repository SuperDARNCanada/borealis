#!/usr/bin/python3

#Copyright SuperDARN Canada 2019

import os
import sys
import copy

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf
from experiment_prototype.decimation_scheme.decimation_scheme import \
    DecimationScheme, DecimationStage, create_firwin_filter_by_attenuation


def decimate_10MHz_scheme():
    """
    Built off of the default scheme used for 45 km with minor changes to
    accommodate the 10MHz input rate instead of 5 MHz.

    :return DecimationScheme: a decimation scheme for use in experiment.
    """

    rates = [10.0e6, 500.0e3, 100.0e3, 50.0e3/3] 
    dm_rates = [20, 5, 6, 5]  
    transition_widths = [300.0e3, 50.0e3, 15.0e3, 1.0e3]  
    # bandwidth is double cutoffs. 
    cutoffs = [20.0e3, 10.0e3, 10.0e3, 5.0e3]
    ripple_dbs = [150.0, 80.0, 35.0, 8.0]  
    scaling_factors = [10.0, 100.0, 100.0, 100.0]  
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



class InterleaveSound(ExperimentPrototype):
    """Interleavesound is a modified version of Interleavedscan with added sounding
    frequency data.

    Interleavedscan was requested in 2016 by Tomo Hori to support the ERG mission.
    On September 13th, 2016 Tomo and Evan sent emails to the darn-swg mailing list regarding
    this request. It was requested to run starting Nov 2016 with the launch of the ERG Japanese
    satellite. It interleaves the beam number, for example a 16-beam radar would proceed like:
    0-4-8-12 - 2-6-10-14 - 1-5-9-13 - 3-711-15 for the forward, and the reverse of that for the
    backward. They were looking to capture doppler velocity oscillations related to Pc3
    geomagnetic pulsations near the cusp."""
    def __init__(self):
        cpid = 197

        forward_beams = [0, 4, 8, 12, 2, 6, 10, 14, 1, 5, 9, 13, 3, 7, 11, 15]
        reverse_beams = [15, 11, 7, 3, 13, 9, 5, 1, 14, 10, 6, 2, 12, 8, 4, 0]
        sounding_beams = [0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15]

        if scf.IS_FORWARD_RADAR:
            beams_to_use = forward_beams
        else:
            beams_to_use = reverse_beams

        slices = []
        
        common_scanbound_spacing = 3.0 # seconds
        common_intt_ms = common_scanbound_spacing * 1.0e3 - 100  # reduce by 100 ms for processing

        slices.append({  # slice_id = 0, the first slice
            "pulse_sequence": scf.SEQUENCE_8P,
            "tau_spacing": scf.TAU_SPACING_8P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": common_intt_ms,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            # this scanbound will be aligned because len(beam_order) = len(scanbound)
            "scanbound" : [i * common_scanbound_spacing for i in range(len(beams_to_use))],
            "txfreq" : scf.COMMON_MODE_FREQ_1, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
            "lag_table": scf.STD_8P_LAG_TABLE, # lag table needed for 8P since not all lags used.
        })

        sounding_scanbound_spacing = 1.5 # seconds
        sounding_intt_ms = sounding_scanbound_spacing * 1.0e3 - 250

        sounding_scanbound = [48 + i * sounding_scanbound_spacing for i in range(8)]
        for num, freq in enumerate(scf.SOUNDING_FREQS):
            slices.append({
                "pulse_sequence": scf.SEQUENCE_8P,
                "tau_spacing": scf.TAU_SPACING_8P,
                "pulse_len": scf.PULSE_LEN_45KM,
                "num_ranges": scf.STD_NUM_RANGES,
                "first_range": scf.STD_FIRST_RANGE,
                "intt": sounding_intt_ms,  # duration of an integration, in ms
                "beam_angle": scf.STD_16_BEAM_ANGLE,
                "beam_order": sounding_beams,
                "scanbound" : sounding_scanbound,
                "txfreq" : freq,
                "acf": True,
                "xcf": True,  # cross-correlation processing
                "acfint": True,  # interferometer acfs
                "lag_table": scf.STD_8P_LAG_TABLE, # lag table needed for 8P since not all lags used.
                })

        super(InterleaveSound, self).__init__(cpid, rx_bandwidth=10.0e6, tx_bandwidth=10.0e6, 
                                              txctrfreq=14000.0, rxctrfreq=14000.0,
                                              decimation_scheme=decimate_10MHz_scheme(),
                                              comment_string=InterleaveSound.__doc__)

        self.add_slice(slices[0])
        self.add_slice(slices[1], {0:'SCAN'})
        for slice_num in range(2,len(slices)):
            self.add_slice(slices[slice_num], {1:'INTTIME'})

