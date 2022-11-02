#!/usr/bin/python3

#Copyright SuperDARN Canada 2019

import os
import sys

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


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
        
        common_scanbound_spacing = 3.0  # seconds
        common_intt_ms = common_scanbound_spacing * 1.0e3 - 100  # reduce by 100 ms for processing

        slices.append({  # slice_id = 0, the first slice
            "pulse_sequence": scf.SEQUENCE_8P,
            "tau_spacing": scf.TAU_SPACING_8P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": common_intt_ms,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": beams_to_use,
            "tx_beam_order": beams_to_use,
            # this scanbound will be aligned because len(beam_order) = len(scanbound)
            "scanbound" : [i * common_scanbound_spacing for i in range(len(beams_to_use))],
            "freq" : scf.COMMON_MODE_FREQ_1, # kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
            "lag_table": scf.STD_8P_LAG_TABLE,  # lag table needed for 8P since not all lags used.
        })

        sounding_scanbound_spacing = 1.5  # seconds
        sounding_intt_ms = sounding_scanbound_spacing * 1.0e3 - 250

        sounding_scanbound = [48 + i * sounding_scanbound_spacing for i in range(8)]
        for freq in scf.SOUNDING_FREQS:
            slices.append({
                "pulse_sequence": scf.SEQUENCE_8P,
                "tau_spacing": scf.TAU_SPACING_8P,
                "pulse_len": scf.PULSE_LEN_45KM,
                "num_ranges": scf.STD_NUM_RANGES,
                "first_range": scf.STD_FIRST_RANGE,
                "intt": sounding_intt_ms,  # duration of an integration, in ms
                "beam_angle": scf.STD_16_BEAM_ANGLE,
                "rx_beam_order": sounding_beams,
                "tx_beam_order": sounding_beams,
                "scanbound" : sounding_scanbound,
                "freq": freq,
                "acf": True,
                "xcf": True,  # cross-correlation processing
                "acfint": True,  # interferometer acfs
                "lag_table": scf.STD_8P_LAG_TABLE,  # lag table needed for 8P since not all lags used
                })

        sum_of_freq = 0
        for slice in slices:
            sum_of_freq += slice['freq']  # kHz, oscillator mixer frequency on the USRP for TX
        rxctrfreq = txctrfreq = int(sum_of_freq / len(slices))

        super(InterleaveSound, self).__init__(cpid, txctrfreq=txctrfreq, rxctrfreq=rxctrfreq,
                                              comment_string=InterleaveSound.__doc__)

        self.add_slice(slices[0])
        self.add_slice(slices[1], {0: 'SCAN'})
        for slice_num in range(2, len(slices)):
            self.add_slice(slices[slice_num], {1: 'AVEPERIOD'})

