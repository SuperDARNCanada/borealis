#!/usr/bin/python3

#Copyright SuperDARN Canada 2021

import os
import sys
import copy

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


class NormalSound(ExperimentPrototype):
    """NormalSound is a modified version of normalscan with added frequency sounding.

    """
    def __init__(self):
        cpid = 157

        sounding_beams = [0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15]

        if scf.IS_FORWARD_RADAR:
            beams_to_use = scf.STD_16_FORWARD_BEAM_ORDER
        else:
            beams_to_use = scf.STD_16_REVERSE_BEAM_ORDER

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
        
        freqrange = (max(scf.SOUNDING_FREQS) - min(scf.SOUNDING_FREQS)) / 2
        centerfreq = min(scf.SOUNDING_FREQS) + freqrange

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

        super(NormalSound, self).__init__(cpid, txctrfreq=centerfreq, rxctrfreq=centerfreq, comment_string=NormalSound.__doc__)

        self.add_slice(slices[0])
        self.add_slice(slices[1], {0:'SCAN'})
        for slice_num in range(2,len(slices)):
            self.add_slice(slices[slice_num], {1:'INTTIME'})

