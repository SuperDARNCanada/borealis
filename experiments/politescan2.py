#!/usr/bin/python

# write an experiment that creates a new control program.

# normalscan and listen has an appended listening integration time 
# at the end of a full scan. 
# integration times are reduced to 3s to allow time for this listening
# integration time. 

import sys
import os

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype

class Politescan2(ExperimentPrototype):

    def __init__(self):
        cpid = 3383
        super(Politescan2, 
              self).__init__(
                  cpid, comment_string='Politescan on two frequencies '
                                       'simultaneously.')

        if scf.IS_FORWARD_RADAR:
            beams_to_use = scf.STD_16_FORWARD_BEAM_ORDER
        else:
            beams_to_use = scf.STD_16_REVERSE_BEAM_ORDER

        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES
        if scf.opts.site_id in ["sas", "pgr"]:
            num_ranges = scf.STD_NUM_RANGES

        self.add_slice({  # slice_id = 0, added first
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            # scanbound ends at 48s.
            "scanbound": [i * 3.5 for i in range(len(beams_to_use))],
            "rxfreq" : 10500, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        })

        self.add_slice({  # slice_id = 1, receive only
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM, 
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,  
            # offset beams so not looking in same direction. 
            "beam_order": beams_to_use,
            "scanbound" : [i * 3.5 for i in range(len(beams_to_use))], 
            "rxfreq" : 13000, #kHz, separate frequency
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }, interfacing_dict={0: 'PULSE'})
