#!/usr/bin/python3

# politescan
# Marci Detwiller Jan 7/2019
# Adapted from ROS politescan (Dieter Andre, Kevin Krieger)
import os
import sys

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf


class Politescan(ExperimentPrototype):

    def __init__(self):
        cpid = 3380
        super(Politescan, self).__init__(cpid)

        if scf.IS_FORWARD_RADAR:
            beams_to_use = scf.STD_16_FORWARD_BEAM_ORDER
        else:
            beams_to_use = scf.STD_16_REVERSE_BEAM_ORDER

        self.add_slice({  # slice_id = 0, there is only one slice.
            "pulse_sequence": scf.SEQUENCE_8P,
            "tau_spacing": scf.TAU_SPACING_8P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": scf.STD_NUM_RANGES,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": scf.INTT_8P,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": beams_to_use,
            "scanbound" : scf.easy_scanbound(scf.INTT_7P, beams_to_use),
            "freq" : 10500, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        })
