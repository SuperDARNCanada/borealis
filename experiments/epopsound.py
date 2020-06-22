"""
Copyright SuperDARN Canada 2020

Keith Kotyk
"""
import os
import sys
import copy
import datetime

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf

EPOP_PASS_FILE = "~/borealis_schedules/{}.epop.passes"

class Epopsound(ExperimentPrototype):
    """Experiment for conjunction with EPOP RRI. This mode creates a transmission that is received
    by RRI"""

    def __init__(self, arg):
        epop_file = EPOP_PASS_FILE.format(scf.opts.site_id)

        with open(epop_file) as f:
            lines = f.readlines()

        time = datetime.datetime.utcnow()

        for line in lines:
            timestamp = int(line[0])

            dt = datetime.datetime.utcfromtimestamp(timestamp)

            if dt < time:
                continue
            else:
                beam = int(line[1])
                marker_period = int(line[2])
                freq = int(line[3])
                break

        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES
        if scf.opts.site_id in ["sas", "pgr"]:
            num_ranges = scf.STD_NUM_RANGES

        beams_to_use = [beam] * marker_period
        slice_0 = {
            "pulse_sequence": scf.SEQUENCE_8P,
            "tau_spacing": scf.TAU_SPACING_8P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intn": 10,
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            "txfreq" : freq, #kHz
            "acf": True,
            "xcf": True,
            "acfint": True,
        }

        beams_to_use = [beam]
        slice_1 = {
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intn": 10,
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            "txfreq" : freq, #kHz
            "acf": True,
            "xcf": True,
            "acfint": True,
        }


        super(Epopsound, self).__init__(comment_string=Epopsound.__docstring__)

