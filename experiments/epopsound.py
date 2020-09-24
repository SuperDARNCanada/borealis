"""
Copyright SuperDARN Canada 2020

Keith Kotyk

This experiment depends on a complementary passes file. The file should have name
{radar}.epop.passes name and located under the directory stored in the BOREALISSCHEDULEPATH env
variable. Lines in the file follow the structure:

utctimestampfromepoch beam marker_period freq(khz)

The closest upcoming timestamp is used, so make sure this mode begins running before the required
line.
"""

import os
import sys
import datetime

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf

EPOP_PASS_FILE = os.environ['BOREALISSCHEDULEPATH'] + "/{}.epop.passes"


class Epopsound(ExperimentPrototype):
    """Experiment for conjunction with EPOP RRI. This mode creates a transmission that is received
    by RRI"""

    def __init__(self, **kwargs):
        cpid = 3371
        epop_file = EPOP_PASS_FILE.format(scf.opts.site_id)

        beam = kwargs['beam']
        marker_period = kwargs['marker_period']
        freq = kwargs['freq']


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
            "txfreq": freq,  # kHz
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
            "txfreq": freq,  # kHz
            "acf": True,
            "xcf": True,
            "acfint": True,
        }


        super(Epopsound, self).__init__(cpid=cpid, txctrfreq=freq, rxctrfreq=freq,
                                        comment_string=Epopsound.__doc__)
        self.add_slice(slice_0)
        self.add_slice(slice_1, interfacing_dict={0: 'SCAN'})

