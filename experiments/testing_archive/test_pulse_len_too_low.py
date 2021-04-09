#!/usr/bin/python

# write an experiment that raises an exception

import sys
import os

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype
from utils.experiment_options.experimentoptions import ExperimentOptions as eo
from experiment_prototype.decimation_scheme.decimation_scheme import \
    DecimationScheme, DecimationStage, create_firwin_filter_by_attenuation

class TestExperiment(ExperimentPrototype):

    def __init__(self):
        cpid = 1

        rates = [5.0e6, 500.0e3, 100.0e3, 50.0e3]  # 50kHz output rate to try for a 20us pulse length
        dm_rates = [10, 5, 2, 1]
        transition_widths = [150.0e3, 40.0e3, 15.0e3, 1.0e3]
        cutoffs = [20.0e3, 10.0e3, 10.0e3, 5.0e3]
        ripple_dbs = [150.0, 80.0, 35.0, 9.0]
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
        decimation_scheme = (DecimationScheme(rates[0], rates[-1]/dm_rates[-1], stages=all_stages))
        super(TestExperiment, self).__init__(
            cpid, output_rx_rate=decimation_scheme.output_sample_rate,
            decimation_scheme=decimation_scheme)

        if scf.IS_FORWARD_RADAR:
            beams_to_use = scf.STD_16_FORWARD_BEAM_ORDER
        else:
            beams_to_use = scf.STD_16_REVERSE_BEAM_ORDER

        if scf.opts.site_id in ["cly", "rkn", "inv"]:
            num_ranges = scf.POLARDARN_NUM_RANGES
        if scf.opts.site_id in ["sas", "pgr"]:
            num_ranges = scf.STD_NUM_RANGES

        slice_1 = {  # slice_id = 0, there is only one slice.
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": int(eo().pulse_ramp_time * 2 * 1e6),  # Too small, should fail. Note that the output rx rate needs to be 50kHz for a 20us pulse length.
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "beam_order": beams_to_use,
            "scanbound": [i * 3.5 for i in range(len(beams_to_use))], #1 min scan
            "txfreq" : scf.COMMON_MODE_FREQ_1, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }
        self.add_slice(slice_1)
