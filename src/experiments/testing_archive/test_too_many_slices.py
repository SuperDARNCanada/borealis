#!/usr/bin/python

# write an experiment that raises an exception

import copy
import sys
import os

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

import experiments.superdarn_common_fields as scf
from experiment_prototype.experiment_prototype import ExperimentPrototype
from utils.options.experimentoptions import ExperimentOptions as eo
from experiment_prototype.decimation_scheme.decimation_scheme import \
    DecimationScheme, DecimationStage, create_firwin_filter_by_attenuation

class TestExperiment(ExperimentPrototype):

    def __init__(self):
        cpid = 1
        
        rates = [5.0e6, 500.0e3, 100.0e3, 50.0e3/3]
        dm_rates = [10, 5, 6, 5]
        transition_widths = [150.0e3, 40.0e3, 15.0e3, 1.0e3]
        cutoffs = [20.0e3, 10.0e3, 10.0e3, 5.0e3]
        ripple_dbs = [150.0, 80.0, 35.0, 9.0]
        scaling_factors = [10.0, 100.0, 100.0, 100.0]
        all_stages = []
        longest_filter_num_taps = 0
        for stage in range(0, len(rates)):
            filter_taps = list(
                scaling_factors[stage] * create_firwin_filter_by_attenuation(
                    rates[stage], transition_widths[stage], cutoffs[stage],
                    ripple_dbs[stage]))
            all_stages.append(DecimationStage(stage, rates[stage],
                              dm_rates[stage], filter_taps))
            if len(filter_taps) > longest_filter_num_taps:
                longest_filter_num_taps = len(filter_taps)

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
            "pulse_len": scf.PULSE_LEN_45KM,
            "num_ranges": num_ranges,
            "first_range": scf.STD_FIRST_RANGE,
            "intt": 3500,  # duration of an integration, in ms
            "beam_angle": scf.STD_16_BEAM_ANGLE,
            "rx_beam_order": beams_to_use,
            "tx_beam_order": beams_to_use,
            "scanbound": [i * 3.5 for i in range(len(beams_to_use))], #1 min scan
            "freq" : scf.COMMON_MODE_FREQ_1, #kHz
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }
        # The check for too many slices is like so:
        # effective_length = 2^x => the next power of two for how many taps there are in a stage
        # if effective_length * num_slices > max_number_of_filter_taps_per_stage:
        # error

        self.add_slice(slice_1)
        self.add_slice(copy.deepcopy(slice_1)) 
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={1: 'AVEPERIOD'})
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={1: 'AVEPERIOD'})
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
        slices = []
        max_num_slices = int(eo().max_number_of_filter_taps_per_stage/longest_filter_num_taps)
        #print("Longest filter: {} taps. Max concurrent slices: {}".format(longest_filter_num_taps, 
        #    max_num_slices))
        for i in range(max_num_slices):
            slices.append(copy.deepcopy(slice_1))
            self.add_slice(slices[i], interfacing_dict={0: 'CONCURRENT'})

        # Just for fun, add a few AVEPERIOD as well
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={1: 'AVEPERIOD'})
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'AVEPERIOD'})
        
        # Add a bunch of slices that are scan interfaced, shouldn't have an effect
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
        self.add_slice(copy.deepcopy(slice_1),interfacing_dict={0: 'SCAN'}) 
