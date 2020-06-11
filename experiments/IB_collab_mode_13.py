#!/usr/bin/python

#Test script for writing new experiment!

# write an experiment that creates a new control program.
import os
import sys
import copy

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

from experiment_prototype.experiment_prototype import ExperimentPrototype
import experiments.superdarn_common_fields as scf
from experiment_prototype.decimation_scheme.decimation_scheme import DecimationScheme, DecimationStage, create_firwin_filter_by_attenuation


def create_15km_scheme(): 
    """
    Frankenstein script by Devin Huyghebaert for 15 km range gates with a special ICEBEAR collab mode

    previous comment:
    Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
    This filter will have a wider receive bandwidth than the previous.
    Pasha recommends a 10kHz bandwidth for the final stage. I believe there will be aliasing caused by this but 
    perhaps the concern is not critical because of the small bandwidth overlapping. I will test this anyways.
    :return DecimationScheme: a decimation scheme for use in experiment.
    """

    rates = [5.0e6, 500.0e3, 100.0e3, 50.0e3] # last stage 50.0e3/3->50.0e3
    dm_rates = [10, 5, 2, 5] # third stage 6->2
    transition_widths = [150.0e3, 40.0e3, 15.0e3, 1.0e3] #did not change
    cutoffs = [20.0e3, 10.0e3, 10.0e3, 5.0e3] # bandwidth is double this.  Did not change
    ripple_dbs = [150.0, 80.0, 35.0, 8.0] # changed last stage 9->8
    scaling_factors = [10.0, 100.0, 100.0, 100.0] # did not change
    all_stages = []

    for stage in range(0,4):
        filter_taps = list(scaling_factors[stage] * create_firwin_filter_by_attenuation(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
        all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

    return (DecimationScheme(5.0e6, 10.0e3, stages=all_stages)) # changed from 10e3/3->10e3


class IBCollabMode(ExperimentPrototype):

    def __init__(self):
        cpid = 3700 #allocated by Marci Detwiller 20200609
        decimation_scheme=create_15km_scheme()

        bangle = scf.STD_16_BEAM_ANGLE
        beams_arr = [0,2,4,6,8,0,2,4,6,8,0,2,4,6,8,0,2,4,6,8,0,2,4,6,8,0,2,4,6,8]

        slice_1 = {  # slice_id = 0, the first slice
            "pulse_sequence": scf.SEQUENCE_7P,
            "tau_spacing": scf.TAU_SPACING_7P,
            "pulse_len": scf.PULSE_LEN_15KM, #(100 us for 15 km)
            "num_ranges": 100, # should only go out to 1500 km w/ 15 km range gates
            "first_range": 90, #closer than standard first range (180 km)
            "intt": 1900,  # duration of an integration, in ms
            "beam_angle": bangle,
            "beam_order": beams_arr,
            "scanbound" : [i * 2.0 for i in range(len(beams_arr))],
            "txfreq" : 13000, #kHz 
            "acf": True,
            "xcf": True,  # cross-correlation processing
            "acfint": True,  # interferometer acfs
        }

        list_of_slices = [slice_1]
        sum_of_freq = 0
        for slice in list_of_slices:
            sum_of_freq += slice['txfreq']# kHz, oscillator mixer frequency on the USRP for TX
        rxctrfreq = txctrfreq = int(sum_of_freq/len(list_of_slices))


        super(IBCollabMode, self).__init__(cpid, txctrfreq=txctrfreq, output_rx_rate=10e3,rxctrfreq=rxctrfreq,decimation_scheme=decimation_scheme,
                comment_string='ICEBEAR, 5 beam, 2s integration, 15 km')

        self.add_slice(slice_1)
