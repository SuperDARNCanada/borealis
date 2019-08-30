#!/usr/bin/python

import os
import sys

sys.path.append(os.environ['BOREALISPATH'])

from experiment_prototype.decimation_scheme.decimation_scheme import DecimationStage, DecimationScheme, \
    create_firwin_filter_by_num_taps, create_firwin_filter_by_attenuation


# Schemes 1-8 are original test schemes circa February 2019. They are no longer in use.

# def create_test_scheme_1():
#   """
#   Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. Filter lengths are: 1284, 1284, 215, 130. 
#   :return DecimationScheme: a decimation scheme for use in experiment.
#   """

#   rates = [5.0e6, 500.0e3, 50.0e3, 10.0e3]
#   dm_rates = [10, 10, 5, 3]
#   transition_widths = [50.0e3, 5.0e3, 3.0e3, 1.0e3]
#   cutoffs = [460.0e3, 46.0e3, 8.0e3, 2.0e3]
#   ripple_dbs = [100.0, 100.0, 100.0, 100.0]

#   all_stages = []
#   for stage in range(0,4):
#       filter_taps = list(create_firwin_filter_by_attenuation(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
#       all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

#   return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


# def create_test_scheme_2():
#   """
#   Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
#   :return DecimationScheme: a decimation scheme for use in experiment.
#   """

#   rates = [5.0e6, 500.0e3, 50.0e3, 10.0e3]
#   dm_rates = [10, 10, 5, 3]
#   transition_widths = [300.0e3, 35.0e3, 7.0e3, 1.0e3]
#   cutoffs = [100.0e3, 5.0e3, 2.0e3, 0.5e3] # bandwidth is double this
#   ripple_dbs = [100.0, 60.0, 20.0, 8.0]

#   all_stages = []

#   for stage in range(0,4):
#       filter_taps = list(create_firwin_filter_by_attenuation(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
#       all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

#   return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


# def create_test_scheme_3(): # tested Feb 11 1800 UTC to 2321 - way too large of filter order 
#   """
#   Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
#   :return DecimationScheme: a decimation scheme for use in experiment.
#   """

#   rates = [5.0e6, 500.0e3, 50.0e3, 10.0e3]
#   dm_rates = [10, 10, 5, 3]
#   transition_widths = [50.0e3, 5.0e3, 3.0e3, 1.0e3]
#   cutoffs = [460.0e3, 46.0e3, 8.0e3, 2.0e3]
#   num_taps = [512, 512, 512, 256]

#   all_stages = []
#   for stage in range(0,4):
#       filter_taps = list(create_firwin_filter_by_num_taps(rates[stage], transition_widths[stage], cutoffs[stage], num_taps[stage]))
#       all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

#   return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


# def create_test_scheme_4(): # tested Feb 11 2321 UTC to Feb 12 1700 UTC - way too large of filter order 
#   """
#   Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
#   :return DecimationScheme: a decimation scheme for use in experiment.
#   """

#   rates = [5.0e6, 500.0e3, 50.0e3, 10.0e3]
#   dm_rates = [10, 10, 5, 3]
#   transition_widths = [50.0e3, 5.0e3, 3.0e3, 1.0e3]
#   cutoffs = [460.0e3, 46.0e3, 8.0e3, 0.5e3]
#   num_taps = [512, 512, 512, 256]

#   all_stages = []
#   for stage in range(0,4):
#       filter_taps = list(create_firwin_filter_by_num_taps(rates[stage], transition_widths[stage], cutoffs[stage], num_taps[stage]))
#       all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

#   return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


# def create_test_scheme_5():
#   """
#   Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
#   :return DecimationScheme: a decimation scheme for use in experiment.
#   """

#   rates = [5.0e6, 500.0e3, 50.0e3, 10.0e3]
#   dm_rates = [10, 10, 5, 3]
#   transition_widths = [300.0e3, 35.0e3, 7.0e3, 1.0e3]
#   cutoffs = [100.0e3, 5.0e3, 2.0e3, 0.5e3] # bandwidth is double this
#   ripple_dbs = [100.0, 100.0, 60.0, 20.0]

#   all_stages = []

#   for stage in range(0,4):
#       filter_taps = list(create_firwin_filter_by_attenuation(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
#       all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

#   return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


def create_test_scheme_6(): # tested Feb 12 1800 - looks ~ 10 dB SNR? 
    """
    Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
    :return DecimationScheme: a decimation scheme for use in experiment.
    """

    rates = [5.0e6, 500.0e3/3, 50.0e3/3, 10.0e3/3]
    dm_rates = [30, 10, 5, 1]
    transition_widths = [130.0e3, 13.0e3, 2.0e3]
    cutoffs = [20.0e3, 2.0e3, 0.5e3] # bandwidth is double this
    ripple_dbs = [100.0, 30.0, 10.0]

    all_stages = []

    for stage in range(0,3):
        filter_taps = list(create_firwin_filter_by_attenuation(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
        all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

    all_stages.append(DecimationStage(3, rates[3], dm_rates[3], [1.0]))
    return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


def create_test_scheme_7(): 
    """
    Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
    :return DecimationScheme: a decimation scheme for use in experiment.
    """

    rates = [5.0e6, 500.0e3/3, 250.0e3/3, 50.0e3/3]
    dm_rates = [30, 2, 5, 5]
    transition_widths = [130.0e3, 50.0e3, 18.0e3, 1.2e3]
    cutoffs = [20.0e3, 2.0e3, 1.0e3, 1.0e3] # bandwidth is double this
    ripple_dbs = [100.0, 100.0, 40.0, 10.0]

    all_stages = []

    for stage in range(0,4):
        filter_taps = list(create_firwin_filter_by_attenuation(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
        all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

    return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


def create_test_scheme_8(): 
    """
    Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
    :return DecimationScheme: a decimation scheme for use in experiment.
    """

    rates = [5.0e6, 500.0e3, 100.0e3, 50.0e3/3]
    dm_rates = [10, 5, 6, 5]
    transition_widths = [200.0e3, 49.0e3, 15.0e3, 1.0e3]
    cutoffs = [2.0e3, 1.0e3, 1.0e3, 1.0e3] # bandwidth is double this
    ripple_dbs = [150.0, 80.0, 35.0, 9.0]

    all_stages = []

    for stage in range(0,4):
        filter_taps = list(create_firwin_filter_by_attenuation(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
        all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

    return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


def create_test_scheme_9(): 
    """
    Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
    This filter will have a wider receive bandwidth than the previous.
    Pasha recommends a 10kHz bandwidth for the final stage. I believe there will be aliasing caused by this but 
    perhaps the concern is not critical because of the small bandwidth overlapping. I will test this anyways.

    :return DecimationScheme: a decimation scheme for use in experiment.
    """

    rates = [5.0e6, 500.0e3, 100.0e3, 50.0e3/3]
    dm_rates = [10, 5, 6, 5]
    transition_widths = [150.0e3, 40.0e3, 15.0e3, 1.0e3]
    cutoffs = [20.0e3, 10.0e3, 10.0e3, 5.0e3] # bandwidth is double this
    ripple_dbs = [150.0, 80.0, 35.0, 9.0]
    scaling_factors = [10.0, 100.0, 100.0, 100.0]
    all_stages = []

    for stage in range(0,4):
        filter_taps = list(scaling_factors[stage] * create_firwin_filter_by_attenuation(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
        all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

    return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))

