#!/usr/bin/python

import os
import sys
from scipy.signal import firwin, remez, kaiserord

sys.path.append(os.environ['BOREALISPATH'])

from experiment_prototype.decimation_scheme.decimation_scheme import DecimationStage, DecimationScheme


def create_test_scheme_1():
	"""
	Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. Filter lengths are: 1284, 1284, 215, 130. 
	:return DecimationScheme: a decimation scheme for use in experiment.
	"""

	rates = [5.0e6, 500.0e3, 50.0e3, 10.0e3]
	dm_rates = [10, 10, 5, 3]
	transition_widths = [50.0e3, 5.0e3, 3.0e3, 1.0e3]
	cutoffs = [460.0e3, 46.0e3, 8.0e3, 2.0e3]
	ripple_dbs = [100.0, 100.0, 100.0, 100.0]

	all_stages = []
	for stage in range(0,4):
		filter_taps = list(create_firwin_filter(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
		all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

	return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


def create_test_scheme_2():
	"""
	Create four stages of FIR filters and a decimation scheme. Returns a decimation scheme of type DecimationScheme. 
	:return DecimationScheme: a decimation scheme for use in experiment.
	"""

	rates = [5.0e6, 500.0e3, 50.0e3, 10.0e3]
	dm_rates = [10, 10, 5, 3]
	transition_widths = [50.0e3, 5.0e3, 3.0e3, 1.0e3]
	cutoffs = [460.0e3, 46.0e3, 8.0e3, 2.0e3]
	ripple_dbs = [80.0, 80.0, 100.0, 100.0]

	all_stages = []
	for stage in range(0,4):
		filter_taps = list(create_firwin_filter(rates[stage], transition_widths[stage], cutoffs[stage], ripple_dbs[stage]))
		all_stages.append(DecimationStage(stage, rates[stage], dm_rates[stage], filter_taps))

	return (DecimationScheme(5.0e6, 10.0e3/3, stages=all_stages))


def create_firwin_filter(sample_rate, transition_width, cutoff_hz, ripple_db, 
	window_type='kaiser'):
	"""
	Create a firwin filter. 

	:param ripple_db: The desired attenuation in the stop band, in dB.
	"""

	# The Nyquist rate of the signal.
	nyq_rate = sample_rate  # because we have complex sampled data. 

	# The desired width of the transition from pass to stop,
	# relative to the Nyquist rate. '
	width_ratio = transition_width/nyq_rate

	# Compute the order and Kaiser parameter for the FIR filter.
	N, beta = kaiserord(ripple_db, width_ratio)
	print(N)

	# Use firwin with a Kaiser window to create a lowpass FIR filter
	if window_type == 'kaiser':
		window = ('kaiser', beta)
	else:
		window = window_type

	taps = firwin(N, cutoff_hz/nyq_rate, window=window)

	return taps