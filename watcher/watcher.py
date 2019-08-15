#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
# Author: Liam Graham
#
# watcher.py
# 2019-08-15
# Monitoring process to flag any problems related to rx/tx power


import inotify as i
import deepdish as dd
import numpy as np


def get_lag0_pwr(iq_record):
	"""
	Gets lag0 power for each antenna and sequence in a record of 
	antenna iq data.
	Args:
		iq_record:	 A single antenna iq data record loaded from file.
	Returns:
		power_array: A numpy array of shape (num_antennas, num_sequences, 70)
					 containing the lag 0 power for each antenna and sequence
					 in the iq_record
	"""
	# Setup and memory allocation
	dims = iq_record["data_dimensions"]
	voltage_samples = iq_record["data"].reshape(dims)
	num_antennas, num_sequences, num_samps = dims
	power_array = np.zeros((num_antennas, num_sequences, 70))
	# get power from voltage samples
	for antenna in range(num_antennas):
		for sequence in range(num_sequences):
			power = (voltage_samples.real**2 + voltage_samples.imag**2)[antenna, sequence, 0:69]
			power_db = 10 * np.log(power)
			# set appropriate section of power_array
			power_array[antenna, sequence, :] = power_db

	return power_array


def build_truth(power_array, threshold):
	"""
	Compares each data point (seq, range) for each antenna against every 
	other antenna and builds an array of booleans of shape
	(num_antennas, num_antennas, num_sequences, num_ranges)
	Each point is the truth value of whether that point is within a difference
	threshold for each antenna. For example, if point (0, 5, 6, 49) is True: 
	then the power level difference between antennas 0 and 5 for sequence 6
	range 50 is within the threshold.
	Args:
		power_array:	An np.ndarray of powers in dB created by get_lag0_pwr.
		threshold:		The acceptable difference in power between antennas.
	Returns:
		the_truth:		Array of booleans as described above.
	"""
	num_antennas, num_sequences, num_ranges = power_array.shape
	the_truth = np.zeros((num_antennas, num_antennas, num_sequences, num_ranges),
							dtype=bool)
	for ant in range(num_antennas):
		for comp in range(num_antennas):
			for seq in range(num_sequences):
				for rng in range(num_ranges):
					if (power_array[ant, seq, rng] - power_array[comp, seq, rng]) < threshold:
						the_truth[ant, comp, seq, rng] = True
					else:
						the_truth[ant, comp, seq, rng] = False

	return the_truth


def check_antennas_iq_file_power(iq_file, threshold):
	"""
	Checks that the power between antennas is reasonably close for each 
	range in a record. If it is not, alert the squad.
	Args:
		iq_file:		The path to the antenna iq file being checked.
		threshold:		The acceptable difference in power between antennas.
	"""

	ant_iq = dd.io.load(iq_file)
	antenna_keys = list(ant_iq.keys())
	first_rec = ant_iq[antenna_keys[0]]
	last_rec = ant_iq[antenna_keys[-1]]

	first_pwr = get_lag0_pwr(first_rec)
	last_pwr = get_lag0_pwr(last_rec)

	first_truth = build_truth(first_pwr, threshold)
	last_truth = build_truth(last_pwr, threshold)
