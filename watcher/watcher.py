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
import sys


def antenna_average(power):
	avg_power = np.mean(power, axis=0)
	avg_db = 10 * np.log(avg_power)
	return avg_db


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
	power_array = np.zeros((num_antennas, num_sequences, 6))
	avg_array = np.zeros((num_sequences, 6))
	# get power from voltage samples
	power = np.sqrt(voltage_samples.real**2 + voltage_samples.imag**2)
	ant_avg = antenna_average(power)
	for antenna in range(num_antennas):
		for sequence in range(num_sequences):
			ant_power = power[antenna, sequence, 1:71]
			power_db = 10 * np.log(ant_power)
			for i in range(6):
				pwr = np.amax(power_db[(i+1)*10:(i+1)*10+9])
				avg = np.amax(ant_avg[sequence, (i+1)*10:(i+1)*10+9])
				# set appropriate section of power_array
				power_array[antenna, sequence, i] = pwr
				avg_array[sequence, i] = avg

	return power_array, avg_array


def build_truth_average(power_array, average_array, threshold):
	"""
	Compares each data point (seq, range) for each antenna against every 
	other antenna and builds an array of booleans of shape
	(num_antennas, num_antennas, num_sequences, num_ranges)
	Each point is the inverse truth value of whether that point is within a difference
	threshold for each antenna. For example, if point (0, 5, 6, 49) is False: 
	then the power level difference between antennas 0 and 5 for sequence 6
	range 50 is within the threshold. Inverse used because it is easier to find
	nonzero elements in numpy.
	Args:
		power_array:	An np.ndarray of powers in dB created by get_lag0_pwr.
		threshold:		The acceptable difference in power between antennas.
	Returns:
		the_truth:		Array of booleans as described above.
	"""
	num_antennas, num_sequences, num_ranges = power_array.shape

	the_truth = np.zeros((num_antennas, num_sequences, num_ranges), dtype=bool)
	for ant in range(num_antennas):
		for seq in range(num_sequences):
			for rng in range(num_ranges):
				if np.abs(power_array[ant, seq, rng] - average_array[seq, rng]) > threshold:
					the_truth[ant, seq, rng] = True


	return the_truth

def build_truth(power_array, threshold):
	"""
	Compares each data point (seq, range) for each antenna against every 
	other antenna and builds an array of booleans of shape
	(num_antennas, num_antennas, num_sequences, num_ranges)
	Each point is the inverse truth value of whether that point is within a difference
	threshold for each antenna. For example, if point (0, 5, 6, 49) is False: 
	then the power level difference between antennas 0 and 5 for sequence 6
	range 50 is within the threshold. Inverse used because it is easier to find
	nonzero elements in numpy.
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
		# start at ant to avoid duplicate results from array symmetry
		for comp in range(ant+1, num_antennas):
			for seq in range(num_sequences):
				for rng in range(num_ranges):
					if np.abs(power_array[ant, seq, rng] - power_array[comp, seq, rng]) > threshold:
						the_truth[ant, comp, seq, rng] = True

	return the_truth


def average_reporter(truth_array):
	"""
	Reports whether there are power discrepancies between antennas based on
	a truth array created by build_truth().
	Args:
		truth_array:	A truth array from build_truth(). Sould have shape:
						(num_antennas, num_antennas, num_sequences, num_ranges)
	"""
	# Get the indices of each False in the array
	num_antennas, num_sequences, num_ranges = truth_array.shape
	ctrs = np.zeros((num_antennas, num_ranges), dtype=int)
	if np.any(truth_array):
		locs = np.transpose(np.nonzero(truth_array))
		for loc in locs:
			ctrs[loc[0], loc[2]] += 1

	total = np.sum(ctrs)
	print(total, "power discrepancies found\n")

	for ant in range(num_antennas):
		for rng in range(num_ranges):
			count = ctrs[ant,rng]
			print(count, "power discrepancies found for antenna", ant, "at range", rng)


def reporter(truth_array):
	"""
	Reports whether there are power discrepancies between antennas based on
	a truth array created by build_truth().
	Args:
		truth_array:	A truth array from build_truth(). Sould have shape:
						(num_antennas, num_antennas, num_sequences, num_ranges)
	"""
	# Get the indices of each False in the array
	num_antennas = truth_array.shape[0]
	ctrs = np.zeros((num_antennas, num_antennas), dtype=int)
	totals = np.zeros((num_antennas), dtype=int)
	if np.any(truth_array):
		locs = np.transpose(np.nonzero(truth_array))
		for loc in locs:
			ctrs[loc[0], loc[1]] += 1
			totals[loc[0]] += 1
			totals[loc[1]] += 1

	total = np.sum(ctrs)
	print(total, "power discrepancies found\n")

	for ant1 in range(num_antennas):
		for ant2 in range(ant1+1, num_antennas):
			count = ctrs[ant1, ant2]
			print(count, "power discrepancies between antenna", ant1, "and antenna", ant2)

	for ant in range(num_antennas):
		print("Antenna", ant, "involved in", totals[ant], "discrepancies")



def check_antennas_iq_file_power(iq_file, threshold):
	"""
	Checks that the power between antennas is reasonably close for each 
	range in a record. If it is not, alert the squad.
	Args:
		iq_file:		The path to the antenna iq file being checked.
		threshold:		The acceptable difference in power between antennas.
	"""

	ant_iq = dd.io.load(iq_file)
	print("Loaded iq")
	antenna_keys = list(ant_iq.keys())
	first_rec = ant_iq[antenna_keys[0]]
	last_rec = ant_iq[antenna_keys[-1]]

	first_pwr, first_avg = get_lag0_pwr(first_rec)
	last_pwr, last_avg = get_lag0_pwr(last_rec)

	# first_truth = build_truth_average(first_pwr, first_avg, threshold)
	# last_truth = build_truth_average(last_pwr, last_avg, threshold)

	first_truth = build_truth(first_pwr, threshold)
	last_truth = build_truth(last_pwr, threshold)

	print("Checking", antenna_keys[0], "\n")
	reporter(first_truth)
	print("Checking", antenna_keys[-1], "\n")
	reporter(last_truth)


if __name__ == "__main__":
	file = sys.argv[1]
	threshold = int(sys.argv[2])
	check_antennas_iq_file_power(file, threshold)
