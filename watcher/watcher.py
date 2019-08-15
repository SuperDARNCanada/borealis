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
		iq_record: A single antenna iq data record loaded from file.
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


def check_antennas_iq_file_power(iq_file):
	"""
	Checks that the power between antennas is reasonably close for each 
	range in a record. If it is not, alert the squad.
	Args:
		iq_file:	The path to the antenna iq file being checked.
	"""

	ant_iq = dd.io.load(iq_file)
	antenna_keys = list(ant_iq.keys())
	first_rec = ant_iq[antenna_keys[0]]
	last_rec = ant_iq[antenna_keys[-1]]

	first_pwr = get_lag0_pwr(first_rec)
	last_pwr = get_lag0_pwr(last_rec)
