#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
# Author: Liam Graham
#
# watcher.py
# 2019-08-20
# Monitoring process to flag any potential problems
# related to rx/tx power


import inotify.adapters
import deepdish as dd
import numpy as np
import sys
import smtplib
import ssl
import argparse


def antenna_average(power):
	"""
	Averages the measurement power in one antenna iq record
	over each antenna
	Args:
		power (ndarray):	A numpy array of shape (num_antennas,
							num_sequences, num_ranges) containing
							the calculated power of the antenna iq
							data.
	Returns:
		avg_db (ndarray): 	A numpy array of shape (num_sequences,
							num_ranges) containing the antenna-averaged
							power in decibels.
	"""
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
				# Get maximum power and maximum average power at each
				# set of ranges
				pwr = np.amax(power_db[(i+1)*10:(i+1)*10+9])
				avg = np.amax(ant_avg[sequence, (i+1)*10:(i+1)*10+9])
				# set appropriate section of power_array and avg_array
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


def flag_antennas(truth_array, proportion=0.5):
	"""
	Finds antennas that may be acting up based on an antenna-antenna comparison
	stored in a truth_array created by build_truth. Checks whether an antenna
	is involved in more power discrepancies than set by the product of num_sequences,
	num_ranges (range sets), num_antennas (remaining) and a user set proportion. If any
	antennas are involved in more discrepancies than this product, that antenna is removed
	from consideration and the check is performed on the remaining antennas.
	Args:
		truth_array (ndarray):	A numpy array of boolean values representing whether or not
								a pair of antennas have matching powers on a specific
								data point, within a threshold.
		proportion (float):		A number describing the proportion of remaining towers with
								which each tower must agree with to not be removed from the
								problem. Default in 0.5.
	"""
	# Set up the problem
	num_antennas, _, num_sequences, num_ranges = truth_array.shape
	antennas = np.zeros((num_antennas), dtype=int)
	locs = np.transpose(np.nonzero(truth_array))
	for loc in locs:
		antennas[loc[0]] += 1
		antennas[loc[1]] += 1

	removed = []
	original_index = np.array(range(num_antennas))
	threshold = proportion * (num_antennas - 1) * num_sequences * num_ranges
	unacceptable = antennas > threshold

	while np.any(unacceptable):
		# Get the index of towers outside the threshold
		indices = np.nonzero(unacceptable)[0]
		truth_array = np.delete(truth_array, indices, 0)
		truth_array = np.delete(truth_array, indices, 1)
		# Should append the original index of the antenna for further analysis
		for idx in indices:
			removed.append(original_index[idx])
		# Keep track of indices that have not yet been removed
		original_index = np.delete(original_index, indices)

		# Update problem
		num_antennas = truth_array.shape[0]
		threshold = proportion * (num_antennas - 1) * num_sequences * num_ranges
		antennas = np.zeros((num_antennas), dtype=int)
		locs = np.transpose(np.nonzero(truth_array))
		for loc in locs:
			antennas[loc[0]] += 1
			antennas[loc[1]] += 1
		unacceptable = antennas > threshold

	return removed


def compare_with_average(antennas, power_array, avg_power):
	"""
	Compares the powers of specific antennas with the antenna-averaged
	power for the whole array. Averages the difference in powers over
	sequences and ranges to give a rough idea of how the antenna is
	performing.
	Args:
		antennas:		A list of antennas to be checked
		power_array:	An array of antenna powers in decibels.
		avg_power:		An array containing the antenna averaged
						power for each measurement.

	Returns:
		power_diffs:	A list of tuples containing the antenna number and 
						sequence and range averaged difference between
						the antenna powers and the average power.
		range_diffs:	A list of tuples containing the antenna number and 
						sequence averaged difference betweenthe antenna
						powers and the average power.
	"""
	antennas = sorted(antennas)
	power_diffs = list()
	range_diffs = list()
	for antenna in antennas:
		print("Analyzing antenna", antenna)
		antenna_power = power_array[antenna]
		# average difference over entire array
		power_diff = np.mean(antenna_power - avg_power)
		# average difference over sequences at each range set
		range_diff = np.mean(antenna_power - avg_power, axis=0)
		power_diffs.append(power_diff)
		range_diffs.append(range_diff)

	return power_diffs, range_diffs


def reporter(flagged, total_power_diff, range_power_diff, history):
	"""
	Reports on the results of the antenna analysis
	Args:
		antennas:			A list containing the flagged antenna numbers
		total_power_diff:	A list containing the sequence and range averaged
							difference between the antenna powers and the average power.
		range_power_diff:	A list containing the sequence averaged difference between
							the antenna powers and the average power.
	"""
	antennas = list()
	for antenna in flagged:
		if antenna not in history:
			antennas.append((antenna, False))
	for antenna in history:
		if antenna not in flagged:
			antennas.append((antenna, True))

	if len(antennas) == 0:
		return
	else:
		report = dict()
		for idx, (antenna, change) in enumerate(antennas):
			report[antenna] = dict()
			if change:
				report[antenna]["state_change"] = "Recovered"
			else:
				report[antenna]["state_change"] = "Failed"
				report[antenna]["total_differences"] = total_power_diff[idx]
				report[antenna]["range_differences"] = range_power_diff[idx].tolist()
		return report


def send_report(report, address):
	"""
	Emails the power report to the addresses in an email file
	Args:
		report:		A report created by reporter()
		address:	The email address to send the report to
	"""
	port = 587
	smtp_server = "smtp.gmail.com"
	sender = "watcherdevel@gmail.com"
	receiver = address
	password = input("Password for watcher email account:")
	message = "Your array report is ready.\n\n"
	antennas = list(report.keys())
	for antenna in antennas:
		message += "Antenna: " + str(antenna) + report[antenna]["state_change"] + "\n"

		if report[antenna]["state_change"] == "Failed":
			message += "Total average difference from array average: " \
						 + str(report[antenna]["total_differences"]) + "\n"
			message += "Differences from array average by range: " \
						 + str(report[antenna]["range_differences"]) + "\n"
		message += "\n"

	context = ssl.create_default_context()

	with smtplib.SMTP(smtp_server, port) as server:
		server.ehlo()
		server.starttls(context=context)
		server.ehlo()
		server.login(sender, password)
		server.sendmail(sender, receiver, message)
	

def check_antennas_iq_file_power(iq_file, threshold, proportion, history):
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

	first_truth = build_truth(first_pwr, threshold)
	last_truth = build_truth(last_pwr, threshold)

	first_flagged = flag_antennas(first_truth, proportion)
	last_flagged = flag_antennas(last_truth, proportion)

	first_power_diffs, first_range_diffs = compare_with_average(first_flagged, first_pwr, first_avg)
	last_power_diffs, last_range_diffs = compare_with_average(last_flagged, last_pwr, last_avg)

	first_report = reporter(first_flagged, first_power_diffs, first_range_diffs, history)
	history = first_flagged
	last_report = reporter(last_flagged, last_power_diffs, last_range_diffs, history)
	history = last_flagged

	reports = [first_report, last_report]
	for report in reports:
		if report is None:
			pass
			print("Nothing to report")
		else:
			send_report(report, "liam.adair.graham@gmail.com")

	return history


def _main():
	parser = argparse.ArgumentParser(description="Automatically generate a report on new antenna iq files")
	parser.add_argument('--threshold', required=True, help='An acceptable decibel difference between antennae')
	parser.add_argument('--proportion', required=True, help='The acceptable proportion of antennas any antenna may\
																differ from')
	args = parser.parse_args()
	threshold = args.threshold
	proportion = args.proportion

	i = inotify.adapters.Inotify()

	i.add_watch('/data')

	# list to hold antenna numbers that were a problem last time
	antenna_history = list()

	while True:
		for event in i.event_gen(yield_nones=False):
			(_, type_names, path, filename) = event

			if (("antennas_iq" in filename) or ("output_ptrs_iq" in filename)) and ("IN_CLOSE_WRITE" in type_names):
				print("Opening...")
				file_path = path + '/' + filename
				thresh = int(threshold)
				prop = float(proportion)
				antenna_history = check_antennas_iq_file_power(file_path, thresh, prop, antenna_history)



if __name__ == "__main__":
	_main()
