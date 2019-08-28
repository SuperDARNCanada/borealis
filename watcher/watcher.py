#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
# Author: Liam Graham
#
# watcher.py
# 2019-08-21
# Monitoring process to flag any potential problems
# related to rx/tx power


import inotify.adapters
import deepdish as dd
import numpy as np
import smtplib
from email.mime.text import MIMEText
import argparse
import random
import logging, logging.handlers


def antenna_average(power):
	"""
	Averages the measurement power in one antenna iq record
	over each antenna
	Args:
		power (ndarray):	A numpy array of shape (num_antennas,
							num_sequences, num_ranges) containing  # TODO: Not really true? The power passed in is an ndarray of num_antennas, num_sequences?
							the calculated power of the antenna iq
							data.
	Returns:
		avg_db (ndarray): 	A numpy array of shape (num_sequences,
							num_ranges) containing the antenna-averaged  # TODO: Fix this
							power in decibels.
	"""
	avg_power = np.mean(power, axis=0)  # Now averaging across antennas, so this is a 1d array of powers at each range.
	avg_db = 10 * np.log(avg_power)
	return avg_db


def get_lag0_pwr(iq_record):
	"""
	Gets lag0 power for each antenna and sequence in a record of 
	antenna iq data.
	Args:
		iq_record:	 A single 'antenna iq' data record loaded from file.
	Returns:
		power_array: A numpy array of shape (num_antennas, num_sequences, 70)  # TODO: Find out what 70 is for
		containing the lag 0 power for each antenna and sequence in the iq_record
	"""
	# Setup and memory allocation
	dims = iq_record["data_dimensions"]
	voltage_samples = iq_record["data"].reshape(dims)
	num_antennas, num_sequences, num_samps = dims
	power_array = np.zeros((num_antennas, 60))  # TODO: Find out what this magic number is for
	# get power from voltage samples  # TODO: Check units and calculation are correct
	power = np.mean(np.sqrt(voltage_samples.real ** 2 + voltage_samples.imag ** 2), axis=1)  # Axis 0 is antennas, axis 1 is sequences, axis 2 is samples so this is a power for each antenna averaged over sequences
	ant_avg = antenna_average(power)[5:65]  # TODO: Find out what magic numbers are for
	for antenna in range(num_antennas):
		ant_power = power[antenna, 5:65]  # TODO: Find out what magic numbers are for
		power_db = 10 * np.log(ant_power)
		power_array[antenna] = power_db

	return power_array, ant_avg


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
	num_antennas, num_ranges = power_array.shape
	the_truth = np.zeros((num_antennas, num_antennas, num_ranges), dtype=bool)
	for ant in range(num_antennas):
		# start at ant to avoid duplicate results from array symmetry
		for comp in range(ant + 1, num_antennas):
			for rng in range(num_ranges):
				if np.abs(power_array[ant, rng] - power_array[comp, rng]) > threshold:
					the_truth[ant, comp, rng] = True

	return the_truth


def flag_antennas(truth_array, proportion):
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
	num_antennas, _, num_ranges = truth_array.shape
	antennas = np.zeros((num_antennas), dtype=int)
	locs = np.transpose(np.nonzero(truth_array))
	for loc in locs:
		antennas[loc[0]] += 1
		antennas[loc[1]] += 1

	removed = []
	original_index = np.array(range(num_antennas))
	threshold = proportion * (num_antennas - 1) * num_ranges
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
		threshold = proportion * (num_antennas - 1) * num_ranges
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
		power_array:	An array of antenna powers in dB.
		avg_power:		An array containing the antenna averaged
						power for each measurement.

	Returns:
		power_diffs:	A list of tuples containing the antenna number and 
						sequence and range averaged difference between
						the antenna powers and the average power.
		range_diffs:	A list of tuples containing the antenna number and 
						sequence averaged difference between the antenna
						powers and the average power.
	"""
	antennas = sorted(antennas)
	power_diffs = list()
	range_diffs = list()
	for antenna in antennas:
		antenna_power = power_array[antenna]
		# average difference over entire array
		power_diff = np.mean(antenna_power - avg_power)
		# average difference over sequences at each range set
		range_diff = antenna_power - avg_power
		power_diffs.append(power_diff)
		range_diffs.append(range_diff)

	return power_diffs, range_diffs


def reporter(flagged, total_power_diff, range_power_diff):
	"""
	Reports on the results of the antenna analysis
	Args:
		flagged:			A list containing the flagged antenna numbers
		total_power_diff:	A list containing the sequence and range averaged
							difference between the antenna powers and the average power.
		range_power_diff:	A list containing the sequence averaged difference between
							the antenna powers and the average power.
	Returns:
		report:				Report on flagged antennas including a total averaged difference
							and averaged differences at each range gate.
	"""
	antennas = [antenna for antenna in flagged]
	if len(antennas) == 0:
		return
	else:
		report = dict()
		for idx, antenna in enumerate(antennas):
			report[antenna] = dict()
			report[antenna]["total_differences"] = total_power_diff[idx]
			report[antenna]["range_differences"] = range_power_diff[idx].tolist()
		return report


def send_report(report, addresses=None):
	"""
	Emails the power report to the addresses in an email file
	Args:
		report:		A report created by reporter()
		addresses:	The email addresses to send the report to, list of strings
	"""
	smtp_server = "localhost"
	email_sender = "antenna_watcher"  # TODO: Better name
	if addresses is None:
		addresses = ['kevin.krieger@usask.ca']
	email_recipients = addresses
	email_subject = "Antenna power report"  # TODO: Get site information and date
	email_message = ""
	antennas = list(report.keys())
	for antenna in antennas:
		email_message += "Antenna: " + str(antenna) + "\n"
		email_message += "Total average difference from array average: " + str(
			report[antenna]["total_differences"]) + "\n"
		email_message += "Differences from array average by range: " + str(report[antenna]["range_differences"]) + "\n"
		email_message += "\n"

	email = MIMEText(email_message)
	email['Subject'] = email_subject
	email['From'] = email_sender
	email['To'] = ', '.join(email_recipients)

	with smtplib.SMTP(smtp_server) as server:
		server.sendmail(email_sender, email_recipients, email.as_string())


def check_antennas_iq_file_power(iq_file, threshold, proportion):
	"""
	Checks that the power between antennas is reasonably close for each 
	range in a record. If it is not, add to the report.
	Args:
		iq_file:		The path to the antenna iq file being checked.
		threshold:		The acceptable difference in power between antennas, in dB
		proportion:		Percentage of antennas that any antenna may mismatch with before being include in the report.
	"""
	ant_iq = dd.io.load(iq_file)
	antenna_keys = list(ant_iq.keys())
	key = random.choice(antenna_keys)
	record = ant_iq[key]
	pwr, avg = get_lag0_pwr(record)
	truth = build_truth(pwr, threshold)
	flagged = flag_antennas(truth, proportion)
	power_diffs, range_diffs = compare_with_average(flagged, pwr, avg)
	report = reporter(flagged, power_diffs, range_diffs)
	return report


def _main():
	proportion_default = 0.2
	parser = argparse.ArgumentParser(description="Automatically generate a power report on new antenna iq files")
	parser.add_argument('-t', '--threshold', type=int, default=1,
						help='An acceptable power difference between antennae in dB')
	parser.add_argument('-p', '--proportion', type=float, default=proportion_default,
						help='The acceptable percentage of antennas any antenna may differ from (float 0.0-1.0)')
	parser.add_argument('-x', '--times', type=int, default=1,
						help='The number of times in a row an antenna can be flagged before a report is sent')
	parser.add_argument('-v', '--verbose', action='store_true',
						help='Script will output more verbose messages')
	parser.add_argument('-f', '--logfile', default=None, help='A path to a logfile to output log messages to')
	parser.add_argument('directory', help='The directory this script should be watching for new antenna iq files')
	args = parser.parse_args()
	threshold = args.threshold
	proportion = args.proportion
	times = args.times
	logfile = args.logfile
	directory = args.directory

	# Set up logger and handlers for a logfile, stream and email
	logger = logging.getLogger(__name__)
	if logfile is not None:
		logger.addHandler(logging.FileHandler(logfile))
	else:
		logger.addHandler(logging.StreamHandler())  # Default stream is sys.stderr
	email_handler = logging.handlers.SMTPHandler(mailhost='localhost',
										fromaddr='antenna_watcher',
										toaddrs=['kevin.krieger@usask.ca'],
										subject='Antenna Power Report',
										credentials=None,
										secure=None)
	email_handler.setLevel(logging.WARN)
	logger.addHandler(email_handler)
	if args.verbose:
		logger.setLevel(logging.DEBUG)
	else:
		logger.setLevel(logging.INFO)

	if proportion < 0.0 or proportion > 1.0:
		logger.info("Proportion: {} doesn't make sense. Pick a float value between 0.0 and 1.0. "
					"Setting to default: {}".format(proportion, proportion_default))

	# Watch input directory and all subdirectories for file events
	i = inotify.adapters.InotifyTree(directory)

	# dictionary to hold antenna numbers that were a problem last time
	antenna_history = dict()
	report = None
	last_file = None
	logger.info("Looping...")

	while True:
		for event in i.event_gen(yield_nones=False):
			(_, type_names, path, filename) = event
			logger.debug(event)
			# When an antennas_iq file is closed and written, or moved to the directory, begin checking logic
			if (("antennas_iq" in filename) or ("output_ptrs_iq" in filename)) and \
				(("IN_CLOSE_WRITE" in type_names) or ("IN_MOVED_TO" in type_names)):

				# Make sure that there are previously created files to be checked, otherwise wait for the next one.
				if last_file is None:
					logger.debug("First file event since script started: {}".format(filename))
					last_file = filename
				else:
					logger.debug("File event for {} so script will analyze {}".format(filename, last_file))
					logger.info("Opening file {}".format(last_file))
					file_path = path + '/' + last_file

					# Generate antenna report on random record in the file
					try:
						report = check_antennas_iq_file_power(file_path, threshold, proportion)
					except OSError as e:
						logger.error("Failed to generate report on {}, due to exception {}".format(file_path, e))

					if report is None:
						logger.info("Nothing to report for {}".format(file_path))
					else:
						# Remove antennas from antenna_history if they were not flagged again
						# TODO: See if you can simplify this antenna history stuff with list comprehensions, etc.
						history_remove = list()
						for antenna in antenna_history:
							if antenna not in report:
								history_remove.append(antenna)
						for antenna in history_remove:
							del antenna_history[antenna]

						report_remove = list()
						# Add reported antennas to history
						for antenna in report:
							if antenna in antenna_history:
								antenna_history[antenna] += 1
							else:
								antenna_history[antenna] = 1
							# Remove antenna from the report if it has not been flagged enough
							if antenna_history[antenna] < times:
								report_remove.append(antenna)
						for antenna in report_remove:
							del report[antenna]

						# Finally, send the report if any antennas remain
						# This means they've been flagged sufficiently often to be reported
						if len(report) > 0:
							send_report(report)
					last_file = filename


if __name__ == "__main__":
	_main()
