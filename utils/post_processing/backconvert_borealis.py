# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Liam Graham

#
# backconvert_borealis.py
# 2019-07-30
# Command line tool for converting restructured and compressed
# Borealis files back to their initial site format

import deepdish as dd
import numpy as np
import os
import sys
import subprocess as sp

def backconvert(data_path):
	"""
	Converts a restructured and compressed hdf5 borealis datafile
	back to its original, record based format.
	Args:
		data_path (str): Path to the data file to be back converted
	"""

	def backconvert_prebfiq(data_record):
		"""
		Converts a restructured antennas iq file back to its
		original site format
		Args:
			data_record (dict): An opened antennas_iq hdf5 file
		"""
		temp = 'temp.hdf5'
		num_records = len(data_record["int_time"])
		keys = np.zeros(num_records)
		ts_dict = dict()
		# get keys from first sequence timestamps
		for rec, seq_ts in enumerate(data_record["sqn_timestamps"]):
			keys[rec] = int(seq_ts[0] * 1000)
		for rec, k in enumerate(keys):
			for f in data_record:
				if type(data_record[f]) is np.ndarray:
					if not (np.shape(data_record[f])[0] == num_records):
						ts_dict[f] = data_record[f]
					else:
						pass
						# FIGURE THIS BIT OUT
				else:
					ts_dict[f] = data_record[f]


	def backconvert_bfiq(data_record):
		"""
		Converts a restructured bfiq file back to its
		original site format
		Args:
			data_record (dict): An opened bfiq hdf5 file
		"""

	def backconvert_rawacf(data_record):
		"""
		Converts a restructured raw acf file back to its
		original site format
		Args:
			data_record (dict): An opened rawacf hdf5 file
		"""

	suffix = data_path.split('.')[-2]

	print("Restructuring", data_path, "...")

	data = dd.io.load(data_path)
	converted = data_path + '.site'

	# touch output file
	try:
		fd = os.open(converted, os.O_CREAT | os.O_EXCL)
		os.close(fd)
	except FileExistsError:
		pass

	if (suffix == 'output_ptrs_iq') or (suffix == 'antennas_iq'):
		print("Loaded a pre bfiq file...")
		backconvert_pre_bfiq(data)
	elif suffix == 'bfiq':
		print("Loaded a bfiq file...")
		backconvert_bfiq(data)
	elif suffix == 'rawacf':
		print("Loaded a raw acf file")
		backconvert_rawacf(data)
	else:
		print(suffix, 'filetypes are not supported')
		return

	print("Success!")
