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
import datetime
import warnings


def backconvert_pre_bfiq(data_record):
	"""
	Converts a restructured antennas iq file back to its
	original site format
	Args:
		data_record (dict): An opened antennas_iq hdf5 file
	"""
	num_records = len(data_record["int_time"])
	ts_dict = dict()
	# get keys from first sequence timestamps
	for rec, seq_ts in enumerate(data_record["sqn_timestamps"]):
		# format dictionary key in the same way it is done
		# in datawrite on site
		sqn_dt_ts = datetime.datetime.utcfromtimestamp(seq_ts[0])
		epoch = datetime.datetime.utcfromtimestamp(0)
		key = str(int((sqn_dt_ts - epoch).total_seconds() * 1000))
		
		ts_dict[key] = dict()
		for f in data_record:
			if not type(data_record[f]) is np.ndarray:
				ts_dict[key][f] = data_record[f]
			else:
				if np.shape(data_record[f])[0] == num_records:
					# pass data fields that are written per record
					pass
				else:
					ts_dict[key][f] = data_record[f]
		# Handle per record fields
		num_sequences = data_record["num_sequences"][rec]
		ts_dict[key]["num_sequences"] = num_sequences
		ts_dict[key]["int_time"] = data_record["int_time"][rec]
		ts_dict[key]["sqn_timestamps"] = data_record["sqn_timestamps"][rec, 0:int(num_sequences)]
		ts_dict[key]["noise_at_freq"] = data_record["noise_at_freq"][rec, 0:int(num_sequences)]
		ts_dict[key]["data_descriptors"] = ts_dict[key]["data_descriptors"][1:]
		ts_dict[key]["data_dimensions"] = data_record["data_dimensions"][rec]

		ts_dict[key]["data"] = np.trim_zeros(data_record["data"][rec].flatten())
	
	return ts_dict

def backconvert_bfiq(data_record):
	"""
	Converts a restructured bfiq file back to its
	original site format
	Args:
		data_record (dict): An opened bfiq hdf5 file
	"""
	num_records = len(data_record["int_time"])
	ts_dict = dict()
	# get keys from first sequence timestamps
	for rec, seq_ts in enumerate(data_record["sqn_timestamps"]):
		# format dictionary key in the same way it is done
		# in datawrite on site
		sqn_dt_ts = datetime.datetime.utcfromtimestamp(seq_ts[0])
		epoch = datetime.datetime.utcfromtimestamp(0)
		key = str(int((sqn_dt_ts - epoch).total_seconds() * 1000))
		
		ts_dict[key] = dict()
		for f in data_record:
			if not type(data_record[f]) is np.ndarray:
				ts_dict[key][f] = data_record[f]
			else:
				if np.shape(data_record[f])[0] == num_records:
					# pass data fields that are written per record
					pass
				else:
					ts_dict[key][f] = data_record[f]
		# Handle per record fields
		num_sequences = data_record["num_sequences"][rec]
		ts_dict[key]["num_sequences"] = num_sequences
		ts_dict[key]["int_time"] = data_record["int_time"][rec]
		ts_dict[key]["sqn_timestamps"] = data_record["sqn_timestamps"][rec, 0:int(num_sequences)]
		ts_dict[key]["noise_at_freq"] = data_record["noise_at_freq"][rec, 0:int(num_sequences)]
		ts_dict[key]["data_descriptors"] = ts_dict[key]["data_descriptors"][1:]
		ts_dict[key]["data_dimensions"] = data_record["data_dimensions"][rec]

		ts_dict[key]["data"] = np.trim_zeros(data_record["data"][rec].flatten())
	
	return ts_dict

def backconvert_rawacf(data_record):
	"""
	Converts a restructured raw acf file back to its
	original site format
	Args:
		data_record (dict): An opened rawacf hdf5 file
	"""
	num_records = len(data_record["int_time"])
	ts_dict = dict()
	# get keys from first sequence timestamps
	for rec, seq_ts in enumerate(data_record["sqn_timestamps"]):
		# format dictionary key in the same way it is done
		# in datawrite on site
		sqn_dt_ts = datetime.datetime.utcfromtimestamp(seq_ts[0])
		epoch = datetime.datetime.utcfromtimestamp(0)
		key = str(int((sqn_dt_ts - epoch).total_seconds() * 1000))
		
		ts_dict[key] = dict()
		for f in data_record:
			if not type(data_record[f]) is np.ndarray:
				ts_dict[key][f] = data_record[f]
			else:
				if np.shape(data_record[f])[0] == num_records:
					# pass data fields that are written per record
					pass
				else:
					ts_dict[key][f] = data_record[f]
		# Handle per record fields
		num_sequences = data_record["num_sequences"][rec]
		ts_dict[key]["num_sequences"] = num_sequences
		ts_dict[key]["int_time"] = data_record["int_time"][rec]
		ts_dict[key]["sqn_timestamps"] = data_record["sqn_timestamps"][rec, 0:int(num_sequences)]
		ts_dict[key]["noise_at_freq"] = data_record["noise_at_freq"][rec, 0:int(num_sequences)]
		ts_dict[key]["correlation_descriptors"] = ts_dict[key]["correlation_descriptors"][1:]

		ts_dict[key]["main_acfs"] = data_record["main_acfs"][rec].flatten()
		ts_dict[key]["intf_acfs"] = data_record["intf_acfs"][rec].flatten()
		ts_dict[key]["xcfs"] = data_record["xcfs"][rec].flatten()
		
	return ts_dict


def write_backconverted(ts_dict, data_path):
	temp_file = 'temp.hdf5'
	site_format_file = data_path + '.site'
	for key in ts_dict:
		time_stamped_dd = {}
		time_stamped_dd[key] = ts_dict[key]
		# touch output file
		try:
			fd = os.open(site_format_file, os.O_CREAT | os.O_EXCL)
			os.close(fd)
		except FileExistsError:
			pass

		dd.io.save(temp_file, time_stamped_dd, compression=None)
		cmd = 'h5copy -i {newfile} -o {fullfile} -s {dtstr} -d {dtstr}'
		cmd = cmd.format(newfile=temp_file, fullfile=site_format_file, dtstr=key)

		sp.call(cmd.split())
		os.remove(temp_file)
		print("Done", key)



def backconvert_data(data_path):
	"""
	Converts a restructured and compressed hdf5 borealis datafile
	back to its original, record based format.
	Args:
		data_path (str): Path to the data file to be back converted
	"""

	path_strings = data_path.split('.')

	print("Restructuring", data_path, "...")

	data = dd.io.load(data_path)

	warnings.simplefilter('ignore')


	if ('output_ptrs_iq' in path_strings) or ('antennas_iq' in path_strings):
		print("Loaded an antenna iq file...")
		ant_iq = backconvert_pre_bfiq(data)
		write_backconverted(ant_iq, data_path)
	elif 'bfiq' in path_strings:
		print("Loaded a bfiq file...")
		bfiq = backconvert_bfiq(data)
		write_backconverted(bfiq, data_path)
	elif 'rawacf' in path_strings:
		print("Loaded a raw acf file")
		raw_acf = backconvert_rawacf(data)
		write_backconverted(raw_acf, data_path)
	else:
		print(suffix, 'filetypes are not supported')
		return

	print("Success!")

if __name__ == "__main__":
	filepath = sys.argv[1]
	backconvert_data(filepath)
