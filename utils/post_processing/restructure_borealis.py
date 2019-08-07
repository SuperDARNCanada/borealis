# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Liam Graham

#
# restructure_borealis.py
# 2019-07-30
# Command line tool for strucuturing Borealis files
# for bfiq, antennas iq, and raw acf data into a smaller,
# faster, and more usable format

import deepdish as dd
import numpy as np
import sys

shared_antiq = ['antenna_arrays_order', 'beam_azms', 'beam_nums', 'borealis_git_hash', 'data_descriptors',
				'data_normalization_factor', 'experiment_comment', 'experiment_id', 'experiment_name',
				'freq', 'intf_antenna_count', 'main_antenna_count', 'num_samps', 'num_slices',
				'pulse_phase_offset', 'pulses', 'rx_sample_rate', 'samples_data_type', 'scan_start_marker',
				'slice_comment', 'station', 'tau_spacing', 'tx_pulse_len']

shared_bfiq =  ['antenna_arrays_order', 'beam_azms', 'beam_nums', 'blanked_samples', 'borealis_git_hash', 
				'data_descriptors', 'data_normalization_factor', 'experiment_comment', 'experiment_id',
				'experiment_name', 'first_range', 'first_range_rtt', 'freq', 'intf_antenna_count', 'lags',
				'main_antenna_count', 'num_ranges', 'num_samps', 'num_slices', 'pulse_phase_offset', 'pulses',
				'range_sep', 'rx_sample_rate', 'samples_data_type', 'scan_start_marker', 'slice_comment',
				'station', 'tau_spacing', 'tx_pulse_len']

shared_rawacf = ['beam_azms', 'beam_nums', 'blanked_samples', 'borealis_git_hash', 'correlation_descriptors',
				 'data_normalization_factor', 'experiment_comment', 'experiment_id', 'experiment_name',
				 'first_range', 'first_range_rtt', 'freq', 'intf_antenna_count', 'lags', 'main_antenna_count',
				 'num_slices', 'pulses', 'range_sep', 'rx_sample_rate', 'samples_data_type', 'scan_start_marker',
				 'slice_comment', 'station', 'tau_spacing', 'tx_pulse_len']

def find_max_sequences(data):
	"""
	Finds the maximum number of sequences between records in a data file
	"""
	first_key = list(data.keys())[0]
	max_seqs = data[first_key]["num_sequences"]
	for k in data:
		if max_seqs < data[k]["num_sequences"]:
			max_seqs = data[k]["num_sequences"]
	return max_seqs


def restructure_pre_bfiq(data_record):
	"""
	Restructuring method for pre bfiq data
	args:
		data_record		a timestamped record loaded from an
		 				hdf5 Borealis pre-bfiq data file
	"""
	data_dict = dict()
	num_records = len(data_record)

	# write shared fields to dictionary
	k = list(data_record.keys())[0]
	for f in shared_antiq:
		data_dict[f] = data_record[k][f]

	# find maximum number of sequences
	max_seqs = find_max_sequences(data_record)
	dims = data_record[k]["data_dimensions"]
	num_antennas = dims[0]
	num_samps = dims[2]

	data_buffer_offset = num_antennas * num_samps * max_seqs
	data_buffer = np.zeros(num_records * data_buffer_offset, dtype=np.complex64)
	data_shape = (num_records, num_antennas, max_seqs, num_samps)

	sqn_buffer_offset = max_seqs
	sqn_ts_buffer = np.zeros(num_records * max_seqs)
	sqn_shape = (num_records, max_seqs)

	noise_buffer_offset = max_seqs
	noise_buffer = np.zeros(num_records * max_seqs)
	noise_shape = (num_records, max_seqs)

	sqn_num_array = np.empty(num_records)
	int_time_array = np.empty(num_records)

	data_dims_array = np.empty((num_records, len(data_record[k]["data_descriptors"])))

	rec_idx = 0
	for k in data_record:
		# handle unshared fields
		int_time_array[rec_idx] = data_record[k]["int_time"]
		sqn_num_array[rec_idx] = data_record[k]["num_sequences"]
		data_dims_array[rec_idx] = data_record[k]["data_dimensions"]

		# insert data into buffer
		record_buffer = data_record[k]["data"]
		data_pos = rec_idx * data_buffer_offset
		data_end = data_pos + len(record_buffer)
		data_buffer[data_pos:data_end] = record_buffer

		# insert sequence timestamps into buffer
		rec_sqn_ts = data_record[k]["sqn_timestamps"]
		sqn_pos = rec_idx * sqn_buffer_offset
		sqn_end = sqn_pos + data_record[k]["num_sequences"]
		sqn_ts_buffer[sqn_pos:sqn_end] = rec_sqn_ts

		rec_noise = data_record[k]["noise_at_freq"]
		noise_pos = rec_idx * noise_buffer_offset
		noise_end = noise_pos + data_record[k]["num_sequences"]
		noise_buffer[noise_pos:noise_end] = rec_noise

		rec_idx += 1

	# write leftover metadata and data
	data_dict["int_time"] = int_time_array
	data_dict["num_sequences"] = sqn_num_array

	data_dict["data"] = data_buffer.reshape(data_shape)
	data_dict["sqn_timestamps"] = sqn_ts_buffer.reshape(sqn_shape)
	data_dict["noise_at_freq"] = noise_buffer.reshape(noise_shape)

	data_dict["data_descriptors"] = np.insert(data_dict["data_descriptors"], 0, "num_records")
	data_dict["data_dimensions"] = data_dims_array

	return data_dict

def restructure_bfiq(data_record):
	"""
	Restructuring method for bfiq data
	args:
		data_record		a timestamped record loaded from an
		 				hdf5 Borealis bfiq data file
	"""
	data_dict = dict()
	num_records = len(data_record)

	# write shared fields to dictionary
	k = list(data_record.keys())[0]
	for f in shared_bfiq:
		data_dict[f] = data_record[k][f]

	# find maximum number of sequences
	max_seqs = find_max_sequences(data_record)
	dims = data_record[k]["data_dimensions"]
	num_arrays = dims[0]
	num_beams = dims[2]
	num_samps = dims[3]

	data_buffer_offset = num_arrays * num_beams * num_samps * max_seqs
	data_buffer = np.zeros(num_records * data_buffer_offset, dtype=np.complex64)
	data_shape = (num_records, num_arrays, max_seqs, num_beams, num_samps)

	sqn_buffer_offset = max_seqs
	sqn_ts_buffer = np.zeros(num_records * max_seqs)
	sqn_shape = (num_records, max_seqs)

	noise_buffer_offset = max_seqs
	noise_buffer = np.zeros(num_records * max_seqs)
	noise_shape = (num_records, max_seqs)

	sqn_num_array = np.empty(num_records)
	int_time_array = np.empty(num_records)

	data_dims_array = np.empty((num_records, len(data_record[k]["data_descriptors"])))

	rec_idx = 0
	for k in data_record:
		# handle unshared fields
		int_time_array[rec_idx] = data_record[k]["int_time"]
		sqn_num_array[rec_idx] = data_record[k]["num_sequences"]
		data_dims_array[rec_idx] = data_record[k]["data_dimensions"]

		# insert data into buffer
		record_buffer = data_record[k]["data"]
		data_pos = rec_idx * data_buffer_offset
		data_end = data_pos + len(record_buffer)
		data_buffer[data_pos:data_end] = record_buffer

		# insert sequence timestamps into buffer
		rec_sqn_ts = data_record[k]["sqn_timestamps"]
		sqn_pos = rec_idx * sqn_buffer_offset
		sqn_end = sqn_pos + data_record[k]["num_sequences"]
		sqn_ts_buffer[sqn_pos:sqn_end] = rec_sqn_ts

		rec_noise = data_record[k]["noise_at_freq"]
		noise_pos = rec_idx * noise_buffer_offset
		noise_end = noise_pos + data_record[k]["num_sequences"]
		noise_buffer[noise_pos:noise_end] = rec_noise

		rec_idx += 1

	# write leftover metadata and data
	data_dict["int_time"] = int_time_array
	data_dict["num_sequences"] = sqn_num_array

	data_dict["data"] = data_buffer.reshape(data_shape)
	data_dict["sqn_timestamps"] = sqn_ts_buffer.reshape(sqn_shape)
	data_dict["noise_at_freq"] = noise_buffer.reshape(noise_shape)

	data_dict["data_descriptors"] = np.insert(data_dict["data_descriptors"], 0, "num_records")
	data_dict["data_dimensions"] = data_dims_array

	return data_dict

def restructure_rawacf(data_record):
	"""
	Restructuring method for rawacf data
	args:
		data_record		a timestamped record loaded from an
		 				hdf5 Borealis rawacf data file
	"""
	data_dict = dict()
	num_records = len(data_record)
	max_seqs = find_max_sequences(data_record)

	# write shared fields to dictionary
	k = list(data_record.keys())[0]
	for f in shared_rawacf:
		data_dict[f] = data_record[k][f]

	# handle unshared data fields
	dims = data_dict["correlation_dimensions"]
	num_beams = dims[0]
	num_ranges = dims[1]
	num_lags = dims[2]
	data_shape = (num_records, num_beams, num_ranges, num_lags)

	noise_buffer_offset = max_seqs
	noise_buffer = np.zeros(num_records * max_seqs)
	noise_shape = (num_records, max_seqs)

	sqn_ts_array = np.empty((num_records, max_seqs))
	sqn_num_array = np.empty(num_records)
	main_array = np.empty(data_shape, dtype=np.complex64)
	intf_array = np.empty(data_shape, dtype=np.complex64)
	xcfs_array = np.empty(data_shape, dtype=np.complex64)

	int_time_array = np.empty(num_records)
	

	rec_idx = 0
	for k in data_record:
		sqn_num_array[rec_idx] = data_record[k]["num_sequences"]
		int_time_array[rec_idx] = data_record[k]["int_time"]

		# some records have fewer than the specified number of sequences
		# get around this by zero padding to the recorded number
		sqn_timestamps = data_record[k]["sqn_timestamps"]
		while len(sqn_timestamps) < max_seqs:
			sqn_timestamps = np.append(sqn_timestamps, 0)
		sqn_ts_array[rec_idx] = sqn_timestamps

		rec_noise = data_record[k]["noise_at_freq"]
		noise_pos = rec_idx * noise_buffer_offset
		noise_end = noise_pos + data_record[k]["num_sequences"]
		noise_buffer[noise_pos:noise_end] = rec_noise

		data_dict["noise_at_freq"] = noise_buffer.reshape(noise_shape)

		main_array[rec_idx] = data_record[k]["main_acfs"].reshape(dims)
		intf_array[rec_idx] = data_record[k]["intf_acfs"].reshape(dims)
		xcfs_array[rec_idx] = data_record[k]["xcfs"].reshape(dims)

		rec_idx += 1

	# write leftover metadata and data
	data_dict["int_time"] = int_time_array
	data_dict["sqn_timestamps"] = sqn_ts_array
	data_dict["num_sequences"] = sqn_num_array
	data_dict["noise_at_freq"] = noise_buffer.reshape(noise_shape)
	data_dict["correlation_descriptors"] = np.insert(data_dict["correlation_descriptors"], 0, "num_records")

	data_dict["main_acfs"] = main_array
	data_dict["intf_acfs"] = intf_array
	data_dict["xcfs"] = xcfs_array

	return data_dict


def write_restructured(data_dict, data_path):
	print("Compressing...")
	dd.io.save(data_path + ".new.test", data_dict, compression='zlib')


def restructure_data(data_path):
	"""
	Restructure the data contained in an hdf5 file to eliminate the record format.
	Rather, data will be contained in a large array according to data dimensions.
	Examples: for rawacfs, this array will be of shape (num_records, num_arrays, num_sequences, num_beams, num_ranges, num_lags)
	Fields from the original record that do not change between records will be stored as fields in one metadata record within
	the file. Other fields will contain the data arrays and other metadata that does change record to record.
	Args:
		data_path:	string containing the path to the data file for restructuring
	Returns:	If valid filetype, returns None and saves the data as a newly
				formatted hdf5 file.
	"""

	path_strings = data_path.split('.')

	print("Restructuring", data_path, "...")

	data = dd.io.load(data_path)

	if ('output_ptrs_iq' in path_strings) or ('antennas_iq' in path_strings):
		print("Loaded an antenna iq file...")
		ant_iq = restructure_pre_bfiq(data)
		write_restructured(ant_iq, data_path)
	elif 'bfiq' in path_strings:
		print("Loaded a bfiq file...")
		bfiq = restructure_bfiq(data)
		write_restructured(bfiq, data_path)
	elif 'rawacf' in path_strings:
		print("Loaded a raw acf file")
		raw_acf = restructure_rawacf(data)
		write_restructured(raw_acf, data_path)
	else:
		print(suffix, 'filetypes are not supported')
		return

	print("Success!")

if __name__ == "__main__":
	filepath = sys.argv[1]
	restructure_data(filepath)
