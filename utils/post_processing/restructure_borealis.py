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


def find_shared(data):
	"""
	Finds fields in an hdf5 file that do not change
	between records
	"""
	shared = list()
	unshared = list()
	start = True

	for k in data:
		if start:
			data_sub = data[k]
			start = False
		else:
			for f in data[k]:
				if type(data[k][f]) is np.ndarray:
					if np.array_equal(data[k][f], data_sub[f]):
						if f not in shared and f not in unshared:
							shared.append(f)
					else:
						unshared.append(f)
				else:
					if data[k][f] == data_sub[f]:
						if f not in shared and f not in unshared:
							shared.append(f)
					else:
						unshared.append(f)
	return shared

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
		shared = find_shared(data_record)
		k = list(data_record.keys())[0]
		for f in shared:
			data_dict[f] = data[k][f]

		# find maximum number of sequences
		max_seqs = find_max_sequences(data_record)
		dims = data_record[k]["data_dimensions"]
		num_antennas = dims[0]
		num_samps = dims[2]

		data_buffer_offset = num_antennas * num_samps * max_seqs
		data_buffer = np.zeros(num_records * data_buffer_offset)
		data_shape = (num_records, num_antennas, max_seqs, num_samps)

		sqn_buffer_offset = max_seqs
		sqn_ts_buffer = np.zeros(num_records * max_seqs)
		sqn_shape = (num_records, max_seqs)

		sqn_num_array = np.empty((num_records))

		rec_idx = 0
		for k in data_record:
			# handle unshared fields
			print(k)

			int_time_array = np.empty(num_records)
			int_time_array[rec_idx] = data_record[k]["int_time"]
			sqn_num_array[rec_idx] = data_record[k]["num_sequences"]

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

			rec_idx += 1

		# write leftover metadata and data
		data_dict["timestamps"] = timestamp_array
		data_dict["timestamp_of_write"] = write_time_array
		data_dict["int_time"] = int_time_array

		data_dict["data"] = data_buffer.reshape(data_shape)
		data_dict["sqn_timestamps"] = sqn_ts_buffer.reshape(sqn_shape)

		data_dict["data_descriptors"] = np.insert(data_dict["data_descriptors"], 0, "num_records")

		dd.io.save(data_path + ".new", data_dict, compression='zlib')

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
		shared = find_shared(data_record)
		k = list(data_record.keys())[0]
		for f in shared:
			data_dict[f] = data[k][f]

		# find maximum number of sequences
		max_seqs = find_max_sequences(data_record)
		dims = data_record[k]["data_dimensions"]
		num_arrays = dims[0]
		num_beams = dims[2]
		num_samps = dims[3]

		data_buffer_offset = num_arrays * num_beams * num_samps * max_seqs
		data_buffer = np.zeros(num_records * data_buffer_offset)
		data_shape = (num_records, num_arrays, max_seqs, num_beams, num_samps)

		sqn_buffer_offset = max_seqs
		sqn_ts_buffer = np.zeros(num_records * max_seqs)
		sqn_shape = (num_records, max_seqs)

		sqn_num_array = np.empty((num_records))

		rec_idx = 0
		for k in data_record:
			# handle unshared fields
			print(k)
			int_time_array = np.empty(num_records)
			int_time_array[rec_idx] = data_record[k]["int_time"]
			sqn_num_array[rec_idx] = data_record[k]["num_sequences"]

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

			rec_idx += 1

		# write leftover metadata and data
		data_dict["timestamps"] = timestamp_array
		data_dict["timestamp_of_write"] = write_time_array
		data_dict["int_time"] = int_time_array

		data_dict["data"] = data_buffer.reshape(data_shape)
		data_dict["sqn_timestamps"] = sqn_ts_buffer.reshape(sqn_shape)

		data_dict["data_descriptors"] = np.insert(data_dict["data_descriptors"], 0, "num_records")

		dd.io.save(data_path + ".new", data_dict, compression='zlib')

	def restructure_rawacf(data_record):
		"""
		Restructuring method for rawacf data
		args:
			data_record		a timestamped record loaded from an
			 				hdf5 Borealis rawacf data file
		"""
		data_dict = dict()
		num_records = len(data_record)

		# write shared fields to dictionary
		shared = find_shared(data_record)
		k = list(data_record.keys())[0]
		for f in shared:
			data_dict[f] = data[k][f]

		# handle unshared data fields
		dims = data_dict["correlation_dimensions"]
		num_beams = dims[0]
		num_ranges = dims[1]
		num_lags = dims[2]
		data_shape = (num_records, num_beams, num_ranges, num_lags)

		num_sequences = data_dict["num_sequences"]

		main_array = np.empty(data_shape)
		intf_array = np.empty(data_shape)
		xcfs_array = np.empty(data_shape)

		int_time_array = np.empty(num_records)
		sqn_ts_array = np.empty((num_records, num_sequences))

		rec_idx = 0
		for k in data_record:
			int_time_array[rec_idx] = data_record[k]["int_time"]

			# some records have fewer than the specified number of sequences
			# get around this by zero padding to the recorded number
			sqn_timestamps = data_record[k]["sqn_timestamps"]
			while len(sqn_timestamps) < num_sequences:
				sqn_timestamps = np.append(sqn_timestamps, 0)
			sqn_ts_array[rec_idx] = sqn_timestamps

			main_array[rec_idx] = data_record[k]["main_acfs"].reshape(dims)
			intf_array[rec_idx] = data_record[k]["intf_acfs"].reshape(dims)
			xcfs_array[rec_idx] = data_record[k]["xcfs"].reshape(dims)

			rec_idx += 1

		# write leftover metadata and data
		data_dict["int_time"] = int_time_array
		data_dict["sqn_timestamps"] = sqn_ts_array

		data_dict["correlation_descriptors"] = np.insert(data_dict["correlation_descriptors"], 0, "num_records")

		data_dict["main_acfs"] = main_array
		data_dict["intf_acfs"] = intf_array
		data_dict["xcfs"] = xcfs_array

		dd.io.save(data_path + ".new", data_dict, compression='zlib')  # TODO: Change all of these to final filename


	suffix = data_path.split('.')[-2]

	print("Restructuring", data_path, "...")

	data = dd.io.load(data_path)

	if (suffix == 'output_ptrs_iq') or (suffix == 'antennas_iq'):
		print("Loaded a pre bfiq file...")
		restructure_pre_bfiq(data)
	elif suffix == 'bfiq':
		print("Loaded a bfiq file...")
		restructure_bfiq(data)
	elif suffix == 'rawacf':
		print("Loaded a raw acf file")
		restructure_rawacf(data)
	else:
		print(suffix, 'filetypes are not supported')
		return

	print("Success!")

if __name__ == "__main__":
	filepath = sys.argv[1]
	restructure_data(filepath)
