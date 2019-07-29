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
						if f not in shared:
							shared.append(f)
					else:
						if f in shared:
							shared.remove(f)
				else:
					if data[k][f] == data_sub[f]:
						if f not in shared:
							shared.append(f)
					else:
						if f in shared:
							shared.remove(f)
	return shared


def restructure_data(data_path):
	"""
	Restructure the data contained in an hdf5 file to eliminate the record format.
	Rather, data will be contained in a large array according to data dimensions.
	Examples: for rawacfs, this array will be of shape (num_records, num_arrays, num_sequences, num_beams, num_ranges, num_lags)
	Fields from the original record that do not change between records will be stored as fields in one metadata record within
	the file. Other fields will contain the data arrays and other metadata that does change record to record.
	"""

	def restructure_pre_bfiq(data_record):
		"""
		Restructuring method for pre bfiq data
		args:
			data_record		a record loaded from an hdf5
							data file
		"""

	def restructure_bfiq(data_record):
		"""
		Restructuring method for bfiq data
		args:
			data_record		a record loaded from an hdf5
							data file
		"""

	def restructure_rawacf(data_record):
		"""
		Restructuring method for rawacf data
		args:
			data_record		a record loaded from an hdf5
							data file
		"""
		data_dict = dict()

		num_records = len(data_record)

		# write shared fields to dictionary
		shared = find_shared(data_record)
		k = list(data_record.keys())[0]
		for f in shared:
			data_dict[f] = data[k][f]

		# handle unshared data fields
		num_beams = data_dict["correlation_dimensions"][0]
		num_ranges = data_dict["correlation_dimensions"][1]
		num_lags = data_dict["correlation_dimensions"][2]
		data_shape = (num_records, num_beams, num_ranges, num_lags)

		dims = data_dict["correlation_dimensions"]

		main_array = np.empty(data_shape)
		intf_array = np.empty(data_shape)
		xcfs_array = np.empty(data_shape)

		timestamp_array = np.empty(num_records)
		write_time_array = np.empty(num_records)
		int_time_array = np.empty(num_records)

		# find largest number of sequences in record
		max_sequences = data_dict["num_sequences"]
		for k in data_record:
			if len(data_record[k]["sqn_timestamps"]) > max_sequences:
				max_sequences = len(data_record[k]["sqn_timestamps"])
		sqn_ts_array = np.empty((num_records, max_sequences))

		rec_idx = 0
		for k in data_record:
			print("Restructuring", k)
			timestamp_array[rec_idx] = k
			write_time_array[rec_idx] = data_record[k]["timestamp_of_write"]
			int_time_array[rec_idx] = data_record[k]["int_time"]

			sqn_timestamps = data_record[k]["sqn_timestamps"]
			if len(sqn_timestamps) < max_sequences:
				while len(sqn_timestamps) < max_sequences:
					sqn_timestamps = np.append(sqn_timestamps, 0)
			sqn_ts_array[rec_idx] = sqn_timestamps

			main_array[rec_idx] = data_record[k]["main_acfs"].reshape(dims)
			intf_array[rec_idx] = data_record[k]["intf_acfs"].reshape(dims)
			xcfs_array[rec_idx] = data_record[k]["xcfs"].reshape(dims)

			rec_idx += 1

		# write leftover metadata and data
		data_dict["timestamps"] = timestamp_array
		data_dict["timestamp_of_write"] = write_time_array
		data_dict["int_time"] = int_time_array
		data_dict["sqn_timestamps"] = sqn_ts_array

		data_dict["correlation_dimensions"] = np.insert(data_dict["correlation_dimensions"], 0, num_records)
		data_dict["correlation_descriptors"] = np.insert(data_dict["correlation_descriptors"], 0, "num_records")

		data_dict["main_acfs"] = main_array
		data_dict["intf_acfs"] = intf_array
		data_dict["xcfs"] = xcfs_array

		dd.io.save(data_path + ".new", data_dict, compression=None)

	suffix = data_path.split('.')[-2]

	data = dd.io.load(data_path)

	if suffix == 'output_ptrs_iq':
		restructure_pre_bfiq(data)
		return
	elif suffix == 'bfiq':
		restructure_bfiq(data)
		return
	elif suffix == 'rawacf':
		restructure_rawacf(data)
		return
	else:
		print(suffix, 'filetypes are not supported')
		return

if __name__ == "__main__":
	filepath = sys.argv[1]
	restructure_data(filepath)
