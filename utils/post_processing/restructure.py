import deepdish as dd
import numpy as np

def restructure_data(data_path):
	"""
	Restructure the data contained in an hdf5 file to eliminate the record format.
	Rather, data will be contained in a large array according to data dimensions.
	Examples: for rawacfs, this array will be of shape (num_records, num_arrays, num_sequences, num_beams, num_ranges, num_lags)
	Fields from the original record that do not change between records will be stored as fields in one metadata record within
	the file. Other fields will contain the data arrays and other metadata that does change record to record.
	"""
	def find_shared(data_record):
		"""
		Finds fields in an hdf5 file that do not change
		between records
		"""
		shared = list()
		unshared = list()
		start = True

		for k in data:
			if start:
				data_sub = data_sub[k]
				start = False
			else:
				print("checking", k)
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
		for f in data_sub:
			if f not in shared:
				unshared.append(f)
		return shared

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

		# write shared fields to dictionary
		shared = find_shared(data_record)
		k = list(data_record.keys())[0]
		for f in shared:
			data_dict[f] = data[k][f]




	suffix = data_path.split('.')[-2]

	data = dd.io.load()

	if suffix == 'output_ptrs_iq':
		restructure_pre_bfiq()
	elif suffix == 'bfiq':
		restructure_bfiq()
	elif suffix == 'rawacf':
		restructure_rawacf()
	else:
		print(suffix, 'filetypes are not supported')
		return
