import deepdish as deep
import numpy as np

def restructure_data(data_path):
	"""
	Restructure the data contained in an hdf5 file to eliminate the record format.
	Rather, data will be contained in a large array according to data dimensions.
	Examples: for rawacfs, this array will be of shape (num_records, num_arrays, num_sequences, num_beams, num_ranges, num_lags)
	Fields from the original record that do not change between records will be stored as fields in one metadata record within
	the file. Other fields will contain the data arrays and other metadata that does change record to record.
	"""
	def restructure_pre_bfiq():
		"""
		Restructuring method for pre bfiq data
		"""

	def restructure_bfiq():
		"""
		Restructuring method for bfiq data
		"""

	def restructure_rawacf():
		"""
		Restructuring method for rawacf data
		"""

	suffix = data_path.split('.')[-2]

	if suffix == 'output_ptrs_iq':
		restructure_pre_bfiq()
	elif suffix == 'bfiq':
		restructure_bfiq()
	elif suffix == 'rawacf':
		restructure_rawacf()
	else:
		print(suffix, 'filetypes are not supported')
		return
