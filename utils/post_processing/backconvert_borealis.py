import deepdish as dd
import numpy as np

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
