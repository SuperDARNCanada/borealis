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

def backconvert_data(data_path):
	"""
	Converts a restructured and compressed hdf5 borealis datafile
	back to its original, record based format.
	Args:
		data_path (str): Path to the data file to be back converted
	"""

	def backconvert_pre_bfiq(data_record, data_path):
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
			temp_file = 'temp.hdf5'
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
			ts_dict[key]["data"] = data_record["data"][rec, :, 0:int(num_sequences), :].flatten()
			ts_dict[key]["data_descriptors"] = ts_dict[key]["data_descriptors"][1:]
			ts_dict[key]["data_dimensions"] = data_record["data_dimensions"][rec]


			# File should be written here

			site_format_file = data_path + '.site'

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

	def backconvert_bfiq(data_record, data_path):
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
			temp_file = 'temp.hdf5'
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
			ts_dict[key]["data"] = data_record["data"][rec, :, 0:int(num_sequences), :, :].flatten()
			ts_dict[key]["data_descriptors"] = ts_dict[key]["data_descriptors"][1:]
			ts_dict[key]["data_dimensions"] = data_record["data_dimensions"][rec]
			# File should be written here

			site_format_file = data_path + '.site'

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

	def backconvert_rawacf(data_record, data_path):
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
			temp_file = 'temp.hdf5'
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
			# File should be written here

			site_format_file = data_path + '.site'

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

	suffix = data_path.split('.')[-3]

	print("Restructuring", data_path, "...")

	data = dd.io.load(data_path)


	if (suffix == 'output_ptrs_iq') or (suffix == 'antennas_iq'):
		print("Loaded a pre bfiq file...")
		backconvert_pre_bfiq(data, data_path)
	elif suffix == 'bfiq':
		print("Loaded a bfiq file...")
		backconvert_bfiq(data, data_path)
	elif suffix == 'rawacf':
		print("Loaded a raw acf file")
		backconvert_rawacf(data, data_path)
	else:
		print(suffix, 'filetypes are not supported')
		return

	print("Success!")

if __name__ == "__main__":
	filepath = sys.argv[1]
	backconvert_data(filepath)
