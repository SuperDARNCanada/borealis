import numpy as np
import deepdish as dd


def autocorrelate_bfiq(ts_dict):
	"""
	Builds the autocorrelation matrices for the beamformed data
	contained in one timestamp dictionary
	"""
	data_buff = ts_dict["data"]
	num_slices = ts_dict["num_slices"]
	num_ant_arrays = ts_dict["data_dimensions"][0]
	num_sequences = ts_dict["data_dimensions"][1]
	num_beams = ts_dict["data_dimensions"][2]
	num_samples = ts_dict["data_dimensions"][3]
	
	samples = np.ndarray((num_ant_arrays,
	 						num_sequences, 
	 						num_beams, 
	 						num_samples,), dtype=np.complex64)
	


if __name__ == "__main__":
bfiq_data = dd.io.load(bfiq_file)  # TODO: RENAME THIS
rawacf = dict()

for timestamp in data:
