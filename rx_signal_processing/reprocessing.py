import numpy as np
import deepdish as dd


def correlate_bfiq_samples(ts_dict):
	"""
	Builds the autocorrelation and cross-correlation matrices for the beamformed data
	contained in one timestamp dictionary
	"""
	data_buff = ts_dict["data"]
	num_slices = ts_dict["num_slices"]
	num_ant_arrays = ts_dict["data_dimensions"][0]
	num_sequences = ts_dict["data_dimensions"][1]
	num_beams = ts_dict["data_dimensions"][2]
	num_samples = ts_dict["data_dimensions"][3]

	lags = ts_dict["lags"]
	num_lags = ts_dict["lags"][0].size
	num_ranges = ts_dict["num_ranges"]
	num_slices = ts_dict["num_slices"]
	
	data_mat = np.reshape(data_buff, (num_ant_arrays,
	 									num_sequences, 
	 									num_beams, 
	 									num_samples,))
	# Get data from each antenna array
	main_data = data_mat[0][:][:][:]
	intf_data = data_mat[1][:][:][:]

	# Preallocate arrays for correlation results
	main_corrs = np.zeros_like(data_mat[0][:][:])
	intf_corrs = np.zeros_like(data_mat[1][:][:])
	cross_corrs = main_corrs

	# Perform autocorrelations of each array, and cross
	# correlation between arrays
	for seq in range(num_sequences):
		for beam in range(num_beams):
			main_samps = data_mat[seq][beam]
			intf_samps = data_mat[seq][beam]

			main_corrs[seq][beam] = np.outer(main_samps, main_samps.conjugate())
			intf_corrs[seq][beam] = np.outer(intf_samps, intf_samps.conjugate())
			cross_corrs[seq][beam] = np.outer(main_samps, intf_samps.conjugate())

			beam_offset = num_beams * num_ranges * num_lags
			first_range_offset = ts_dict["first_range"] // ts_dict["range_sep"]

			# Select out the lags for each range gate
			main_small = intf_small = cross_small = np.array((num_lags, num_ranges,))

			for rng in range(num_ranges):
				for lag in range(num_lags):
					# tau spacing in us, sample rate in hz
					tau_in_samples = np.ceil(ts_dict["tau_spacing"] * 1e-6 * 
												ts_dict["output_sample_rate"])
					p1_offset = lags[lag][0] * tau_in_samples
					p2_offset = lags[lag][1] * tau_in_samples
					
					dim_1_offset = rng + first_range_offset + p1_offset
					dim_2_offset = rng + first_range_offset + p2_offset

					main_small[lag, rng] = main_corrs[dim_1_offset, dim_2_offset]
					intf_small[lag, rng] = intf_corrs[dim_1_offset, dim_2_offset]
					cross_small[lag, rng] = cross_corrs[dim_1_offset, dim_2_offset]
					


if __name__ == "__main__":
bfiq_data = dd.io.load(bfiq_file)  # TODO: RENAME THIS
rawacf = dict()

for timestamp in data:
