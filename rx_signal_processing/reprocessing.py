import numpy as np
import deepdish as dd
import sys


def correlate_samples(ts_dict):
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
	num_lags = np.shape(ts_dict["lags"])[0]
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
	main_corrs = intf_corrs = cross_corrs = np.zeros((num_sequences,
													 	num_beams, 
													 	num_samples, 
													 	num_samples), 
														dtype=np.complex64)

	# Preallocate arrays for results of range-lag selection
	out_main = out_intf = out_cross = np.zeros((num_sequences,
												num_beams,
												num_ranges,
												num_lags), 
												dtype=np.complex64)

	# Perform autocorrelations of each array, and cross
	# correlation between arrays
	for seq in range(num_sequences):
		for beam in range(num_beams):
			main_samps = main_data[seq, beam]
			intf_samps = intf_data[seq, beam]

			main_corrs[seq, beam] = np.outer(main_samps, main_samps.conjugate())
			intf_corrs[seq, beam] = np.outer(intf_samps, intf_samps.conjugate())
			cross_corrs[seq, beam] = np.outer(main_samps, intf_samps.conjugate())

			beam_offset = num_beams * num_ranges * num_lags
			first_range_offset = ts_dict["first_range"] // ts_dict["range_sep"]

			# Select out the lags for each range gate
			main_small = intf_small = cross_small = np.zeros((num_ranges, num_lags,), 
																dtype=np.complex64)

			for rng in range(num_ranges):
				for lag in range(num_lags):
					# tau spacing in us, sample rate in hz
					tau_in_samples = np.ceil(ts_dict["tau_spacing"] * 1e-6 * 
												ts_dict["rx_sample_rate"])
					p1_offset = lags[lag, 0] * tau_in_samples
					p2_offset = lags[lag, 1] * tau_in_samples
					
					dim_1_offset = int(rng + first_range_offset + p1_offset)
					dim_2_offset = int(rng + first_range_offset + p2_offset)

					main_small[rng, lag] = main_corrs[seq, beam, dim_1_offset, dim_2_offset]
					intf_small[rng, lag] = intf_corrs[seq, beam, dim_1_offset, dim_2_offset]
					cross_small[rng, lag] = cross_corrs[seq, beam, dim_1_offset, dim_2_offset]

			# replace full correlation matrix with resized range-lag matrix
			out_main[seq, beam] = main_small
			out_intf[seq, beam] = intf_small
			out_cross[seq, beam] = cross_small

	# average each correlation matrix over sequences dimension
	out_main = np.mean(out_main, axis=0)
	out_intf = np.mean(out_intf, axis=0)
	out_cross = np.mean(out_cross, axis=0)

	return out_main, out_intf, out_cross


if __name__ == "__main__":
	bfiq_filepath = sys.argv[1]
	acfs_filepath = sys.argv[2]

	bfiq = dd.io.load(bfiq_filepath, group="/1560967198922")
	acfs = dd.io.load(acfs_filepath, group="/1560967198922")

	my_main, my_intf, my_xcfs = correlate_samples(bfiq)

