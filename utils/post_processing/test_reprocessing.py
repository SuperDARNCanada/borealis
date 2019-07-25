import numpy as np
import deepdish as dd
# from bfiq_to_rawacf import correlate_samples

acfs = dd.io.load("data/20190619.1600.02.sas.0.rawacf.hdf5")
test = dd.io.load("data/test_acf.hdf5")

failed = []

# for k in bfiq:

# 	my_main, my_intf, my_xcfs = correlate_samples(bfiq[k])

# 	num_beams = acfs[k]["correlation_dimensions"][0]
# 	num_ranges = acfs[k]["correlation_dimensions"][1]
# 	num_lags = acfs[k]["correlation_dimensions"][2]

# 	main_acfs = np.reshape(acfs[k]["main_acfs"], (num_beams, num_ranges, num_lags))
# 	intf_acfs = np.reshape(acfs[k]["intf_acfs"], (num_beams, num_ranges, num_lags))
# 	xcfs_acfs = np.reshape(acfs[k]["xcfs"], (num_beams, num_ranges, num_lags))

# 	main = np.array_equal(my_main, main_acfs)
# 	intf = np.array_equal(my_intf, intf_acfs)
# 	crss = np.array_equal(my_xcfs, xcfs_acfs)

# 	if main and intf and crss:
# 		print(k, "passed")

# 	else:
# 		print(k, "failed")
# 		failed.append(k)

shared_fields = ['beam_azms', 'beam_nums', 'blanked_samples', 'lags', 'noise_at_freq',
					'pulses', 'sqn_timestamps', 'borealis_git_hash', 
					'data_normalization_factor', 'experiment_comment', 
					'experiment_id', 'experiment_name', 'first_range', 
					'first_range_rtt', 'freq', 'int_time', 'intf_antenna_count', 
					'main_antenna_count', 'num_sequences', 'num_slices', 'range_sep', 
					'rx_sample_rate', 'samples_data_type', 'scan_start_marker', 
					'slice_comment', 'station', 'tau_spacing', 'timestamp_of_write', 'tx_pulse_len']

for k in acfs:
	for f in shared_fields:
		if type(acfs[k][f]) is np.ndarray:
			if np.array_equal(acfs[k][f], test[k][f]):
				print(k, "passed")
			else:
				print(k, "failed")
				failed.append((k,f))

		else:
			if acfs[k][f] == test[k][f]:
				print(k, "passed")
			else:
				print(k, "failed")
				failed.append((k,f))

print(failed)
