import numpy as np
import deepdish as dd
from reprocessing import correlate_samples

acfs = dd.io.load("data/20190619.1600.02.sas.0.rawacf.hdf5")
bfiq = dd.io.load("data/20190619.1600.02.sas.0.bfiq.hdf5")

failed = []

for k in bfiq:

	my_main, my_intf, my_xcfs = correlate_samples(bfiq[k])

	num_beams = acfs[k]["correlation_dimensions"][0]
	num_ranges = acfs[k]["correlation_dimensions"][1]
	num_lags = acfs[k]["correlation_dimensions"][2]

	main_acfs = np.reshape(acfs[k]["main_acfs"], (num_beams, num_ranges, num_lags))
	intf_acfs = np.reshape(acfs[k]["intf_acfs"], (num_beams, num_ranges, num_lags))
	xcfs_acfs = np.reshape(acfs[k]["xcfs"], (num_beams, num_ranges, num_lags))

	main = np.array_equal(my_main, main_acfs)
	intf = np.array_equal(my_intf, intf_acfs)
	crss = np.array_equal(my_xcfs, xcfs_acfs)

	if main and intf and crss:
		print(k, "passed")

	else:
		print(k, "failed")
		failed.append(k)
