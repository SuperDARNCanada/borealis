import numpy as np
import deepdish as dd
from reprocessing import correlate_samples

def test_correlate():
	acfs = dd.io.load("../data/20190619.1600.02.sas.0.rawacf.hdf5", group="/1560967198922")
	bfiq = dd.io.load("../data/20190619.1600.02.sas.0.bfiq.hdf5", group="/1560967198922")

	my_main, my_intf, my_xcfs = correlate_samples(bfiq)

	num_beams = acfs["correlation_dimensions"][0]
	num_ranges = acfs["correlation_dimensions"][1]
	num_lags = acfs["correlation_dimensions"][2]

	main_acfs = np.reshape(acfs["main_acfs"], (num_beams, num_ranges, num_lags))
	intf_acfs = np.reshape(acfs["intf_acfs"], (num_beams, num_ranges, num_lags))
	xcfs_acfs = np.reshape(acfs["xcfs"], (num_beams, num_ranges, num_lags))

	np.testing.assert_array_equal(my_main, main_acfs)
	np.testing.assert_array_equal(my_intf, intf_acfs)
	np.testing.assert_array_equal(my_xcfs, xcfs)
