import numpy as np
import deepdish as dd
from tools import correlate_samples

acfs = dd.io.load("data/20190619.1600.02.sas.0.rawacf.hdf5")
# bfiq = dd.io.load("data/20190619.1600.02.sas.0.bfiq.hdf5")
test = dd.io.load("data/20190619.1600.02.sas.0.rawacf.hdf5.test")

failed = []

for k in bfiq:
	print("testing", k)
	my_main, my_intf, my_xcfs = correlate_samples(bfiq[k])

	num_beams = acfs[k]["correlation_dimensions"][0]
	num_ranges = acfs[k]["correlation_dimensions"][1]
	num_lags = acfs[k]["correlation_dimensions"][2]

	# main_acfs = np.reshape(acfs[k]["main_acfs"], (num_beams, num_ranges, num_lags))
	# intf_acfs = np.reshape(acfs[k]["intf_acfs"], (num_beams, num_ranges, num_lags))
	# xcfs_acfs = np.reshape(acfs[k]["xcfs"], (num_beams, num_ranges, num_lags))

	main_acfs = acfs[k]["main_acfs"]
	intf_acfs = acfs[k]["intf_acfs"]
	xcfs_acfs = acfs[k]["xcfs"]

	my_main = my_main.flatten()
	my_intf = my_intf.flatten()
	my_xcfs = my_xcfs.flatten()

	main = np.array_equal(my_main, main_acfs)
	intf = np.array_equal(my_intf, intf_acfs)
	crss = np.array_equal(my_xcfs, xcfs_acfs)

	if not (main and intf and crss):
		failed.append(k)

f = open("failed_correlations.txt", "w+")
for i in range(len(failed)):
	fail_str = "timestamp: " + str(failed[i]) + " failed\n"

failed = []
for k in acfs:
	for f in acfs[k]:
		if f == 'experiment_comment':
			continue
		if type(acfs[k][f]) is np.ndarray:
			if not (np.array_equal(acfs[k][f], test[k][f])):
				failed.append((k,f))
		else:
			if not (acfs[k][f] == test[k][f]):
				failed.append((k,f))

f = open("failed_correlations.txt", "w+")

for i in range(len(failed)):
	fail_str = 'timestamp: ' + str(failed[i][0]) + ' failed with field: ' + str(failed[i][1]) + "\n"
	f.write(fail_str)

f.close()
print(len(failed))
