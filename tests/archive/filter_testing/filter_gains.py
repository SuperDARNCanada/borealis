# check the filter and plot the response given a boxcar representing the pulse before decimation.


import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

with open("/home/radar/filter3coefficients.dat", "r") as f:
    filter1_data = f.read()

numtaps = len(filter1_data.splitlines())
filter1_gain = 0.0
filter_taps = []
for line in filter1_data.splitlines():
    filter_coeff = eval(line)
    filter_taps.append(filter_coeff[0])
    filter1_gain += float(filter_coeff[0])

print("Gain of filter 3: {}".format(filter1_gain))
print("Numtaps: {}".format(numtaps))

plt.plot(np.arange(numtaps), filter_taps)
plt.title("Filter Response")
plt.show()

# show 'impulse' response at decimated sample rate.

boxcar = [0.0] * 360
boxcar.extend([1.0] * 30)
boxcar.extend([0.0] * 360)

output = signal.convolve(boxcar, filter_taps, mode="full")

# get all possible scenarios depending on the location of the pulse echo in the data
for start_sample in range(0, 30):
    decimated_output = output[start_sample::30]
    plt.plot(np.arange(len(decimated_output)), decimated_output)

plt.title("Decimated Pulse Response")
plt.show()
