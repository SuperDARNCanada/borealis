import matplotlib.pyplot as plt
import pydarnio
import glob
import os
import datetime as dt
import sys
import random

sys.path.append("../plot_borealis_hdf5_data/")
from plotting_borealis_data_utils import (
    plot_antennas_iq_data,
    plot_bf_iq_data,
    fft_and_plot_rawrf_data,
)


data_path = "/data/borealis_data/20241212/"
files = [os.path.basename(x) for x in glob.glob(f"{data_path}/*")]
slice_id = 0
record = None

allowable_file_extensions = ["h5"]
allowable_file_types = [
    "antennas_iq",
    "bfiq",
    "rawrf",
    "stage_0_iq",
    "stage_1_iq",
    "stage_2_iq",
]

# if there are multiple file times, only the first file time in the files list
# will be loaded
df_metadata = files[0].split(".")
# First 3 items hold the file timestamp
date_string = f"{df_metadata[0]}.{df_metadata[1]}.{df_metadata[2]}"
date = dt.datetime.strptime(date_string, "%Y%m%d.%H%M.%S")

# Only continue with files that all have the same timestamp
# and or valid types and file extensions
data_files = []
for f in files:
    df_metadata = f.split(".")
    if date_string not in f:
        continue
    file_extension = df_metadata[-1]
    file_type = df_metadata[-2]
    if file_extension not in allowable_file_extensions:
        print(f"Extension .{file_extension} is not supported")
        continue
    if file_type not in allowable_file_types:
        print(f"File type {file_type} is not supported")
        continue
    if df_metadata[4].isnumeric():
        if slice_id != int(df_metadata[4]):
            print(f"Evaluating slice {slice_id}. Skipping slice {int(df_metadata[4])}")
            continue
    data_files.append(f)

data = dict()
reader = pydarnio.BorealisV1Read
for df in data_files:
    df_metadata = df.split(".")
    file_type = df_metadata[-2]
    data[file_type] = reader.read_records(f"{data_path}/{df}")

# pick a record to compare if record is None
possible_records = range(len(data["antennas_iq"][0]))
if (record is None) or (record not in possible_records):
    record = random.choice(possible_records)
print(f"Comparing record {record}")

print([data["rawrf"][0][x]["freq"] for x in range(len(data["rawrf"][0]))])
print(
    data["rawrf"][1]["rx_center_freq"],
    data["rawrf"][0][record]["freq"],
    (data["rawrf"][1]["rx_center_freq"] - data["rawrf"][0][record]["freq"]) * -1,
)

# Plot antennas IQ stages
for stage in ["stage_0_iq", "stage_1_iq", "stage_2_iq"]:
    plot_antennas_iq_data(data[stage][0][record], data[stage][1], stage)

# plot antennas IQ and beamformed IQ data
# plot_antennas_iq_data(data['antennas_iq'][0][record], data['antennas_iq'][1], 'antennas_iq')
plot_bf_iq_data(data["bfiq"][0][record], data["bfiq"][1], "bfiq")

# TODO: Is the statement below still accurate?
# In loopback data, we expect to see the peak of the FFT at our transmitted frequency. We expect to see
# other peaks at 666.6 or 416.6 Hz off of the txfreq, because this is the periodicity of our signal (1.5ms or 2.4ms = tau)

# Plot fft of antennas iq and beamformed iq
# fft_samps, xf, fig = fft_and_plot_antennas_iq(data['antennas_iq'][0][record], data['antennas_iq'][1],"antennas_iq",)
# fft_samps, xf, fig = fft_and_plot_bfiq_data(data['bfiq'][0][record], data['bfiq'][1], 'bfiq')
plt.show()
plt.close()

# Vizualize fft and time domain of RAWRF data
fft_and_plot_rawrf_data(
    data["rawrf"][0][record], data["rawrf"][1], sequence=1, start_sample=0
)
plt.show()
plt.close()

fig, ax = plt.subplots(1, 1)
ax.plot(data["rawrf"][0][record]["rawrf_data"][0, 15].real)
plt.show()
plt.close()
print("Finished loopback tests")
