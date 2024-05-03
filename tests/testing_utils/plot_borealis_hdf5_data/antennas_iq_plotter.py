#!/usr/bin/env python3

import sys
import deepdish
import random
from plotting_borealis_data_utils import plot_antennas_iq_data

filename = sys.argv[1]

data = deepdish.io.load(filename)

record_name = random.choice(list(data.keys()))
print(record_name)
antennas_iq = data[record_name]


plot_antennas_iq_data(antennas_iq, "antennas_iq")
