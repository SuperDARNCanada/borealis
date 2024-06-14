#!/usr/bin/env python3

import sys
import deepdish
import random

from plotting_borealis_data_utils import plot_output_tx_data

filename = sys.argv[1]

data = deepdish.io.load(filename)

record_name = random.choice(list(data.keys()))
print(record_name)
tx = data[record_name]

plot_output_tx_data(tx, "tx_data")
