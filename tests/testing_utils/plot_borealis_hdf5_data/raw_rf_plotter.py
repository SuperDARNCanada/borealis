#!/usr/bin/env python3

import sys
import deepdish
import random

from plotting_borealis_data_utils import plot_output_raw_data

filename = sys.argv[1]

data = deepdish.io.load(filename)

record_name = random.choice(list(data.keys()))
# record_name = '1547660180625'
print(record_name)
raw_rf_data = data[record_name]

plot_output_raw_data(raw_rf_data, "raw_rf_data", start_sample=0, end_sample=430000)
