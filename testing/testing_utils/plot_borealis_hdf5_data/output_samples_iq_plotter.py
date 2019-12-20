#!/usr/bin/env python3

import sys
import deepdish
import random
import numpy as np
import matplotlib.pyplot as plt
from plotting_borealis_data_utils import plot_output_samples_iq_data

filename = sys.argv[1]

data = deepdish.io.load(filename)

record_name = random.choice(list(data.keys()))
print(record_name)
output_samples_iq = data[record_name]


plot_output_samples_iq_data(output_samples_iq, 'output_samples_iq')