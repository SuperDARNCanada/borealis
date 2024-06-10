#!/usr/bin/env python3

import os
import sys
import subprocess as sp

directory = sys.argv[1]

for file in os.listdir(directory):
    if file.endswith("bfiq.hdf5"):
        command = (
            "python3 /home/radar/borealis/tools/dsp_testing/test_beamforming.py " + file
        )
        string_output = sp.check_output(command.split()).decode("utf-8")
        print(string_output)
