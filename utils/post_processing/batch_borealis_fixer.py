# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

import argparse
import sys
import os
import copy
import glob
import subprocess as sp
import numpy as np
import warnings
import tables
from multiprocessing import Process
import deepdish as dd

from borealis_fixer import file_updater

def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
     
    :returns: the usage message
    """

    usage_message = """ borealis_fixer.py [-h] fixed_dat_dir path_regex 
    
    **** NOT TO BE USED IN PRODUCTION ****
    **** USE WITH CAUTION ****

    Batch modify borealis files with updated data fields. Modify the script where
    indicated to update the file. Used in commissioning phase of Borealis when 
    data fields were not finalized."""

    return usage_message


def script_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("fixed_data_dir", nargs=1, help="Path to place the updated file in.")
    parser.add_argument("path_regex", nargs='+', help="Path regex you want to match. Will"
        " find the files that match to modify. Alternatively, list files separately and "
        " all listed will be processed.")
    return parser


if __name__ == "__main__":
    parser = script_parser()
    args = parser.parse_args()

    files_to_update = args.path_regex # should be a list
    
    jobs = []

    files_left = True
    filename_index = 0
    num_processes = 4

    fixed_data_dir = args.fixed_data_dir[0] # only 1

    while files_left:
        for procnum in range(num_processes):
            try:
                filename = files_to_update[filename_index + procnum]
                print('Fixing: ' + filename)
            except IndexError:
                if filename_index + procnum == 0:
                    print('No files found to check!')
                    raise
                files_left = False
                break
            p = Process(target=file_updater, args=(filename, fixed_data_dir))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        filename_index += num_processes
