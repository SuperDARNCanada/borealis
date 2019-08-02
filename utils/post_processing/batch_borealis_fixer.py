# Copyright SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

import argparse
import sys
import os
import copy
import subprocess as sp
import numpy as np
import warnings
import tables
from multiprocessing import Pool
import deepdish as dd

from borealis_fixer import file_updater

def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
     
    :returns: the usage message
    """

    usage_message = """ borealis_fixer.py [-h] path_regex fixed_dat_dir
    
    **** NOT TO BE USED IN PRODUCTION ****
    **** USE WITH CAUTION ****

    Batch modify borealis files with updated data fields. Modify the script where
    indicated to update the file. Used in commissioning phase of Borealis when 
    data fields were not finalized."""

    return usage_message


def script_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("path_regex", help="Path regex you want to match. Will"
    	" find the files that match to modify.")
    parser.add_argument("fixed_data_dir", help="Path to place the updated file in.")
    return parser


if __name__ == "__main__":
    parser = script_parser()
    args = parser.parse_args()

    update_pool = Pool(processes=4)

    files_to_update = glob.glob(args.path_regex)
    
    update_pool.map(file_updater, rawacf_hdf5_files)
