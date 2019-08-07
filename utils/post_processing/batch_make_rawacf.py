# Copyright SuperDARN Canada, University of Saskatchewan
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

from make_rawacf import rawacf_processor

def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
     
    :returns: the usage message
    """

    usage_message = """ batch_make_rawacf.py [-h] rawacf_directory path_regex
    
    Process bfiq files to rawacf Borealis files and place in the given directory. If the 
    input file is bzip2 compressed, the output file will also be bzip2 compressed before 
    exiting.
    """

    return usage_message


def script_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("rawacf_directory", nargs=1, help="Path to place the rawacf file in.")
    parser.add_argument("path_regex", nargs='+', help="Path regex you want to match. Will"
        " find the files that match to modify. Alternatively, list files separately and "
        " all listed will be processed. Should all be bfiq.")
    return parser


if __name__ == "__main__":
    parser = script_parser()
    args = parser.parse_args()

    files_to_update = args.path_regex # should be a list
    jobs = []

    rawacf_directory = args.rawacf_directory[0] # only 1 arg
    files_left = True
    filename_index = 0
    num_processes = 5

    while files_left:
        for procnum in range(num_processes):
            try:
                filename = files_to_update[filename_index + procnum]
                print('Rawacf Processing: ' + filename)
            except IndexError:
                if filename_index + procnum == 0:
                    print('No files found to check!')
                    raise
                files_left = False
                break
            p = Process(target=rawacf_processor, args=(filename, rawacf_directory))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        filename_index += num_processes
