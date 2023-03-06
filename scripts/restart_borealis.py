#!/usr/bin/python
"""
Python script to check data being written and restart Borealis in case it's not

Classes
-------

Methods
-------


References
----------

    :copyright: 2020 SuperDARN Canada
    :author: Kevin Krieger
"""

import argparse
import os
import sys
from .general import load_config
from datetime import datetime as dt
import glob
import subprocess
import time


def get_args():
    """
    Supports the command-line arguments listed below.
    """
    parser = argparse.ArgumentParser(description="Borealis Check")
    parser.add_argument('-r', '--restart-after-seconds', type=int, default=300,
                        help='How many seconds can the data file be out of date before attempting '
                             'to restart the radar? Default 300 seconds (5 minutes)')
    parser.add_argument('-p', '--borealis-path', required=False, help='Path to Borealis directory',
                        dest='borealis_path', default='/home/radar/borealis/')
    args = parser.parse_args()
    return args


def main():
    # Handling arguments
    args = get_args()
    restart_after_seconds = args.restart_after_seconds
    borealis_path = args.borealis_path

    # Gather the borealis configuration information
    raw_config = load_config()

    data_directory = raw_config["data_directory"]

    # Get today's date and look for the current data file being written
    today = dt.utcnow().strftime("%Y%m%d")
    today_data_files = glob.glob(f"{data_directory}/{today}/*")
    # If there are no files yet today, then just use the start of the day as the newest file write time
    if len(today_data_files) == 0:
        new_file_write_time = dt.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        new_file_write_time = float(new_file_write_time.strftime("%s"))
    else:
        newest_file = max(today_data_files, key=os.path.getmtime)
        new_file_write_time = os.path.getmtime(newest_file)
    now_utc_seconds = float(dt.utcnow().strftime("%s"))

    # How many seconds ago was the last write to a data file?
    last_data_write = now_utc_seconds - new_file_write_time
    print('Write: {}, Now: {}, Diff: {} s' 
          ''.format(dt.utcfromtimestamp(new_file_write_time).strftime('%Y%m%d.%H%M:%S'), 
                    dt.utcfromtimestamp(now_utc_seconds).strftime('%Y%m%d.%H%M:%S'),
                    last_data_write))

    # if under the threshold it is OK, if not then there's a problem
    print(f"{last_data_write} seconds since last write")
    if float(last_data_write) <= float(restart_after_seconds):
        sys.exit(0)
    else:
        # Now we attempt to restart Borealis
        stop_borealis = subprocess.Popen(f"{borealis_path}/borealis/scripts/stop_radar.sh",
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = stop_borealis.communicate()
        # Check out the output to make sure it's all good (empty output means it's all good)
        if error:
            print(f'Attempting to restart Borealis: {error}')

        time.sleep(5)

        # Now call the start radar script, reads will block, so no need to communicate with
        # this process.
        start_borealis = subprocess.Popen(f"{borealis_path}/borealis/scripts/start_radar.sh",
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print('Borealis stop_radar.sh and start_radar.sh called')
        sys.exit(0)


if __name__ == '__main__':
    main()

