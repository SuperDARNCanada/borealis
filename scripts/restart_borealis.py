#!/usr/bin/python

"""
    restart_borealis
    ~~~~~~~~~~~~~~~~

    Python script to check data being written and restart Borealis in case it's not

    :copyright: 2020 SuperDARN Canada
    :author: Kevin Krieger
"""

import argparse
import os
import sys
import json
from datetime import datetime as dt
import glob
import subprocess
import time
from textwrap import indent


def get_args():
    """
    Supports the command-line arguments listed below.
    """
    # Gather the borealis configuration information
    if not os.environ["BOREALISPATH"]:
        raise ValueError("BOREALISPATH env variable not set")
    if not os.environ["RADAR_ID"]:
        raise ValueError("RADAR_ID env variable not set")
    BOREALISPATH = os.environ["BOREALISPATH"]
    RADAR_ID = os.environ["RADAR_ID"]

    # Config file parsing needed for data directory location
    path = f"{BOREALISPATH}/config/{RADAR_ID}/{RADAR_ID}_config.ini"
    try:
        with open(path, "r") as data:
            raw_config = json.load(data)
    except IOError:
        raise (f"IOError on config file at {path}")

    parser = argparse.ArgumentParser(
        description="Python script to check data being written and "
        "restart Borealis in case it's not"
    )
    parser.add_argument(
        "-r",
        "--restart-after-seconds",
        type=int,
        default=300,
        help="How many seconds can the data file be out of date before attempting "
        "to restart the radar? Default 300 seconds (5 minutes)",
    )
    parser.add_argument(
        "-p",
        "--borealis-path",
        required=False,
        dest="borealis_path",
        default=BOREALISPATH,
        help="Path to Borealis directory. Default " "BOREALISPATH environment variable",
    )
    parser.add_argument(
        "-d",
        "--data-directory",
        required=False,
        dest="data_directory",
        default=raw_config["data_directory"],
        help="Path to Borealis data directory. Defaults to data_directory within "
        "config file",
    )
    args = parser.parse_args()
    return args


def main():
    # Handling arguments
    args = get_args()
    restart_after_seconds = args.restart_after_seconds
    borealis_path = args.borealis_path
    data_directory = args.data_directory

    # Get today's date and look for the current data file being written
    today = dt.utcnow().strftime("%Y%m%d")
    today_data_files = glob.glob(f"{data_directory}/{today}/*")
    # If there are no files yet today, then just use the start of the day as the newest file write time
    if len(today_data_files) == 0:
        new_file_write_time = dt.utcnow().replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        new_file_write_time = float(new_file_write_time.strftime("%s"))
    else:
        newest_file = max(today_data_files, key=os.path.getmtime)
        new_file_write_time = os.path.getmtime(newest_file)
    now_utc_seconds = float(dt.utcnow().strftime("%s"))

    # How many seconds ago was the last write to a data file?
    last_data_write = now_utc_seconds - new_file_write_time
    print(
        f"Last write time: {dt.utcfromtimestamp(new_file_write_time).strftime('%Y-%m-%dT%H:%M:%S')}, "
        f"Current time: {dt.utcfromtimestamp(now_utc_seconds).strftime('%Y-%m-%dT%H:%M:%S')}, "
        f"Difference: {last_data_write} s"
    )

    # if under the threshold it is OK, if not then there's a problem
    if float(last_data_write) <= float(restart_after_seconds):
        print(
            f"{last_data_write} s within {restart_after_seconds} s threshold "
            "- no restart neccessary"
        )
        sys.exit(0)
    else:
        print(
            f"{last_data_write} s greater than {restart_after_seconds} s threshold "
            "- attempting to restart Borealis"
        )
        # Now we attempt to restart Borealis
        stop_borealis = subprocess.Popen(
            f"{borealis_path}/scripts/stop_radar.sh",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output, error = stop_borealis.communicate()
        print("Borealis stop_radar.sh called")
        print(indent(output, "    "))
        # Check that the stop_radar.sh script was successful (empty error output means it worked)
        if error:
            print("Error with stop_radar.sh:")
            print(indent(error, "      "))

        time.sleep(1)

        # Now call the start radar script, reads will block, so no need to communicate with
        # this process.
        start_borealis = subprocess.Popen(
            f"{borealis_path}/scripts/start_radar.sh",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        output, error = start_borealis.communicate()
        print("Borealis start_radar.sh called")
        print(indent(output, "    "))
        if error:
            print("Error with start_radar:")
            print(indent(error, "      "))

        sys.exit(1)


if __name__ == "__main__":
    main()
