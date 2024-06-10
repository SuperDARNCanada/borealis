#!/usr/bin/python3

"""
    local_scd_server.py
    ~~~~~~~~~~~~~~~~~~~
    Monitors for new SWG files and adds the SWG info to the scd if there is an update.

    :copyright: 2022 SuperDARN Canada
"""

import subprocess as sp
import sys
import os
import datetime
import time
import argparse

BOREALISPATH = os.environ["BOREALISPATH"]
sys.path.append(f"{BOREALISPATH}/scheduler")
import scd_utils


SWG_GIT_REPO_DIR = "schedules"
SWG_GIT_REPO = "https://github.com/SuperDARN/schedules.git"

EXPERIMENTS = {
    "sas": {
        "common_time": "twofsound",
        "discretionary_time": "twofsound",
        "htr_common_time": "twofsound",
        "themis_time": "themisscan",
        "special_time_normal": "twofsound",
        "rbsp_time": "rbspscan",
        "no_switching_time": "normalscan",
        "interleaved_time": "interleavedscan",
        "normalsound_time": "normalsound",
    },
    "pgr": {
        "common_time": "twofsound",
        "discretionary_time": "twofsound",
        "htr_common_time": "twofsound",
        "themis_time": "themisscan",
        "special_time_normal": "twofsound",
        "rbsp_time": "rbspscan",
        "no_switching_time": "normalscan",
        "interleaved_time": "interleavedscan",
        "normalsound_time": "normalsound",
    },
    "rkn": {
        "common_time": "twofsound",
        "discretionary_time": "twofsound",
        "htr_common_time": "twofsound",
        "themis_time": "themisscan",
        "special_time_normal": "twofsound",
        "rbsp_time": "rbspscan",
        "no_switching_time": "normalscan",
        "interleaved_time": "interleavedscan",
        "normalsound_time": "normalsound",
    },
    "inv": {
        "common_time": "twofsound",
        "discretionary_time": "twofsound",
        "htr_common_time": "twofsound",
        "themis_time": "themisscan",
        "special_time_normal": "twofsound",
        "rbsp_time": "rbspscan",
        "no_switching_time": "normalscan",
        "interleaved_time": "interleavedscan",
        "normalsound_time": "normalsound",
    },
    "cly": {
        "common_time": "twofsound",
        "discretionary_time": "twofsound",
        "htr_common_time": "twofsound",
        "themis_time": "themisscan",
        "special_time_normal": "twofsound",
        "rbsp_time": "rbspscan",
        "no_switching_time": "normalscan",
        "interleaved_time": "interleavedscan",
        "normalsound_time": "normalsound",
    },
    "lab": {
        "common_time": "twofsound",
        "discretionary_time": "twofsound",
        "htr_common_time": "twofsound",
        "themis_time": "themisscan",
        "special_time_normal": "twofsound",
        "rbsp_time": "rbspscan",
        "no_switching_time": "normalscan",
        "interleaved_time": "interleavedscan",
        "normalsound_time": "normalsound",
    },
}


class SWG(object):
    """Holds the data needed for processing a SWG file."""

    def __init__(self, scd_dir):
        super().__init__()
        self.scd_dir = scd_dir

        # Determine if the git repo for schedules exists, and clone it if it doesn't
        try:
            cmd = f"git -C {self.scd_dir}/{SWG_GIT_REPO_DIR} rev-parse"
            sp.check_output(cmd, shell=True)

        except sp.CalledProcessError:
            cmd = f"cd {self.scd_dir}; git clone {SWG_GIT_REPO}"

            sp.call(cmd, shell=True)

    def new_swg_file_available(self):
        """
        Checks if a new swg file is uploaded via git.

        :returns:   True, if new git update is available.
        :rtype:     bool
        """

        # This command will return the number of new commits available in main. This signals that
        # there are new SWG files available.

        cmd = f"cd {self.scd_dir}/{SWG_GIT_REPO_DIR}; git fetch; git log ..origin/main --oneline | wc -l"
        shell_output = sp.check_output(cmd, shell=True)

        return bool(int(shell_output))

    def pull_new_swg_file(self):
        """Uses git to grab the new scd updates."""
        cmd = f"cd {self.scd_dir}/{SWG_GIT_REPO_DIR}; git pull origin main"
        print("Pulling schedule repository")
        shell_output = sp.check_output(cmd, shell=True)
        print(f"Result: {shell_output}")

    def parse_swg_to_scd(self, modes, radar, first_run):
        """
        Reads the new SWG file and parses into a set of parameters than can be used for borealis
        scheduling.

        :param  modes:      Holds the modes that correspond to the SWG requests.
        :type   modes:      dict
        :param  radar:      Radar acronym.
        :type   radar:      str
        :param  first_run:  Is this the first run? If so - start with current month, otherwise next month.
        :type   first_run:  bool

        :returns:   List of all the parsed parameters.
        :rtype:     list
        """
        print("Parsing schedule files")

        if first_run:
            month_to_use = datetime.datetime.utcnow()
        else:
            month_to_use = scd_utils.get_next_month_from_date()

        year = month_to_use.strftime("%Y")
        month = month_to_use.strftime("%m")
        yearmonth = year + month
        swg_file = f"{self.scd_dir}/{SWG_GIT_REPO_DIR}/{year}/{yearmonth}.swg"
        print(f"Reading schedule file {swg_file}")
        with open(swg_file, "r") as f:
            swg_lines = f.readlines()

        skip_line = False
        parsed_params = []
        for idx, line in enumerate(swg_lines):
            # Init mode_to_use and mode_type to None every loop, so we can error check
            mode_to_use = None
            mode_type = None
            # Skip line is used for special time radar lines
            if skip_line:
                skip_line = False
                continue

            # We've hit the SCD notes and no longer need to parse
            if "# Notes:" in line:
                break

            # Skip only white space lines
            if not line.strip():
                continue

            # Lines starting with '#' or whitespace and '#' are comments
            if line.strip()[0] == "#":
                continue

            # First line is month and year
            if idx == 0:
                continue

            items = line.split()

            start_day = items[0][0:2]
            start_hr = items[0][3:]

            if "Common Time" in line:
                mode_type = "common"
                # 2018 11 23 no longer scheduling twofsound as common time during 'no switching'
                if "no switching" in line:
                    mode_to_use = modes["no_switching_time"]
                elif "normalsound" in line:
                    mode_to_use = modes["normalsound_time"]
                elif "interleavescan" in line:
                    mode_to_use = modes["interleaved_time"]
                else:
                    mode_to_use = modes["htr_common_time"]

            if "Special Time" in line:
                mode_type = "special"
                if "ALL" in line or radar.upper() in line:
                    if "THEMIS" in line:
                        mode_to_use = modes["themis_time"]
                    elif "ST-APOG" in line or "RBSP" in line:
                        mode_to_use = modes["rbsp_time"]
                    elif "ARASE" in line or "interleavescan" in line:
                        mode_to_use = modes["interleaved_time"]
                    elif "normalsound" in line:
                        mode_to_use = modes["normalsound_time"]
                    else:
                        print("Unknown Special Time: using default common time")
                        mode_to_use = modes["htr_common_time"]
                else:
                    mode_to_use = modes["special_time_normal"]

            if "Discretionary Time" in line:
                mode_type = "discretionary"
                mode_to_use = modes["discretionary_time"]

            if not mode_to_use or not mode_type:
                raise ValueError(f"SWG line couldn't be parsed: {line}")
            param = {
                f"yyyymmdd": f"{year}{month}{start_day}",
                f"hhmm": f"{start_hr}:00",
                "experiment": mode_to_use,
                "scheduling_mode": mode_type,
            }
            parsed_params.append(param)
            print(f"Found schedule line: {param}")

        return parsed_params


def main():
    """ """
    parser = argparse.ArgumentParser(
        description="Automatically schedules new events from the SWG"
    )
    parser.add_argument("--scd-dir", required=True, help="The scd working directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force an update to the schedules for the next month",
    )
    parser.add_argument(
        "--first-run",
        action="store_true",
        help="This will generate the first set of schedule files if running on a fresh directory. "
        "If the next month schedule is available, you will need to roll back the SWG schedule "
        "folder back to the last commit before running in continuous operation.",
    )

    args = parser.parse_args()

    scd_dir = args.scd_dir
    scd_logs = scd_dir + "/logs"

    if not os.path.exists(scd_dir):
        os.makedirs(scd_dir)

    if not os.path.exists(scd_logs):
        os.makedirs(scd_logs)

    sites = list(EXPERIMENTS.keys())
    site_scds = [scd_utils.SCDUtils(f"{scd_dir}/{s}.scd", s) for s in sites]
    swg = SWG(scd_dir)

    force_next_month = args.force

    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{current_time} - Starting local_scd_server.py...")

    if args.first_run:
        # Create the .scd files for each site if running for first time
        for s in sites:
            filename = f"{scd_dir}/{s}.scd"
            with open(filename, "a"):
                pass

    while True:
        if swg.new_swg_file_available() or args.first_run or force_next_month:
            current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"{current_time} - New swg file available")

            swg.pull_new_swg_file()

            site_experiments = [
                swg.parse_swg_to_scd(EXPERIMENTS[s], s, args.first_run) for s in sites
            ]

            errors = False
            today = datetime.datetime.utcnow()
            scd_error_log = today.strftime("/scd_errors.%Y%m%d")

            for se, site_scd in zip(site_experiments, site_scds):
                for ex in se:
                    try:
                        print(
                            f"add_line date: {ex['yyyymmdd']}, "
                            f"with experiment: {ex['experiment']}, "
                            f"mode: {ex['scheduling_mode']}"
                        )
                        site_scd.add_line(
                            ex["yyyymmdd"],
                            ex["hhmm"],
                            ex["experiment"],
                            ex["scheduling_mode"],
                        )
                    except ValueError as err:
                        error_msg = (
                            f"{today.strftime('%c')} {site_scd.scd_filename}: Unable to add line:\n"
                            f"\t {ex['yyyymmdd']} {ex['hhmm']} {ex['experiment']} {ex['scheduling_mode']}\n"
                            f"\t Exception thrown:\n"
                            f"\t\t {str(err)}\n"
                        )

                        with open(scd_logs + scd_error_log, "a") as f:
                            f.write(error_msg)

                        errors = True
                        break
                    except FileNotFoundError:
                        error_msg = (
                            f"SCD filename: {site_scd.scd_filename} is missing!!!\n"
                        )

                        with open(scd_logs + scd_error_log, "a") as f:
                            f.write(error_msg)

                        errors = True
                        break

            if not errors:
                success_msg = "All swg lines successfully scheduled.\n"
                for site, site_scd in zip(sites, site_scds):
                    yyyymmdd = today.strftime("%Y%m%d")
                    hhmm = today.strftime("%H:%M")

                    new_lines = site_scd.get_relevant_lines(yyyymmdd, hhmm)

                    text_lines = [site_scd.fmt_line(x) for x in new_lines]

                    success_msg += f"\t{site}\n"
                    for line in text_lines:
                        success_msg += f"\t\t{line}\n"

                with open(scd_logs + scd_error_log, "a") as f:
                    f.write(success_msg)

            if args.first_run:
                break

            force_next_month = False

        else:
            time.sleep(300)


if __name__ == "__main__":
    main()
