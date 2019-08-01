#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
#
# local_scd_server.py
# 2019-04-18
# Moniters for new SWG files and adds the SWG info to the scd if there is an update.
#
import subprocess as sp
import scd_utils
import email_utils
import os
import datetime
import time
import argparse

SWG_GIT_REPO_DIR = 'schedules'
SWG_GIT_REPO = "https://github.com/SuperDARN/schedules.git"

EXPERIMENTS = {
    "sas" : {
              "common_time" : "twofsound",
              "discretionary_time" : "twofsound",
              "htr_common_time" : "twofsound",
              "themis_time" : "themisscan",
              "special_time_normal" : "twofsound",
              "rbsp_time" : "rbspscan",
              "no_switching_time" : "normalscan",
              "interleaved_time" : "interleavedscan"
    },
    "pgr" : {
              "common_time" : "twofsound",
              "discretionary_time" : "twofsound",
              "htr_common_time" : " twofsound",
              "themis_time" : "themisscan",
              "special_time_normal" : "normalscan",
              "rbsp_time" : "rbspscan",
              "no_switching_time" : "normalscan",
              "interleaved_time" : "interleavedscan"
    },
    "rkn" : {
              "common_time" : "twofsound",
              "discretionary_time" : "twofsound",
              "htr_common_time" : "twofsound",
              "themis_time" : "themisscan",
              "special_time_normal" : "twofsound",
              "rbsp_time" : "rbspscan",
              "no_switching_time" : "normalscan",
              "interleaved_time" : "interleavedscan"
    },
    "inv" : {
              "common_time" : "twofsound",
              "discretionary_time" : "twofsound",
              "htr_common_time" : "twofsound",
              "themis_time" : "themisscan",
              "special_time_normal" : "twofsound",
              "rbsp_time" : "rbspscan",
              "no_switching_time" : "normalscan",
              "interleaved_time" : "interleavedscan"
    },
    "cly" : {
              "common_time" : "twofsound",
              "discretionary_time" : "twofsound",
              "htr_common_time" : "twofsound",
              "themis_time" : "themisscan",
              "special_time_normal" : "twofsound",
              "rbsp_time" : "rbspscan",
              "no_switching_time" : "normalscan",
              "interleaved_time" : "interleavedscan"
    }
}
class SWG(object):
    """Holds the data needed for processing a SWG file.

    Attributes:
        scd_dir (str): Path to the SCD files dir.
    """
    def __init__(self, scd_dir):
        super(SWG, self).__init__()
        self.scd_dir = scd_dir

        try:
            cmd = "git -C {}/{} rev-parse".format(self.scd_dir, SWG_GIT_REPO_DIR)
            sp.check_output(cmd, shell=True)
        except sp.CalledProcessError as e:
            cmd = 'cd {}; git clone {}'.format(self.scd_dir, SWG_GIT_REPO)
            sp.call(cmd, shell=True)


    def new_swg_file_available(self):
        """Checks if a new swg file is uploaded via git.

        Returns:
            TYPE: True, if new git update is available.
        """

        # This command will return the number of new commits available in master. This signals that
        # there are new SWG files available.
        cmd = "cd {}/{}; git log ..origin/master --oneline | wc -l".format(self.scd_dir,
                                                                        SWG_GIT_REPO_DIR)
        shell_output = sp.check_output(cmd, shell=True)

        return bool(int(shell_output))

    def pull_new_swg_file(self):
        """Uses git to grab the new scd updates.

        """
        cmd = "cd {}/{}; git pull origin master".format(self.scd_dir, SWG_GIT_REPO_DIR)
        sp.call(cmd, shell=True)

    def get_next_month(self):
        """Finds the datetime of the next month.

        Returns:
            TYPE: datetime object.
        """
        today = datetime.datetime.utcnow()

        counter = 1
        new_date = today + datetime.timedelta(days=counter)
        while new_date.month == today.month:
            counter += 1
            new_date = today + datetime.timedelta(days=counter)

        return new_date

    def parse_swg_to_scd(self, modes, radar, first_run):
        """Reads the new SWG file and parses into a set of parameters than can be used for borealis
        scheduling.

        Args:
            modes (Dict): Holds the modes that correspond to the SWG requests.
            radar (String): Radar acronym.

        Returns:
            TYPE: List of all the parsed parameters.
        """


        if first_run:
            month_to_use = datetime.datetime.utcnow()
        else:
            month_to_use = next_month = self.get_next_month()

        year = month_to_use.strftime("%Y")
        month = month_to_use.strftime("%m")

        swg_file = "{scd_dir}/{swg_dir}/{yyyy}/{yyyymm}.swg".format(scd_dir=self.scd_dir,
                                                                    swg_dir=SWG_GIT_REPO_DIR,
                                                                    yyyy=year,
                                                                    yyyymm=year+month)

        with open(swg_file, 'r') as f:
            swg_lines = f.readlines()

        skip_line = False
        parsed_params = []
        for idx,line in enumerate(swg_lines):

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

            #Lines starting with '#' are comments
            if line[0] == "#":
                continue

            items = line.split()

            #First line is month and year
            if idx == 0:
                continue

            start_day = items[0][0:2]
            start_hr = items[0][3:]
            end_day = items[1][0:2]
            end_hr = items[1][3:]

            if "Common Time" in line:

                # 2018 11 23 no longer scheduling twofsound as common time.
                if "no switching" in line:
                    mode_to_use = modes["no_switching_time"]
                else:
                    mode_to_use = modes["htr_common_time"]

            if "Special Time" in line:

                if "THEMIS" in line:
                        mode_to_use = modes["themis_time"]

                if "ST-APOG" in line:
                    if radar.upper() in swg_lines[idx+1]:
                        mode_to_use = modes["rbsp_time"]
                    else:
                        mode_to_use = modes["special_time_normal"]

                if "ARASE" in line:
                    if radar.upper() in swg_lines[idx+1]:
                        if "themis" in swg_lines[idx+1]:
                            mode_to_use = modes["themis_time"]
                        if "interleaved" in swg_lines[idx+1]:
                            mode_to_use = modes["interleaved_time"]
                    else:
                        mode_to_use = modes["special_time_normal"]


                else:
                    print("Unknown Special Time: using default common time")
                    mode_to_use = modes["htr_common_time"]

                # Skip next line
                skip_line = True

            if "Discretionary Time" in line:
                mode_to_use = modes["discretionary_time"]

            param = {"yyyymmdd": "{}{}{}".format(year, month, start_day),
                     "hhmm" : "{}:00".format(start_hr),
                     "experiment" : mode_to_use}
            parsed_params.append(param)

        return parsed_params



def main():
    parser = argparse.ArgumentParser(description="Automatically schedules new events from the SWG")
    parser.add_argument('--emails-filepath',required=True, help='A list of emails to send logs to')
    parser.add_argument('--scd-dir', required=True, help='The scd working directory')
    parser.add_argument('--first-run', action="store_true", help='This will generate the first set'
                                                                 ' of schedule files if running on'
                                                                 ' a fresh directory. If the next'
                                                                 ' month schedule is available,'
                                                                 ' you will need to roll back the'
                                                                 ' SWG schedule folder back to the'
                                                                 ' last commit before running in'
                                                                 ' continuous operation.')
    args = parser.parse_args()

    scd_dir = args.scd_dir
    scd_logs = scd_dir + "/logs"

    emailer = email_utils.Emailer(args.emails_filepath)

    if not os.path.exists(scd_dir):
        os.makedirs(scd_dir)

    if not os.path.exists(scd_logs):
        os.makedirs(scd_logs)

    sites = list(EXPERIMENTS.keys())
    site_scds = [scd_utils.SCDUtils("{}.scd".format(s)) for s in sites]
    swg = SWG(scd_dir)

    while True:
        if swg.new_swg_file_available() or args.first_run:
            swg.pull_new_swg_file()

            site_experiments = [swg.parse_swg_to_scd(EXPERIMENTS[s], s, args.first_run)
                                for s in sites]

            errors = False
            today = datetime.datetime.utcnow()
            scd_error_log = "/scd_errors.{}{}{}".format(today.year, today.month, today.day)
            for se, site_scd in zip(site_experiments, site_scds):
                for ex in se:
                    try:
                        site_scd.add_line(ex['yyyymmdd'], ex['hhmm'], ex['experiment'])
                    except ValueError as e:
                        error_msg = ("{logtime}: Unable to add line with parameters:\n"
                                     "\t {date} {time} {experiment}\n"
                                     "\t Exception thrown:\n"
                                     "\t\t {exception}\n")
                        error_msg = error_msg.format(logtime = today.strftime("%c"),
                                                        date=ex['yyyymmdd'],
                                                        time=ex['hhmm'],
                                                        experiment=ex['experiment'],
                                                        exception=str(e))

                        with open(scd_logs + scd_error_log, 'a') as f:
                            f.write(error_msg)

                        errors = True
                    except FileNotFoundError as e:
                        error_msg = "SCD filename: {} is missing!!!".format(site_scd.scd_filename)
                        with open(scd_logs + scd_error_log, 'a') as f:
                            f.write(error_msg)

                        errors = True


            subject = "Scheduling report for swg lines"
            if not errors:
                success_msg = "All swg lines successfully scheduled.\n"
                for site, site_scd in zip(sites, site_scds):
                    yyyymmdd = today.strftime("%Y%m%d")
                    hhmm = today.strftime("%H:%M")

                    new_lines = site_scd.get_relevant_lines(yyyymmdd, hhmm)

                    text_lines = [site_scd.fmt_line(x) for x in new_lines]

                    success_msg += "\t{}\n".format(site)
                    for line in text_lines:
                        success_msg += "\t\t{}\n".format(line)

                with open(scd_logs + scd_error_log, 'a') as f:
                        f.write(success_msg)
            else:
                errors = False

            emailer.email_log(subject, scd_logs + scd_error_log)

            if args.first_run:
                break;

        else:
            time.sleep(300)


if __name__ == '__main__':
    main()
