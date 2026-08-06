"""
make_atq.py
~~~~~~~~~~~
This script takes the Borealis schedule file (ending in .scd) and converts the schedule into
commands that are scheduled using the system ``at`` service.

Logs are printed to stdout. Specific logs for each time the schedule is updated are also created in
``${HOME}/logs/make_atq/``.

This script can be configured to run automatically when the .scd file is modified, using Linux
system services. The following files are required for this:
    - A file called ``.borealis.env`` defining the following variables
        - BOREALISPATH (example: /home/radar/borealis/)
        - RADAR_ID (example: sas)
        - PYTHON_VERSION (example: 3.12)
        - VIRTUAL_ENV (example: /home/radar/borealis/borealis_env3.12)
    - ``monitor-schedule.path`` located in ``/etc/systemd/system/``
    - ``schedule-radar.service`` located in ``/etc/systemd/system/``
"""

import argparse
import datetime as dt
import os
import subprocess as sp

from scd_utils import ScheduleError, SCDUtils


def tee_print(msg: str, logfile):
    """
    Prints to stdout and to a logfile.
    """
    print(msg)
    print(msg, file=logfile)


def backup_atq(time_of_interest: dt.datetime, scd_dir: str, site_id: str):
    """
    Creates a backup of the current ``atq`` schedule, saving it to
    ``[scd_dir]/atq_backup/YYYYmmdd-HHMM.[site_id].atq``.

    :param time_of_interest:    Time that the schedule is being evaluated for.
    :type  time_of_interest:    dt.datetime
    :param scd_dir:             Path to the directory containing schedule files.
    :type  scd_dir:             str
    :param site_id:             Three-letter radar identifier, e.g. "sas"
    :type  site_id:             str
    """
    backup_dir = f"{scd_dir}/atq_backups"
    backup_time_str = time_of_interest.strftime("%Y%m%d-%H%M")

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    backup_file = f"{backup_dir}/{backup_time_str}.{site_id}.atq"
    output = sp.check_output(SCDUtils.get_atq_cmd, shell=True)

    with open(backup_file, "wb") as f:
        f.write(output)


def make_schedule(scd_dir: str, site_id: str):
    """
    Reads the schedule file and uses ``at`` to schedule all relevant lines from the schedule.

    If the schedule is empty or contains invalid lines, it will send a Slack message alerting
    that the operation has failed.

    :param scd_dir: Path to the directory containing schedule files.
    :type  scd_dir: str
    :param site_id: Three-letter radar identifier, e.g. "sas"
    :type  site_id: str
    """

    scd_file = f"{scd_dir}/{site_id}.scd"
    scd_util = SCDUtils(scd_file, site_id)

    time_of_interest = dt.datetime.now(dt.timezone.utc)
    year = time_of_interest.strftime("%Y")
    month = time_of_interest.strftime("%m")

    log_dir = f"{os.environ['HOME']}/logs/make_atq/{year}/{month}/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_time_str = time_of_interest.strftime("%Y%m%d")
    log_file = f"{log_dir}/{log_time_str}.make_atq.log"
    log_msg_header = f"Updated at {time_of_interest}\n"

    start_time = time_of_interest.strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a") as f:
        f.write("#" * 80 + "\n\n")
        tee_print(f"\n{start_time} - Scheduler booted", f)
        tee_print("Making schedule...", f)

    backup_atq(time_of_interest, scd_dir, site_id)

    try:
        new_atq_str = scd_util.parse_and_schedule(time_of_interest)
        with open(log_file, "a") as f:
            f.write(log_msg_header)
            f.write(new_atq_str)
            f.write("\n")
    except ScheduleError as e:
        logtime = time_of_interest.strftime("%c")
        error_msg = (
            f"{logtime}: Unable to make schedule\n\tException thrown:\n\t\t{str(e)}\n"
        )
        with open(log_file, "a") as f:
            f.write(log_msg_header)
            f.write(error_msg)
            tee_print("Failed to make schedule, sending alert to Slack.", f)
        relevant_lines = scd_util.get_relevant_lines(time_of_interest)
        message = f"make_atq @ {site_id}: Failed to schedule {[str(x) for x in relevant_lines]}"
        command = f"""
            curl --silent --header "Content-type: application/json" --data "{{'text':{message}}}" "${{!SLACK_WEBHOOK}}"
        """

        sp.call(command.split(), shell=True)


def parser():
    argparser = argparse.ArgumentParser(description="Schedules new SCD file entries")
    argparser.add_argument("scd_dir", help="The scd working directory")
    return argparser


if __name__ == "__main__":
    args = parser().parse_args()

    scd_dir = args.scd_dir
    site_id = os.environ["RADAR_ID"]

    make_schedule(scd_dir, site_id)
