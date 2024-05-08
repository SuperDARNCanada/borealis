#!/usr/bin/python3

"""
remote_server.py
~~~~~~~~~~~~~~~~
This process runs on the Borealis computer at each radar site. This process should be running in
the background whenever the radar is on, doing the following:

- On start up, it schedules Borealis based on the existing schedule (.scd) file for the respective
  site. This is done using the Linux `at` service and the `atq` command.
- Using inotify, remote_server.py then watches the .scd file for the respective site for any changes.
  If the .scd file is modified, the scheduled Borealis runs are updated.

Logs are printed to stdout. Specific logs for each time the schedule is updated are also created
in borealis_schedules/logs/ and are emailed to verify if any issues occur.

:copyright: 2019 SuperDARN Canada
"""

import inotify.adapters
import os
import datetime
import argparse
import copy
import subprocess as sp
import sys

import scd_utils

sys.path.append(f'{os.environ["BOREALISPATH"]}/src')
from utils.options import Options


def format_to_atq(
    dt, experiment, scheduling_mode, first_event_flag=False, kwargs="", embargo=False
):
    """
    Turns an experiment line from the scd into a formatted atq command.

    :param  dt:                 Datetime of the experiment
    :type   dt:                 datetime
    :param  experiment:         The experiment to run
    :type   experiment:         str
    :param  scheduling_mode:    The scheduling mode to run
    :type   scheduling_mode:    str
    :param  first_event_flag:   Flag to signal whether the experiment is the first to run (Default value = False)
    :type   first_event_flag:   bool
    :param  kwargs:             String of keyword arguments to run steamed hams (Default value = '')
    :type   kwargs:             str
    :param  embargo:            Option to embargo the data (makes the CPID negative)
    :type   embargo:            bool

    :returns:   Formatted atq str.
    :rtype:     str
    """
    borealis_path = os.environ["BOREALISPATH"]

    start_cmd = f"echo 'screen -d -m -S starter {borealis_path}/scripts/steamed_hams.py {experiment} release {scheduling_mode}"
    if embargo:
        start_cmd += f" --embargo"
    if kwargs:
        start_cmd += f" --kwargs {kwargs}"
    start_cmd += "'"  # Terminate the echo string

    if first_event_flag:
        cmd_str = start_cmd + " | at now + 1 minute"
    else:
        cmd_str = start_cmd + " | at -t %Y%m%d%H%M"
    cmd_str = dt.strftime(cmd_str)
    return cmd_str


def convert_scd_to_timeline(scd_lines, time_of_interest):
    """
    Creates a true timeline from the scd lines, accounting for priority and duration of each line.
    Will reorder and add breaks to lines to account for differing priorities and durations. Keep the
    same line format.

    Line dict keys are:
        - timestamp(ms since epoch)
        - time(datetime)
        - duration(minutes)
        - prio(priority)
        - experiment
        - scheduling_mode

    The true timeline queued_lines dictionary differs from the scd_lines list by the following:

        - duration is parsed, adding in events so that all event durations are equal to the next
          event's start time, subtract the current event's start time.
        - priority is parsed so that there is only ever one event at any time (no overlap)
        - therefore the only event in the true timeline with infinite duration is the last event.
        - the keys of the true timeline dict are the original scd_lines order of the lines
          (integer). This allows the preservation of which events in the true timeline were
          scheduled in the same original line. This can be useful for plotting (same color = same
          scd scheduled line). The items in queued_lines dict are lists of all of the events
          corresponding to that original line's order. These events have the same keys as the lines
          in scd_lines.

    :param  scd_lines:          List of sorted lines by timestamp and priority, scd lines to try
                                convert to a timeline.
    :type   scd_lines:          list
    :param  time_of_interest:   The datetime holding the time of scheduling.
    :type   time_of_interest:   Datetime

    :returns:   Tuple containing the following items:

            - queued_lines: Groups of entries belonging to the same experiment.
            - warnings: List of warnings produced by the function
    :rtype:     tuple(list, list)

    """

    inf_dur_line = None
    queued_lines = []

    # Add the ordering of the lines to each line. This is so we can group entries that get split
    # up as the same originally scheduled experiment line in the plot.
    for i, line in enumerate(scd_lines):
        line["order"] = i

    def calculate_new_last_line_params():
        """
        when the last line is of set duration, find its finish time so that
        the infinite duration line can be set to run again at that point.
        """
        last_queued = queued_lines[-1]
        queued_dur_td = datetime.timedelta(minutes=int(last_queued["duration"]))
        queued_finish_time = last_queued["time"] + queued_dur_td
        return last_queued, queued_finish_time

    warnings = []
    last_queued_line = {}
    for idx, scd_line in enumerate(scd_lines):
        # if we have lines queued up, grab the last one to compare to.
        if queued_lines:
            last_queued_line, queued_finish = calculate_new_last_line_params()
        # handling infinite duration lines
        if scd_line["duration"] == "-":
            # if no line set, just assign it
            if not inf_dur_line:
                inf_dur_line = scd_line
            else:
                # case for switching infinite duration lines.
                if int(scd_line["prio"]) >= int(inf_dur_line["prio"]):
                    # if no queued lines yet, just take the time up to the new line and add it,
                    # else check against the last line to see if it comes before or after.
                    if not queued_lines:
                        time_diff = scd_line["time"] - inf_dur_line["time"]
                        inf_dur_line["duration"] = time_diff.total_seconds() // 60
                        queued_lines.append(inf_dur_line)
                    else:
                        duration = int(last_queued_line["duration"])
                        new_time = last_queued_line["time"] + datetime.timedelta(
                            minutes=duration
                        )
                        if scd_line["time"] > new_time:
                            inf_dur_line["time"] = new_time
                            time_diff = scd_line["time"] - new_time
                            inf_dur_line["duration"] = time_diff.total_seconds() // 60
                            queued_lines.append(copy.deepcopy(inf_dur_line))
                    inf_dur_line = scd_line
                else:
                    warning_msg = (
                        f"Unable to schedule {scd_line['experiment']} at {scd_line['time']}."
                        " An infinite duration line with higher priority exists"
                    )
                    warnings.append(warning_msg)

        else:  # line has a set duration
            duration_td = datetime.timedelta(minutes=int(scd_line["duration"]))
            finish_time = scd_line["time"] + duration_td

            if finish_time < time_of_interest:
                continue

            # if no lines added yet, just add the line to the queue. Check if an inf dur line is
            # running.
            if not queued_lines:
                if not inf_dur_line:
                    queued_lines.append(scd_line)
                else:
                    if int(scd_line["prio"]) > int(inf_dur_line["prio"]):
                        if scd_line["time"] > time_of_interest:
                            time_diff = scd_line["time"] - inf_dur_line["time"]
                            new_line = copy.deepcopy(inf_dur_line)
                            new_line["duration"] = time_diff.total_seconds() // 60
                            queued_lines.append(new_line)
                        queued_lines.append(scd_line)

                        finish_time = scd_line["time"] + duration_td
                        inf_dur_line["time"] = finish_time
            else:
                # loop to find where to insert this line. Hold all following lines.
                holder = []
                while scd_line["time"] < queued_lines[-1]["time"]:
                    holder.append(queued_lines.pop())

                last_queued_line, queued_finish = calculate_new_last_line_params()
                holder.append(scd_line)

                # Durations and priorities change when lines can run and sometimes lines get
                # broken up. We continually loop to readjust the timeline to account for the
                # priorities and durations of new lines added.
                while holder:  # or first_time:

                    item_to_add = holder.pop()
                    duration_td = datetime.timedelta(
                        minutes=int(item_to_add["duration"])
                    )
                    finish_time = item_to_add["time"] + duration_td

                    while item_to_add["time"] < queued_lines[-1]["time"]:
                        holder.append(queued_lines.pop())

                    last_queued_line, queued_finish = calculate_new_last_line_params()

                    # if the line comes directly after the last line, we can add it.
                    if item_to_add["time"] == queued_finish:
                        queued_lines.append(item_to_add)
                    # if the time of the line starts before the last line ends, we need to may need
                    # to make adjustments.
                    elif item_to_add["time"] < queued_finish:
                        # if the line finishes before the last line and is a higher priority,
                        # we split the last line and insert the new line.
                        if finish_time < queued_finish:
                            if int(item_to_add["prio"]) > int(last_queued_line["prio"]):
                                queued_copy = copy.deepcopy(last_queued_line)

                                # if time is >, then we can split the first part and insert
                                # the line is. Otherwise, we can directly overwrite the first
                                # part.
                                if item_to_add["time"] > last_queued_line["time"]:
                                    first_dur = (
                                        item_to_add["time"] - last_queued_line["time"]
                                    )
                                    last_queued_line["duration"] = (
                                        first_dur.total_seconds() // 60
                                    )
                                    queued_lines.append(item_to_add)
                                else:
                                    queued_lines.append(item_to_add)

                                remaining_duration = queued_finish - finish_time

                                queued_copy["time"] = finish_time
                                queued_copy["duration"] = (
                                    remaining_duration.total_seconds() // 60
                                )
                                queued_lines.append(queued_copy)

                        else:
                            # if the finish time is > than the last line and the prio is higher,
                            # we can overwrite the last piece, otherwise we directly overwrite the
                            # whole line.
                            if int(item_to_add["prio"]) > int(last_queued_line["prio"]):
                                if item_to_add["time"] > last_queued_line["time"]:
                                    new_duration = (
                                        item_to_add["time"] - last_queued_line["time"]
                                    )
                                    last_queued_line["duration"] = (
                                        new_duration.total_seconds() // 60
                                    )
                                    queued_lines.append(item_to_add)
                                else:
                                    queued_lines.append(item_to_add)
                            else:
                                # if the prio is lower, then we only the schedule the piece that
                                # doesn't overlap.
                                item_to_add["time"] = queued_finish
                                new_duration = finish_time - queued_finish
                                item_to_add["duration"] = (
                                    new_duration.total_seconds() // 60
                                )
                                queued_lines.append(item_to_add)
                    else:
                        time_diff = item_to_add["time"] - queued_finish
                        new_line = copy.deepcopy(inf_dur_line)
                        new_line["duration"] = time_diff.total_seconds() // 60
                        new_line["time"] = queued_finish
                        queued_lines.append(new_line)
                        queued_lines.append(item_to_add)

        if idx == len(scd_lines) - 1:
            if queued_lines:  # infinite duration line starts after the last queued line
                last_queued_line, queued_finish = calculate_new_last_line_params()
                inf_dur_line["time"] = queued_finish
            queued_lines.append(inf_dur_line)

    return queued_lines, warnings


def timeline_to_atq(timeline, scd_dir, time_of_interest, site_id):
    """
    Converts the created timeline to actual atq commands.

    Log and backup the existing atq, remove old events and then schedule everything recent. The
    first entry should be the currently running event, so it gets scheduled immediately. This
    function only backs up the commands that have not run yet.

    :param  timeline:           A list holding all timeline events.
    :type   timeline:           list
    :param  scd_dir:            The directory with SCD files.
    :type   scd_dir:            str
    :param  time_of_interest:   The datetime holding the time of scheduling.
    :type   time_of_interest:   Datetime
    :param  site_id:            Site identifier for logs.
    :type   site_id:            str

    :returns:   output of the executed atq command
    :rtype:     bytes
    """

    # This command is basically: for j in atq job number, print job num, time and command
    # More detail: sort the atq first by year, then month name ('-M flag), then day of month
    # Then hour, minute and second. Finally, just get the atq index (job #) in first column
    # then, iterate through all jobs in the atq, list them to standard output, get the last 2 lines
    get_atq_cmd = (
        "for j in $(atq | sort -k6,6 -k3,3M -k4,4 -k5,5 |cut -f 1); "
        'do atq |grep -P "^$j\t"; at -c "$j" | tail -n 2; done'
    )

    output = sp.check_output(get_atq_cmd, shell=True)

    backup_time_str = time_of_interest.strftime("%Y.%m.%d.%H.%M")
    backup_dir = f"{scd_dir}/atq_backups"

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    backup_file = f"{backup_dir}/{backup_time_str}.{site_id}.atq"

    with open(backup_file, "wb") as f:
        f.write(output)

    clear_command = "for i in `atq | awk '{print $1}'`;do atrm $i;done"

    sp.call(clear_command, shell=True)

    atq = []
    first_event = True
    for event in timeline:
        if first_event:
            atq.append(
                format_to_atq(
                    event["time"],
                    event["experiment"],
                    event["scheduling_mode"],
                    True,
                    event["kwargs"],
                    event["embargo"],
                )
            )
            first_event = False
        else:
            atq.append(
                format_to_atq(
                    event["time"],
                    event["experiment"],
                    event["scheduling_mode"],
                    False,
                    event["kwargs"],
                    event["embargo"],
                )
            )
    for cmd in atq:
        sp.call(cmd, shell=True)

    return sp.check_output(get_atq_cmd, shell=True)


def _main():
    """ """
    parser = argparse.ArgumentParser(
        description="Automatically schedules new SCD file entries"
    )
    parser.add_argument("--scd-dir", required=True, help="The scd working directory")

    args = parser.parse_args()

    scd_dir = args.scd_dir

    inot = inotify.adapters.Inotify()

    options = Options()
    site_id = options.site_id

    scd_file = f"{scd_dir}/{site_id}.scd"

    inot.add_watch(scd_file)
    scd_util = scd_utils.SCDUtils(scd_file, site_id)

    log_dir = f"{scd_dir}/logs"

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    def make_schedule():
        print("Making schedule...")

        time_of_interest = datetime.datetime.utcnow()

        log_time_str = time_of_interest.strftime("%Y.%m.%d.%H.%M")
        log_file = f"{log_dir}/{site_id}_remote_server.{log_time_str}.log"

        log_msg_header = f"Updated at {time_of_interest}\n"
        try:
            yyyymmdd = time_of_interest.strftime("%Y%m%d")
            hhmm = time_of_interest.strftime("%H:%M")
            relevant_lines = scd_util.get_relevant_lines(yyyymmdd, hhmm)
            for line in relevant_lines:
                scd_util.test_line(line)
        except (IndexError, ValueError) as e:
            logtime = time_of_interest.strftime("%c")
            error_msg = f"{logtime}: Unable to make schedule\n\t Exception thrown:\n\t\t {str(e)}\n"
            with open(log_file, "w") as f:
                f.write(log_msg_header)
                f.write(error_msg)

        else:

            timeline, warnings = convert_scd_to_timeline(
                relevant_lines, time_of_interest
            )
            new_atq_str = timeline_to_atq(timeline, scd_dir, time_of_interest, site_id)

            with open(log_file, "wb") as f:
                f.write(log_msg_header.encode())
                f.write(new_atq_str)

                f.write("\n".encode())
                for warning in warnings:
                    f.write(f"\n{warning}".encode())

    start_time = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    print(f"\n{start_time} - Scheduler booted")
    print(f"Inotify monitoring schedule file {scd_file}")

    # Make the schedule on restart of application
    make_schedule()
    new_notify = False
    path = ""
    while True:
        # "IN_IGNORED" was removing watch points and wouldn't monitor the path. This regens it.
        if new_notify:
            inot = inotify.adapters.Inotify()
            inot.add_watch(scd_file)
            new_notify = False

        events = inot.event_gen(yield_nones=False, timeout_s=10)
        events = list(events)

        if events:
            event_types = []
            for event in events:
                (_, type_names, path, filename) = event
                event_types.extend(type_names)

            # File has been copied
            print(f"Events triggered: {event_types}]")
            if site_id in path:
                if all(
                    i in event_types for i in ["IN_OPEN", "IN_ACCESS", "IN_CLOSE_WRITE"]
                ):
                    scd_utils.SCDUtils(path, site_id)
                    make_schedule()

                # Nextcloud/Vim triggers
                if all(
                    i in event_types
                    for i in ["IN_ATTRIB", "IN_DELETE_SELF", "IN_IGNORED"]
                ):
                    scd_utils.SCDUtils(path, site_id)
                    make_schedule()
                    new_notify = True


if __name__ == "__main__":
    _main()
