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

import bisect
import datetime
import inotify.adapters
import os
import datetime as dt
import argparse
import copy
import subprocess as sp
import sys

import scd_utils
from scd_utils import ScheduleLine

sys.path.append(f'{os.environ["BOREALISPATH"]}/src')
from utils.options import Options


def format_to_atq(
    event: ScheduleLine,
    first_event_flag=False,
):
    """
    Turns an experiment line from the scd into a formatted atq command.

    :param  event:              Schedule entry
    :type   event:              ScheduleLine
    :param  first_event_flag:   Flag to signal whether the experiment is the first to run (Default value = False)
    :type   first_event_flag:   bool

    :returns:   Formatted atq str.
    :rtype:     str
    """
    borealis_path = os.environ["BOREALISPATH"]

    start_cmd = f"echo 'screen -d -m -S starter {borealis_path}/scripts/steamed_hams.py {event.experiment} release {event.scheduling_mode}"
    if event.embargo:
        start_cmd += " --embargo"
    if event.rawacf_format is not None:
        start_cmd += f" --rawacf-format {event.rawacf_format}"
    if event.kwargs:
        start_cmd += f" --kwargs {event.kwargs}"
    start_cmd += "'"  # Terminate the echo string

    if first_event_flag:
        cmd_str = start_cmd + " | at now + 1 minute"
    else:
        cmd_str = start_cmd + " | at -t %Y%m%d%H%M"
    cmd_str = event.timestamp.strftime(cmd_str)
    return cmd_str


def resolve_schedule(scd_lines):
    """
    Creates a true timeline from the scd lines, accounting for priority and duration of each line.
    Will reorder and add breaks to lines to account for differing priorities and durations. Keep the
    same line format.

    :param  scd_lines:          List of all lines to schedule
    :type   scd_lines:          list

    :returns: All distinct scheduling events, in chronological order
    :rtype: list

    """

    def reduce_intervals(current_list: list[tuple], value: tuple):
        """
        current_list: (start, end) tuples that are sorted, with no overlaps (i.e. start[k] > end[k-1])
        value: (start, end) tuple
        Finds the new master list of (start, end) tuples with inclusion of value, and the list of (start, end) times
        that value filled in.

        E.g. current_list = [(0, 1), (2, 3), (5, 6)], value = (1, 2), output = [(0, 3), (5, 6)] and [(1, 2)]
        e.g. current_list = [(0, 1), (2, 3), (5, 6)], value = (0, 7), output = [(0, 7)] and [(1, 2), (3, 5), (6, 7)]
        """
        start_times = [x[0] for x in current_list]
        end_times = [x[1] for x in current_list]
        # finds index such that all items <= are to the left
        start_idx = bisect.bisect(start_times, value[0])
        end_idx = bisect.bisect(end_times, value[1])
        if end_idx < start_idx:
            # occurs if value is completely contained by an element of current_list
            # e.g. current_list = [(1, 4)], value = (2, 3), start_idx = 1, end_idx = 0
            return current_list, []

        item_idx = bisect.bisect_left(start_times, value[0])
        reduced_list = current_list[:item_idx] + current_list[end_idx:]
        enclosed_times = current_list[item_idx:end_idx]  # rest of current_list

        # e.g. current_list = [(0, 1), (2, 3), (5, 6)] and value = (1, 2) so item_idx = 1, end_idx = 1
        # then reduced_list = [(0, 1), (5, 6)] and enclosed_times = [(2, 3)]
        reduced_list.insert(item_idx, value)  # now [(0, 1), (1, 2), (5, 6)]
        times_for_value = [value]
        if item_idx > 0 and reduced_list[item_idx - 1][1] >= reduced_list[item_idx][0]:
            # have to combine elements (0, 1) and (1, 2) into (0, 2), so reduced_list = [(0, 2), (2, 3), (5, 6)]
            val = reduced_list.pop(item_idx)
            times_for_value[0] = (
                reduced_list[item_idx - 1][1],
                val[1],
            )  # truncate the start
            reduced_list[item_idx - 1] = (
                reduced_list[item_idx - 1][0],
                times_for_value[0][1],
            )
            item_idx -= 1
        if (
            item_idx < len(reduced_list) - 1
            and reduced_list[item_idx + 1][0] <= value[1]
        ):
            # e.g. reduced_list = [(0, 1), (1, 2)] with item_idx = 0, want result of [(0, 2)]
            val = reduced_list.pop(item_idx + 1)
            times_for_value[0] = (times_for_value[0][0], val[0])  # truncate at the end
            reduced_list[item_idx] = (reduced_list[item_idx][0], val[1])

        # now have to split times_for_value with all the items that were fully enclosed by it
        for x in enclosed_times:
            if x[0] <= times_for_value[-1][1]:
                # block starts before values run ends
                end_val = times_for_value.pop()
                if end_val[0] < x[0]:
                    # Add in the first bit of end_val before this enclosed block starts
                    times_for_value.append((end_val[0], x[0]))
                times_for_value.append((x[0], end_val[1]))
                # add back in the bit between x starting and end_val ending, which will be truncated in the next if statement as required.
            if x[1] <= times_for_value[-1][1]:
                end_val = times_for_value.pop()
            if end_val[1] > x[1]:
                times_for_value.append((x[1], end_val[1]))

        return reduced_list, times_for_value

    sorted_lines = sorted(scd_lines, key=lambda x: x.timestamp)
    sorted_lines.reverse()
    sorted_lines = sorted(key=lambda x: x.priority, reverse=True)
    # at this stage, lines are sorted by priority, then by reverse timestamp for equal priority,
    # then for two lines with equal priority and timestamp, by reverse order in the schedule file.

    scheduled = []  # list of ScheduleLine objects, with no overlap between them
    scheduled_times = []  # complete list of (start, end) times that have an experiment scheduled
    for line in sorted_lines:
        start = line.timestamp
        if line.duration == "-":
            end = datetime.datetime.max
        else:
            end = start + line.duration

        scheduled_times, times_for_line = reduce_intervals(
            scheduled_times, (start, end)
        )
        for block in times_for_line:
            new_line = copy.deepcopy(line)
            new_line.duration = block[1] - block[0]
            new_line.timestamp = block[0]
            scheduled.append(new_line)
    scheduled.sort(key=lambda x: x.timestamp)

    return scheduled


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
        atq.append(
            format_to_atq(
                event,
                first_event,
            )
        )
        first_event = False
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

        time_of_interest = dt.datetime.utcnow()

        log_time_str = time_of_interest.strftime("%Y.%m.%d.%H.%M")
        log_file = f"{log_dir}/{site_id}_remote_server.{log_time_str}.log"

        log_msg_header = f"Updated at {time_of_interest}\n"

        yyyymmdd = time_of_interest.strftime("%Y%m%d")
        hhmm = time_of_interest.strftime("%H:%M")
        relevant_lines = scd_util.get_relevant_lines(yyyymmdd, hhmm)
        try:
            i = 0
            for i, line in enumerate(relevant_lines):
                line.test(site_id)
        except (IndexError, ValueError) as e:
            logtime = time_of_interest.strftime("%c")
            error_msg = f"{logtime}: Unable to make schedule\n\t Exception thrown:\n\t\t {str(e)}\n"
            with open(log_file, "w") as f:
                f.write(log_msg_header)
                f.write(error_msg)
            message = f"remote_server @ {site_id}: Failed to schedule {str(relevant_lines[i])}"
            command = f"""curl --silent --header "Content-type: application/json" --data "{{'text':{message}}}" "${{!SLACK_WEBHOOK_{site_id.upper()}}}" """
            sp.call(command.split(), shell=True)

        else:
            timeline = resolve_schedule(relevant_lines)
            new_atq_str = timeline_to_atq(timeline, scd_dir, time_of_interest, site_id)

            with open(log_file, "wb") as f:
                f.write(log_msg_header.encode())
                f.write(new_atq_str)

                f.write("\n".encode())

    start_time = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
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
