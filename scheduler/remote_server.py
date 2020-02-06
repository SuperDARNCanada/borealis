"""
remote_server.py
Copyright SuperDARN Canada 2019

Using inotify to determine if changes to the SCD file are made, this script will automatically
parse new changes and update the schedule via Linux's atq. Plots and logs are produced to verify
if any issues occur.
"""

import inotify.adapters
import time
import scd_utils
import email_utils
import os
import datetime
import argparse
import collections
import math
import copy
import random
import subprocess as sp
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import remote_server_options as rso

def format_to_atq(dt, experiment, first_event_flag=False):
    """Turns an experiment line from the scd into a formatted atq command.

    Args:
        dt (datetime): Datetime of the experiment
        experiment (str): The experiment to run
        first_event_flag (bool, optional): Flag to signal whether the experiment is the first to
        run

    Returns:
        str: Formatted atq str.
    """

    start_cmd = "echo 'screen -d -m -S starter {borealis_path}/steamed_hams.py {experiment} release'"
    start_cmd = start_cmd.format(borealis_path=os.environ['BOREALISPATH'],experiment=experiment)
    if first_event_flag:
        cmd_str = start_cmd + " | at now + 1 minute"
    else:
        cmd_str = start_cmd + " | at -t %Y%m%d%H%M"

    cmd_str = dt.strftime(cmd_str)
    return cmd_str

def get_next_month_from_date(date):
        """Finds the datetime of the next month.

        Returns:
            datetime: datetime object of the next month.
        """

        counter = 1
        new_date = date + datetime.timedelta(days=counter)
        while new_date.month == date.month:
            counter += 1
            new_date = date + datetime.timedelta(days=counter)

        return new_date

def timeline_to_dict(timeline):
    """
    Converts the timeline list to an ordered dict for scheduling and
    colour mapping
    Returns:
        OrderedDict: an ordered dict containing the timeline
    """
    timeline_dict = collections.OrderedDict()
    for line in timeline:
        if not line['order'] in timeline_dict:
            timeline_dict[line['order']] = []

        timeline_dict[line['order']].append(line)
    return timeline_dict

def plot_timeline(timeline, scd_dir, time_of_interest, site_id):
    """Plots the timeline to better visualize runtime.

    Args:
        timeline (list): A list of entries ordered chronologically as scheduled
        scd_dir (str): The scd directory path.
        time_of_interest (datetime): The datetime holding the time of scheduling.
        site_id (str): Site identifier for logs.

    Returns:
        (str, str): Paths to the saved plots.
    """
    fig, ax = plt.subplots()

    first_date, last_date = None, None

    timeline_list = [0] * len(timeline)

    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    def split_event(event):
        """
        Recursively splits an event that runs during two or more days into two events
        in order to handle plotting.
        """
        if event['start'].day == event['end'].day:
            return [event]
        else:
            new_event = dict()
            new_event['color'] = event['color']
            new_event['label'] = event['label']

            td = datetime.timedelta(days=1)
            midnight = datetime.datetime.combine(event['start'] + td, datetime.datetime.min.time())

            first_dur = midnight - event['start']
            second_dur = event['end'] - midnight

            # handle the new event first
            new_event['start'] = midnight
            new_event['duration'] = second_dur
            new_event['end'] = event['end']

            # now handle the old event
            event['duration'] = first_dur
            event['end'] = midnight

            return [event] + split_event(new_event)


    # make random colors
    timeline_dict = timeline_to_dict(timeline)
    cmap = get_cmap(len(timeline_dict.items()))
    colors = [cmap(i) for i in range(len(timeline_dict.items()))]
    random.shuffle(colors)

    # put the colors in and create event records with only start and end times,
    # durations, colors, and labels
    for i, (_, events) in enumerate(timeline_dict.items()):
        for event in events:
            event['color'] = colors[i]

    for i, event in enumerate(timeline):
        event_item = dict()

        event_item['color'] = event['color']
        event_item['label'] = event['experiment']

        event_item['start'] = event['time']

        if event['duration'] == '-':
            td = get_next_month_from_date(event['time']) - event['time']
        else:
            td = datetime.timedelta(minutes=int(event['duration']))

        event_item['end'] = event_item['start'] + td
        event_item['duration'] = td

        timeline_list[i] = event_item

        if i == 0:
            first_date = event['time'].date()
        if i == len(timeline_list) - 1:
            last_date = event['time'] + td
            plot_last = (last_date + datetime.timedelta(hours=12)).date()

    event_list = []
    # loop through events again, splitting them where necessary
    for event in timeline_list:
        event_list += split_event(event)

    for event in event_list:
        day_offset = event['start'].date() - first_date
        start = event['start'] - day_offset
        ax.barh(event['start'].date(), event['duration'], 0.16, left=start, color=event['color'], align='edge')
        ax.text(x=(start + (event['duration'] / 2)), y=event['start'].date(), s=event['label'], color='k', rotation=45, ha='right', va='top', fontsize=8)


    hours = mdates.HourLocator(byhour=[0,6,12,18,24])
    days = mdates.DayLocator()
    minutes = mdates.MinuteLocator()
    x_fmt = mdates.DateFormatter('%H:%M')
    y_fmt = mdates.DateFormatter('%m-%d')

    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(x_fmt)
    plt.xticks(rotation=45)
    ax.set_xlabel('Time of Day', fontsize=12)

    ax.yaxis.set_major_locator(days)
    ax.yaxis.set_major_formatter(y_fmt)
    ax.yaxis.set_minor_locator(hours)
    ax.set_ylim(first_date, plot_last)
    ax.set_ylabel('Date, MM-DD', rotation='vertical', fontsize=12)

    ax.set_title('Schedule from {} to {}'.format(first_date, last_date.date()) )


    pretty_date_str = time_of_interest.strftime("%Y-%m-%d")
    pretty_time_str = time_of_interest.strftime("%H:%M")

    ax.annotate('Generated on {} at {} UTC'.format(pretty_date_str, pretty_time_str),
                                                xy=(1,1), xycoords='axes fraction', fontsize=12, ha='right', va='top')

    plot_time_str = time_of_interest.strftime("%Y.%m.%d.%H.%M")
    plot_dir = "{}/timeline_plots".format(scd_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_file = "{}/{}.{}.png".format(plot_dir, site_id, plot_time_str)
    fig.set_size_inches(14,8)
    fig.savefig(plot_file, dpi=80)

    pkl_file = "{}/{}.{}.pickle".format(plot_dir, site_id, plot_time_str)
    pkl.dump(fig, open(pkl_file, 'wb'))

    return (plot_file, pkl_file)





def convert_scd_to_timeline(scd_lines, time_of_interest):
    """ Creates a true timeline from the scd lines, accounting for priority and duration of each
    line. Will reorder and add breaks to lines to account for differing priorities and durations.
    Keep the same line format.

    Line dict keys are:
        timestamp(ms since epoch)
        time(datetime)
        duration(minutes)
        prio(priority)
        experiment

    The true timeline queued_lines dictionary differs from the scd_lines list by the following:
        - duration is parsed, adding in events so that all event durations are equal to the next event's
        start time, subtract the current event's start time.
        - priority is parsed so that there is only ever one event at any time (no overlap)
        - therefore the only event in the true timeline with infinite duration is the last event.
        - the keys of the true timeline dict are the original scd_lines order of the lines (integer). This allows
        the preservation of which events in the true timeline were scheduled in the same original line.
        This can be useful for plotting (same color = same scd scheduled line). The items in queued_lines
        dict are lists of all of the events corresponding to that original line's order. These events have the same
        keys as the lines in scd_lines.

    Args:
        scd_lines (list): List of sorted lines by timestamp and priority,
                          scd lines to try convert to a timeline.
        time_of_interest (datetime): The datetime holding the time of scheduling.

    Returns:
        queued_lines (dict): Groups of entries belonging to the same experiment.
        warnings (list):     Warning strings.
    """

    inf_dur_line = None
    queued_lines = []

    # Add the ordering of the lines to each line. This is so we can group entries that get split
    # up as the same originally scheduled experiment line in the plot.
    for i, line in enumerate(scd_lines):
        line['order'] = i

    def calculate_new_last_line_params():
        last_queued_line = queued_lines[-1]
        queued_dur_td = datetime.timedelta(minutes=int(last_queued_line['duration']))
        queued_finish = last_queued_line['time'] + queued_dur_td
        return last_queued_line, queued_finish

    warnings = []
    for idx, scd_line in enumerate(scd_lines):
        # if we have lines queued up, grab the last one to compare to.
        if queued_lines:
            last_queued_line, queued_finish = calculate_new_last_line_params()

        # handling infinite duration lines
        if scd_line['duration'] == '-':
            # if no line set, just assign it
            if not inf_dur_line:
                inf_dur_line = scd_line
            else:
                # case for switching infinite duration lines.
                if int(scd_line['prio']) >= int(inf_dur_line['prio']):
                    # if no queued lines yet, just take the time up to the new line and add it,
                    # else check against the last line to see if it comes before or after.
                    if not queued_lines:
                        time_diff = scd_line['time'] - inf_dur_line['time']
                        inf_dur_line['duration'] = time_diff.total_seconds()//60
                        queued_lines.append(inf_dur_line)
                    else:
                        duration = int(last_queued_line['duration'])
                        new_time = last_queued_line['time'] + datetime.timedelta(minutes=duration)
                        if scd_line['time'] > new_time:
                            inf_dur_line['time'] = new_time
                            time_diff = scd_line['time'] - new_time
                            inf_dur_line['duration'] = time_diff.total_seconds()//60
                            queued_lines.append(copy.deepcopy(inf_dur_line))
                    inf_dur_line = scd_line
                else:
                    warning_msg = "Unable to schedule {exp} at {dt}. A infinite duration line "\
                                  "with higher priority exists".format(exp=scd_line['experiment'],
                                                                        dt=scd_line['time'])
                    warnings.append(warnings)

        else:
            duration_td = datetime.timedelta(minutes=int(scd_line['duration']))
            finish_time = scd_line['time'] + duration_td

            if finish_time < time_of_interest:
                continue

            # if no lines added yet, just add the line to the queue. Check if an inf dur line is
            # running.
            if not queued_lines:
                if not inf_dur_line:
                    queued_lines.append(scd_line)
                else:
                    if int(scd_line['prio']) > int(inf_dur_line['prio']):
                        time_diff = scd_line['time'] - inf_dur_line['time']
                        new_line = copy.deepcopy(inf_dur_line)
                        new_line['duration'] = time_diff.total_seconds()//60
                        queued_lines.append(new_line)
                        queued_lines.append(scd_line)

                        finish_time = scd_line['time'] + duration_td
                        inf_dur_line['time'] = finish_time
            else:
                # loop to find where to insert this line. Hold all following lines.
                holder = []
                while scd_line['time'] < queued_lines[-1]['time']:
                    holder.append(queued_lines.pop())

                last_queued_line, queued_finish = calculate_new_last_line_params()
                holder.append(scd_line)

                # Durations and priorities change when lines can run and sometimes lines get
                # broken up. We continually loop to readjust the timeline to account for the
                # priorities and durations of new lines added.
                while holder:# or first_time:

                    item_to_add = holder.pop()
                    duration_td = datetime.timedelta(minutes=int(item_to_add['duration']))
                    finish_time = item_to_add['time'] + duration_td

                    while item_to_add['time'] < queued_lines[-1]['time']:
                        holder.append(queued_lines.pop())

                    last_queued_line, queued_finish = calculate_new_last_line_params()

                    # if the line comes directly after the last line, we can add it.
                    if item_to_add['time'] == queued_finish:
                        queued_lines.append(item_to_add)
                    # if the time of the line starts before the last line ends, we need to may need
                    # to make adjustments.
                    elif item_to_add['time'] < queued_finish:
                        # if the line finishes before the last line and is a higher priority,
                        # we split the last line up and insert the new line.
                        if finish_time < queued_finish:
                            if int(item_to_add['prio']) > int(last_queued_line['prio']):
                                queued_copy = copy.deepcopy(last_queued_line)

                                # if time is >, then we can split the first part and insert
                                # the line is. Otherwise, we can directly overwrite the first
                                # part.
                                if item_to_add['time'] > last_queued_line['time']:
                                    first_dur = item_to_add['time'] - last_queued_line['time']
                                    last_queued_line['duration'] = first_dur.total_seconds()//60
                                    queued_lines.append(item_to_add)
                                else:
                                    queued_lines.append(item_to_add)

                                remaining_duration = queued_finish - finish_time

                                queued_copy['time'] = finish_time
                                queued_copy['duration'] = remaining_duration.total_seconds()//60
                                queued_lines.append(queued_copy)

                        else:
                            # if the finish time is > than the last line and the prio is higher,
                            # we can overwrite the last piece, otherwise we directly overwrite the
                            # whole line.
                            if int(item_to_add['prio']) > int(last_queued_line['prio']):
                                if item_to_add['time'] > last_queued_line['time']:
                                    new_duration = item_to_add['time'] - last_queued_line['time']
                                    last_queued_line['duration'] = new_duration.total_seconds()//60
                                    queued_lines.append(item_to_add)
                                else:
                                    queued_lines.append(item_to_add)
                            else:
                                # if the prio is lower, then we only the schedule the piece that
                                # doesn't overlap.
                                item_to_add['time'] = queued_finish
                                new_duration = finish_time - queued_finish
                                item_to_add['duration'] = new_duration.total_seconds()//60
                                queued_lines.append(item_to_add)
                    else:
                        time_diff = item_to_add['time'] - queued_finish
                        new_line = copy.deepcopy(inf_dur_line)
                        new_line['duration'] = time_diff.total_seconds()//60
                        new_line['time'] = queued_finish
                        queued_lines.append(new_line)
                        queued_lines.append(item_to_add)

        if idx == len(scd_lines) - 1:
            last_queued_line, queued_finish = calculate_new_last_line_params()

            inf_dur_line['time'] = queued_finish
            queued_lines.append(inf_dur_line)

    return queued_lines, warnings

def timeline_to_atq(timeline, scd_dir, time_of_interest, site_id):
    """ Converts the created timeline to actual atq commands.

    Args:
        timeline (List): A list holding all timeline events.
        scd_dir (str): The directory with SCD files.
        time_of_interest (datetime): The datetime holding the time of scheduling.
        site_id (str): Site identifier for logs.

    Log and backup the existing atq, remove old events and then schedule everything recent. The
    first entry should be the currently running event, so it gets scheduled immediately. This
    function only backs up the commands that have not run yet.
    """

    # This command is basically: for j in atq job number, print job num, time and command
    get_atq_cmd = 'for j in $(atq | sort -k6,6 -k3,3M -k4,4 -k5,5 |cut -f 1);'\
    'do atq |grep -P "^$j\t"; at -c "$j" | tail -n 2; done'

    output = sp.check_output(get_atq_cmd, shell=True)

    backup_time_str = time_of_interest.strftime("%Y.%m.%d.%H.%M")
    backup_dir = "{}/atq_backups".format(scd_dir)

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    backup_file = "{}/{}.{}.atq".format(backup_dir, backup_time_str, site_id)

    with open(backup_file, 'wb') as f:
        f.write(output)

    clear_command = "for i in `atq | awk '{print $1}'`;do atrm $i;done"

    sp.call(clear_command, shell=True)

    atq = []
    first_event = True
    for event in timeline:
        if first_event:
            atq.append(format_to_atq(event['time'], event['experiment'], True))
            first_event = False
        else:
            atq.append(format_to_atq(event['time'], event['experiment']))
    for cmd in atq:
        sp.call(cmd, shell=True)

    return sp.check_output(get_atq_cmd, shell=True)

def get_relevant_lines(scd_util, time_of_interest):
    """
    @brief      Gets the relevant lines.

    @param      scd_util  The scd utility object that holds the scd lines.

    @return     The relevant lines.

    Does a search for relevant lines. If the first line returned isnt an infinite duration, we need
    to look back until we find an infinite duration line, as that should be the last line to
    continue running if we need it.
    """

    found = False
    time = time_of_interest

    yyyymmdd = time_of_interest.strftime("%Y%m%d")
    hhmm = time_of_interest.strftime("%H:%M")


    relevant_lines = scd_util.get_relevant_lines(yyyymmdd, hhmm)
    while not found:

        if not relevant_lines:
            msg = "Error in schedule: Could not find any relevant_lines. Either no lines exist " \
            "or an infinite duration line could not be found."

            raise ValueError(msg)

        lines_dict = {}
        for line in relevant_lines:
            if line['timestamp'] not in lines_dict:
                lines_dict[line['timestamp']] = []
            lines_dict[line['timestamp']].append(line)

        for line in lines_dict[list(lines_dict)[0]]:
            if line['duration'] == '-':
                found = True

        if found != True:
            time -= datetime.timedelta(days=1)

            yyyymmdd = time.strftime("%Y%m%d")
            hhmm = time.strftime("%H:%M")

            new_relevant_lines = scd_util.get_relevant_lines(yyyymmdd, hhmm)
            if not new_relevant_lines:
                msg = "Error in schedule: Could not find any relevant_lines. Either no lines exist "\
                "or an infinite duration line could not be found."

                raise ValueError(msg)

            lines_diff = [d for d in new_relevant_lines if d not in relevant_lines]

            for line in reversed(lines_diff):
                if line['duration'] == '-':
                    relevant_lines.insert(0, line)
                    found = True

    return relevant_lines


def _main():
    parser = argparse.ArgumentParser(description="Automatically schedules new SCD file entries")
    parser.add_argument('--emails-filepath',required=True, help='A list of emails to send logs to')
    parser.add_argument('--scd-dir', required=True, help='The scd working directory')

    args = parser.parse_args()

    scd_dir = args.scd_dir

    emailer = email_utils.Emailer(args.emails_filepath)

    inot = inotify.adapters.Inotify()

    options = rso.RemoteServerOptions()
    site_id = options.site_id

    scd_file = '{}/{}.scd'.format(scd_dir, site_id)

    inot.add_watch(scd_file)
    scd_util = scd_utils.SCDUtils(scd_file)

    log_dir = "{}/logs".format(scd_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    def make_schedule():
        time_of_interest = datetime.datetime.utcnow()

        log_time_str = time_of_interest.strftime("%Y.%m.%d.%H.%M")
        log_file = "{}/{}_remote_server.{}.log".format(log_dir, site_id, log_time_str)

        log_msg_header = "Updated at {}\n".format(time_of_interest)
        try:
            relevant_lines = get_relevant_lines(scd_util, time_of_interest)
        except ValueError as e:
            error_msg = ("{logtime}: Unable to make schedule\n"
                         "\t Exception thrown:\n"
                         "\t\t {exception}\n")
            error_msg = error_msg.format(logtime = time_of_interest.strftime("%c"),
                                            exception=str(e))
            with open(log_file, 'w') as f:
                f.write(log_msg_header)
                f.write(error_msg)

            subject = "Unable to make schedule at {}".format(site_id)

            emailer.email_log(subject, log_file)
        else:

            timeline, warnings = convert_scd_to_timeline(relevant_lines, time_of_interest)
            plot_path, pickle_path = plot_timeline(timeline, scd_dir, time_of_interest, site_id)
            new_atq_str = timeline_to_atq(timeline, scd_dir, time_of_interest, site_id)

            with open(log_file, 'wb') as f:
                f.write(log_msg_header.encode())
                f.write(new_atq_str)

                f.write("\n".encode())
                for warning in warnings:
                    f.write("\n{}".format(warning).encode())

            subject = "Successfully scheduled commands at {}".format(site_id)
            emailer.email_log(subject, log_file, [plot_path, pickle_path])



    # Make the schedule on restart of application
    make_schedule()
    new_notify = False
    while True:
        # "IN_IGNORED" was removing watch points and wouldnt monitor the path. This regens it.
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
            print(event_types)
            if site_id in path:
                if all(i in event_types for i in ["IN_OPEN", "IN_ACCESS", "IN_CLOSE_WRITE"]):
                    scd_utils.SCDUtils(path)
                    make_schedule()

                # Nextcloud/Vim triggers
                if all(i in event_types for i in ["IN_ATTRIB", "IN_DELETE_SELF", "IN_IGNORED"]):
                    scd_utils.SCDUtils(path)
                    make_schedule()
                    new_notify = True


if __name__ == '__main__':
    _main()
