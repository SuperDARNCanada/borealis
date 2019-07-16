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
    if first_event_flag:
        cmd_str = "echo '{borealis_path}/start_radar.sh {experiment}' | at now"
        cmd_str = cmd_str.format(borealis_path=os.environ['BOREALISPATH'],experiment=experiment)
        cmd_str = dt.strftime(cmd_str)
    else:
        cmd_str = "echo '{borealis_path}/start_radar.sh {experiment}' | at -t %Y%m%d%H%M"
        cmd_str = cmd_str.format(borealis_path=os.environ['BOREALISPATH'],experiment=experiment)
        cmd_str = dt.strftime(cmd_str)

    return cmd_str

def get_next_month(date):
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

def plot_timeline(timeline_list, timeline_dict, scd_dir, now):
    """Plots the timeline to better visualize runtime.

    Args:
        timeline_dict (dict): A dict of grouped entries.
        scd_dir (str): The scd directory path.
        now (datetime): The datetime holding the time of scheduling.

    Returns:
        (str, str): Paths to the saved plots.
    """
    fig, ax = plt.subplots()

    event_list = []
    first_date, last_date = None, None


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
    cmap = get_cmap(len(timeline_dict.items()))
    colors = [cmap(i) for i in range(len(timeline_dict.items()))]
    random.shuffle(colors)

    # put the colors in and create event records with only start and end times,
    # durations, colors, and labels
    for i, (_, events) in enumerate(timeline_dict.items()):
        for event in events:
            event['color'] = colors[i]

    for i, event in enumerate(timeline_list):
        event_item = dict()

        event_item['color'] = event['color']
        event_item['label'] = event['experiment']

        event_item['start'] = event['time']

        if event['duration'] == '-':
            td = get_next_month(event['time']) - event['time']
        else:
            td = datetime.timedelta(minutes=int(event['duration']))

        event_item['end'] = event_item['start'] + td
        event_item['duration'] = td

        timeline_list[i] = event_item

        if i == 0:
            first_date = event['time'].date()
        if i == len(timeline_list) - 1:
            last_date = event['time'] + td
            last_date = last_date.date()

    # loop through events again, splitting them where necessary
    for event in timeline_list:
        event_list += split_event(event)

    for event in event_list:
        day_offset = event['start'].date() - first_date
        start = event['start'] - day_offset
        ax.barh(event['start'].date(), event['duration'], left=start, color=event['color'])
        # ax.text((ev['start'] + ev['duration']) / 2, ev['start'].day, ev['label'])

    hours = mdates.HourLocator(byhour=[0,6,12,18,24])
    days = mdates.DayLocator()
    minutes = mdates.MinuteLocator()
    x_fmt = mdates.DateFormatter('%H:%M')
    y_fmt = mdates.DateFormatter('%m-%d')

    ax.xaxis.set_major_locator(hours)
    ax.xaxis.set_major_formatter(x_fmt)

    # ax.xaxis.set_minor_locator(minutes)

    ax.yaxis.set_major_locator(days)
    ax.yaxis.set_major_formatter(y_fmt)
    ax.set_ylim(first_date, last_date)

    plt.xticks(rotation=45)

    plt.show()

    plot_time_str = now.strftime("%Y.%m.%d.%H.%M")
    plot_dir = "{}/timeline_plots".format(scd_dir)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    plot_file = "{}/{}.png".format(plot_dir, plot_time_str)

    fig.set_size_inches(14,8)
    fig.savefig(plot_file, dpi=80)

    pkl_file = "{}/{}.pickle".format(plot_dir, plot_time_str)
    pkl.dump(fig, open(pkl_file, 'wb'))

    return (plot_file, pkl_file)





def convert_scd_to_timeline(scd_lines):
    """ Creates a true timeline from the scd lines, accounting for priority and duration of each
    line.

    Args:
        scd_lines (list): List of scd lines to try convert to a timeline.

    Returns:
        dict: Groups of entries belonging to the same experiment.
    """

    inf_dur_line = None
    queued_lines = []

    # Add the ordering of the lines to each line. This is so we can group entries that get split
    # up on one line for plotting.
    for i, line in enumerate(scd_lines):
        line['order'] = i

    def calculate_new_last_line_params():
        last_tt_line = queued_lines[-1]
        queued_dur_td = datetime.timedelta(minutes=int(last_tt_line['duration']))
        queued_finish = last_tt_line['time'] + queued_dur_td
        return last_tt_line, queued_finish

    for idx, tt in enumerate(scd_lines):
        # if we have lines queued up, grab the last one to compare to.
        if queued_lines:
            last_tt_line, queued_finish = calculate_new_last_line_params()

        # handling infinite duration lines
        if tt['duration'] == '-':
            # if no line set, just assign it
            if not inf_dur_line:
                inf_dur_line = tt
            else:
                # case for switching infinite duration lines.
                if int(tt['prio']) >= int(inf_dur_line['prio']):
                    # if no queued lines yet, just take the time up to the new line and add it,
                    # else check against the last line to see if it comes before or after.
                    if not queued_lines:
                        time_diff = tt['time'] - inf_dur_line['time']
                        inf_dur_line['duration'] = time_diff.total_seconds()//60
                        queued_lines.append(inf_dur_line)
                    else:
                        duration = int(last_tt_line['duration'])
                        new_time = last_tt_line['time'] + datetime.timedelta(minutes=duration)
                        if tt['time'] > new_time:
                            inf_dur_line['time'] = new_time
                            time_diff = tt['time'] - new_time
                            inf_dur_line['duration'] = time_diff.total_seconds()//60
                            queued_lines.append(copy.deepcopy(inf_dur_line))
                    inf_dur_line = tt
        else:
            duration_td = datetime.timedelta(minutes=int(tt['duration']))
            finish_time = tt['time'] + duration_td

            # if no lines added yet, just add the line to the queue. Check if an inf dur line is
            # running.
            if not queued_lines:
                if not inf_dur_line:
                    queued_lines.append(tt)
                else:
                    if int(tt['prio']) > int(inf_dur_line['prio']):
                        time_diff = tt['time'] - inf_dur_line['time']
                        new_line = copy.deepcopy(inf_dur_line)
                        new_line['duration'] = time_diff.total_seconds()//60
                        queued_lines.append(new_line)
                        queued_lines.append(tt)

                        finish_time = tt['time'] + duration_td
                        inf_dur_line['time'] = finish_time
            else:
                # loop to find where to insert this line. Hold all following lines.
                holder = []
                while tt['time'] < queued_lines[-1]['time']:
                    holder.append(queued_lines.pop())

                last_tt_line, queued_finish = calculate_new_last_line_params()
                holder.append(tt)

                # Durations and priorities change when lines can run and sometimes lines get
                # broken up. We continually loop to readjust the timeline to account for the
                # priorities and durations of new lines added.
                while holder:# or first_time:

                    item_to_add = holder.pop()
                    duration_td = datetime.timedelta(minutes=int(item_to_add['duration']))
                    finish_time = item_to_add['time'] + duration_td

                    while item_to_add['time'] < queued_lines[-1]['time']:
                        holder.append(queued_lines.pop())

                    last_tt_line, queued_finish = calculate_new_last_line_params()

                    # if the line comes directly after the last line, we can add it.
                    if item_to_add['time'] == queued_finish:
                        queued_lines.append(item_to_add)
                    # if the time of the line starts before the last line ends, we need to may need
                    # to make adjustments.
                    elif item_to_add['time'] < queued_finish:
                        # if the line finishes before the last line and is a higher priority,
                        # we split the last line up and insert the new line.
                        if finish_time < queued_finish:
                            if int(item_to_add['prio']) > int(last_tt_line['prio']):
                                queued_copy = copy.deepcopy(last_tt_line)

                                # if time is >, then we can split the first part and insert
                                # the line is. Otherwise, we can directly overwrite the first
                                # part.
                                if item_to_add['time'] > last_tt_line['time']:
                                    first_duration = item_to_add['time'] - last_tt_line['time']
                                    last_tt_line['duration'] = first_duration.total_seconds()//60
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
                            if int(item_to_add['prio']) > int(last_tt_line['prio']):
                                if item_to_add['time'] > last_tt_line['time']:
                                    new_duration = item_to_add['time'] - last_tt_line['time']
                                    last_tt_line['duration'] = new_duration.total_seconds()//60
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
            last_tt_line, queued_finish = calculate_new_last_line_params()

            inf_dur_line['time'] = queued_finish
            queued_lines.append(inf_dur_line)

    queued_dict = collections.OrderedDict()
    for line in queued_lines:
        if not line['order'] in queued_dict:
            queued_dict[line['order']] = []

        queued_dict[line['order']].append(line)

    return queued_dict, queued_lines

def timeline_to_atq(timeline, scd_dir, now):
    """ Converts the created timeline to actual atq commands.

    Args:
        timeline (dict): A dictionary holding all timeline events.
        scd_dir (str): The directory with SCD files.
        now (datetime): The datetime holding the time of scheduling.

    Log and backup the existing atq, remove old events and then schedule everything recent. The
    first entry should be the currently running event, so it gets scheduled immediately.
    """
    backup_cmd = 'for j in $(atq | sort -k6,6 -k3,3M -k4,4 -k5,5 |cut -f 1);'\
    'do atq |grep -P "^$j\t"; at -c "$j" | tail -n 2; done'

    output = sp.check_output(backup_cmd, shell=True)

    backup_time_str = now.strftime("%Y.%m.%d.%H.%M")
    backup_dir = "{}/atq_backups".format(scd_dir)

    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)

    backup_file = "{}/{}.atq".format(backup_dir, backup_time_str)

    with open(backup_file, 'wb') as f:
        f.write(output)

    clear_command = "for i in `atq | awk '{print $1}'`;do atrm $i;done"

    sp.call(clear_command, shell=True)

    atq = []
    first_event = True
    for _,events in timeline.items():
        for event in events:
            if first_event:
                atq.append(format_to_atq(event['time'], event['experiment'], True))
                first_event = False
            else:
                atq.append(format_to_atq(event['time'], event['experiment']))

    for cmd in atq:
        sp.call(cmd, shell=True)

    return sp.check_output(backup_cmd, shell=True)

def get_relevant_lines(scd_util, now):
    """
    @brief      Gets the relevant lines.

    @param      scd_util  The scd utility object that holds the scd lines.

    @return     The relevant lines.

    Does a search for relevant lines. If the first line returned isnt an infinite duration, we need
    to look back until we find an infinite duration line, as that should be the last line to
    continue running if we need it.
    """

    found = False
    time = now

    yyyymmdd = now.strftime("%Y%m%d")
    hhmm = now.strftime("%H:%M")


    relevant_lines = scd_util.get_relevant_lines(yyyymmdd, hhmm)
    while not found:

        if not relevant_lines:
            msg = "Error in schedule: Could not find any relevant_lines. Either no lines exist "
            "or an infinite duration line could not be found."

            raise ValueError(msg)

        if relevant_lines[0]['duration'] == '-':
            found = True
        else:
            time -= datetime.timedelta(days=1)

            yyyymmdd = time.strftime("%Y%m%d")
            hhmm = time.strftime("%H:%M")

            new_relevant_lines = scd_util.get_relevant_lines(yyyymmdd, hhmm)
            if not new_relevant_lines:
                msg = "Error in schedule: Could not find any relevant_lines. Either no lines exist "
                "or an infinite duration line could not be found."

                raise ValueError(msg)

            lines_diff = [d for d in new_relevant_lines if d not in relevant_lines]

            for line in reversed(lines_diff):
                if line['duration'] == '-':
                    relevant_lines.insert(0, line)
                    found = True


    # Since SCDUtils relevant lines will return the last line run if the supplied time
    # is in the beginning of two lines, we then only schedule that time if it's duration is
    # infinite, otherwise we filter it out since we missed it's time.
    relevant_lines = [l for l in relevant_lines if l['time'] > now or l['duration'] == '-']
    return relevant_lines


def _main():
    parser = argparse.ArgumentParser(description="Automatically schedules new SCD file entries")
    parser.add_argument('--emails-filepath',required=True, help='A list of emails to send logs to')
    parser.add_argument('--scd-dir', required=True, help='The scd working directory')

    args = parser.parse_args()

    scd_dir = args.scd_dir

    emailer = email_utils.Emailer(args.emails_filepath)

    i = inotify.adapters.Inotify()

    options = rso.RemoteServerOptions()
    site_id = options.site_id

    scd_file = '{}/{}.scd'.format(scd_dir, site_id)

    i.add_watch(scd_file)
    scd_util = scd_utils.SCDUtils(scd_file)

    log_dir = "{}/logs".format(scd_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)


    def make_schedule():
        now = datetime.datetime.utcnow()

        log_time_str = now.strftime("%Y.%m.%d.%H.%M")
        log_file = "{}/{}.log".format(log_dir, log_time_str)

        try:
            relevant_lines = get_relevant_lines(scd_util, now)
        except ValueError as e:
            error_msg = ("{logtime}: Unable to make schedule\n"
                         "\t Exception thrown:\n"
                         "\t\t {exception}\n")
            error_msg = error_msg.format(logtime = now.strftime("%c"),
                                            exception=str(e))
            with open(log_file, 'w') as f:
                f.write(error_msg)

            subject = "Unable to make schedule at {}".format(site_id)

            emailer.email_log(subject, log_file)
        else:
            timeline, timeline_list = convert_scd_to_timeline(relevant_lines)
            plot_path, pickle_path = plot_timeline(timeline_list, timeline, scd_dir, now)
            new_atq_str = timeline_to_atq(timeline, scd_dir, now)

            with open(log_file, 'wb') as f:
                f.write(new_atq_str)

            subject = "Successfully scheduled commands at {}".format(site_id)
            emailer.email_log(subject, log_file, [plot_path, pickle_path])



    # Make the schedule on restart of application
    make_schedule()
    while True:
        for event in i.event_gen(yield_nones=False):
            (_, type_names, path, filename) = event

            if site_id in path and "IN_CLOSE_WRITE" in type_names:
                scd_utils.SCDUtils(path)

                make_schedule()



if __name__ == '__main__':
    _main()