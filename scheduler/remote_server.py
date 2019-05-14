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

import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import remote_server_options as rso

SCD_FILES = ["sas.scd", "pgr.scd", "cly.scd", "rkn.scd", "inv.scd"]

def format_to_atq(dt, experiment):
    cmd_str = "at -t %Y%m%d%H%M start_script {experiment}"
    cmd_str = cmd_str.format(experiment=experiment)
    cmd_str = dt.strftime(cmd_str)

    return cmd_str

def get_next_month(date):
        """Finds the datetime of the next month.

        Returns:
            TYPE: datetime object.
        """

        counter = 1
        new_date = date + datetime.timedelta(days=counter)
        while new_date.month == date.month:
            counter += 1
            new_date = date + datetime.timedelta(days=counter)

        return new_date

def plot_timeline(timeline):
    fig, ax = plt.subplots()

    event_labels = []
    first_date, last_date = None, None
    for i, event in enumerate(timeline):
        event_times = []
        event_label = ""
        #for event in events:
        time_start = mdates.date2num(event['time'])

        if event['duration'] == '-':
            td = get_next_month(event['time']) - event['time']
        else:
            td = datetime.timedelta(minutes=int(event['duration']))


        time_end = td.total_seconds()/(24 * 60 * 60)
        event_times.append((time_start, time_end))
        event_label = event['experiment']


        if i == 0:
            first_date = event['time']
        if i == len(timeline) - 1:
            last_date = event['time'] + td

        event_labels.append(event_label)
        ax.broken_barh(event_times, ((i+1)*10, 4))

    ax.set_yticks([(i+1)*10 + 2 for i in range(len(event_labels))])
    ax.set_yticklabels(event_labels)
    ax.set_ylim(5, 50)

    hours = mdates.HourLocator(byhour=[0,6,12,18,24])
    days = mdates.DayLocator()
    fmt = mdates.DateFormatter('%m-%d')

    ax.xaxis.set_major_locator(days)
    ax.xaxis.set_major_formatter(fmt)
    ax.xaxis.set_minor_locator(hours)

    ax.set_xlim(first_date, last_date)
    plt.xticks(rotation=45)

    plt.show()


def convert_scd_to_atq(scd_lines):
    #backup file

    scd_dict = collections.OrderedDict()

    for scd_line in scd_lines:
        if not scd_line['timestamp'] in scd_dict:
            scd_dict[scd_line['timestamp']] = []

        scd_dict[scd_line['timestamp']].append(scd_line)


    atq = []
    active_lines = []

    current_priority = None
    total_duration = None

    timestamp_timeline = []
    for timestamp, lines in scd_dict.items():
        timestamp_to_timeline = []
        for line in lines:
            if not timestamp_to_timeline:
                current_priority = int(line['prio'])
                total_duration = line['duration']
                timestamp_to_timeline.append(line)
            else:
                if int(line['prio']) == current_priority:
                    #TODO ERROR
                    continue
                elif int(line['prio']) < current_priority:
                    if total_duration == '-':
                        continue
                        # TODO Line can't be scheduled

                    else:
                        duration = int(total_duration)

                        def new_timestamp(x):
                            epoch = datetime.datetime.utcfromtimestamp(0)
                            return int((x - epoch).total_seconds() * 1000)

                        if line['duration'] == '-':
                            new_time = line['time'] + datetime.timedelta(minutes=duration)
                            line['time'] = new_time
                            line['timestamp'] = new_timestamp(line['time'])
                            total_duration = line['duration']
                        else:
                            if int(line['duration']) > duration:
                                new_time = line['time'] + datetime.timedelta(minutes=duration)
                                new_line_duration = int(line['duration']) - duration
                                line['time'] = new_time
                                line['duration'] = new_line_duration
                                line['timestamp'] = new_timestamp(line['time'])
                                total_duration = new_line_duration + duration
                            else:
                                continue

                        timestamp_to_timeline.append(line)

                else:
                    pass
                    # TODO line out of order
        #print(timestamp_to_timeline)
        timestamp_timeline.extend(timestamp_to_timeline)

    timestamp_timeline = sorted(timestamp_timeline, key=lambda x: x['prio'], reverse=True)
    timestamp_timeline = sorted(timestamp_timeline, key=lambda x: x['timestamp'])


    inf_dur_line = None
    queued_lines = []

    for idx, tt in enumerate(timestamp_timeline):
        if tt['duration'] == '-':
            if not inf_dur_line:
                inf_dur_line = tt
            else:
                if int(tt['prio']) >= int(inf_dur_line['prio']):
                    if not queued_lines:
                        time_diff = tt['time'] - inf_dur_line['time']
                        inf_dur_line['duration'] = time_diff.seconds//60
                        active_lines.append(inf_dur_line)
                    else:
                        dur = int(queued_lines[-1]['duration'])
                        new_time = queued_lines[-1]['time'] + datetime.timedelta(minutes=duration)
                        inf_dur_line['time'] = new_time
                        time_diff = tt['time'] - new_time
                        inf_dur_line['duration'] = time_diff.seconds//60
                        print(inf_dur_line)
                        queued_lines.append(copy.deepcopy(inf_dur_line))
                    inf_dur_line = tt
        else:
            duration_td = datetime.timedelta(minutes=int(tt['duration']))
            finish_time = tt['time'] + duration_td

            if not queued_lines:
                if not inf_dur_line:
                    queued_lines.append(tt)
                else:
                    if int(tt['prio']) > int(inf_dur_line['prio']):
                        time_diff = tt['time'] - inf_dur_line['time']
                        new_line = copy.deepcopy(inf_dur_line)
                        new_line['duration'] = time_diff.seconds//60
                        queued_lines.append(new_line)
                        queued_lines.append(tt)

                        inf_dur_line['time'] = finish_time
            else:
                holder = []
                while tt['time'] <= queued_lines[-1]['time']:
                    holder.append(queued_lines.pop())

                item_to_add = tt
                while holder:
                    queued_dur_td = datetime.timedelta(minutes=int(queued_lines[-1]['duration']))
                    queued_finish = queued_lines[-1]['time'] + queued_dur_td

                    if item_to_add['time'] < queued_finish:
                        if int(item_to_add['prio']) > int(queued_lines[-1]['prio']):
                            if finish_time < queued_finish:
                                queued_copy = copy.deepcopy(queued_lines[-1])

                                first_duration = item_to_add['time'] - queued_lines[-1]['time']
                                queued_lines[-1]['duration'] = first_duration.seconds//60

                                queued_lines.append(item_to_add)

                                remaining_duration = queued_finish - finish_time

                                queued_copy['time'] = finish_time
                                queued_copy['duration'] = remaining_duration.seconds//60
                                queued_lines.append(queued_copy)

                            else:
                                new_duration = queued_lines[-1]['time'] - item_to_add['time']
                                queued_lines[-1]['duration'] = new_duration
                                queued_lines.append(item_to_add)
                    else:
                        queued_lines.append(item_to_add)


                    if holder:
                        item_to_add = holder.pop()
                        holder = []
                        while item_to_add['time'] <= queued_lines[-1]['time']:
                            holder.append(queued_lines.pop())

    print(queued_lines)
    plot_timeline(queued_lines)


def _main():
    parser = argparse.ArgumentParser(description="Automatically schedules new SCD file entries")
    parser.add_argument('--emails-filepath',required=True, help='A list of emails to send logs to')
    parser.add_argument('--scd_dir', required=True, help='The scd working directory')

    args = parser.parse_args()

    scd_dir = args.scd_dir

    emailer = email_utils.Emailer(args.emails_filepath)

    i = inotify.adapters.Inotify()

    for scd in SCD_FILES:
        print(scd)
        i.add_watch(scd_dir + '/' + scd)

    options = rso.RemoteServerOptions()
    site_id = options.site_id

    scd_util = scd_utils.SCDUtils("{}.scd".format(site_id))

    while True:
        for event in i.event_gen(yield_nones=False):
            (_, type_names, path, filename) = event

            print(type_names, path, filename)
            if site_id in path and "IN_CLOSE_WRITE" in type_names:
                scd_utils.SCDUtils(path)

                now = datetime.datetime.utcnow()
                yyyymmdd = now.strftime("%Y%m%d")
                hhmm = now.strftime("%H:%M")

                relevant_lines = scd_util.get_relevant_lines(yyyymmdd, hhmm)

                convert_scd_to_atq(relevant_lines)





            # print("PATH=[{}] FILENAME=[{}] EVENT_TYPES={}".format(
            #       path, filename, type_names))



if __name__ == '__main__':
    _main()