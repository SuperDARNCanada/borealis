#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
#
# scd_utils.py
# 2019-04-18
# Utilites for working with scd files.

import datetime as dt
import collections
import shutil
import locale
import sys

class SCDUtils(object):
    """Contains utilities for working with SCD files. SCD files are schedule files for Borealis.

    Attributes:
        line_fmt (str): String format for scd line.
        scd_default (dict): Default event to run if no other infinite duration line is scheduled.
        scd_dt_fmt (str): String format for parsing/writing datetimes.
        scd_filename (str): The filename of schedule to use.
    """

    def __init__(self, scd_filename):
        super(SCDUtils, self).__init__()
        self.scd_filename = scd_filename
        self.scd_dt_fmt = "%Y%m%d %H:%M"
        self.line_fmt = "{datetime} {duration} {prio} {experiment}"
        self.scd_default = self.check_line('20000101', '00:00', 'normalscan', '0', '-')

    def check_line(self, yyyymmdd, hhmm, experiment, prio, duration):
        """Checks the line parameters to see if they are valid and then returns a dict with all
        the valid fields.

        Args:
            yyyymmdd (str): year/month/day string.
            hhmm (str): hour/minute string.
            experiment (str): The experiment to run.
            prio (str or int): priority value.
            duration (str): an optional duration to run for.

        Returns:
            TYPE: Dict of line params.

        Raises:
            ValueError: If line parameters are invalid or if line is a duplicate.
        """

        try:
            # create datetime from args to see if valid
            time = dt.datetime.strptime(yyyymmdd + " " + hhmm, self.scd_dt_fmt)
        except:
            raise ValueError("Can not create datetime from supplied formats")

        try:
            int(prio)
        except ValueError as e:
            raise ValueError("Unable to cast priority {} as int.".format(prio))

        if not (0 <= int(prio) <= 20):
            raise ValueError("Priority is out of bounds. 0 <= prio <= 20.")


        if duration != "-":
            try:
                int(duration)
            except ValueError as e:
                raise ValueError("Unable to cast duration {} as int".format(duration))

        epoch = dt.datetime.utcfromtimestamp(0)
        epoch_milliseconds = int((time - epoch).total_seconds() * 1000)

        return {"timestamp" : epoch_milliseconds,
                "time" : time,
                "duration" : str(duration),
                "prio" : str(prio),
                "experiment" : experiment}



    def read_scd(self):
        """Read and parse the Borealis schedule file.

        Returns:
            TYPE: list of dicts.

        Raises:
            ValueError: on lines with obvious errors in them.
            Will also throw file errors if there are problems opening SCD file.
        """
        with open(self.scd_filename, "r") as f:
            raw_scd = f.readlines()

        raw_scd = [line.split() for line in raw_scd]

        scd_lines = []

        # add the default infinite duration line 
        scd_lines.append(self.scd_default)

        for num, line in enumerate(raw_scd):
            if len(line) != 5:
                raise ValueError("Line {} has too many arguments".format(num))

            scd_lines.append(self.check_line(line[0], line[1], line[4], line[3], line[2]))

        if len(scd_lines) == 1:
            print('WARNING: SCD file empty; default normalscan will run')

        return scd_lines

    def fmt_line(self, line_dict):
        """Formats a dictionary with line info into a text line for file.

        Args:
            line_dict (dict): A dict that holds all the line info.

        Returns:
            TYPE: Formatted string.
        """
        line_str = self.line_fmt.format(datetime=line_dict["time"].strftime(self.scd_dt_fmt),
                                        prio=line_dict["prio"],
                                        experiment=line_dict["experiment"],
                                        duration=line_dict["duration"])
        return line_str

    def write_scd(self, scd_lines):
        """Creates SCD text lines and writes to file. Creates a backup of the old file before
        writing.

        Args:
            scd_lines (list): A list dicts that contain the line info.
        """
        text_lines = [self.fmt_line(x) for x in scd_lines]

        shutil.copy(self.scd_filename, self.scd_filename+".bak")

        with open(self.scd_filename, 'w') as f:
            for line in text_lines:
                f.write("{}\n".format(line))


    def add_line(self, yyyymmdd, hhmm, experiment, prio=0, duration='-'):
        """Adds a new line to the SCD.

        Args:
            yyyymmdd (str): year/month/day string.
            hhmm (str): hour/minute string.
            experiment (str): The experiment to run.
            prio (int or str, optional): priority value.
            duration (str, optional): an optional duration to run for.

        Raises:
            ValueError: If line parameters are invalid or if line is a duplicate.
        """

        new_line = self.check_line(yyyymmdd, hhmm, experiment, prio, duration)

        scd_lines = self.read_scd()

        if new_line in scd_lines:
            raise ValueError("Line is a duplicate of an existing line")

        if any([(new_line['timestamp'] == line['timestamp'] and
                    new_line['prio'] == line['prio']) for line in scd_lines]):
            raise ValueError("Priority already exists at this time")


        scd_lines.append(new_line)

        # sort priorities in reverse so that they are descending order. Then sort everything by
        # timestamp
        new_scd = sorted(scd_lines, key=lambda x: x['prio'], reverse=True)
        new_scd = sorted(new_scd, key=lambda x: x['timestamp'])

        self.write_scd(new_scd)

    def remove_line(self, yyyymmdd, hhmm, experiment, prio=0, duration='-'):
        """Summary

        Args:
            yyyymmdd (str): year/month/day string.
            hhmm (str): hour/minute string.
            experiment (str): The experiment to run.
            prio (int or str, optional): priority value.
            duration (str, optional): an optional duration to run for.

        Raises:
            ValueError: If line parameters are invalid or if line does not exist.
        """

        line_to_rm = self.check_line(yyyymmdd, hhmm, experiment, prio, duration)

        scd_lines = self.read_scd()
        try:
            scd_lines.remove(line_to_rm)
        except:
            raise ValueError("Line does not exist in SCD")

        self.write_scd(scd_lines)

    def get_relevant_lines(self, yyyymmdd, hhmm):
        """Gets the currently scheduled and future lines given a supplied time. If the provided time
        is equal to a scheduled line time, it provides that line and all future lines. If the
        provided time is between schedule line times, it provides any lines in the schedule with the
        most recent timestamp and all future lines.  If the provided time is before any lines in the
        schedule, it provides all schedule lines.

        Args:
            yyyymmdd (str): year/month/day string.
            hhmm (str): hour/minute string.

        Returns:
            TYPE: List of relevant dicts of line info.

        Raises:
            ValueError: If datetime could not be created from supplied arguments.
            IndexError: If schedule file is empty
        """

        try:
            # create datetime from args to see if valid
            time = dt.datetime.strptime(yyyymmdd + " " + hhmm, self.scd_dt_fmt)
        except:
            raise ValueError("Can not create datetime from supplied formats")

        scd_lines = self.read_scd()

        if not scd_lines:
            raise IndexError("Schedule file is empty. No lines can be returned")

        epoch = dt.datetime.utcfromtimestamp(0)
        epoch_milliseconds = int((time - epoch).total_seconds() * 1000)

        equals = False
        prev_line_appended = False
        relevant_lines = []
        for idx, line in enumerate(scd_lines):
            if line['timestamp'] == epoch_milliseconds:
                equals = True
                relevant_lines.append(line)
            elif line['timestamp'] > epoch_milliseconds:
                if equals:
                    relevant_lines.append(line)
                else:
                    if not prev_line_appended:
                        if idx != 0:
                            last_line_timestamp = scd_lines[idx-1]['timestamp']
                            temp_list = scd_lines[:]
                            for t in temp_list:
                                if t['timestamp'] == last_line_timestamp:
                                    relevant_lines.append(t)
                        prev_line_appended = True
                    relevant_lines.append(line)
            else:
                continue

        return relevant_lines



if __name__ == "__main__":
    filename = sys.argv[1]

    scd_util = SCDUtils(filename)

    scd_util.add_line("20190404", "10:43", "twofsound")
    #scd_util.add_line("04/04/2019", "10:43", "twofsound")
    scd_util.add_line("20190407", "10:43", "twofsound")
    scd_util.add_line("20190414", "10:43", "twofsound")
    scd_util.add_line("20190414", "10:43", "twofsound", prio=2)
    scd_util.add_line("20190414", "10:43", "twofsound", prio=1, duration=89)
    #scd_util.add_line("20190414", "10:43", "twofsound", prio=1, duration=24)
    scd_util.add_line("20190414", "11:43", "twofsound", duration=46)
    scd_util.add_line("20190414", "00:43", "twofsound")
    scd_util.add_line("20190408", "15:43", "twofsound", duration=57)


    scd_util.remove_line("20190414", "10:43", "twofsound")


    print(scd_util.get_relevant_lines("20190414", "10:44"))

