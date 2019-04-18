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

# Standardizes datetime format to use the en_US formatting in the Python docs.
locale.setlocale(locale.LC_TIME, "en_US")

class SCDUtils(object):
    """Contains utilities for working with SCD files.

    Attributes:
        line_fmt (str): String format for scd line.
        scd_dt_fmt (str): String format for parsing/writing datetimes.
        scd_filename (TYPE): The filename of schedule to use.
    """

    def __init__(self, scd_filename):
        super(SCDUtils, self).__init__()
        self.scd_filename = scd_filename
        self.scd_dt_fmt = "%x %H:%M"
        self.line_fmt = "{datetime} {duration} {prio} {experiment}"

    def read_scd(self):
        """Read and parse the SCD file.

        Returns:
            TYPE: Dict

        Raises:
            ValueError: on lines with obvious errors in them.
            Will also throw file errors if there are problems opening SCD file.
        """
        with open(self.scd_filename, "r") as f:
            raw_scd = f.readlines()

        raw_scd = [line.split() for line in raw_scd]

        scd_lines = []
        for num, line in enumerate(raw_scd):
            if len(line) != 5:
                raise ValueError("Line {} has too many arguments".format(num))

            try:
                time = dt.datetime.strptime(line[0] + " " + line[1], self.scd_dt_fmt)
            except:
                raise ValueError("Unable to create datetime from line {}".format(num))

            duration = line[2]

            prio = int(line[3])
            if not (0 <= prio <= 20):
                raise ValueError("Priority in line {} is out of bounds".format(num))

            experiment = line[4]

            epoch = dt.datetime.utcfromtimestamp(0)
            epoch_milliseconds = int((time - epoch).total_seconds() * 1000)

            scd_lines.append({"timestamp" : epoch_milliseconds,
                                "time" : time,
                                "duration" : str(duration),
                                "prio" : str(prio),
                                "experiment" : experiment})

        return scd_lines

    def fmt_line(self, line_dict):
        """Formats a dictionary with line info into a text line for file.

        Args:
            line_dict (TYPE): A dict that holds all the line info.

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
            scd_lines (TYPE): A list dicts that contain the line info.
        """
        text_lines = [self.fmt_line(x) for x in scd_lines]

        shutil.copy(self.scd_filename, self.scd_filename+".bak")

        with open(self.scd_filename, 'w') as f:
            for line in text_lines:
                f.write("{}\n".format(line))


    def add_line(self, mmddyy, hhmm, experiment, prio=0, duration='-'):
        """Adds a new line to the SCD.

        Args:
            mmddyy (TYPE): month/day/year string.
            hhmm (TYPE): hour/minute string.
            prio (TYPE): priority value.
            experiment (TYPE): The experiment to run.
            duration (str, optional): an optional duration to run for.

        Raises:
            ValueError: If line parameters are invalid or if line is a duplicate.
        """

        try:
            # create datetime from args to see if valid
            time = dt.datetime.strptime(mmddyy + " " + hhmm, self.scd_dt_fmt)
        except:
            raise ValueError("Can not create datetime from supplied formats")

        if not (0 <= prio <= 20):
            raise ValueError("Priority is out of bounds. 0 <= prio <= 20.")


        scd_lines = self.read_scd()

        epoch = dt.datetime.utcfromtimestamp(0)
        epoch_milliseconds = int((time - epoch).total_seconds() * 1000)

        new_line = {"timestamp" : epoch_milliseconds,
                    "time" : time,
                    "duration" : str(duration),
                    "prio" : str(prio),
                    "experiment" : experiment}

        if new_line in scd_lines:
            raise ValueError("Line is a duplicate of an existing line")

        scd_lines.append(new_line)

        # sort first by timestamp, then by duration with default duration first, then by priority.
        # duration sorting is funky cause the default value is not a int.
        new_scd = sorted(scd_lines, key=lambda x:(x['timestamp'],
                                                    (x['duration'] != '-', x['duration']),
                                                    x['prio']))

        self.write_scd(new_scd)

    def remove_line(self, mmddyy, hhmm, experiment, prio=0, duration='-'):
        """Summary

        Args:
            mmddyy (TYPE): month/day/year string.
            hhmm (TYPE): hour/minute string.
            prio (TYPE): priority value.
            experiment (TYPE): The experiment to run.
            duration (str, optional): an optional duration to run for.

        Raises:
            ValueError: If line parameters are invalid or if line does not exist.
        """

        try:
            # create datetime from args to see if valid
            time = dt.datetime.strptime(mmddyy + " " + hhmm, self.scd_dt_fmt)
        except:
            raise ValueError("Can not create datetime from supplied formats")

        if not (0 <= prio <= 20):
            raise ValueError("Priority is out of bounds. 0 <= prio <= 20.")

        scd_lines = self.read_scd()

        epoch = dt.datetime.utcfromtimestamp(0)
        epoch_milliseconds = int((time - epoch).total_seconds() * 1000)

        line_to_rm = {"timestamp" : epoch_milliseconds,
                        "time" : time,
                        "duration" : str(duration),
                        "prio" : str(prio),
                        "experiment" : experiment}

        try:
            scd_lines.remove(line_to_rm)
        except:
            raise ValueError("Line does not exist in SCD")

        self.write_scd(scd_lines)

    def get_relevant_lines(self, mmddyy, hhmm):
        """Gets any relevant future lines given a supplied time.

        Args:
            mmddyy (TYPE): month/day/year string.
            hhmm (TYPE): hour/minute string.

        Returns:
            TYPE: List of relevant dicts of line info.

        Raises:
            ValueError: If datetime could not be created from supplied arguments.
        """

        try:
            # create datetime from args to see if valid
            time = dt.datetime.strptime(mmddyy + " " + hhmm, self.scd_dt_fmt)
        except:
            raise ValueError("Can not create datetime from supplied formats")

        scd_lines = self.read_scd()

        epoch = dt.datetime.utcfromtimestamp(0)
        epoch_milliseconds = int((time - epoch).total_seconds() * 1000)

        relevant_lines = [x for x in scd_lines if x['timestamp'] >= epoch_milliseconds]

        return relevant_lines



if __name__ == "__main__":
    filename = "test_scd.txt"

    scd_util = SCDUtils(filename)

    scd_util.add_line("04/04/2019", "10:43", "twofsound")
    #scd_util.add_line("04/04/2019", "10:43","twofsound")
    scd_util.add_line("04/07/2019", "10:43", "twofsound")
    scd_util.add_line("04/14/2019", "10:43", "twofsound")
    scd_util.add_line("04/14/2019", "10:43", "twofsound", 2)
    scd_util.add_line("04/14/2019", "10:43", "twofsound", 1, 89)
    scd_util.add_line("04/14/2019", "10:43", "twofsound", 1, 24)
    scd_util.add_line("04/14/2019", "11:43", "twofsound", 46)
    scd_util.add_line("04/14/2019", "00:43", "twofsound")
    scd_util.add_line("04/08/2019", "15:43", "twofsound", 57)


    scd_util.remove_line("04/14/2019", "10:43", "twofsound")

    print(scd_util.get_relevant_lines("04/14/2019", "10:43"))

