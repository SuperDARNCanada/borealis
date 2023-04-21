#!/usr/bin/python3
"""
    scd_utils.py
    ~~~~~~~~~~~~
    Utilities for working with scd files

    :copyright: 2019 SuperDARN Canada
"""
import os
import datetime as dt
import shutil
import sys
import unittest

borealis_path = os.environ['BOREALISPATH']
sys.path.append(f"{borealis_path}/tests/experiments")
import experiment_unittests


def get_next_month_from_date(date=None):
    """Finds the datetime of the next month.

    Args
        date - Default today. Datetime to get next month from

    Returns:
        TYPE: datetime object.
    """
    if date is None:
        date = dt.datetime.utcnow()

    counter = 1
    new_date = date + dt.timedelta(days=counter)
    while new_date.month == date.month:
        counter += 1
        new_date = date + dt.timedelta(days=counter)

    return new_date


class SCDUtils(object):
    """
    Contains utilities for working with SCD files. SCD files are schedule files for Borealis.
    
    :param  scd_filename:   Schedule file name
    :type:  scd_filename:   str
    :param  scd_dt_fmt:     String format for parsing/writing datetimes.
    :type:  scd_dt_fmt:     str
    :param  line_fmt:       String format for scd line.
    :type:  line_fmt:       str
    :param  scd_default:    Default event to run if no other infinite duration line is scheduled.
    :type:  scd_default:    dict
    """

    def __init__(self, scd_filename):
        super().__init__()
        self.scd_filename = scd_filename
        self.scd_dt_fmt = "%Y%m%d %H:%M"
        self.line_fmt = "{datetime} {duration} {prio} {experiment} {scheduling_mode} {kwargs_string}"
        self.scd_default = self.check_line('20000101', '00:00', 'normalscan', 'common', '0', '-')

    def check_line(self, yyyymmdd, hhmm, experiment, scheduling_mode, prio, duration, kwargs_string=''):
        """
        Checks the line parameters to see if they are valid and then returns a dict with all the
        valid fields.

        :param  yyyymmdd:           year/month/day string.
        :type   yyyymmdd:           str
        :param  hhmm:               hour/minute string.
        :type   hhmm:               str
        :param  experiment:         The experiment to run.
        :type   experiment:         str
        :param  scheduling_mode:    The type of scheduling mode.
        :type   scheduling_mode:    str
        :param  prio:               priority value.
        :type   prio:               str or int
        :param  duration:           an optional duration to run for.
        :type   duration:           str
        :param  kwargs_string:      kwargs for the experiment instantiation. (Default value = '')
        :type   kwargs_string:      str

        :returns:   Dict of line params.
        :rtype:     dict

        :raises     ValueError: If line parameters are invalid or if line is a duplicate.
        """

        # create datetime from args to see if valid. Value error for incorrect format
        time = dt.datetime.strptime(yyyymmdd + " " + hhmm, self.scd_dt_fmt)

        if not isinstance(kwargs_string, str):
            raise ValueError("kwargs_string should be a string")

        if not (0 <= int(prio) <= 20):
            raise ValueError("Priority is out of bounds. 0 <= prio <= 20.")

        if duration != "-":
            if isinstance(duration, float) or int(duration) < 1:
                raise ValueError("Duration should be an integer > 0, or '-'")
            duration = int(duration)

        epoch = dt.datetime.utcfromtimestamp(0)
        epoch_milliseconds = int((time - epoch).total_seconds() * 1000)

        possible_scheduling_modes = ['common', 'special', 'discretionary']
        if scheduling_mode not in possible_scheduling_modes:
            raise ValueError(f"Unknown scheduling mode type {scheduling_mode} not in {possible_scheduling_modes}")

        # See if the experiment itself would run
        args = ['--site_id', self.scd_filename[:2], '--experiment', experiment]
        test_program = experiment_unittests.main(args, buffer=True, print_results=False)
        if len(test_program.result.failures) != 0 or len(test_program.result.errors) != 0:
            raise ValueError("Experiment could not be scheduled due to errors in experiment.\n"
                             f"{test_program.result.failures[0][1]}")

        return {"timestamp": epoch_milliseconds,
                "time": time,
                "duration": str(duration),
                "prio": str(prio),
                "experiment": experiment,
                "scheduling_mode": scheduling_mode,
                "kwargs_string": kwargs_string}

    def read_scd(self):
        """
        Read and parse the Borealis schedule file.

        :returns:   list of dicts containing schedule info
        :rtype:     list(dict)

        :raises ValueError: if any lines have obvious errors
        :raises OSError:    if SCD file cannot be opened
        """
        with open(self.scd_filename, "r") as f:
            raw_scd = f.readlines()

        raw_scd = [line.split() for line in raw_scd]

        scd_lines = []

        for num, line in enumerate(raw_scd):
            if len(line) not in [6, 7]:
                raise ValueError(f"Line {num} has incorrect number of arguments; requires 6 or 7. Line: {line}")
            # date time experiment mode priority duration (kwargs if any)
            if len(line) == 6:
                scd_lines.append(self.check_line(line[0], line[1], line[4], line[5], line[3], line[2]))
            else:
                scd_lines.append(self.check_line(line[0], line[1], line[4], line[5], line[3], line[2], line[6]))

        if len(scd_lines) == 0:
            print('WARNING: SCD file empty; default normalscan will run')
            # add the default infinite duration line 
            scd_lines.append(self.scd_default)

        return scd_lines

    def fmt_line(self, line_dict):
        """
        Formats a dictionary with line info into a text line for file.

        :param  line_dict: A dict that holds all the line info.
        :type   line_dict: dict

        :returns:   Formatted string.
        :rtype:     str
        """
        line_str = self.line_fmt.format(datetime=line_dict["time"].strftime(self.scd_dt_fmt),
                                        prio=line_dict["prio"],
                                        experiment=line_dict["experiment"],
                                        scheduling_mode=line_dict["scheduling_mode"],
                                        duration=line_dict["duration"],
                                        kwargs_string=line_dict["kwargs_string"])
        return line_str

    def write_scd(self, scd_lines):
        """
        Creates SCD text lines and writes to file. Creates a backup of the old file before
        writing.

        Raises:
            PermissionError - When there are not sufficient permissions with the scd file
            FileNotFoundError - When the scd file doesn't exist
            IsADirectoryError - When the scd file given is a directory

        :param  scd_lines: A list of dicts that contain the schedule line info.
        :type   scd_lines: list(dict)
        """
        text_lines = [self.fmt_line(x) for x in scd_lines]

        shutil.copy(self.scd_filename, self.scd_filename+".bak")

        with open(self.scd_filename, 'w') as f:
            for line in text_lines:
                f.write(f"{line}\n")

    def add_line(self, yyyymmdd, hhmm, experiment, scheduling_mode, prio=0, 
                 duration='-', kwargs_string=''):
        """
        Adds a new line to the schedule.

        :param  yyyymmdd:           year/month/day string.
        :type   yyyymmdd:           str
        :param  hhmm:               hour/minute string.
        :type   hhmm:               str
        :param  experiment:         The experiment to run.
        :type   experiment:         str
        :param  scheduling_mode:    The mode type running for this time period.
        :type   scheduling_mode:    str
        :param  prio:               priority value. (Default value = 0)
        :type   prio:               int or str
        :param  duration:           duration to run for. (Default value = '-')
        :type   duration:           str
        :param  kwargs_string:      kwargs for the experiment instantiation. (Default value = '')
        :type   kwargs_string:      str

        :raises ValueError: If line parameters are invalid or if line is a duplicate.
        """
        new_line = self.check_line(yyyymmdd, hhmm, experiment, scheduling_mode, prio, duration, kwargs_string)

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

    def remove_line(self, yyyymmdd, hhmm, experiment, scheduling_mode, prio=0, 
                    duration='-', kwargs_string=''):
        """
        Removes a line from the schedule

        :param  yyyymmdd:           year/month/day string.
        :type   yyyymmdd:           str
        :param  hhmm:               hour/minute string.
        :type   hhmm:               str
        :param  experiment:         The experiment to run.
        :type   experiment:         str
        :param  scheduling_mode:    The mode type running for this time period.
        :type   scheduling_mode:    str
        :param  prio:               priority value. (Default value = 0)
        :type   prio:               int or str
        :param  duration:           an optional duration to run for. (Default value = '-')
        :type   duration:           str
        :param  kwargs_string:      kwargs for the experiment instantiation. (Default value = '')
        :type   kwargs_string:      str

        :raises ValueError: If line parameters are invalid or if line does not exist.
        """

        line_to_rm = self.check_line(yyyymmdd, hhmm, experiment, scheduling_mode, prio, duration, kwargs_string)

        scd_lines = self.read_scd()
        try:
            scd_lines.remove(line_to_rm)
        except ValueError:
            raise ValueError("Line does not exist in SCD")

        self.write_scd(scd_lines)

    def get_relevant_lines(self, yyyymmdd, hhmm):
        """
        Gets the currently scheduled and future lines given a supplied time. If the provided time is
        equal to a scheduled line time, it provides that line and all future lines. If the provided
        time is between schedule line times, it provides any lines in the schedule with the most
        recent timestamp and all future lines.  If the provided time is before any lines in the
        schedule, it provides all schedule lines.

        :param  yyyymmdd:   year/month/day string.
        :type   yyyymmdd:   str
        :param  hhmm:       hour/minute string.
        :type   hhmm:       str

        :returns:   List of relevant dicts of line info.
        :rtype:     list(dict)

        :raises ValueError: If datetime could not be created from supplied arguments.
        :raises IndexError: If schedule file is empty
        """

        try:
            # create datetime from args to see if valid
            time = dt.datetime.strptime(yyyymmdd + " " + hhmm, self.scd_dt_fmt)
        except ValueError:
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
