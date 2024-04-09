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


class SCDUtils:
    """
    Contains utilities for working with SCD files. SCD files are schedule files for Borealis.
    """

    """String format for parsing and writing datetimes"""
    scd_dt_fmt = "%Y%m%d %H:%M"

    """String format for scd line"""
    line_fmt = "{datetime} {duration} {prio} {experiment} {scheduling_mode} {embargo} {kwargs}"


    def __init__(self, scd_filename):
        """
        :param  scd_filename:   Schedule file name
        :type:  scd_filename:   str
        """
        self.scd_filename = scd_filename

        """Default event to run if no other infinite duration line is scheduled"""
        self.scd_default = self.check_line('20000101', '00:00', 'normalscan', 'common', '0', '-')

    def check_line(self, yyyymmdd, hhmm, experiment, scheduling_mode, prio, duration, kwargs='', embargo=False):
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
        :param  kwargs:      kwargs for the experiment instantiation. Default None
        :type   kwargs:      str
        :param  embargo:            flag for embargoing files. (Default value = False)
        :type   embargo:            bool

        :returns:   Dict of line params.
        :rtype:     dict

        :raises     ValueError: If line parameters are invalid or if line is a duplicate.
        """

        # create datetime from args to see if valid. Value error for incorrect format
        time = dt.datetime.strptime(yyyymmdd + " " + hhmm, self.scd_dt_fmt)

        if not isinstance(kwargs, str):
            raise ValueError("kwargs should be a string")

        if not (0 <= int(prio) <= 20):
            raise ValueError("Priority is out of bounds. 0 <= prio <= 20.")

        if duration != "-":
            if isinstance(duration, float) or int(duration) < 1:
                raise ValueError("Duration should be an integer > 0, or '-'")
            duration = int(duration)
        else:
            if int(prio) > 0:
                raise ValueError("Infinite duration lines must have priority 0")

        epoch = dt.datetime.utcfromtimestamp(0)
        epoch_milliseconds = int((time - epoch).total_seconds() * 1000)

        possible_scheduling_modes = ['common', 'special', 'discretionary']
        if scheduling_mode not in possible_scheduling_modes:
            raise ValueError(f"Unknown scheduling mode type {scheduling_mode} not in {possible_scheduling_modes}")

        # Don't bother testing past experiments, formats/settings/capabilities/etc. could have changed
        if not dt.datetime.utcnow() - time > dt.timedelta(days=1):  # Test if experiment is in future or past day
            # See if the experiment itself would run
            # This is a full path to /.../{site}.scd file, only want {site}
            site_name = os.path.basename(self.scd_filename).replace('.scd', '')
            args = ['--site_id', site_name,
                    '--experiments', experiment,
                    '--kwargs', kwargs,
                    '--module', 'experiment_unittests']
            test_program = experiment_unittests.run_tests(args, buffer=True, print_results=False)
            if len(test_program.result.failures) != 0 or len(test_program.result.errors) != 0:
                raise ValueError("Experiment could not be scheduled due to errors in experiment.\n"
                                 f"Errors: {test_program.result.errors}\n"
                                 f"Failures: {test_program.result.failures}")

        return {"timestamp": epoch_milliseconds,
                "time": time,
                "duration": str(duration),
                "prio": str(prio),
                "experiment": experiment,
                "scheduling_mode": scheduling_mode,
                "kwargs": kwargs,
                "embargo": embargo}

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
            kwarg_entries = line[6:]
            embargo_flag = '--embargo' in kwarg_entries
            if embargo_flag:
                kwarg_entries.remove('--embargo')
            kwargs = " ".join(kwarg_entries)

            # date time experiment mode priority duration [kwargs]
            scd_lines.append(self.check_line(line[0], line[1], line[4], line[5], line[3], line[2], kwargs,
                                             embargo=embargo_flag))

        if len(scd_lines) == 0:
            print('WARNING: SCD file empty; default normalscan will run')
            # add the default infinite duration line 
            scd_lines.append(self.scd_default)

        return scd_lines

    @classmethod
    def fmt_line(cls, line_dict):
        """
        Formats a dictionary with line info into a text line for file.

        :param  line_dict: A dict that holds all the line info.
        :type   line_dict: dict

        :returns:   Formatted string.
        :rtype:     str
        """
        line_str = cls.line_fmt.format(datetime=line_dict["time"].strftime(cls.scd_dt_fmt),
                                       prio=line_dict["prio"],
                                       experiment=line_dict["experiment"],
                                       scheduling_mode=line_dict["scheduling_mode"],
                                       duration=line_dict["duration"],
                                       embargo='--embargo' if line_dict["embargo"] else '',
                                       kwargs=line_dict["kwargs"])
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
                 duration='-', kwargs='', embargo=False):
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
        :param  kwargs:      kwargs for the experiment instantiation. (Default value = '')
        :type   kwargs:      str
        :param  embargo:            flag for embargoing files. (Default value = False)
        :type   embargo:            bool

        :raises ValueError: If line parameters are invalid or if line is a duplicate.
        """
        new_line = self.check_line(yyyymmdd, hhmm, experiment, scheduling_mode, prio, duration, kwargs,
                                   embargo=embargo)

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
                    duration='-', kwargs='', embargo=False):
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
        :param  kwargs:      kwargs for the experiment instantiation. (Default value = '')
        :type   kwargs:      str
        :param  embargo:            flag for embargoing files. (Default value = False)
        :type   embargo:            bool

        :raises ValueError: If line parameters are invalid or if line does not exist.
        """

        line_to_rm = self.check_line(yyyymmdd, hhmm, experiment, scheduling_mode, prio, duration, kwargs,
                                     embargo=embargo)

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
        time is between schedule line times, it provides any lines in the schedule from the past that
        haven't ended yet, plus the most recently timestamped infinite-duration line, plus all future
        lines. If the provided time is before any lines in the schedule, it provides all schedule lines.

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

        # Sort the lines by timestamp, and for equal times, in reverse priority
        scd_lines = sorted(scd_lines, key=lambda x: x['prio'], reverse=True)
        scd_lines = sorted(scd_lines, key=lambda x: x['timestamp'])

        if not scd_lines:
            raise IndexError("Schedule file is empty. No lines can be returned")

        epoch = dt.datetime.utcfromtimestamp(0)
        epoch_milliseconds = int((time - epoch).total_seconds() * 1000)

        relevant_lines = []
        past_infinite_line_added = False
        for line in reversed(scd_lines):
            if line['timestamp'] > epoch_milliseconds:
                relevant_lines.append(line)
            elif line['timestamp'] == epoch_milliseconds:
                relevant_lines.append(line)
            else:
                # Include the most recent infinite line
                if line['duration'] == '-':
                    if not past_infinite_line_added:
                        relevant_lines.append(line)
                        past_infinite_line_added = True
                else:
                    # If the line ends after the current time, include the line
                    duration_ms = int(line['duration']) * 60 * 1000
                    line_end = line['timestamp'] + duration_ms
                    if line_end >= epoch_milliseconds:
                        relevant_lines.append(line)

        # Put the lines into chronological order (oldest to newest)
        relevant_lines.reverse()
        return relevant_lines


if __name__ == "__main__":
    filename = sys.argv[1]

    scd_util = SCDUtils(filename)
