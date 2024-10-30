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
from typing_extensions import Annotated, Union, Literal, Self, Optional

from pydantic.dataclasses import dataclass
from pydantic import field_validator, Field, model_validator

borealis_path = os.environ["BOREALISPATH"]
sys.path.append(f"{borealis_path}/tests/experiments")
import experiment_unittests


class LineConfig:
    """
    This class configures pydantic options for ScheduleLine.

    validate_assignment: Whether to run all validators for a field whenever field is changed (init or after init)
    extra: Whether to allow extra fields not defined when instantiating
    arbitrary_types_allowed: Whether to allow arbitrary types like user-defined classes (e.g. Options, DecimationScheme)
    """

    validate_assignment = True
    extra = "allow"
    arbitrary_types_allowed = False


@dataclass(config=LineConfig)
class ScheduleLine:
    timestamp: dt.datetime
    duration: Union[str, dt.timedelta]
    experiment: str
    priority: Annotated[int, Field(ge=0, le=20)]
    scheduling_mode: Literal["common", "discretionary", "special"]
    kwargs: list[str] = Field(default_factory=list)
    embargo: bool = False
    rawacf_format: Optional[Literal["dmap", "hdf5"]] = None

    def __str__(self):
        dur = self.duration
        if isinstance(dur, dt.timedelta):
            dur = int(round(dur.total_seconds() / 60))
        line = (
            f"{self.timestamp.strftime('%Y%m%d %H:%M')}"
            f" {dur}"
            f" {self.priority}"
            f" {self.experiment}"
            f" {self.scheduling_mode}"
            f"{' --embargo' if self.embargo else ''}"
            f"{' --rawacf_format=' + self.rawacf_format if self.rawacf_format is not None else ''}"
            f"{' ' + ' '.join(self.kwargs) if len(self.kwargs) > 0 else ''}"
        )
        return line

    @field_validator("duration")
    @classmethod
    def check_duration(cls, v: Union[str, dt.timedelta]) -> Union[str, dt.timedelta]:
        """Verifies duration is either `'-'`, or a positive timedelta"""
        if isinstance(v, str) and v != "-":
            raise ValueError("only '-' supported for string-type duration")
        elif isinstance(v, dt.timedelta):
            if v.total_seconds() <= 0.0:
                raise ValueError("must be positive")
            v = dt.timedelta(minutes=int(v.total_seconds() // 60))
            if v.total_seconds() <= 60.0:
                raise ValueError("must be greater than one minute")
        return v

    @model_validator(mode="after")
    def check_inf_line_priority(self) -> Self:
        if self.duration == "-" and self.priority > 0:
            raise ValueError("infinite duration lines must have priority = 0")
        return self

    def test(self, site_id: str):
        """
        Check validity of fields and run the line through experiment unit tests to check that the experiment will run.

        :param   site_id: Three-letter site code
        :type    site_id: str
        """

        args = [
            "--site_id",
            site_id,
            "--experiments",
            self.experiment,
            "--kwargs",
            " ".join(self.kwargs),
            "--module",
            "experiment_unittests",
        ]
        test_program = experiment_unittests.run_tests(
            args, buffer=True, print_results=False
        )
        if (
            len(test_program.result.failures) != 0
            or len(test_program.result.errors) != 0
        ):
            raise ValueError(
                "Experiment could not be scheduled due to errors in experiment.\n"
                f"Errors: {test_program.result.errors}\n"
                f"Failures: {test_program.result.failures}"
            )

    @classmethod
    def from_str(cls, line: str) -> Self:
        """
        Parses a line from the schedule file
        """
        fields = line.split()

        if fields[2] == "-":
            duration = "-"
        else:
            duration = dt.timedelta(minutes=int(fields[2]))

        kwargs = fields[6:]
        embargo = "--embargo" in kwargs
        if embargo:
            kwargs.remove("--embargo")

        raw_format = None
        raw_format_flag = ["rawacf_format" in x for x in kwargs]
        if any(raw_format_flag):
            idx = raw_format_flag.index(True)
            if len(kwargs[idx].split("=")) == 1:
                # supplied as --rawacf_format <format>
                raw_format = kwargs.pop(idx + 1)
                kwargs.pop(idx)
            else:
                # supplied as --rawacf_format=<format>
                raw_format = kwargs.pop(idx).split("=")[1]

        scd_line = ScheduleLine(
            timestamp=dt.datetime.strptime(f"{fields[0]} {fields[1]}", "%Y%m%d %H:%M"),
            duration=duration,
            priority=int(fields[3]),
            experiment=fields[4],
            scheduling_mode=fields[5],
            embargo=embargo,
            kwargs=kwargs,
            rawacf_format=raw_format,
        )
        return scd_line


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

    def __init__(self, scd_filename, site_id):
        """
        :param  scd_filename:   Schedule file name
        :type   scd_filename:   str
        :param       site_id:   Three-letter radar site ID, for testing the experiment
        :type        site_id:   str
        """
        self.scd_filename = scd_filename
        self.site_id = site_id

        """Default event to run if no other infinite duration line is scheduled"""
        self.scd_default = self.create_line(
            "20000101", "00:00", "normalscan", "common", 0, "-", []
        )

    def create_line(
        self,
        yyyymmdd,
        hhmm,
        experiment,
        scheduling_mode,
        prio,
        duration,
        kwargs,
        embargo=False,
        rawacf_format=None,
    ) -> ScheduleLine:
        """
        Creates a line dictionary from inputs, turning the date and time into a timestamp since epoch.

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
        :param  duration:           a duration to run for.
        :type   duration:           Union[str, dt.timedelta]
        :param  kwargs:             kwargs for the experiment instantiation.
        :type   kwargs:             list[str]
        :param  embargo:            flag for embargoing files. (Default value = False)
        :type   embargo:            bool
        :param  rawacf_format:      The file format to save rawacf files in.
        :type   rawacf_format:      str

        :returns:   Line details
        :rtype:     ScheduleLine
        """
        # create datetime from args to see if valid. Value error for incorrect format
        time = dt.datetime.strptime(yyyymmdd + " " + hhmm, self.scd_dt_fmt)

        return ScheduleLine(
            timestamp=time,
            duration=duration,
            priority=prio,
            experiment=experiment,
            scheduling_mode=scheduling_mode,
            kwargs=kwargs,
            embargo=embargo,
            rawacf_format=rawacf_format,
        )

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

        scd_lines = [ScheduleLine.from_str(line.strip()) for line in raw_scd]

        if len(scd_lines) == 0:
            print("WARNING: SCD file empty; default normalscan will run")
            # add the default infinite duration line
            scd_lines.append(self.scd_default)

        return scd_lines

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
        text_lines = [str(x) for x in scd_lines]

        shutil.copy(self.scd_filename, self.scd_filename + ".bak")

        with open(self.scd_filename, "w") as f:
            for line in text_lines:
                f.write(f"{line}\n")

    def add_line(
        self,
        yyyymmdd,
        hhmm,
        experiment,
        scheduling_mode,
        prio=0,
        duration="-",
        kwargs=None,
        embargo=False,
        rawacf_format=None,
    ):
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
        :param  kwargs:             kwargs for the experiment instantiation. (Default value = '')
        :type   kwargs:             list[str]
        :param  embargo:            flag for embargoing files. (Default value = False)
        :type   embargo:            bool
        :param  rawacf_format:      File format to use when writing rawacf files.
        :type   rawacf_format:      str

        :raises ValueError: If line parameters are invalid or if line is a duplicate.
        """
        new_line = self.create_line(
            yyyymmdd,
            hhmm,
            experiment,
            scheduling_mode,
            prio,
            duration,
            kwargs,
            embargo=embargo,
            rawacf_format=rawacf_format,
        )

        scd_lines = self.read_scd()

        if new_line in scd_lines:
            raise ValueError("Line is a duplicate of an existing line")

        if any(
            [
                (
                    new_line.timestamp == line.timestamp
                    and new_line.priority == line.priority
                )
                for line in scd_lines
            ]
        ):
            raise ValueError("Priority already exists at this time")

        try:
            new_line.test(self.site_id)
        except ValueError as e:
            raise ValueError("Unable to add line:\n", str(e))

        scd_lines.append(new_line)

        # sort priorities in reverse so that they are descending order. Then sort everything by timestamp
        new_scd = sorted(scd_lines, key=lambda x: x.priority, reverse=True)
        new_scd = sorted(new_scd, key=lambda x: x.timestamp)

        self.write_scd(new_scd)

    def remove_line(
        self,
        yyyymmdd,
        hhmm,
        experiment,
        scheduling_mode,
        prio=0,
        duration="-",
        kwargs=None,
        embargo=False,
        rawacf_format=None,
    ):
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
        :param  kwargs:             kwargs for the experiment instantiation. (Default value = '')
        :type   kwargs:             list[str]
        :param  embargo:            flag for embargoing files. (Default value = False)
        :type   embargo:            bool
        :param  rawacf_format:      File format to use when writing rawacf files.
        :type   rawacf_format:      str

        :raises ValueError: If line parameters are invalid or if line does not exist.
        """
        if kwargs is None:
            kwargs = list()
        line_to_rm = self.create_line(
            yyyymmdd,
            hhmm,
            experiment,
            scheduling_mode,
            prio,
            duration,
            kwargs,
            embargo=embargo,
            rawacf_format=rawacf_format,
        )

        scd_lines = self.read_scd()
        if line_to_rm in scd_lines:
            scd_lines.remove(line_to_rm)
        else:
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
        scd_lines = sorted(scd_lines, key=lambda x: x.priority, reverse=True)
        scd_lines = sorted(scd_lines, key=lambda x: x.timestamp)

        if not scd_lines:
            raise IndexError("Schedule file is empty. No lines can be returned")

        relevant_lines = []
        past_infinite_line_added = False
        for line in reversed(scd_lines):
            if line.timestamp >= time:
                relevant_lines.append(line)
            else:
                # Include the most recent infinite line
                if line.duration == "-":
                    if not past_infinite_line_added:
                        relevant_lines.append(line)
                        past_infinite_line_added = True
                else:
                    # If the line ends after the current time, include the line
                    if line.timestamp + line.duration >= time:
                        relevant_lines.append(line)

        # Put the lines into chronological order (oldest to newest)
        relevant_lines.reverse()
        return relevant_lines
