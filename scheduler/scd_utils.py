#!/usr/bin/python3
"""
scd_utils.py
~~~~~~~~~~~~
Utilities for working with scd files

:copyright: 2019 SuperDARN Canada
"""

import bisect
import copy
import datetime as dt
import os
import shutil
import subprocess as sp
import sys
from typing_extensions import Annotated, Union, Literal, Self, Optional

from pydantic.dataclasses import dataclass
from pydantic import field_validator, Field, model_validator, ConfigDict

borealis_path = os.environ["BOREALISPATH"]
sys.path.append(f"{borealis_path}/tests/experiments")


class ScheduleError(Exception):
    """
    Error in the schedule.
    """

    pass


@dataclass(
    config=ConfigDict(
        validate_assignment=True, extra="allow", arbitrary_types_allowed=False
    )
)
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

    def format_to_atq(self, first_event_flag=False):
        call = (
            f"echo 'screen -d -m -S starter {borealis_path}/scripts/steamed_hams.py"
            f" {self.experiment}"
            f" release"
            f" {self.scheduling_mode}"
            f"{' --embargo' if self.embargo else ''}"
            f"{' --rawacf-format=' + self.rawacf_format if self.rawacf_format is not None else ''}"
            f"{' --kwargs ' + ' '.join(self.kwargs) if len(self.kwargs) > 0 else ''}'"
        )

        if first_event_flag:
            cmd_str = call + " | at now + 1 minute"
        else:
            cmd_str = call + self.timestamp.strftime(" | at -t %Y%m%d%H%M")
        return cmd_str

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

    @field_validator("timestamp")
    @classmethod
    def check_timestamp(cls, v: dt.datetime) -> dt.datetime:
        """Verifies v is a time-aware datetime object"""
        if v.tzinfo != dt.timezone.utc:
            raise ValueError("timestamp must have UTC timezone")

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
            site_id,
            "--experiments",
            self.experiment,
        ]

        if len(self.kwargs) > 0:
            args.extend(
                [
                    "--kwargs",
                    " ".join(self.kwargs),
                ]
            )
        try:
            sp.run(
                [
                    "python3",
                    f"{os.environ['BOREALISPATH']}/tests/experiments/test_as_site.py",
                ]
                + args,
                check=True,
            )
        except sp.CalledProcessError as e:
            raise ScheduleError(
                "Experiment could not be scheduled due to errors in experiment.\n"
                + str(e)
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

        line_timestamp = dt.datetime.strptime(
            f"{fields[0]} {fields[1]}", "%Y%m%d %H:%M"
        )
        line_timestamp = line_timestamp.replace(tzinfo=dt.timezone.utc)
        scd_line = ScheduleLine(
            timestamp=line_timestamp,
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
        date = dt.datetime.now(dt.timezone.utc)

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

    scd_dt_fmt = "%Y%m%d %H:%M"
    """String format for parsing and writing datetimes"""

    clear_command = "for i in `atq | awk '{print $1}'`;do atrm $i;done"
    """Clear the atq."""

    get_atq_cmd = """
        for j in $(atq | sort -k6,6 -k3,3M -k4,4 -k5,5 | cut -f 1); do
           atq |grep -P "^$j\t";
           at -c "$j" | tail -n 2;
        done
    """
    """
    This command is basically: for j in atq job number, print job num, time and command
    More detail: sort the atq first by year, then month name ('-M flag), then day of month
    Then hour, minute and second. Finally, just get the atq index (job #) in first column
    then, iterate through all jobs in the atq, list them to standard output, get the last 2 lines
    """

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

    @classmethod
    def clear_atq(cls):
        """
        Remove all scheduled commands in the ``atq``.
        """
        sp.call(cls.clear_command, shell=True)

    @classmethod
    def get_atq(cls):
        """
        Retrieve all scheduled commands in the ``atq``.
        """
        retval = sp.run(
            cls.get_atq_cmd, capture_output=True, check=True, shell=True
        ).stdout
        if retval is None:
            retval = b""
        return retval

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
        time = dt.datetime.strptime(yyyymmdd + " " + hhmm, self.scd_dt_fmt).replace(
            tzinfo=dt.timezone.utc
        )

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
            raise ScheduleError("Priority already exists at this time")

        try:
            new_line.test(self.site_id)
        except ScheduleError as e:
            raise ScheduleError("Unable to add line:\n", str(e))
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
            raise ScheduleError("Line does not exist in SCD")

        self.write_scd(scd_lines)

    def get_relevant_lines(self, time_of_interest: dt.datetime):
        """
        Gets the currently scheduled and future lines given a supplied time. If the provided time is
        equal to a scheduled line time, it provides that line and all future lines. If the provided
        time is between schedule line times, it provides any lines in the schedule from the past that
        haven't ended yet, plus the most recently timestamped infinite-duration line, plus all future
        lines. If the provided time is before any lines in the schedule, it provides all schedule lines.

        :param  time_of_interest: Current time
        :type   time_of_interest: dt.datetime

        :returns:   List of relevant lines.
        :rtype:     list[ScheduleLine]

        :raises ScheduleError: If schedule file is empty.
        """

        scd_lines = self.read_scd()

        # Sort the lines by timestamp, and for equal times, in reverse priority
        scd_lines = sorted(scd_lines, key=lambda x: x.priority, reverse=True)
        scd_lines = sorted(scd_lines, key=lambda x: x.timestamp)

        if not scd_lines:
            raise ScheduleError("Schedule file is empty. No lines can be returned")

        relevant_lines = []
        past_infinite_line_added = False
        for line in reversed(scd_lines):
            if line.timestamp >= time_of_interest:
                relevant_lines.append(line)
            else:
                # Include the most recent infinite line
                if line.duration == "-":
                    if not past_infinite_line_added:
                        relevant_lines.append(line)
                        past_infinite_line_added = True
                else:
                    # If the line ends after the current time, include the line
                    if line.timestamp + line.duration >= time_of_interest:
                        relevant_lines.append(line)

        # Put the lines into chronological order (oldest to newest)
        relevant_lines.reverse()
        return relevant_lines

    @staticmethod
    def resolve_schedule(scd_lines, time_of_interest):
        """
        Creates a true timeline from the scd lines, accounting for priority and duration of each line.
        Will reorder and add breaks to lines to account for differing priorities and durations. Keep the
        same line format.

        :param  scd_lines:          All lines to schedule
        :type   scd_lines:          list[ScheduleLine]
        :param  time_of_interest:   Time to use when finding all schedule lines that should run (i.e. now and in the future)
        :type   time_of_interest:   dt.datetime

        :returns: All distinct scheduling events, in chronological order
        :rtype: list[ScheduleLine]
        """

        def reduce_intervals(current_list: list[tuple], value: tuple):
            """
            current_list: (start, end) tuples that are sorted, with no overlaps (i.e. start[k] > end[k-1])
            value: (start, end) tuple
            Finds the new master list of (start, end) tuples with inclusion of value, and the list of (start, end) times
            that value filled in.

            E.g. current_list = [(0, 1), (2, 3), (5, 6)], value = (1, 3), output = [(0, 3), (5, 6)] and [(1, 2)]
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

            # bisect_left() finds the index such that all items < are to the left (equal values to the right)
            item_idx = bisect.bisect_left(start_times, value[0])
            reduced_list = current_list[:item_idx] + current_list[end_idx:]
            enclosed_times = current_list[item_idx:end_idx]  # rest of current_list

            # e.g. current_list = [(0, 1), (2, 3), (5, 6)] and value = (1, 3) so item_idx = 1, end_idx = 2
            # then reduced_list = [(0, 1), (5, 6)] and enclosed_times = [(2, 3)]
            reduced_list.insert(item_idx, value)  # now [(0, 1), (1, 3), (5, 6)]
            times_filled = [value]
            if (
                item_idx > 0
                and reduced_list[item_idx - 1][1] >= reduced_list[item_idx][0]
            ):
                # have to combine elements (0, 1) and (1, 3) into (0, 3), so reduced_list = [(0, 3), (5, 6)]
                val = reduced_list.pop(item_idx)
                times_filled[0] = (
                    reduced_list[item_idx - 1][1],
                    val[1],
                )  # truncate the start
                reduced_list[item_idx - 1] = (
                    reduced_list[item_idx - 1][0],
                    times_filled[0][1],
                )
                item_idx -= 1
            if (
                item_idx < len(reduced_list) - 1
                and reduced_list[item_idx + 1][0] <= value[1]
            ):
                # e.g. reduced_list = [(0, 1), (1, 2)] with item_idx = 0, want result of [(0, 2)]
                val = reduced_list.pop(item_idx + 1)
                times_filled[0] = (times_filled[0][0], val[0])  # truncate at the end
                reduced_list[item_idx] = (reduced_list[item_idx][0], val[1])

            # now have to split times_filled with all the items that were fully enclosed by it
            # e.g. enclosed_times = [(2, 3)], times_filled starts as [(1, 3)], we want to remove
            # enclosed_times from times_filled. Thus, a result of [(1, 2)]
            for x in enclosed_times:
                if x[0] < times_filled[-1][1]:
                    # block starts before values run ends
                    end_val = times_filled.pop()
                    if end_val[0] < x[0]:
                        # Add in the first bit of end_val before this enclosed block starts
                        times_filled.append((end_val[0], x[0]))
                    if end_val[1] > x[1]:
                        # Add in the rest of end_val after this enclosed block ends
                        times_filled.append((x[1], end_val[1]))

            return reduced_list, times_filled

        sorted_lines = sorted(scd_lines, key=lambda x: x.timestamp)
        sorted_lines.reverse()
        sorted_lines.sort(key=lambda x: x.priority, reverse=True)
        # at this stage, lines are sorted by priority, then by reverse timestamp for equal priority,
        # then for two lines with equal priority and timestamp, by reverse order in the schedule file.

        scheduled = []  # list of ScheduleLine objects, with no overlap between them
        scheduled_times = []  # complete list of (start, end) times that have an experiment scheduled
        for line in sorted_lines:
            start = line.timestamp
            if line.duration == "-":
                end = dt.datetime.max.replace(tzinfo=dt.timezone.utc)
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

        # If there are multiple events scheduled that start in the past, then remove all but the most recent one.
        # E.g. there's a past infinite line, and you're in the middle of a finite experiment line. This would have
        # both the infinite and finite lines starting in the past, so we want to throw out the infinite one since the
        # finite one supersedes it
        in_past = [x for x in scheduled if x.timestamp < time_of_interest]
        if len(in_past) > 1:
            scheduled = scheduled[len(in_past) - 1 :]

        return scheduled

    @classmethod
    def timeline_to_atq(cls, timeline: list[ScheduleLine]):
        """
        Converts the created timeline to actual atq commands.

        Remove old events and then schedule everything recent. The first entry should be the currently running event,
        so it gets scheduled immediately.

        :param  timeline:   A list holding all timeline events.
        :type   timeline:   list[ScheduleLine]

        :returns:   output of the executed atq command
        :rtype:     bytes
        """

        cls.clear_atq()

        atq = []
        first_event = True
        for event in timeline:
            atq_call = event.format_to_atq(first_event)
            atq.append(atq_call)
            first_event = False
        for cmd in atq:
            sp.call(cmd, shell=True)

        return cls.get_atq()

    def parse_and_schedule(self, time_of_interest: dt.datetime = None):
        """
        Parses the schedule file, and schedules all relevant lines using ``at``.

        :param time_of_interest: Current time.
        :type  time_of_interest: dt.Datetime

        :raises ScheduleError: if any of the relevant schedule lines are invalid, or the schedule file is empty.
        """
        if time_of_interest is None:
            time_of_interest = dt.datetime.now(dt.timezone.utc)

        relevant_lines = self.get_relevant_lines(time_of_interest)
        for i, line in enumerate(relevant_lines):
            line.test(self.site_id)

        timeline = self.resolve_schedule(relevant_lines, time_of_interest)
        new_atq_str = self.timeline_to_atq(timeline)

        return new_atq_str
