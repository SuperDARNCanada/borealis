"""
Test module for the scheduler code.
Run via: 'python3 test_scheduler.py'

:copyright: 2022 SuperDARN Canada
:author: Kevin Krieger
"""

import shutil
import unittest
import os
from pathlib import Path
import sys
import tempfile
import datetime
import subprocess as sp
import re

if "BOREALISPATH" not in os.environ:
    BOREALISPATH = (
        f"{Path(__file__).resolve().parents[2]}"  # Two directories up from this file
    )
    os.environ["BOREALISPATH"] = BOREALISPATH
else:
    BOREALISPATH = os.environ["BOREALISPATH"]

sys.path.append(BOREALISPATH)

from scheduler import scd_utils
from scheduler import local_scd_server, remote_server


class TestSchedulerUtils(unittest.TestCase):
    """
    unittest class to test the scheduler utilities module. All test methods must begin with the word 'test' to be run
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.site_id = "sas"
        # A file that has a bunch of lines in the scd file that should all pass all tests
        self.good_scd_file = (
            f"{os.environ['BOREALISPATH']}/tests/scheduler/good_scd_file.scd"
        )
        # A file that has 5 arguments (missing the scheduling mode)
        self.incorrect_args_scd = (
            f"{os.environ['BOREALISPATH']}/tests/scheduler/incorrect_args.scd"
        )
        self.line_fmt = (
            "{self.dt} {self.dur} {self.prio} {self.exp} {self.mode} {self.kwargs}"
        )
        self.scd_dt_fmt = "%Y%m%d %H:%M"
        self.yyyymmdd = "20221014"
        self.hhmm = "12:34"
        self.dt = "20221011 00:00"
        self.dur = 120
        self.prio = 10
        self.exp = "normalscan"
        self.mode = "common"
        self.kwargs = ""
        self.no_perms = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        self.no_perms_file = self.no_perms.name
        os.chmod(self.no_perms_file, 0o000)
        self.no_perms.close()
        time_of_interest = datetime.datetime(2000, 1, 1, 6, 30)
        time_of_interest2 = datetime.datetime(2022, 4, 5, 16, 56)
        self.linedict = {
            "time": time_of_interest,
            "prio": 0,
            "experiment": "normalscan",
            "scheduling_mode": "common",
            "duration": 60,
            "kwargs": "-",
            "embargo": False,
        }
        self.linestr = "20000101 06:30 60 0 normalscan common  -\n"
        self.linedict2 = {
            "time": time_of_interest2,
            "prio": 15,
            "experiment": "twofsound",
            "scheduling_mode": "discretionary",
            "duration": 360,
            "kwargs": "freq1=10500 freq2=13100",
            "embargo": False,
        }
        self.linestr2 = (
            "20220405 16:56 360 15 twofsound discretionary  freq1=10500 freq2=13100\n"
        )
        self.maxDiff = None

    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

    # get_next_month_from_date tests
    def test_wrong_type(self):
        """
        Test getting the next month from a given date with wrong type
        """
        date = 20221114
        with self.assertRaises(TypeError):
            scd_utils.get_next_month_from_date(date)

    def test_next_month(self):
        """
        Test getting the next month from a given date
        """
        date = datetime.datetime(2022, 1, 2)
        date2 = datetime.datetime(2022, 2, 1)
        self.assertEqual(scd_utils.get_next_month_from_date(date), date2)
        date = datetime.datetime(2022, 12, 13)
        date2 = datetime.datetime(2023, 1, 1)
        self.assertEqual(scd_utils.get_next_month_from_date(date), date2)
        date = datetime.datetime(2022, 12, 5)
        date2 = datetime.datetime(2023, 1, 1)
        self.assertEqual(scd_utils.get_next_month_from_date(date), date2)

    # create_line tests

    def test_invalid_yyyymmdd(self):
        """
        Test an invalid start date
        """
        yyyymmdd = None
        # Invalid type for yyyymmdd should raise a TypeError in strptime.
        # it will also raise ValueError if the date_string and format can't be parsed by time.strptime()
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        with self.assertRaises(TypeError):
            scdu.create_line(
                yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
        yyyymmdd = 0  # TypeError
        with self.assertRaises(TypeError):
            scdu.create_line(
                yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
        yyyymmdd = "20211"  # ValueError
        with self.assertRaises(ValueError):
            scdu.create_line(
                yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
        yyyymmdd = "20221512"  # ValueError, 15th month
        with self.assertRaises(ValueError):
            scdu.create_line(
                yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
        yyyymmdd = "20220230"  # ValueError, 30th day in Feb
        with self.assertRaises(ValueError):
            scdu.create_line(
                yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )

    def test_invalid_hhmm(self):
        """
        Test an invalid start hour and minute. required format: "HH:MM"
        """
        hhmm = None
        # Invalid type for hhmm should raise a TypeError in strptime.
        # it will also raise ValueError if the date_string and format can't be parsed by time.strptime()
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        with self.assertRaises(TypeError):
            scdu.create_line(
                self.yyyymmdd,
                hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
        hhmm = 0  # TypeError
        with self.assertRaises(TypeError):
            scdu.create_line(
                self.yyyymmdd,
                hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
        hhmm = "2011"  # ValueError
        with self.assertRaises(ValueError):
            scdu.create_line(
                self.yyyymmdd,
                hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
        hhmm = "2500"  # ValueError, 25th hour
        with self.assertRaises(ValueError):
            scdu.create_line(
                self.yyyymmdd,
                hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )

    # test_line tests

    def test_invalid_experiment(self):
        """
        Test an invalid experiment name - Must be an experiment named in the repo
        """
        exp = "non-existent_Experiment"
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)
        exp = None
        with self.assertRaises(TypeError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)
        exp = 5
        with self.assertRaises(TypeError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)

    def test_invalid_mode(self):
        """
        Test an invalid mode (possible: ['common', 'special', 'discretionary'])
        """
        mode = "notamode"
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)
        mode = None
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)
        mode = 0
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)

    def test_invalid_prio(self):
        """
        Test an invalid priority (string/integer from 0 to 20 inclusive)
        """
        prio = "blah"
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)
        prio = None
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)
        prio = -1
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)
        prio = 21
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                prio,
                self.dur,
                self.kwargs,
            )
            scdu.test_line(line)

    def test_invalid_duration(self):
        """
        Test an invalid duration (optional: needs to be a positive integer/string minutes, or '-')
        """
        dur = 0
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                dur,
                self.kwargs,
            )
            scdu.test_line(line)
        dur = -1
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                dur,
                self.kwargs,
            )
            scdu.test_line(line)
        dur = -50
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                dur,
                self.kwargs,
            )
            scdu.test_line(line)
        dur = 53.09
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                dur,
                self.kwargs,
            )
            scdu.test_line(line)
        dur = ""
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                dur,
                self.kwargs,
            )
            scdu.test_line(line)
        dur = None
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                dur,
                self.kwargs,
            )
            scdu.test_line(line)

    def test_invalid_kwargs_string(self):
        """
        Test an invalid kwargs string (optional arguments that may be passed to an experiment's kwargs)
        """
        kwargs_str = "this doesnt mean anything to the experiment"
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                kwargs_str,
            )
            scdu.test_line(line)
        kwargs_str = 0
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                kwargs_str,
            )
            scdu.test_line(line)
        kwargs_str = -1
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                kwargs_str,
            )
            scdu.test_line(line)
        kwargs_str = 56.9
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                kwargs_str,
            )
            scdu.test_line(line)
        kwargs_str = None
        with self.assertRaises(ValueError):
            line = scdu.create_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                kwargs_str,
            )
            scdu.test_line(line)

    # read_scd tests
    def test_invalid_num_args(self):
        """
        Test an invalid number of arguments, requires 6 or 7 args
        """
        scdu = scd_utils.SCDUtils(self.incorrect_args_scd, self.site_id)
        with self.assertRaises(IndexError):
            scdu.read_scd()

    def test_bad_scd_file(self):
        """
        Test giving a non-existent scd file to the module
        """
        scdfile = "thisfilelikelydoesntexist90758hjna;ksjn"
        scdu = scd_utils.SCDUtils(scdfile, self.site_id)
        with self.assertRaises(IOError):
            scdu.read_scd()

    def test_empty_scd_file(self):
        """
        Test giving an empty scd file to the module, prints warning, returns and appends default
        line to the scd file
        """
        scdfile = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scdfile.name, self.site_id)
        scdfile.close()
        self.assertEqual(
            scdu.read_scd(),
            [scdu.create_line("20000101", "00:00", "normalscan", "common", "0", "-")],
        )

    # fmt_line tests
    def test_fmt_line(self):
        """
        Test that the result from fmt_line agrees with what it should
        """
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        self.assertEqual(scdu.fmt_line(self.linedict), self.linestr.strip())
        self.assertEqual(scdu.fmt_line(self.linedict2), self.linestr2.strip())

    # write_scd tests
    def test_no_perms(self):
        """
        Test that a scd file without good permissions to read or write will cause PermissionError
        """
        scdu = scd_utils.SCDUtils(self.no_perms_file, self.site_id)
        lines = []
        with self.assertRaises(PermissionError):
            scdu.write_scd(lines)

    def test_file_dne(self):
        """
        Test write_scd with no actual file for the scd
        """
        scdu = scd_utils.SCDUtils("blkja;lskjdf;lkhj", self.site_id)
        lines = []
        with self.assertRaises(FileNotFoundError):
            scdu.write_scd(lines)

    def test_file_is_dir(self):
        """
        Test write_scd with a directory given for the scd file
        """
        scdu = scd_utils.SCDUtils(os.environ["HOME"], self.site_id)
        lines = []
        with self.assertRaises(IsADirectoryError):
            scdu.write_scd(lines)

    def test_scd_backup(self):
        """
        Test write_scd with a good line, so that it creates a .bak of the scd file
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scd_file.close()
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        lines = [self.linedict, self.linedict2]
        scdu.write_scd(lines)

        # Verify there's a .bak file parallel with the temp file with nothing in it
        # Verify there's a scd file with the added lines to it
        bak_file = scd_file.name + ".bak"
        self.assertTrue(os.path.exists(bak_file))
        with open(bak_file, "r") as f, open(scd_file.name, "r") as s:
            bak_lines = f.readlines()
            scd_lines = s.readlines()
            self.assertEqual([self.linestr, self.linestr2], scd_lines)
            self.assertEqual(bak_lines, [])

    # add_line tests
    def test_duplicate_line(self):
        """
        Test trying to add duplicate line
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        scd_file.close()
        with self.assertRaisesRegex(
            ValueError, "Line is a duplicate of an existing line"
        ):
            scdu.add_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
            scdu.add_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )

    def test_priority_time_dup(self):
        """
        Test trying to add line with the same time and priority as an existing line
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        scd_file.close()
        with self.assertRaisesRegex(ValueError, "Priority already exists at this time"):
            scdu.add_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )
            scdu.add_line(
                self.yyyymmdd,
                self.hhmm,
                "twofsound",
                "special",
                self.prio,
                "1440",
                self.kwargs,
            )

    def test_add_line(self):
        """
        Test trying to add a line to an empty file. This will add the default normalscan first
        """
        # TODO: Should the behaviour of add_line be adding a default line first if it's an empty file?
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        scd_file.close()
        scdu.add_line(
            self.yyyymmdd,
            self.hhmm,
            self.exp,
            self.mode,
            self.prio,
            self.dur,
            self.kwargs,
        )
        with open(scd_file.name, "r") as f:
            lines = f.readlines()
        self.assertEqual(
            len(lines) - 1, 1
        )  # -1 to handle the extra default line that was added in read_scd()
        line = lines[1].split()  # Get the second line, which is the one we added
        self.assertEqual(line[0], self.yyyymmdd)
        self.assertEqual(line[1], self.hhmm)
        self.assertEqual(line[2], str(self.dur))
        self.assertEqual(line[3], str(self.prio))
        self.assertEqual(line[4], self.exp)
        self.assertEqual(line[5], self.mode)

    def test_add_lines_sorted(self):
        """
        Test trying to add multiple lines to a file and have them properly sorted by time and priority
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        scd_file.close()
        counter = 0
        for yyyymmdd in range(20221001, 1, 20221031):
            for hhmm in range(59, -1, 0):
                counter += 1
                hmstr = f"{hhmm:0<4}"[:2] + ":" + f"{hhmm:0<4}"[2:]
                scdu.add_line(
                    str(f"{yyyymmdd:0>8}"),
                    hmstr,
                    self.exp,
                    self.mode,
                    self.prio,
                    self.dur,
                    self.kwargs,
                )
                if hhmm % 2 == 0:
                    prio = 4
                else:
                    prio = 7
                scdu.add_line(
                    str(f"{yyyymmdd:0>8}"),
                    hmstr,
                    self.exp,
                    self.mode,
                    prio,
                    self.dur,
                    self.kwargs,
                )

        with open(scd_file.name, "r") as f:
            lines = f.readlines()

        # The lines should be sorted by date, time then priority
        self.assertEqual(len(lines), counter)
        for i in range(0, 1, 20221031 - 20221001):  # use i, j as indices, so start at 0
            for j in range(0, 1, 59):
                if (i * j + j + 1) > counter:
                    # If we don't have any more indices left, get out of the loop
                    break
                line = lines[i * j + j].split()
                nextline = lines[i * j + j + 1].split()
                self.assertGreaterEqual(line[0], nextline[0])
                self.assertGreaterEqual(line[1], nextline[1])
                self.assertEqual(line[2], self.dur)
                if j % 2 == 0 and nextline[3] == 0:
                    self.assertGreaterEqual(line[3], self.prio)
                    self.assertEqual(nextline[3], 4)
                elif j % 2 == 0 and nextline[3] == 4:
                    self.assertEqual(self.prio, line[3])
                    self.assertEqual(nextline[3], 7)
                elif j % 2 == 0 and nextline[3] == 7:
                    self.assertEqual(self.prio, line[3])
                    self.assertEqual(nextline[3], 7)

                self.assertEqual(line[4], self.exp)
                self.assertEqual(line[5], self.mode)
                self.assertEqual(line[6], self.kwargs)

    # remove_line tests
    def test_remove_lines(self):
        """
        Test trying to remove lines from an SCD file
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        scd_lines = [
            "20200917 00:00 - 0 normalscan common  \n",
            "20200921 00:00 - 0 normalscan discretionary  \n",
            "20200924 00:00 - 0 normalscan common  freq1=10500\n",
            "20200926 00:00 - 0 normalscan common  \n",
        ]
        l0 = scd_lines[0].split()
        l1 = scd_lines[1].split()
        l2 = scd_lines[2].split()
        l3 = scd_lines[3].split()

        for line in scd_lines:
            scd_file.write(line)
        scd_file.close()

        # Remove 0th line and test
        yyyymmdd = l0[0]
        hhmm = l0[1]
        dur = l0[2]
        prio = l0[3]
        exp = l0[4]
        mode = l0[5]

        scdu.remove_line(yyyymmdd, hhmm, exp, mode, prio, dur)

        with open(scd_file.name, "r") as f:
            file_lines = f.readlines()

        self.assertEqual(len(file_lines), len(scd_lines) - 1)
        self.assertEqual(file_lines[0], scd_lines[1])
        self.assertEqual(file_lines[1], scd_lines[2])
        self.assertEqual(file_lines[2], scd_lines[3])

        yyyymmdd = l1[0]
        hhmm = l1[1]
        dur = l1[2]
        prio = l1[3]
        exp = l1[4]
        mode = l1[5]
        scdu.remove_line(yyyymmdd, hhmm, exp, mode, prio, dur)

        with open(scd_file.name, "r") as f:
            file_lines = f.readlines()

        self.assertEqual(len(file_lines), len(scd_lines) - 2)
        self.assertEqual(file_lines[0], scd_lines[2])
        self.assertEqual(file_lines[1], scd_lines[3])

        yyyymmdd = l2[0]
        hhmm = l2[1]
        dur = l2[2]
        prio = l2[3]
        exp = l2[4]
        mode = l2[5]
        kwarg = l2[6]
        scdu.remove_line(yyyymmdd, hhmm, exp, mode, prio, dur, kwarg)

        with open(scd_file.name, "r") as f:
            file_lines = f.readlines()

        self.assertEqual(len(file_lines), len(scd_lines) - 3)
        self.assertEqual(file_lines[0], scd_lines[3])

        yyyymmdd = l3[0]
        hhmm = l3[1]
        dur = l3[2]
        prio = l3[3]
        exp = l3[4]
        mode = l3[5]
        scdu.remove_line(yyyymmdd, hhmm, exp, mode, prio, dur)

        with open(scd_file.name, "r") as f:
            file_lines = f.readlines()

        self.assertEqual(len(file_lines), len(scd_lines) - 4)

    def test_remove_line_dne(self):
        """
        Test trying to remove lines from an SCD file that don't exist. Should raise ValueError
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        scd_lines = [
            "20200917 00:00 - 0 normalscan common\n",
            "20200921 00:00 - 0 normalscan discretionary\n",
            "20200924 00:00 - 0 normalscan common freq=10500\n",
            "20200926 00:00 - 0 normalscan common\n",
        ]

        for line in scd_lines:
            scd_file.write(line)
        scd_file.close()

        with self.assertRaises(ValueError):
            scdu.remove_line(
                self.yyyymmdd,
                self.hhmm,
                self.exp,
                self.mode,
                self.prio,
                self.dur,
                self.kwargs,
            )

    # get_relevant_lines tests
    def test_incorrect_datetime_fmt(self):
        """
        Test trying to input an incorrect datetime string, should raise ValueError or TypeError
        """
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)

        with self.assertRaises(ValueError):
            scdu.get_relevant_lines("202", "blah")
        with self.assertRaises(TypeError):
            scdu.get_relevant_lines("20225555", None)
        with self.assertRaises(TypeError):
            scdu.get_relevant_lines(5, "5890")

    def test_empty_file(self):
        """
        Test trying to get relevant lines from an empty file, should be empty, as even though the
        read_scd() call in get_relevant_lines returns the default scd, the for loop
        doesn't ever add it to relevant lines if the time of interest is in the future
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scd_file.close()
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        self.assertEqual(
            scdu.get_relevant_lines("20061101", "00:00"), [scdu.scd_default]
        )

    def test_empty_file_w_default(self):
        """
        Test trying to get relevant lines from an empty file, should be the default scd as the time of interest
        is in the past from the default line
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scd_file.close()
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        self.assertEqual(
            scdu.get_relevant_lines("19991101", "00:00"),
            [scdu.create_line("20000101", "00:00", "normalscan", "common", "0", "-")],
        )

    def test_no_lines_relevant(self):
        """
        Test trying to get relevant lines from a file with all lines in the past
        """
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        lines = scdu.get_relevant_lines("21001113", "00:00")
        self.assertEqual(len(lines), 1)

    def test_one_line_relevant(self):
        """
        Test trying to get relevant lines from a file with one line in the future
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        test_exp_line = "20200917 00:00 89 19 ulfscan common\n"
        scd_file.write(test_exp_line)
        scd_file.close()
        lines = scdu.get_relevant_lines("20200916", "23:59")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0]["prio"], "19")
        self.assertEqual(lines[0]["duration"], "89")
        self.assertEqual(lines[0]["experiment"], "ulfscan")
        self.assertEqual(lines[0]["scheduling_mode"], "common")

    def test_some_lines_relevant(self):
        """
        Test trying to get relevant lines from a file with some lines in the future
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        test_scd_lines = [
            "20200917 00:00 - 0 normalscan common\n",
            "20200921 00:00 - 0 normalscan discretionary\n",
            "20200924 00:00 - 0 twofsound common freq=10500\n",
            "20200926 00:00 60 2 politescan special\n",
        ]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = scdu.get_relevant_lines("20200921", "00:01")
        self.assertEqual(len(lines), 3)

        # Should be the second line from test_scd_lines
        self.assertEqual(lines[0]["duration"], "-")
        self.assertEqual(lines[0]["prio"], "0")
        self.assertEqual(lines[0]["experiment"], "normalscan")
        self.assertEqual(lines[0]["scheduling_mode"], "discretionary")

        self.assertEqual(lines[1]["duration"], "-")
        self.assertEqual(lines[1]["prio"], "0")
        self.assertEqual(lines[1]["experiment"], "twofsound")
        self.assertEqual(lines[1]["scheduling_mode"], "common")
        self.assertEqual(lines[1]["kwargs"], "freq=10500")

        self.assertEqual(lines[2]["duration"], "60")
        self.assertEqual(lines[2]["prio"], "2")
        self.assertEqual(lines[2]["experiment"], "politescan")
        self.assertEqual(lines[2]["scheduling_mode"], "special")

    def test_all_lines_relevant_matched(self):
        """
        Test trying to get relevant lines from a file with all lines in the future or present
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        test_scd_lines = [
            "20200917 00:00 - 0 normalscan common\n",
            "20200921 00:00 - 0 normalscan discretionary\n",
            "20200924 00:00 - 0 twofsound common freq=10500\n",
            "20200926 00:00 60 2 politescan special\n",
        ]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = scdu.get_relevant_lines("20200917", "00:00")
        self.assertEqual(len(lines), 4)

        self.assertEqual(lines[0]["duration"], "-")
        self.assertEqual(lines[0]["prio"], "0")
        self.assertEqual(lines[0]["experiment"], "normalscan")
        self.assertEqual(lines[0]["scheduling_mode"], "common")

        self.assertEqual(lines[1]["duration"], "-")
        self.assertEqual(lines[1]["prio"], "0")
        self.assertEqual(lines[1]["experiment"], "normalscan")
        self.assertEqual(lines[1]["scheduling_mode"], "discretionary")

        self.assertEqual(lines[2]["duration"], "-")
        self.assertEqual(lines[2]["prio"], "0")
        self.assertEqual(lines[2]["experiment"], "twofsound")
        self.assertEqual(lines[2]["scheduling_mode"], "common")
        self.assertEqual(lines[2]["kwargs"], "freq=10500")

        self.assertEqual(lines[3]["duration"], "60")
        self.assertEqual(lines[3]["prio"], "2")
        self.assertEqual(lines[3]["experiment"], "politescan")
        self.assertEqual(lines[3]["scheduling_mode"], "special")

    def test_all_lines_relevant(self):
        """
        Test trying to get relevant lines from a file with all lines in the future
        """
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        test_scd_lines = [
            "20200917 00:00 - 0 normalscan common\n",
            "20200921 00:00 - 0 normalscan discretionary\n",
            "20200924 00:00 - 0 twofsound common freq=10500\n",
            "20200926 00:00 60 2 politescan special\n",
        ]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = scdu.get_relevant_lines("20200916", "00:00")
        self.assertEqual(len(lines), 4)

        self.assertEqual(lines[0]["duration"], "-")
        self.assertEqual(lines[0]["prio"], "0")
        self.assertEqual(lines[0]["experiment"], "normalscan")
        self.assertEqual(lines[0]["scheduling_mode"], "common")

        self.assertEqual(lines[1]["duration"], "-")
        self.assertEqual(lines[1]["prio"], "0")
        self.assertEqual(lines[1]["experiment"], "normalscan")
        self.assertEqual(lines[1]["scheduling_mode"], "discretionary")

        self.assertEqual(lines[2]["duration"], "-")
        self.assertEqual(lines[2]["prio"], "0")
        self.assertEqual(lines[2]["experiment"], "twofsound")
        self.assertEqual(lines[2]["scheduling_mode"], "common")
        self.assertEqual(lines[2]["kwargs"], "freq=10500")

        self.assertEqual(lines[3]["duration"], "60")
        self.assertEqual(lines[3]["prio"], "2")
        self.assertEqual(lines[3]["experiment"], "politescan")
        self.assertEqual(lines[3]["scheduling_mode"], "special")

    def test_one_relevant_line(self):
        """
        Use a time-of-interest that is far in the future so there is only one relevant line (should be last line)
        """
        time_of_interest = datetime.datetime(2050, 11, 14, 0, 1)
        scdu = scd_utils.SCDUtils(self.good_scd_file, self.site_id)
        lines = scdu.get_relevant_lines(
            time_of_interest.strftime("%Y%m%d"), time_of_interest.strftime("%H:%M")
        )
        self.assertEqual(
            [scdu.fmt_line(line).strip() for line in lines],
            ["20220929 12:00 - 0 normalscan common"],
        )

    def test_no_relevant_lines(self):
        """
        Use a time-of-interest that is far in the future with an SCD file without any inf duration lines
        """
        time_of_interest = datetime.datetime(2050, 11, 14, 0, 1)
        scdfile = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdfile.write(
            "20220929 12:00 360 0 normalscan common"
        )  # A 360 minute duration line
        scdfile.close()
        scdu = scd_utils.SCDUtils(scdfile.name, self.site_id)
        lines = scdu.get_relevant_lines(
            time_of_interest.strftime("%Y%m%d"), time_of_interest.strftime("%H:%M")
        )
        self.assertEqual(lines, [])

    def test_one_prev_relevant_line(self):
        """
        Use a time-of-interest that is far in the future with an SCD file with one inf duration line in the past
        """
        time_of_interest = datetime.datetime(2050, 11, 14, 0, 1)
        scdfile = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdfile.write(
            "20210903 00:00 - 0 twofsound common\n"
        )  # An infinite duration line
        scdfile.write(
            "20220929 12:00 60 0 normalscan common\n"
        )  # A 60 minute duration line
        scdfile.write(
            "20220929 13:00 60 0 normalscan common\n"
        )  # A 60 minute duration line
        scdfile.close()
        scdu = scd_utils.SCDUtils(scdfile.name, self.site_id)
        lines = scdu.get_relevant_lines(
            *time_of_interest.strftime("%Y%m%d %H:%M").split()
        )
        self.assertEqual(
            [scdu.fmt_line(line).strip() for line in lines],
            ["20210903 00:00 - 0 twofsound common"],
        )

    def test_no_inf_dur_relevant(self):
        """
        Test trying to get relevant lines from a file with one line in the future but no infinite duration lines
        """
        time_of_interest = datetime.datetime(2020, 9, 25, 0, 1)
        scd_file = tempfile.NamedTemporaryFile(mode="w+t", delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name, self.site_id)
        test_scd_lines = [
            "20200917 00:00 60 0 normalscan common\n",
            "20200921 00:00 1440 0 normalscan discretionary\n",
            "20200924 00:00 360 0 normalscan common freq=10500\n",
            "20200926 00:00 120 0 normalscan common\n",
        ]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = scdu.get_relevant_lines(
            *time_of_interest.strftime("%Y%m%d %H:%M").split()
        )
        self.assertEqual(len(lines), 1)
        self.assertEqual(scdu.fmt_line(lines[0]).strip(), test_scd_lines[-1].strip())


class TestRemoteServer(unittest.TestCase):
    """
    unittest class to test the remote server module.
    All test methods must begin with the word 'test' to be run
    """

    def __init__(self, *args, **kwargs):
        """
        Init some reused variables
        """
        super().__init__(*args, **kwargs)

        self.site_id = "lab"
        self.good_config = f"{os.environ['BOREALISPATH']}/config/sas/sas_config.ini"
        self.good_scd_file = (
            f"{os.environ['BOREALISPATH']}/tests/scheduler/good_scd_file.scd"
        )
        self.maxDiff = None

    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

    # format_to_atq tests
    def test_make_atq_commands(self):
        """
        Test creating atq commands from scd lines
        """
        # Atq commands are: [command to run] | at [now+ x minute | -t %Y%m%d%H%M]
        time_of_interest = datetime.datetime(2022, 9, 8, 12, 34)
        atq_str = remote_server.format_to_atq(
            time_of_interest, "some weird experiment with options", "some mode"
        )
        self.assertEqual(
            atq_str,
            f"echo 'screen -d -m -S starter {os.environ['BOREALISPATH']}/scripts/steamed_hams.py "
            "some weird experiment with options release some mode' | "
            "at -t 202209081234",
        )
        atq_str = remote_server.format_to_atq(
            time_of_interest, "exp", "md", first_event_flag=True
        )
        self.assertEqual(
            atq_str,
            f"echo 'screen -d -m -S starter {os.environ['BOREALISPATH']}/scripts/steamed_hams.py "
            "exp release md' | "
            "at now + 1 minute",
        )
        time_of_interest = datetime.datetime(2019, 4, 3, 9, 56)
        atq_str = remote_server.format_to_atq(
            time_of_interest, "exp", "md", kwargs="this is the kwargs"
        )
        self.assertEqual(
            atq_str,
            f"echo 'screen -d -m -S starter {os.environ['BOREALISPATH']}/scripts/steamed_hams.py "
            "exp release md --kwargs this is the kwargs' | "
            "at -t 201904030956",
        )

    # timeline_to_atq tests

    @unittest.skip
    def test_empty_timeline_to_atq(self):
        """
        Test converting a timeline to atq command string
        The atq looks like this:
        1183    Wed Nov 16 00:00:00 2022 a radar
        1184    Fri Nov 18 00:00:00 2022 a radar
        1185    Mon Nov 21 00:00:00 2022 a radar
        1186    Mon Nov 21 12:00:00 2022 a radar

        And the details look like this:
        1214    Fri Dec 30 00:00:00 2022 a radar
        screen -d -m -S starter /home/radar/borealis//steamed_hams.py interleavedscan release special

        The timeline being an empty list should remove all atq commands and reschedule future events
        The function should create a backup atq file with all current and future jobs
        timeline is a list of events, which are dicts with 'time', 'experiment', 'scheduling_mode', 'kwargs_string'
        """
        get_atq_cmd = (
            "for j in $(atq | sort -k6,6 -k3,3M -k4,4 -k5,5 |cut -f 1); "
            'do atq |grep -P "^$j\t"; at -c "$j" | tail -n 2; done'
        )
        current_atq = sp.check_output(get_atq_cmd, shell=True)
        print(f"Current atq: {current_atq}")

        # Remove any existing atq backups directory
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        atq_dir = f"{scd_dir}/atq_backups/"
        try:
            shutil.rmtree(atq_dir)
        except FileNotFoundError:
            pass

        # This is what we're testing - an empty timeline list, and calling the timeline_to_atq
        # function with this empty list should not fail, and shouldn't change the atq
        timeline_list = []
        site_id = os.environ["RADAR_ID"]
        time_of_interest = datetime.datetime(2022, 11, 14, 3, 30)
        backup_time = time_of_interest.strftime("%Y.%m.%d.%H.%M")
        atq_command = remote_server.timeline_to_atq(
            timeline_list, scd_dir, time_of_interest, site_id
        )
        print(f"Final atq: {atq_command}")
        self.assertEqual(current_atq, atq_command)
        self.assertTrue(os.path.exists(atq_dir))
        self.assertTrue(os.path.exists(f"{atq_dir}/{backup_time}.{site_id}.atq"))
        shutil.rmtree(atq_dir)

    @unittest.skip
    def test_timeline_to_atq(self):
        """
        Test converting a timeline to atq command string
        The atq looks like this:
        [job #] [datetime of execution] [queue ('a' is default, '=' means currently running)] [user]
        1183    Wed Nov 16 00:00:00 2022 a radar
        1184    Fri Nov 18 00:00:00 2022 a radar
        1185    Mon Nov 21 00:00:00 2022 a radar
        1186    Mon Nov 21 12:00:00 2022 a radar

        And the details look like this:
        1214    Fri Dec 30 00:00:00 2022 a radar
        screen -d -m -S starter /home/radar/borealis//steamed_hams.py interleavedscan release special

        The function should create a backup atq file with all current and future jobs
        timeline is a list of events, which are dicts with 'time', 'experiment', 'scheduling_mode', 'kwargs_string'
        """
        # Remove any existing atq backups directory
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        atq_dir = f"{scd_dir}/atq_backups/"
        try:
            shutil.rmtree(atq_dir)
        except FileNotFoundError:
            pass

        t1 = datetime.datetime(2099, 10, 9, 5, 30)
        t2 = datetime.datetime(2099, 10, 10, 0, 30)
        t3 = datetime.datetime(2099, 10, 11, 8, 0)

        # Of the three timeline events, the first will be considered the 'first event' in format_to_atq,
        # which means it gets scheduled NOW
        events = [
            {
                "time": t1,
                "experiment": "politescan",
                "scheduling_mode": "discretionary",
                "kwargs_string": "-",
            },
            {
                "time": t2,
                "experiment": "normalscan",
                "scheduling_mode": "special",
                "kwargs_string": "hi=96",
            },
            {
                "time": t3,
                "experiment": "twofsound",
                "scheduling_mode": "common",
                "kwargs_string": "-",
            },
        ]
        site_id = os.environ["RADAR_ID"]
        time_of_interest = datetime.datetime(2022, 11, 14, 3, 30)
        backup_time = time_of_interest.strftime("%Y.%m.%d.%H.%M")
        atq_commands = remote_server.timeline_to_atq(
            events, scd_dir, time_of_interest, site_id
        )

        # The atq should have the three events added, but the last one should have a date that is now + 1 minutes
        now_plus_one_min = (
            datetime.datetime.now() + datetime.timedelta(minutes=1)
        ).strftime("%a %b %-d %H:%M:00 %Y")
        new_atq = (
            f"11\t{now_plus_one_min} a radar\nscreen -d -m -S starter "
            "/home/radar/borealis/scripts/steamed_hams.py politescan release discretionary --kwargs_string -\n\n"
            "12\tSat Oct 10 00:30:00 2099 a radar\nscreen -d -m -S starter "
            "/home/radar/borealis/scripts/steamed_hams.py normalscan release special --kwargs_string hi=96\n\n"
            "13\tSun Oct 11 08:00:00 2099 a radar\nscreen -d -m -S starter "
            "/home/radar/borealis/scripts/steamed_hams.py twofsound release common --kwargs_string -\n\n"
        )

        # First remove the job numbers (matched by \d+), which are always before the tab character (\t)
        new_commands = str(re.sub("\d+\t", "", new_atq)).split()
        prev_commands = str(re.sub("\d+\t", "", atq_commands.decode("ascii"))).split()
        self.assertEqual(prev_commands, new_commands)
        self.assertTrue(os.path.exists(atq_dir))
        self.assertTrue(os.path.exists(f"{atq_dir}/{backup_time}.{site_id}.atq"))

        # Now remove the atq directory, remove the future at jobs submitted, and return
        shutil.rmtree(atq_dir)
        # TODO: Remove added jobs


class TestLocalServer(unittest.TestCase):
    """
    unittest class to test the local server module. All test methods must begin with the word 'test' to be run
    """

    def __init__(self, *args, **kwargs):
        """
        Set up variables and data used for unit testing
        """
        super().__init__(*args, **kwargs)
        self.maxDiff = None

    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

    # SWG tests:
    # init tests
    def test_swg_init(self):
        """
        Test initializing the SWG class
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        swg_dir = scd_dir + "schedules/"

        # Ensure the swg dir (the git repo for schedules) doesn't exist before
        try:
            shutil.rmtree(swg_dir)
        except FileNotFoundError:
            pass
        self.assertFalse(os.path.exists(swg_dir))

        local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(swg_dir))
        self.assertTrue(os.path.exists(swg_dir))
        # Remove the swg dir again
        shutil.rmtree(swg_dir)
        self.assertFalse(os.path.exists(swg_dir))

    # new_swg_file_available
    def test_swg_new_file(self):
        """
        Test new file exists method, which checks for new swg file uploads via git and returns True or False
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        swg_dir = scd_dir + "schedules/"
        swg = local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(scd_dir))
        new = swg.new_swg_file_available()
        self.assertFalse(
            new
        )  # Assume that the repo is up-to-date always, not sure how else to test this
        # Remove the swg dir again
        shutil.rmtree(swg_dir)
        self.assertFalse(os.path.exists(swg_dir))

    # pull_new_swg_file
    # def test_swg_pull(self):  # Nothing to really test here
    # """
    # Test pulling new git files
    # """
    # scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
    # swg = local_scd_server.SWG(scd_dir)
    # self.assertTrue(os.path.exists(scd_dir))

    # parse_swg_to_scd
    def test_swg_dne(self):
        """
        Test parsing the SWG file that DNE. Should fail with FileNotFoundError
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        site_id = os.environ["RADAR_ID"]

        modes = local_scd_server.EXPERIMENTS[site_id]
        swg = local_scd_server.SWG(scd_dir)

        # Make sure to remove any existing test schedules dir that was just cloned
        try:
            shutil.rmtree(scd_dir + "schedules/")
        except FileNotFoundError:
            pass

        self.assertTrue(os.path.exists(scd_dir))
        with self.assertRaises(FileNotFoundError):
            params = swg.parse_swg_to_scd(modes, site_id, first_run=True)

    @unittest.skip  # TODO: Should the scheduler check for gaps in the schedule?
    def test_missing_hours(self):
        """
        Test parsing the SWG file. Should fail due to a gap in the schedule of several hours.
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        site_id = os.environ["RADAR_ID"]
        swg_file = f"{os.environ['BOREALISPATH']}/tests/scheduler/missing_hours.swg"
        swg_dir = scd_dir + "schedules/"

        # Need to ensure we put in the current month to the schedule file and set first run to True
        mm_yyyy = datetime.datetime.today().strftime("%B %Y")
        yyyy = datetime.datetime.today().strftime("%Y")
        yyyymm = datetime.datetime.today().strftime("%Y%m")
        new_swg_file = f"{scd_dir}/schedules/{yyyy}/{yyyymm}"
        with open(swg_file, "r") as f:
            swg_data = f.read()
        with open(new_swg_file, "w") as f:
            f.write(mm_yyyy + swg_data)

        modes = local_scd_server.EXPERIMENTS[site_id]
        swg = local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(scd_dir))
        self.assertTrue(os.path.exists(swg_file))
        self.assertTrue(os.path.exists(new_swg_file))
        with self.assertRaises(ValueError):
            params = swg.parse_swg_to_scd(modes, site_id, first_run=True)

        # Remove the files we wrote
        shutil.rmtree(new_swg_file)
        # Remove the swg dir again
        shutil.rmtree(swg_dir)
        self.assertFalse(os.path.exists(swg_dir))

    def test_bad_experiment(self):
        """
        Test parsing the SWG file. Should fail when it encounters a line with a non-existent experiment
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        site_id = os.environ["RADAR_ID"]
        swg_file = f"{os.environ['BOREALISPATH']}/tests/scheduler/bad_experiment.swg"
        swg_dir = scd_dir + "schedules/"

        # Need to ensure we put in the current month to the schedule file and set first run to True
        mm_yyyy = datetime.datetime.today().strftime("%B %Y")
        yyyy = datetime.datetime.today().strftime("%Y")
        yyyymm = datetime.datetime.today().strftime("%Y%m")
        new_swg_file = f"{scd_dir}/schedules/{yyyy}/{yyyymm}.swg"
        with open(swg_file, "r") as f:
            swg_data = f.read()
        if not os.path.exists(os.path.dirname(new_swg_file)):
            os.makedirs(os.path.dirname(new_swg_file))
        with open(new_swg_file, "w") as f:
            f.write(f"{mm_yyyy}\n{swg_data}")

        modes = local_scd_server.EXPERIMENTS[site_id]
        swg = local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(scd_dir))
        self.assertTrue(os.path.exists(swg_file))
        self.assertTrue(os.path.exists(new_swg_file))
        with self.assertRaises(ValueError):
            params = swg.parse_swg_to_scd(modes, site_id, first_run=True)

        # Remove the swg dir again
        shutil.rmtree(swg_dir)
        self.assertFalse(os.path.exists(swg_dir))

    def test_swg_parse(self):
        """
        Test parsing the SWG file. Should work properly and return a list of parsed parameters corresponding to
        the swg file input
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        site_id = os.environ["RADAR_ID"]
        swg_file = (
            f"{os.environ['BOREALISPATH']}/tests/scheduler/complicated_schedule.swg"
        )
        swg_dir = scd_dir + "schedules/"

        # Need to ensure we put in the current month to the schedule file and set first run to True
        mm_yyyy = datetime.datetime.today().strftime("%B %Y")
        yyyy = datetime.datetime.today().strftime("%Y")
        yyyymm = datetime.datetime.today().strftime("%Y%m")
        new_swg_file = f"{swg_dir}/{yyyy}/{yyyymm}.swg"
        modes = local_scd_server.EXPERIMENTS[site_id]
        swg = local_scd_server.SWG(scd_dir)

        with open(swg_file, "r") as f:
            swg_data = f.read()

        with open(new_swg_file, "w") as f:
            f.write(mm_yyyy + swg_data)

        self.assertTrue(os.path.exists(scd_dir))
        self.assertTrue(os.path.exists(swg_file))
        self.assertTrue(os.path.exists(new_swg_file))
        parsed_params = swg.parse_swg_to_scd(modes, site_id, first_run=True)
        print(parsed_params)
        self.assertTrue(isinstance(parsed_params, list))
        # Remove the swg dir again
        shutil.rmtree(swg_dir)
        self.assertFalse(os.path.exists(swg_dir))


if __name__ == "__main__":
    unittest.main()
