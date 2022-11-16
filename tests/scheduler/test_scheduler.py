"""
Test module for the scheduler code.
Run via: 'python3 test_scheduler.py'

:copyright: 2022 SuperDARN Canada
:author: Kevin Krieger
"""
import shutil
import unittest
import os
import sys
import tempfile
import json
import datetime
import subprocess as sp

if not os.environ['BOREALISPATH']:
    BOREALISPATH = f"{os.environ['HOME']}/PycharmProjects/borealis/"
else:
    BOREALISPATH = os.environ['BOREALISPATH']

sys.path.append(BOREALISPATH)

from scheduler import email_utils, scd_utils
from scheduler import remote_server_options as rso
from scheduler import local_scd_server, remote_server


class TestSchedulerUtils(unittest.TestCase):
    """
    unittest class to test the scheduler utilities module. All test methods must begin with the word 'test' to be run
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # A file that has a bunch of lines in the scd file that should all pass all tests
        self.good_scd_file = f"{os.environ['BOREALISPATH']}/tests/scheduler/good_scd_file.scd"
        # A file that has 5 arguments (missing the scheduling mode)
        self.incorrect_args_scd = f"{os.environ['BOREALISPATH']}/tests/scheduler/incorrect_args.scd"
        self.line_fmt = "{self.dt} {self.dur} {self.prio} {self.exp} {self.mode} {self.kwargs}"
        self.scd_dt_fmt = "%Y%m%d %H:%M"
        self.yyyymmdd = '20221014'
        self.hhmm = '12:34'
        self.dt = '20221011 00:00'
        self.dur = 120
        self.prio = 10
        self.exp = 'normalscan'
        self.mode = 'common'
        self.kwargs = '-'
        self.no_perms = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        self.no_perms_file = self.no_perms.name
        os.chmod(self.no_perms_file, 0o000)
        self.no_perms.close()
        self.linedict = {"time": '20000101 06:30', "prio": 0, "experiment": 'normalscan', "scheduling_mode": 'common',
                         "duration": 60, "kwargs_string": '-'}
        self.linestr = "20000101 06:30 60 0 normalscan common -"
        self.linedict2 = {"time": '20220405 16:56', "prio": 15, "experiment": 'twofsound',
                          "scheduling_mode": 'discretionary', "duration": 360,
                          "kwargs_string": 'freq1=10500 freq2=13100'}
        self.linestr2 = "20220405 16:56 360 15 twofsound discretionary freq1=10500 freq2=13100"

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
        with self.assertRaises(AttributeError):
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

# check_line tests

    def test_invalid_yyyymmdd(self):
        """
        Test an invalid start date
        """
        yyyymmdd = None
        # Invalid type for yyyymmdd should raise a TypeError in strptime.
        # it will also raise ValueError if the date_string and format can't be parsed by time.strptime()
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        with self.assertRaises(TypeError):
            scdu.check_line(yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
        yyyymmdd = 0   # TypeError
        with self.assertRaises(TypeError):
            scdu.check_line(yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
        yyyymmdd = '20211'  # ValueError
        with self.assertRaises(ValueError):
            scdu.check_line(yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
        yyyymmdd = '20221512'  # ValueError, 15th month
        with self.assertRaises(ValueError):
            scdu.check_line(yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
        yyyymmdd = '20220230'  # ValueError, 30th day in Feb
        with self.assertRaises(ValueError):
            scdu.check_line(yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)

    def test_invalid_hhmm(self):
        """
        Test an invalid start hour and minute. required format: "HH:MM"
        """
        hhmm = None
        # Invalid type for hhmm should raise a TypeError in strptime.
        # it will also raise ValueError if the date_string and format can't be parsed by time.strptime()
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        with self.assertRaises(TypeError):
            scdu.check_line(self.yyyymmdd, hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
        hhmm = 0  # TypeError
        with self.assertRaises(TypeError):
            scdu.check_line(self.yyyymmdd, hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
        hhmm = '2011'  # ValueError
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
        hhmm = '2500'  # ValueError, 25th hour
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)

    def test_invalid_experiment(self):
        """
        Test an invalid experiment name - Must be an experiment named in the repo
        """
        exp = 'non-existent_Experiment'
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, exp, self.mode, self.prio, self.dur, self.kwargs)
        exp = None
        with self.assertRaises(TypeError):
            scdu.check_line(self.yyyymmdd, self.hhmm, exp, self.mode, self.prio, self.dur, self.kwargs)
        exp = 5
        with self.assertRaises(TypeError):
            scdu.check_line(self.yyyymmdd, self.hhmm, exp, self.mode, self.prio, self.dur, self.kwargs)

    def test_invalid_mode(self):
        """
        Test an invalid mode (possible: ['common', 'special', 'discretionary'])
        """
        mode = 'notamode'
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, mode, self.prio, self.dur, self.kwargs)
        mode = None
        with self.assertRaises(TypeError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, mode, self.prio, self.dur, self.kwargs)
        mode = 0
        with self.assertRaises(TypeError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, mode, self.prio, self.dur, self.kwargs)

    def test_invalid_prio(self):
        """
        Test an invalid priority (string/integer from 0 to 20 inclusive)
        """
        prio = 'blah'
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, prio, self.dur, self.kwargs)
        prio = None
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, prio, self.dur, self.kwargs)
        prio = -1
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, prio, self.dur, self.kwargs)
        prio = 21
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, prio, self.dur, self.kwargs)

    def test_invalid_duration(self):
        """
        Test an invalid duration (optional: needs to be a positive integer/string minutes, or '-')
        """
        dur = 0
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, dur, self.kwargs)
        dur = -1
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, dur, self.kwargs)
        dur = -50
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, dur, self.kwargs)
        dur = 53.09
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, dur, self.kwargs)
        dur = ''
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, dur, self.kwargs)
        dur = None
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, dur, self.kwargs)

    def test_invalid_kwargs_string(self):
        """
        Test an invalid kwargs string (optional arguments that may be passed to an experiment's kwargs)
        """
        kwargs_str = 'this doesnt mean anything to the experiment'
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, kwargs_str)
        kwargs_str = 0
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, kwargs_str)
        kwargs_str = -1
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, kwargs_str)
        kwargs_str = 56.9
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, kwargs_str)
        kwargs_str = None
        with self.assertRaises(ValueError):
            scdu.check_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, kwargs_str)

# read_scd tests
    def test_invalid_num_args(self):
        """
        Test an invalid number of arguments, requires 6 or 7 args
        """
        scdu = scd_utils.SCDUtils(self.incorrect_args_scd)
        with self.assertRaises(ValueError):
            scdu.read_scd()

    def test_bad_scd_file(self):
        """
        Test giving a non-existent scd file to the module
        """
        scdfile = 'thisfilelikelydoesntexist90758hjna;ksjn'
        scdu = scd_utils.SCDUtils(scdfile)
        with self.assertRaises(IOError):
            scdu.read_scd()

    def test_empty_scd_file(self):
        """
        Test giving an empty scd file to the module, prints warning, returns and appends default
        line to the scd file
        """
        scdfile = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scdfile.name)
        scdfile.close()
        self.assertEqual(scdu.read_scd(), scdu.check_line('20000101', '00:00', 'normalscan', 'common', '0', '-'))

# fmt_line tests
    def test_fmt_line(self):
        """
        Test that the result from fmt_line agrees with what it should
        """
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        self.assertEqual(scdu.fmt_line(self.linedict), self.linestr)
        self.assertEqual(scdu.fmt_line(self.linedict2), self.linestr2)

# write_scd tests
    def test_no_perms(self):
        """
        Test that a scd file without good permissions to read or write will cause PermissionError
        """
        scdu = scd_utils.SCDUtils(self.no_perms_file)
        lines = []
        with self.assertRaises(PermissionError):
            scdu.write_scd(lines)

    def test_file_dne(self):
        """
        Test write_scd with no actual file for the scd
        """
        scdu = scd_utils.SCDUtils('blkja;lskjdf;lkhj')
        lines = []
        with self.assertRaises(FileNotFoundError):
            scdu.write_scd(lines)

    def test_file_is_dir(self):
        """
        Test write_scd with a directory given for the scd file
        """
        scdu = scd_utils.SCDUtils(os.environ['HOME'])
        lines = []
        with self.assertRaises(IsADirectoryError):
            scdu.write_scd(lines)

    def test_scd_backup(self):
        """
        Test write_scd with a good line, so that it creates a .bak of the scd file
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scd_file.close()
        scdu = scd_utils.SCDUtils(scd_file.name)
        lines = [self.linedict, self.linedict2]
        scdu.write_scd(lines)

        # Verify there's a .bak file parallel with the temp file with nothing in it
        # Verify there's a scd file with the added lines to it
        bak_file = scd_file.name + '.bak'
        self.assertTrue(os.path.exists(bak_file))
        with open(bak_file, 'r') as f, open(scd_file, 'r') as s:
            bak_lines = f.readlines()
            scd_lines = s.readlines()
            self.assertEqual([self.linestr, self.linestr2], scd_lines)
            self.assertEqual(bak_lines, [])

# add_line tests
    def test_duplicate_line(self):
        """
        Test trying to add duplicate line
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_file.close()
        with self.assertRaisesRegex(ValueError, "Line is a duplicate of an existing line"):
            scdu.add_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
            scdu.add_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)

    def test_priority_time_dup(self):
        """
        Test trying to add line with the same time and priority as an existing line
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_file.close()
        with self.assertRaisesRegex(ValueError, "Priority already exists at this time"):
            scdu.add_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
            scdu.add_line(self.yyyymmdd, self.hhmm, 'twofsound', 'special', self.prio, '1440', self.kwargs)

    def test_add_line(self):
        """
        Test trying to add a line to a file
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_file.close()
        scdu.add_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
        with open(scd_file.name, 'r') as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)
        line = lines[0].split()
        self.assertEqual(line[0], self.yyyymmdd)
        self.assertEqual(line[1], self.hhmm)
        self.assertEqual(line[2], self.dur)
        self.assertEqual(line[3], self.prio)
        self.assertEqual(line[4], self.exp)
        self.assertEqual(line[5], self.mode)
        self.assertEqual(line[6], self.kwargs)

    def test_add_lines_sorted(self):
        """
        Test trying to add multiple lines to a file and have them properly sorted by time and priority
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_file.close()
        counter = 0
        for yyyymmdd in range(20221001, 1, 20221031):
            for hhmm in range(59, -1, 0):
                counter += 1
                hmstr = f"{hhmm:0<4}"[:2] + ':' + f"{hhmm:0<4}"[2:]
                scdu.add_line(str(f"{yyyymmdd:0>8}"), hmstr, self.exp, self.mode, self.prio, self.dur, self.kwargs)
                if hhmm % 2 == 0:
                    prio = 4
                else:
                    prio = 7
                scdu.add_line(str(f"{yyyymmdd:0>8}"), hmstr, self.exp, self.mode, prio, self.dur, self.kwargs)

        with open(scd_file.name, 'r') as f:
            lines = f.readlines()

        # The lines should be sorted by date, time then priority
        self.assertEqual(len(lines), counter)
        for i in range(0, 1, 20221031-20221001):  # use i, j as indices, so start at 0
            for j in range(0, 1, 59):
                line = lines[i*j + j].split()
                if (i*j + j+1) > counter:
                    # If we don't have any more indices left, get out of the loop
                    break
                nextline = lines[i*j + j + 1].split()
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
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n", 
                     "20200924 00:00 - 0 normalscan common freq1=10500\n", "20200926 00:00 - 0 normalscan common\n"]
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
        
        with open(scd_file.name, 'r') as f:
            file_lines = f.readlines()
    
        self.assertEqual(len(file_lines), len(scd_lines)-1)
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

        with open(scd_file.name, 'r') as f:
            file_lines = f.readlines()

        self.assertEqual(len(file_lines), len(scd_lines)-2)
        self.assertEqual(file_lines[0], scd_lines[2])
        self.assertEqual(file_lines[1], scd_lines[3])

        yyyymmdd = l2[0]
        hhmm = l2[1]
        dur = l2[2]
        prio = l2[3]
        exp = l2[4]
        mode = l2[5]
        scdu.remove_line(yyyymmdd, hhmm, exp, mode, prio, dur)

        with open(scd_file.name, 'r') as f:
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

        with open(scd_file.name, 'r') as f:
            file_lines = f.readlines()

        self.assertEqual(len(file_lines), len(scd_lines) - 4)

    def test_remove_line_dne(self):
        """
        Test trying to remove lines from an SCD file that don't exist. Should raise ValueError
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n",
                     "20200924 00:00 - 0 normalscan common freq=10500\n", "20200926 00:00 - 0 normalscan common\n"]

        for line in scd_lines:
            scd_file.write(line)
        scd_file.close()

        with self.assertRaises(ValueError):
            scdu.remove_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)

# get_relevant_lines tests
    def test_incorrect_datetime_fmt(self):
        """
        Test trying to input an incorrect datetime string, should raise ValueError
        """
        scdu = scd_utils.SCDUtils(self.good_scd_file)

        with self.assertRaises(ValueError):
            scdu.get_relevant_lines("202", "blah")
        with self.assertRaises(ValueError):
            scdu.get_relevant_lines("20225555", None)
        with self.assertRaises(ValueError):
            scdu.get_relevant_lines(5, '5890')

    def test_empty_file(self):
        """
        Test trying to get relevant lines from an empty file, should raise IndexError
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        with self.assertRaises(IndexError):
            scdu.get_relevant_lines("20061101", "00:00")

    def test_no_lines_relevant(self):
        """
        Test trying to get relevant lines from a file with all lines in the past
        """
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        lines = scdu.get_relevant_lines("21001113", "00:00")
        self.assertEqual(len(lines), 0)

    def test_one_line_relevant(self):
        """
        Test trying to get relevant lines from a file with one line in the future
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_exp_line = "20200917 00:00 - 0 normalscan common\n"
        scd_file.write(test_exp_line)
        scd_file.close()
        lines = scdu.get_relevant_lines("20200916", "23:59")
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], test_exp_line)

    def test_some_lines_relevant(self):
        """
        Test trying to get relevant lines from a file with some lines in the future
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n",
                          "20200924 00:00 - 0 normalscan common freq=10500\n", "20200926 00:00 - 0 normalscan common\n"]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = scdu.get_relevant_lines("20200917", "00:01")
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[0], test_scd_lines[1])
        self.assertEqual(lines[1], test_scd_lines[2])
        self.assertEqual(lines[2], test_scd_lines[3])

    def test_all_lines_relevant_matched(self):
        """
        Test trying to get relevant lines from a file with all lines in the future or present
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n",
                          "20200924 00:00 - 0 normalscan common freq=10500\n", "20200926 00:00 - 0 normalscan common\n"]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = scdu.get_relevant_lines("20200917", "00:00")
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], test_scd_lines[0])
        self.assertEqual(lines[1], test_scd_lines[1])
        self.assertEqual(lines[2], test_scd_lines[2])
        self.assertEqual(lines[3], test_scd_lines[3])

    def test_all_lines_relevant(self):
        """
        Test trying to get relevant lines from a file with all lines in the future
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n",
                          "20200924 00:00 - 0 normalscan common freq=10500\n", "20200926 00:00 - 0 normalscan common\n"]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = scdu.get_relevant_lines("20200916", "00:00")
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], test_scd_lines[0])
        self.assertEqual(lines[1], test_scd_lines[1])
        self.assertEqual(lines[2], test_scd_lines[2])
        self.assertEqual(lines[3], test_scd_lines[3])


class TestRemoteServer(unittest.TestCase):
    """
    unittest class to test the remote server and remote server options modules.
    All test methods must begin with the word 'test' to be run
    """
    def __init__(self, *args, **kwargs):
        """
        Init some reused variables
        """
        super().__init__(*args, **kwargs)

        self.good_config = f"{os.environ['BOREALISPATH']}/config/sas/sas_config.ini"
        self.good_scd_file = f"{os.environ['BOREALISPATH']}/tests/scheduler/good_scd_file.scd"

    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

# remote server options tests
    def test_remote_server_options(self):
        """
        Test creating an options object
        """
        config = f"{os.environ['BOREALISPATH']}/tests/scheduler/good_config_file.ini"
        ops = rso.RemoteServerOptions(config)
        self.assertEqual(ops.site_id, 'tst')

    def test_no_borealispath(self):
        """
        Test creating an options object without a BOREALISPATH set up, raises ValueError
        """
        # Need to remove the environment variable, reset for other tests
        os.environ.pop('BOREALISPATH')
        sys.path.remove(BOREALISPATH)
        del os.environ['BOREALISPATH']
        os.unsetenv('BOREALISPATH')
        with self.assertRaisesRegex(ValueError, "BOREALISPATH"):
            ops = rso.RemoteServerOptions()
            self.assertEqual(ops.site_id, 'sas')
        os.environ['BOREALISPATH'] = BOREALISPATH
        sys.path.append(BOREALISPATH)

    def test_bad_config_file(self):
        """
        Test creating an options object without a good config file. Raises IOError
        """
        bad_config = f"{os.environ['BOREALISPATH']}/tests/scheduler/bad_config.ini"
        with self.assertRaises(json.JSONDecodeError):
            rso.RemoteServerOptions(config_path=bad_config)

    def test_empty_config_file(self):
        """
        Test creating an options object an empty config file. Raises IOError
        """
        bad_config = f"{os.environ['BOREALISPATH']}/tests/scheduler/empty_config.ini"
        with self.assertRaises(IOError):
            rso.RemoteServerOptions(config_path=bad_config)

    def test_config_file_dne(self):
        """
        Test creating an options object with a config file that DNE
        """
        bad_config = "/not/config/file/location"
        with self.assertRaises(IOError):
            rso.RemoteServerOptions(config_path=bad_config)

# format_to_atq tests
    def test_make_atq_commands(self):
        """
        Test creating atq commands from scd lines
        """
        # Atq commands are: [command to run] | at [now+ x minute | -t %Y%m%d%H%M]
        time_of_interest = datetime.datetime(2022, 9, 8, 12, 34)
        atq_str = remote_server.format_to_atq(time_of_interest, "some weird experiment with options", "some mode")
        self.assertEqual(atq_str, "echo 'screen -d -m -S starter /home/radar/borealis/scripts/steamed_hams.py "
                                  "some weird experiment with options release some mode | "
                                  "at -t 202209081234")
        atq_str = remote_server.format_to_atq(time_of_interest, "exp", "md", first_event_flag=True)
        self.assertEqual(atq_str, "echo 'screen -d -m -S starter /home/radar/borealis/scripts/steamed_hams.py "
                                  "exp release md | "
                                  "at now + 1 minute")
        time_of_interest = datetime.datetime(2019, 4, 3, 9, 56)
        atq_str = remote_server.format_to_atq(time_of_interest, "exp", "md", kwargs_string="this is the kwargs")
        self.assertEqual(atq_str, "echo 'screen -d -m -S starter /home/radar/borealis/scripts/steamed_hams.py "
                                  "exp release md --kwargs_string this is the kwargs | "
                                  "at -t 201904030956")

# plot_timeline tests
    # timeline_to_dict -> nested method so no unittesting
#    def test_timeline_to_dict(self):
#        """
#        Test getting an ordered dict back given timeline list
#        """
#        random_list = [{'order': 10}, {'order': 0}, {'order': 1}, {'order': 1}, {'order': 17}, {'order': 5}]

    # get_cmap -> nested method so no unittesting
#    def test_get_cmap(self):
#        """
#       Test the simple cmap function
#       """
#
#
    # split_event -> nested method so no unittesting
#    def test_split_event(self):
#        """
#        Test splitting an event recursively (long event over two or more days)
#        """

#    def test_plot_timeline(self): # TODO
#        """
#        After this test runs, there should be saved plots and a pickle file of the plot
#        """
#        timeline = []
#        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
#        timeline_of_interest = datetime.datetime()
#        site_id = os.environ['RADAR_ID']

# convert_scd_to_timeline tests # TODO
#    def test_scd_to_timeline_inf_duration(self):
#        """
#        Test scd to timeline. Last one should be infinite duration.
#        """
#        scd_lines = []
#        remote_server.convert_scd_to_timeline(scd_lines, time_of_interest)

# calculate_new_last_line_params -> Nested method so no unittesting

# timeline_to_atq tests
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

        The timeline being an empty list should still work as expected with no change to the atq
        The function should create a backup atq file with all current and future jobs
        timeline is a list of events, which are dicts with 'time', 'experiment', 'scheduling_mode', 'kwargs_string'
        """
        # Get the current atq, as it should be unchanged from previously
        get_atq_cmd = 'for j in $(atq | sort -k6,6 -k3,3M -k4,4 -k5,5 |cut -f 1); ' \
                      'do atq |grep -P "^$j\t"; at -c "$j" | tail -n 2; done'
        current_atq = sp.check_output(get_atq_cmd, shell=True)

        # Remove any existing atq backups directory
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        atq_dir = f"{scd_dir}/atq_backups/"
        shutil.rmtree(atq_dir)

        timeline_list = []
        site_id = os.environ['RADAR_ID']
        time_of_interest = datetime.datetime(2022, 11, 14, 3, 30)
        backup_time = time_of_interest.strftime("%Y.%m.%d.%H.%M")
        atq_command = remote_server.timeline_to_atq(timeline_list, scd_dir, time_of_interest, site_id)
        self.assertEqual(atq_command, current_atq)
        self.assertTrue(os.path.exists(atq_dir))
        self.assertTrue(os.path.exists(f"{atq_dir}/{backup_time}.{site_id}.atq"))
        shutil.rmtree(atq_dir)

    def test_timeline_to_atq(self):
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

        The function should create a backup atq file with all current and future jobs
        timeline is a list of events, which are dicts with 'time', 'experiment', 'scheduling_mode', 'kwargs_string'
        """
        # Get the current atq, as it should be unchanged from previously
        get_atq_cmd = 'for j in $(atq | sort -k6,6 -k3,3M -k4,4 -k5,5 |cut -f 1); ' \
                      'do atq |grep -P "^$j\t"; at -c "$j" | tail -n 2; done'
        current_atq = sp.check_output(get_atq_cmd, shell=True)

        # Remove any existing atq backups directory
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        atq_dir = f"{scd_dir}/atq_backups/"
        shutil.rmtree(atq_dir)

        t1 = datetime.datetime(2022, 10, 9, 5, 30)
        t2 = datetime.datetime(2022, 10, 10, 0, 30)
        t3 = datetime.datetime(2022, 10, 11, 8, 0)

        # Of the three timeline events, the first will be considered the 'first event' in format_to_atq,
        # which means it gets scheduled NOW
        events = [{'time': t1, 'experiment': "politescan", "scheduling_mode": "discretionary", "kwargs_string": "-"},
                  {'time': t2, 'experiment': "normalscan", "scheduling_mode": "special", "kwargs_string": "hi=96"},
                  {'time': t3, 'experiment': "twofsound", "scheduling_mode": "common", "kwargs_string": "-"}]
        site_id = os.environ['RADAR_ID']
        time_of_interest = datetime.datetime(2022, 11, 14, 3, 30)
        backup_time = time_of_interest.strftime("%Y.%m.%d.%H.%M")
        atq_command = remote_server.timeline_to_atq(events, scd_dir, time_of_interest, site_id)

        # The atq should be the same as the events added were all in the past
        self.assertEqual(atq_command, current_atq)
        self.assertTrue(os.path.exists(atq_dir))
        self.assertTrue(os.path.exists(f"{atq_dir}/{backup_time}.{site_id}.atq"))
        shutil.rmtree(atq_dir)

# get_relevant_lines tests
    def test_inc_datetime(self):
        """
        Test trying to input an incorrect datetime type, should raise AttributeError
        """
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        with self.assertRaises(AttributeError):
            remote_server.get_relevant_lines(scdu, "blah")
        with self.assertRaises(AttributeError):
            remote_server.get_relevant_lines(scdu, None)
        with self.assertRaises(AttributeError):
            remote_server.get_relevant_lines(scdu, 5890)

    def test_one_relevant_lines(self):
        """
        Use a time-of-interest that is far in the future so there is only one relevant line (should be last line)
        """
        time_of_interest = datetime.datetime(2050, 11, 14, 0, 1)
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        with self.assertRaises(SystemExit):
            lines = remote_server.get_relevant_lines(scdu, time_of_interest)
            self.assertEqual(lines, ["20220929 12:00 - 0 normalscan common"])

    def test_no_relevant_lines(self):
        """
        Use a time-of-interest that is far in the future with an SCD file without any inf duration lines
        """
        time_of_interest = datetime.datetime(2050, 11, 14, 0, 1)
        scdfile = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdfile.write("20220929 12:00 360 0 normalscan common")  # A 360 minute duration line
        scdfile.close()
        scdu = scd_utils.SCDUtils(scdfile.name)
        with self.assertRaises(SystemExit):
            lines = remote_server.get_relevant_lines(scdu, time_of_interest)
            self.assertEqual(lines, [])

    def test_one_prev_relevant_line(self):
        """
        Use a time-of-interest that is far in the future with an SCD file with one inf duration line in the past
        """
        time_of_interest = datetime.datetime(2050, 11, 14, 0, 1)
        scdfile = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdfile.write("20210903 00:00 - 0 twofsound common")  # An infinite duration line
        scdfile.write("20220929 12:00 60 0 normalscan common")  # A 60 minute duration line
        scdfile.write("20220929 13:00 60 0 normalscan common")  # A 60 minute duration line
        scdfile.close()
        scdu = scd_utils.SCDUtils(scdfile.name)
        with self.assertRaises(SystemExit):
            lines = remote_server.get_relevant_lines(scdu, time_of_interest)
            self.assertEqual(lines, ["20210903 00:00 - 0 twofsound common"])

    def test_empty_scdu_file(self):
        """
        Test trying to get relevant lines from an empty file, should raise IndexError
        """
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        with self.assertRaises(IndexError):
            remote_server.get_relevant_lines(scdu, "20221114 00:00")

    def test_no_scdu_lines_relevant(self):
        """
        Test trying to get relevant lines from a file
        """
        scdu = scd_utils.SCDUtils(self.good_scd_file)
        lines = remote_server.get_relevant_lines(scdu, "00:00")
        self.assertEqual(len(lines), 0)

    def test_one_line_relevant(self):
        """
        Test trying to get relevant lines from a file with one line in the future
        """
        time_of_interest = datetime.datetime(2022, 11, 14, 0, 1)
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_exp_line = "20221115 03:00 - 0 normalscan common\n"
        scd_file.write(test_exp_line)
        scd_file.close()
        lines = remote_server.get_relevant_lines(scdu, time_of_interest)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], test_exp_line)

    def test_no_inf_dur_relevant(self):
        """
        Test trying to get relevant lines from a file with one line in the future but no infinite duration lines
        """
        time_of_interest = datetime.datetime(2020, 9, 25, 0, 1)
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_scd_lines = ["20200917 00:00 60 0 normalscan common\n", "20200921 00:00 1440 0 normalscan discretionary\n",
                          "20200924 00:00 360 0 normalscan common freq=10500\n",
                          "20200926 00:00 120 0 normalscan common\n"]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = remote_server.get_relevant_lines(scdu, time_of_interest)
        self.assertEqual(len(lines), 1)
        self.assertEqual(lines[0], test_scd_lines)

    def test_some_lines_relevant(self):
        """
        Test trying to get relevant lines from a file with some lines in the future
        """
        time_of_interest = datetime.datetime(2020, 9, 23, 0, 1)
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n",
                          "20200924 00:00 - 0 normalscan common freq=10500\n",
                          "20200926 00:00 - 0 normalscan common\n"]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = remote_server.get_relevant_lines(scdu, time_of_interest)
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0], test_scd_lines[2])
        self.assertEqual(lines[1], test_scd_lines[3])

    def test_all_lines_relevant_matched(self):
        """
        Test trying to get relevant lines from a file with all lines in the future or present
        """
        time_of_interest = datetime.datetime(2020, 9, 17, 0, 0)
        scd_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n",
                          "20200924 00:00 - 0 normalscan common freq=10500\n",
                          "20200926 00:00 - 0 normalscan common\n"]
        for test_line in test_scd_lines:
            scd_file.write(test_line)
        scd_file.close()
        lines = remote_server.get_relevant_lines(scdu, time_of_interest)
        self.assertEqual(len(lines), 4)
        self.assertEqual(lines[0], test_scd_lines[0])
        self.assertEqual(lines[1], test_scd_lines[1])
        self.assertEqual(lines[2], test_scd_lines[2])
        self.assertEqual(lines[3], test_scd_lines[3])


class TestLocalServer(unittest.TestCase):
    """
    unittest class to test the local server module. All test methods must begin with the word 'test' to be run
    """

    def __init__(self, *args, **kwargs):
        """
        Set up variables and data used for unit testing
        """
        super().__init__(*args, **kwargs)

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
        # Ensure the scd dir and therefore the git repo for schedules doesn't exist before
        shutil.rmtree(scd_dir)
        self.assertFalse(os.path.exists(scd_dir))
        local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(scd_dir))
        self.assertTrue(os.path.exists(scd_dir+"schedules/"))

    # new_swg_file_available
    def test_swg_new_file(self):
        """
        Test new file exists method, which checks for new swg file uploads via git and returns True or False
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        swg = local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(scd_dir))
        new = swg.new_swg_file_available()
        self.assertFalse(new)  # Assume that the repo is up-to-date always, not sure how else to test this

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
        site_id = os.environ['RADAR_ID']

        modes = local_scd_server.EXPERIMENTS[site_id]
        swg = local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(scd_dir))
        with self.assertRaises(FileNotFoundError):
            params = swg.parse_swg_to_scd(modes, site_id, first_run=True)
            self.assertEqual(params, None)

    def test_missing_hours(self):
        """
        Test parsing the SWG file. Should fail due to a gap in the schedule of several hours.
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        site_id = os.environ['RADAR_ID']
        swg_file = f"{os.environ['BOREALISPATH']}/tests/scheduler/missing_hours.swg"

        # Need to ensure we put in the current month to the schedule file and set first run to True
        mm_yyyy = datetime.datetime.today().strftime("%B %Y")
        yyyy = datetime.datetime.today().strftime("%Y")
        yyyymm = datetime.datetime.today().strftime("%Y%m")
        new_swg_file = f"{swg_file}/schedules/{yyyy}/{yyyymm}"
        with open(swg_file, 'r') as f:
            swg_data = f.read()
        with open(new_swg_file, 'w') as f:
            f.write(mm_yyyy + swg_data)

        modes = local_scd_server.EXPERIMENTS[site_id]
        swg = local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(scd_dir))
        self.assertTrue(os.path.exists(swg_file))
        self.assertTrue(os.path.exists(new_swg_file))
        with self.assertRaises(ValueError):
            params = swg.parse_swg_to_scd(modes, site_id, first_run=True)
            self.assertEqual(params, None)

        # Remove the file we wrote
        shutil.rmtree(new_swg_file)

    def test_bad_experiment(self):
        """
        Test parsing the SWG file. Should fail when it encounters a line with a non-existent experiment
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        site_id = os.environ['RADAR_ID']
        swg_file = f"{os.environ['BOREALISPATH']}/tests/scheduler/bad_experiment.swg"

        # Need to ensure we put in the current month to the schedule file and set first run to True
        mm_yyyy = datetime.datetime.today().strftime("%B %Y")
        yyyy = datetime.datetime.today().strftime("%Y")
        yyyymm = datetime.datetime.today().strftime("%Y%m")
        new_swg_file = f"{swg_file}/schedules/{yyyy}/{yyyymm}"
        with open(swg_file, 'r') as f:
            swg_data = f.read()
        with open(new_swg_file, 'w') as f:
            f.write(mm_yyyy + swg_data)

        modes = local_scd_server.EXPERIMENTS[site_id]
        swg = local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(scd_dir))
        self.assertTrue(os.path.exists(swg_file))
        self.assertTrue(os.path.exists(new_swg_file))
        with self.assertRaises(ValueError):
            params = swg.parse_swg_to_scd(modes, site_id, first_run=True)
            self.assertEqual(params, None)

        # Remove the file we wrote
        shutil.rmtree(new_swg_file)

    def test_swg_parse(self):
        """
        Test parsing the SWG file. Should work properly and return a list of parsed parameters corresponding to
        the swg file input
        """
        scd_dir = f"{os.environ['BOREALISPATH']}/tests/scheduler/"
        site_id = os.environ['RADAR_ID']
        swg_file = f"{os.environ['BOREALISPATH']}/tests/scheduler/complicated_schedule.swg"

        # Need to ensure we put in the current month to the schedule file and set first run to True
        mm_yyyy = datetime.datetime.today().strftime("%B %Y")
        yyyy = datetime.datetime.today().strftime("%Y")
        yyyymm = datetime.datetime.today().strftime("%Y%m")
        new_swg_file = f"{swg_file}/schedules/{yyyy}/{yyyymm}"
        with open(swg_file, 'r') as f:
            swg_data = f.read()
        with open(new_swg_file, 'w') as f:
            f.write(mm_yyyy + swg_data)

        modes = local_scd_server.EXPERIMENTS[site_id]
        swg = local_scd_server.SWG(scd_dir)
        self.assertTrue(os.path.exists(scd_dir))
        self.assertTrue(os.path.exists(swg_file))
        self.assertTrue(os.path.exists(new_swg_file))
        parsed_params = swg.parse_swg_to_scd(modes, site_id, first_run=True)
        self.assertTrue(isinstance(parsed_params, list))


class TestSchedulerEmailer(unittest.TestCase):
    """
    unittest class to test the scheduler emailing module. All test methods must begin with the word 'test' to be run
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.emails = "kevin.krieger@usask.ca\nkevinjkrieger@gmail.com"
        self.email_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        self.email_file.write(self.emails)
        self.email_file.close()

        self.logfile = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        self.logfile.write('Not much of a logfile,\n but here we are')
        self.logfile.close()

        self.no_perms = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        self.no_perms_file = self.no_perms.name
        os.chmod(self.no_perms_file, 0o000)
        self.no_perms.close()

        self.attachments = [f"{os.environ['BOREALISPATH']}/docs/uml_diagrams/scheduler/local.drawio.png",
                            f"{os.environ['BOREALISPATH']}/docs/uml_diagrams/scheduler/remote.drawio.png",
                            f"{os.environ['BOREALISPATH']}/docs/source/scheduling.rst"]

    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

    def test_no_arg(self):
        """
        Test calling the scheduler emailer with empty arg string, it expects a filename
        """
        with self.assertRaisesRegex(ValueError, "OSError opening emails text file: "):
            email_utils.Emailer('')

    def test_dir(self):
        """
        Test calling the scheduler emailer with a directory, it expects a filename
        """
        with self.assertRaisesRegex(ValueError, "OSError opening emails text file: "):
            email_utils.Emailer(os.environ['HOME'])

    def test_no_permissions(self):
        """
        Test calling the scheduler emailer with a file without permissions
        """

        with self.assertRaisesRegex(ValueError, "OSError opening emails text file: "):
            email_utils.Emailer(self.no_perms_file)

    def test_not_owner(self):
        """
        Test calling the scheduler emailer with a file with wrong owner
        """
        with self.assertRaisesRegex(ValueError, "OSError opening emails text file: "):
            email_utils.Emailer('/root/')

    def test_no_logfile(self):
        """
        Test calling the scheduler emailer with a non-existent log file
        Should send email with body of email containing error message
        """
        with open(self.email_file) as f:
            e = email_utils.Emailer(f.name)
            subject = 'Unittest scheduler emailer, no log file'
            with self.assertRaises(SystemExit):
                e.email_log(subject, 'albj;ljkas;ldj;oij_nonexistentlogfilename')

    def test_bad_logfile(self):
        """
        Test calling the scheduler emailer with a log file that can't be opened
        Should send email with body of email containing error message
        """
        with open(self.email_file) as f:
            e = email_utils.Emailer(f.name)
            subject = 'Unittest scheduler emailer, bad log file'
            with self.assertRaises(SystemExit):
                e.email_log(subject, self.no_perms_file)

    def test_email_works(self):
        """
        Test with everything working properly
        """
        with open(self.email_file) as f:
            e = email_utils.Emailer(f.name)
            subject = 'Unittest scheduler emailer'
            with self.assertRaises(SystemExit):
                e.email_log(subject, self.logfile.name)

    def test_email_attachments_work(self):
        """
        Test with everything working properly, including several attachments
        """
        with tempfile.NamedTemporaryFile(mode='w+t', delete=False) as f:
            f.write(self.emails)
            e = email_utils.Emailer(f.name)
            subject = 'Unittest scheduler emailer'
            with self.assertRaises(SystemExit):
                e.email_log(subject, self.logfile.name, attachments=self.attachments)


if __name__ == "__main__":
    unittest.main()
