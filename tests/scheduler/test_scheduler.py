"""
Test module for the scheduler code.
Run via: 'python3 test_scheduler.py'

:copyright: 2022 SuperDARN Canada
:author: Kevin Krieger
"""

import unittest
import os
import sys
import tempfile

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

    def __init__(self):
        super().__init__()
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
        self.no_perms = tempfile.NamedTemporaryFile()
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
        scdfile = tempfile.NamedTemporaryFile()
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
        scd_file = tempfile.NamedTemporaryFile()
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
        scd_file = tempfile.NamedTemporaryFile()
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_file.close()
        with self.assertRaisesRegex(ValueError, "Line is a duplicate of an existing line"):
            scdu.add_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
            scdu.add_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)

    def test_priority_time_dup(self):
        """
        Test trying to add line with the same time and priority as an existing line
        """
        scd_file = tempfile.NamedTemporaryFile()
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_file.close()
        with self.assertRaisesRegex(ValueError, "Priority already exists at this time"):
            scdu.add_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
            scdu.add_line(self.yyyymmdd, self.hhmm, 'twofsound', 'special', self.prio, '1440', self.kwargs)

    def test_add_line(self):
        """
        Test trying to add a line to a file
        """
        scd_file = tempfile.NamedTemporaryFile()
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_file.close()
        scdu.add_line(self.yyyymmdd, self.hhmm, self.exp, self.mode, self.prio, self.dur, self.kwargs)
        lines = []
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
        scd_file = tempfile.NamedTemporaryFile()
        scdu = scd_utils.SCDUtils(scd_file.name)
        scd_file.close()
        counter = 0
        for yyyymmdd in range(20221001, 1, 20221031):
            for hhmm in range(59, -1, 0):
                counter += 1
                scdu.add_line(str(f"{yyyymmdd:0>8}"), hmstr, self.exp, self.mode, self.prio, self.dur, self.kwargs)
                hmstr = f"{hhmm:0<4}"[:2] + ':' + f"{hhmm:0<4}"[2:]
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
        Test trying to remove lines from an scd file
        """
        scd_file = tempfile.NamedTemporaryFile()
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
        Test trying to remove lines from an scd file that don't exist. Should raise ValueError
        """
        scd_file = tempfile.NamedTemporaryFile()
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
        scd_file = tempfile.NamedTemporaryFile()
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
        scd_file = tempfile.NamedTemporaryFile()
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
        scd_file = tempfile.NamedTemporaryFile()
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n",
                          "20200924 00:00 - 0 normalscan common freq=10500\n", "20200926 00:00 - 0 normalscan common\n"]
        scd_file.write(test_scd_lines)
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
        scd_file = tempfile.NamedTemporaryFile()
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n",
                          "20200924 00:00 - 0 normalscan common freq=10500\n", "20200926 00:00 - 0 normalscan common\n"]
        scd_file.write(test_scd_lines)
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
        scd_file = tempfile.NamedTemporaryFile()
        scdu = scd_utils.SCDUtils(scd_file.name)
        test_scd_lines = ["20200917 00:00 - 0 normalscan common\n", "20200921 00:00 - 0 normalscan discretionary\n",
                          "20200924 00:00 - 0 normalscan common freq=10500\n", "20200926 00:00 - 0 normalscan common\n"]
        scd_file.write(test_scd_lines)
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
    def __init__(self):
        """
        Init some reused variables
        """
        self.good_config = f"{os.environ['BOREALISPATH']}/config/sas/sas_config.ini"

    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

# remote server options tests
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
            self.assertEqual(ops.site_id(), 'sas')
        os.environ['BOREALISPATH'] = BOREALISPATH
        sys.path.append(BOREALISPATH)

    def test_bad_config_file(self):
        """
        Test creating an options object without a good config file. Raises IOError
        """
        bad_config = tempfile.NamedTemporaryFile()

    # TODO:

# format_to_atq tests
    def test_make_atq_commands(self):
        """
        Test creating atq commands from scd lines
        """
# TODO:

# get_next_month_from_data tests
    def test_get_next_month(self):
        """
        Test getting the next month from a given date
        """
# TODO

# timeline_to_dict tests
    def test_timeline_to_dict(self):
        """
        Test getting an ordered dict back given timeline
        """
        # TODO

# plot_timeline tests # TODO This is a huge function, could probably be broken up into multiple
    # get_cmap
    def test_get_cmap(self):
        """
        Test the simple cmap function
        """
        # TODO:

    # split_event
    def test_split_event(self):
        """
        Test splitting an event recursively (long event over two or more days)
        """
        # TODO:


# convert_scd_to_timeline tests # TODO This is a huge function, could probably be broken up into multiple
    # calculate_new_last_line_params

# timeline_to_atq tests # TODO

# get_relevant_lines tests # TODO

# _main(): make_schedule tests # TODO

class TestLocalServer(unittest.TestCase):
    """
    unittest class to test the local server module. All test methods must begin with the word 'test' to be run
    """
    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

# get_next_month tests # TODO

# SWG tests: # TODO
    # init tests
    # new_swg_file_available
    # pull_new_swg_file
    # parse_swg_to_scd

# main() tests # TODO

class TestSchedulerEmailer(unittest.TestCase):
    """
    unittest class to test the scheduler emailing module. All test methods must begin with the word 'test' to be run
    """
    def __init__(self):
        super().__init__()

        self.emails = "kevin.krieger@usask.ca\nkevinjkrieger@gmail.com"
        self.email_file = tempfile.NamedTemporaryFile()
        self.email_file.write(self.emails)
        self.email_file.close()

        self.logfile = tempfile.NamedTemporaryFile()
        self.logfile.write('Not much of a logfile,\n but here we are')
        self.logfile.close()

        self.no_perms = tempfile.NamedTemporaryFile()
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
        with tempfile.NamedTemporaryFile() as f:
            f.write(self.emails)
            e = email_utils.Emailer(f.name)
            subject = 'Unittest scheduler emailer'
            with self.assertRaises(SystemExit):
                e.email_log(subject, self.logfile.name, attachments=self.attachments)


if __name__ == "__main__":
    unittest.main()
