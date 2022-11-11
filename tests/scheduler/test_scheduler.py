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


    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

# __init__ tests # TODO
    def test_bad_scd_file(self):
        """
        Test giving a non-existent scd file to the module
        """
        scdu = scd_utils.SCDUtils('thisfilelikelydoesntexist90758hjna;ksjn')

# check_line tests # TODO
    def test_invalid_yyyymmdd(self):
        """
        Test an invalid start date
        """

    def test_invalid_hhmm(self):
        """
        Test an invalid start hour and minute
        """

    def test_invalid_experiment(self):
        """
        Test an invalid experiment name - Must be an experiment named in the repo
        """
    def test_invalid_mode(self):
        """
        Test an invalid mode (possible: ['common', 'special', 'discretionary'])
        """
    def test_invalid_prio(self):
        """
        Test an invalid priority (integer from 0 to 20 inclusive)
        """
    def test_invalid_duration(self):
        """
        Test an invalid duration (optional: needs to be an integer minutes)
        """
    def test_invalid_kwargs_string(self):
        """
        Test an invalid kwargs string (optional arguments that may be passed to an experiment's kwargs)
        """

    def test_non_int_prio(self):
        """
        Test an invalid priority type
        """

# read_scd tests # TODO
    def test_invalid_num_args(self):
        """
        Test an invalid number of arguments, requires 6 or 7 args
        """

# fmt_line tests
# TODO

# write_scd tests
# TODO

# add_line tests
# TODO

# remove_line tests
# TODO

# get_relevant_lines tests
# TODO


class TestRemoteServer(unittest.TestCase):
    """
    unittest class to test the remote server and remote server options modules.
    All test methods must begin with the word 'test' to be run
    """
    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

# remote server options tests # TODO

# format_to_atq tests # TODO

# get_next_month_from_data tests # TODO

# timeline_to_dict tests # TODO

# plot_timeline tests # TODO
    # get_cmap
    # split_event

# convert_scd_to_timeline tests # TODO
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
