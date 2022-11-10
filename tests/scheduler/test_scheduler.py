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
    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)


class TestRemoteScheduler(unittest.TestCase):
    """
    unittest class to test the remote scheduler module. All test methods must begin with the word 'test' to be run
    """
    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)


class TestLocalScheduler(unittest.TestCase):
    """
    unittest class to test the local scheduler module. All test methods must begin with the word 'test' to be run
    """
    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)


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
        """
        with open(self.email_file) as f:
            e = email_utils.Emailer(f.name)
            subject = 'Unittest scheduler emailer'
            e.email_log(subject, self.logfile.name)

    def test_email_works(self):
        """
        Test with everything working properly
        """
        with open(self.email_file) as f:
            e = email_utils.Emailer(f.name)
            subject = 'Unittest scheduler emailer'
            e.email_log(subject, self.logfile.name)

    def test_email_attachments_work(self):
        """
        Test with everything working properly, including several attachments
        """
        with tempfile.NamedTemporaryFile() as f:
            f.write(self.emails)
            e = email_utils.Emailer(f.name)
            subject = 'Unittest scheduler emailer'
            e.email_log(subject, self.logfile.name, attachments=self.attachments)


if __name__ == "__main__":
    unittest.main()
