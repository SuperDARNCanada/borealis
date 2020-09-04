"""
Test module for the experiment_handler/experiment_prototype code.
It is run simply via 'python3 test_experiment_exceptions.py' and will go through all tests
in the experiment_tests.csv file as well as the hardcoded tests here that don't fit nicely into
a csv file.

The csv file format is:
[#][experiment file module import name]::[regex error message]

The [#] is an optional comment, and that line will be removed
An example of a test line is:
testing_archive.my_test_experiment.py::Regex line that * matches the ExperimentException err msg

References:
https://stackoverflow.com/questions/32899/how-do-you-generate-dynamic-parameterized-unit-tests-in-python
https://docs.python.org/3/library/unittest.html

:copyright: 2020 SuperDARN Canada
:author: Kevin Krieger
"""


import unittest
import os
import sys
import argparse

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)
# Need to hardcode this, as unittest does weird things when you supply an argument on command line,
# or if you use argparse. There is probably a better way
input_test_file = "PATH_TO_/experiment_tests.csv"

# Call experiment handler main function like so: eh.main(['normalscan', 'discretionary'])
import experiment_handler.experiment_handler as eh
from experiment_prototype.experiment_exception import ExperimentException


def ehmain(experiment='normalscan', scheduling_mode='discretionary'):
    """
    Convenience method to call the experiment handler with arguments
    """
    eh.main([experiment, scheduling_mode])


class TestExperimentEnvSetup(unittest.TestCase):
    """
    A unittest class to test the environment setup for the experiment_handler module.
    All test methods must begin with the word 'test' to be run by unittest.
    """
    def setUp(self):
        """
        This function is called before every test_* method (every test case in unittest lingo)
        """
        print("Method: ", self._testMethodName)

    def test_no_args(self):
        """
        Test calling the experiment handler without any command line arguments, which returns 2
        """
        with self.assertRaisesRegex(SystemExit, "2"):
            eh.main([])

    def test_borealispath(self):
        """
        Test failure to have BOREALISPATH in env
        """
        # Need to remove the environment variable, reset for other tests
        del os.environ['BOREALISPATH']
        with self.assertRaisesRegex(KeyError, "BOREALISPATH"):
            ehmain()
        os.environ['BOREALISPATH'] = BOREALISPATH

    def test_config_file(self):
        """
        Test the code that checks for the config file
        """
        # Rename the config file temporarily
        os.rename(BOREALISPATH + 'config.ini', BOREALISPATH + '_config.ini')
        with self.assertRaisesRegex(ExperimentException, "Cannot open config file at "):
            ehmain()
        # experiment_prototype.experiment_exception.ExperimentException: Cannot open config file
        # at /home/kevin/PycharmProjects/borealis//config.ini

        # Now rename the config file and move on
        os.rename(BOREALISPATH + '_config.ini', BOREALISPATH + 'config.ini')

    def test_hdw_file(self):
        """
        Test the code that checks for the hdw.dat file
        """
        # Rename the hdw.dat file temporarily
        os.rename(BOREALISPATH + 'hdw.dat.sas', BOREALISPATH + '_hdw.dat.sas')
        with self.assertRaisesRegex(ExperimentException, "Cannot open hdw.dat.[a-z]{3} file at"):
             ehmain()
        # experiment_prototype.experiment_exception.ExperimentException: Cannot open hdw.dat.sas
        # file at /home/kevin/PycharmProjects/borealis//hdw.dat.sas

        # Now rename the hdw.dat file and move on
        os.rename(BOREALISPATH + '_hdw.dat.sas', BOREALISPATH + 'hdw.dat.sas')


class TestExperimentExceptions(unittest.TestCase):
    """
       A unittest class to test various ways for an experiment to fail for the experiment_handler
       module. All test methods must begin with the word 'test' to be run by unittest.
    """
    def setUp(self):
        """
        This function is called before every test_* method (every test case in unittest lingo)
        """
        print("Method: ", self._testMethodName)


def test_generator(module_name, exception_msg_regex):
    """
    Generate a single test for the given module name and exception message
    :param module_name: Experiment module name, string (i.e. 'normalscan')
    :param exception_msg_regex: Error msg the experiment module is expected to return, regex string
    """
    def test(self):
        with self.assertRaisesRegex(ExperimentException, exception_msg_regex):
            ehmain(experiment=module_name)
    return test


if __name__ == '__main__':
    # Redirect stderr because it's annoying
    # null = open(os.devnull, 'w')
    # sys.stderr = null

    # Open the file given on command line with a set of tests, one per line.
    # File format is: [experiment module]::[string regex message that the experiment will raise]
    # Generate a single test for each of the lines in the file.
    try:
        with open(input_test_file) as test_suite_list:
            for test in test_suite_list.readlines():
                # Remove comment lines and empty lines
                if test.startswith('#') or test.strip() == '':
                    continue
                # Separate on double colon to ensure the regex msg isn't split
                exp_module_name = test.split('::')[0]
                exp_exception_msg_regex = test.split('::')[1]
                test = test_generator(exp_module_name, exp_exception_msg_regex.strip())
                # setattr is used to add properly named test methods to TestExperimentExceptions
                setattr(TestExperimentExceptions, exp_module_name, test)
        print("Done building tests")
    except TypeError:
        print("No extra tests supplied, only performing basic tests")

    unittest.main()
