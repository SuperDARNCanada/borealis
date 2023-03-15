"""
Test module for the experiment_handler/experiment_prototype code.
It is run simply via 'python3 experiment_unittests.py' and will go through all tests
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
https://www.bnmetrics.com/blog/dynamic-import-in-python3

:copyright: 2020 SuperDARN Canada
:author: Kevin Krieger
"""


import unittest
import os
import sys
import inspect
import pkgutil
from pathlib import Path
from importlib import import_module

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(BOREALISPATH)

# Need to hardcode this, as unittest does weird things when you supply an argument on command line,
# or if you use argparse. There is probably a better way
input_test_file = f"{BOREALISPATH}/tests/testing_utils/experiments/experiment_tests.csv"

import experiment_handler as eh
from experiment_prototype.experiment_exception import ExperimentException
from experiment_prototype.experiment_prototype import ExperimentPrototype
import borealis_experiments.superdarn_common_fields as scf

def ehmain(experiment_name='normalscan', scheduling_mode='discretionary'):
    """
    Calls the functions within experiment handler that verify an experiment

    :param  experiment_name: The module name of the experiment to be verified. Experiment name must
                             be in module format (i.e. testing_archive.test_example for unit tests)
                             to work properly
    :type   experiment_name: str
    :param  scheduling_mode: The scheduling mode to run. Defaults to 'discretionary'
    :type   scheduling_mode: str
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            Experiment = eh.retrieve_experiment(experiment_name)
            exp = Experiment()
            exp._set_scheduling_mode(scheduling_mode)
            exp.build_scans()
        finally:
            sys.stdout = old_stdout


class TestExperimentEnvSetup(unittest.TestCase):
    """
    A unittest class to test the environment setup for the experiment_handler module.
    All test methods must begin with the word 'test' to be run by unittest.
    """
    def setUp(self):
        """
        This function is called before every test_* method within this class (every test case in
        unittest lingo)
        """
        print("\nMethod: ", self._testMethodName)

    def test_no_args(self):
        """
        Test calling the experiment handler without any command line arguments, which returns 2
        """
        with self.assertRaisesRegex(SystemExit, "2"):
            eh.main([])

    @unittest.skip("Skip for TODO reason")
    def test_borealispath(self):
       """
       Test failure to have BOREALISPATH in env
       """
       # Need to remove the environment variable, reset for other tests
       os.environ.pop('BOREALISPATH')
       sys.path.remove(BOREALISPATH)
       del os.environ['BOREALISPATH']
       os.unsetenv('BOREALISPATH')
       with self.assertRaisesRegex(KeyError, "BOREALISPATH"):
           ehmain()
       os.environ['BOREALISPATH'] = BOREALISPATH
       sys.path.append(BOREALISPATH)

    def test_config_file(self):
        """
        Test the code that checks for the config file
        """
        # Rename the config file temporarily
        site_id = scf.options.site_id
        os.rename(f"{BOREALISPATH}/config/{site_id}/{site_id}_config.ini", f"{BOREALISPATH}/_config.ini")
        with self.assertRaisesRegex(ValueError, "Cannot open config file at "):
            ehmain()

        # Now rename the config file and move on
        os.rename(f"{BOREALISPATH}/_config.ini", f"{BOREALISPATH}/config/{site_id}/{site_id}_config.ini")

    @unittest.skip("Cannot test this while hdw.dat files are in /usr/local/hdw")
    def test_hdw_file(self):
        """
        Test the code that checks for the hdw.dat file
        """
        site_name = scf.options.site_id
        hdw_path = scf.options.hdw_path
        # Rename the hdw.dat file temporarily
        os.rename(f"{hdw_path}/hdw.dat.{site_name}", f"{hdw_path}/_hdw.dat.{site_name}")

        with self.assertRaisesRegex(ValueError, "Cannot open hdw.dat.[a-z]{3} file at"):
             ehmain()

        # Now rename the hdw.dat file and move on
        os.rename(f"{hdw_path}/_hdw.dat.{site_name}", f"{hdw_path}/hdw.dat.{site_name}")


class TestExperimentExceptions(unittest.TestCase):
    """
    A unittest class to test various ways for an experiment to fail for the experiment_handler
    module. Tests will check that exceptions are correctly thrown for each failure case. All test
    methods must begin with the word 'test' to be run by unittest.
    """
    def setUp(self):
        """
        This function is called before every test_* method (every test case in unittest lingo)
        """
        print("\nException Test: ", self._testMethodName)


def build_unit_tests():
    """
    Create individual unit tests for all test cases specified in experiment_tests.csv

    Open the file hardcoded above with a set of tests, one per line.
    File format is: [experiment module]::[string regex message that the experiment will raise]
    Generate a single test for each of the lines in the file.
    """
    try:
        with open(input_test_file) as test_suite_list:
            for test_line in test_suite_list.readlines():
                # Remove comment lines and empty lines
                if test_line.startswith('#') or test_line.strip() == '':
                    continue
                # Separate on double colon to ensure the regex msg isn't split
                exp_module_name = test_line.split('::')[0]  # Names all start with "test_"
                exp_exception_msg_regex = test_line.split('::')[1]
                test = exception_test_generator(exp_module_name, exp_exception_msg_regex.strip())
                # setattr makes a properly named test method within TestExperimentExceptions which 
                # can be run by unittest.main()
                setattr(TestExperimentExceptions, exp_module_name, test)
        print("Done building exception unit tests")
    except TypeError:
        print(f"Could not open test file {input_test_file}, only performing basic tests")

def exception_test_generator(module_name, exception_msg_regex):
    """
    Generate a single test for the given module name and exception message

    :param module_name: Experiment module name, i.e. 'normalscan'
    :type module_name: str
    :param exception_msg_regex: Regex error msg the experiment module is expected to return
    :type exception_msg_regex: str
    """
    def test(self):
        with self.assertRaisesRegex(ExperimentException, exception_msg_regex):
            ehmain(experiment_name=module_name)
    return test


class TestExperiments(unittest.TestCase):
    """
    A unittest class to test all Borealis experiments and verify that none of them are built
    incorrectly. Tests are verified using code within experiment handler. All test methods must
    begin with the word 'test' to be run by unittest.
    """
    def setUp(self):
        """
        This function is called before every test_* method (every test case in unittest lingo)
        """
        print("\nExperiment Test: ", self._testMethodName)


def build_experiment_tests():
    """
    Create individual unit tests for all experiments within the base borealis_experiments/
    directory. All experiments are run to ensure no exceptions are thrown when they are built
    """
    experiment_package = 'borealis_experiments'
    experiment_path = f"{BOREALISPATH}/src/{experiment_package}/"
    if not os.path.exists(experiment_path):
        raise OSError(f"Error: experiment path {experiment_path} is invalid")

    # Iterate through all modules in the borealis_experiments directory
    for (_, name, _) in pkgutil.iter_modules([Path(experiment_path)]):
        imported_module = import_module('.' + name, package=experiment_package)
        # Loop through all attributes of each found module
        for i in dir(imported_module):
            attribute = getattr(imported_module, i)
            # To verify that an attribute is a runnable experiment, check that the attribute is 
            # a class and inherits from ExperimentPrototype
            if inspect.isclass(attribute) and issubclass(attribute, ExperimentPrototype):
                # Only create a test if the current attribute is the experiment itself
                if 'ExperimentPrototype' not in str(attribute):
                    test = experiment_test_generator(name)
                    # setattr make the "test" function a method within TestExperiments called 
                    # "test_[name]" which can be run via unittest.main()
                    setattr(TestExperiments, f"test_{name}", test)

def experiment_test_generator(module_name):
    """
    Generate a single test for a given experiment name. The test will try to run the experiment, 
    and if any exceptions are thrown (i.e. the experiment is built incorrectly) the test will fail.

    :param module_name: Experiment module name (i.e. 'normalscan')
    :type module_name: str
    """
    def test(self):
        try:
            ehmain(experiment_name=module_name)
        except Exception as err:
            self.fail(err)
    return test

if __name__ == '__main__':
    
    # Redirect stderr because it's annoying
    # null = open(os.devnull, 'w')
    # sys.stderr = null

    build_unit_tests()
    build_experiment_tests()
    unittest.main()
