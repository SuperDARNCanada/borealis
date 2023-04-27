"""
Test module for the experiment_handler/experiment_prototype code.

This script can be run most simply via 'python3 experiment_unittests.py'. This will run through all
experiments defined in the Borealis experiments top level directory, experiment exception tests
defined in the testing_archive directory, as well as any other tests hard coded into this script.
Any experiment that raises an exception when building will show up as a failed test here.

This script can also be run to test individual experiments by using the --experiment flag. For
example: `python3 experiment_unittests.py --experiment normalscan normalsound` will only test the
normalscan and normalsound experiments. Any experiments specified must exist within the Borealis
experiments top level directory (i.e. src/borealis_experiments).

Other command line options include:

- Specifying what radar site to run the tests as

References:
https://stackoverflow.com/questions/32899/how-do-you-generate-dynamic-parameterized-unit-tests-in-python
https://docs.python.org/3/library/unittest.html
https://www.bnmetrics.com/blog/dynamic-import-in-python3

:copyright: 2023 SuperDARN Canada
:author: Kevin Krieger, Theodore Kolkman
"""

import argparse
import unittest
import os
import sys
import inspect
import pkgutil
from pathlib import Path
from importlib import import_module

BOREALISPATH = os.environ['BOREALISPATH']
sys.path.append(f"{BOREALISPATH}/src")

import experiment_handler as eh
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

    @unittest.skip("Skip because it is annoying")
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


class TestExperimentArchive(unittest.TestCase):
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


class TestActiveExperiments(unittest.TestCase):
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


def build_unit_tests():
    """
    Create individual unit tests for all test cases specified in testing_archive directory of experiments path.
    """
    experiment_package = 'testing_archive'
    experiment_path = f"{BOREALISPATH}/src/borealis_experiments/{experiment_package}/"
    if not os.path.exists(experiment_path):
        raise OSError(f"Error: experiment path {experiment_path} is invalid")

    # Iterate through all modules in the borealis_experiments directory
    for (_, name, _) in pkgutil.iter_modules([Path(experiment_path)]):
        imported_module = import_module('.' + name, package=f'borealis_experiments.{experiment_package}')
        # Loop through all attributes of each found module
        for i in dir(imported_module):
            attribute = getattr(imported_module, i)
            # To verify that an attribute is a runnable experiment, check that the attribute is
            # a class and inherits from ExperimentPrototype
            if inspect.isclass(attribute) and issubclass(attribute, ExperimentPrototype):
                # Only create a test if the current attribute is the experiment itself
                if 'ExperimentPrototype' not in str(attribute):
                    if hasattr(attribute, 'error_message'):
                        # If expected to fail, should have a classmethod called "error_message"
                        # that contains the error message raised
                        exp_exception, msg = getattr(attribute, 'error_message')()
                        test = exception_test_generator('testing_archive.' + name, exp_exception, msg)
                    else:   # No exception expected - this is a positive test
                        test = experiment_test_generator('testing_archive.' + name)
                    # setattr makes a properly named test method within TestExperimentArchive which
                    # can be run by unittest.main()
                    setattr(TestExperimentArchive, name, test)
                    break

    print("Done building unit tests")


def exception_test_generator(module_name, exception, exception_message):
    """
    Generate a single test for the given module name and exception message

    :param module_name:         Experiment module name, i.e. 'normalscan'
    :type  module_name:         str
    :param exception:           Exception that is expected to be raised
    :type  exception:           BaseException
    :param exception_message:   Message from the Exception raised.
    :type  exception_message:   str
    """
    def test(self):
        with self.assertRaisesRegex(exception, exception_message):
            ehmain(experiment_name=module_name)
    return test


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
                    # setattr make the "test" function a method within TestActiveExperiments called
                    # "test_[name]" which can be run via unittest.main()
                    setattr(TestActiveExperiments, f"test_{name}", test)
    print("Done building experiment tests")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--site_id", required=False, default="sas", 
                        choices=["sas", "pgr", "inv", "rkn", "cly", "lab"], 
                        help="Site ID of site to test experiments as. Defaults to sas.")
    parser.add_argument("--experiment", required=False, nargs="+", default=None, 
                        help="Only run the experiments specified after this option. Experiments \
                        specified must exist within the top-level Borealis experiments directory.")

    args, extra_args = parser.parse_known_args()
    os.environ["RADAR_ID"] = args.site_id
    experiments = args.experiment

    if len(extra_args) != 0:
        print(f"Unknown command line arguments {extra_args}")
        parser.print_help()
        exit(1)

    if experiments is None:  # Run all unit tests and experiment tests
        build_unit_tests()
        build_experiment_tests()
        unittest.main(argv=sys.argv[:1])
    else:  # Only test specified experiments
        build_experiment_tests()
        exp_tests = []
        for exp in experiments:
            # Check experiment exists 
            if hasattr(TestActiveExperiments, f"test_{exp}"):
                # Create correct string to test the experiment with unittest
                exp_tests.append(f"TestActiveExperiments.test_{exp}")
            else:
                print(f"Could not find experiment {exp}. Exiting...")
                exit(1)
        unittest.main(argv=sys.argv[:1] + exp_tests)