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
from importlib import import_module
import importlib.util
import json

# Need the path append to import within this file
BOREALISPATH = os.environ["BOREALISPATH"]
sys.path.append(f"{BOREALISPATH}/src")


def redirect_to_devnull(func, *args, **kwargs):
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    return result


def ehmain(experiment_name="normalscan", scheduling_mode="discretionary", **kwargs):
    """
    Calls the functions within experiment handler that verify an experiment

    :param  experiment_name: The module name of the experiment to be verified. Experiment name must
                             be in module format (i.e. testing_archive.test_example for unit tests)
                             to work properly
    :type   experiment_name: str
    :param  scheduling_mode: The scheduling mode to run. Defaults to 'discretionary'
    :type   scheduling_mode: str
    :param  kwargs: The keyword arguments for the experiment
    :type   kwargs: dict
    """
    from utils import log_config

    log_config.log(
        console=False, logfile=False, aggregator=False
    )  # Prevent logging in experiment

    import experiment_handler as eh

    experiment = eh.retrieve_experiment(experiment_name)
    exp = experiment(**kwargs)
    exp._set_scheduling_mode(scheduling_mode)
    exp.build_scans()


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
        import experiment_handler as eh

        with self.assertRaisesRegex(SystemExit, "2"):
            eh.main([])

    @unittest.skip("Skip for TODO reason")
    def test_borealispath(self):
        """
        Test failure to have BOREALISPATH in env
        """
        # Need to remove the environment variable, reset for other tests
        os.environ.pop("BOREALISPATH")
        sys.path.remove(BOREALISPATH)
        del os.environ["BOREALISPATH"]
        os.unsetenv("BOREALISPATH")
        with self.assertRaisesRegex(KeyError, "BOREALISPATH"):
            ehmain()
        os.environ["BOREALISPATH"] = BOREALISPATH
        sys.path.append(BOREALISPATH)

    @unittest.skip("Skip because it is annoying")
    def test_config_file(self):
        """
        Test the code that checks for the config file
        """
        # Rename the config file temporarily
        site_id = scf.options.site_id
        os.rename(
            f"{BOREALISPATH}/config/{site_id}/{site_id}_config.ini",
            f"{BOREALISPATH}/_config.ini",
        )
        with self.assertRaisesRegex(ValueError, "Cannot open config file at "):
            ehmain()

        # Now rename the config file and move on
        os.rename(
            f"{BOREALISPATH}/_config.ini",
            f"{BOREALISPATH}/config/{site_id}/{site_id}_config.ini",
        )

    @unittest.skip("Cannot test this while hdw.dat files are in /usr/local/hdw")
    def test_hdw_file(self):
        """
        Test the code that checks for the hdw.dat file
        """
        import borealis_experiments.superdarn_common_fields as scf

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
    from experiment_prototype.experiment_prototype import ExperimentPrototype

    experiment_package = "testing_archive"
    experiment_path = f"{BOREALISPATH}/src/borealis_experiments/{experiment_package}/"
    if not os.path.exists(experiment_path):
        raise OSError(f"Error: experiment path {experiment_path} is invalid")

    # Iterate through all modules in the borealis_experiments directory
    for _, name, _ in pkgutil.iter_modules([experiment_path]):
        imported_module = import_module(
            "." + name, package=f"borealis_experiments.{experiment_package}"
        )
        # Loop through all attributes of each found module
        for i in dir(imported_module):
            attribute = getattr(imported_module, i)
            # To verify that an attribute is a runnable experiment, check that the attribute is
            # a class and inherits from ExperimentPrototype
            if inspect.isclass(attribute) and issubclass(
                attribute, ExperimentPrototype
            ):
                # Only create a test if the current attribute is the experiment itself
                if "ExperimentPrototype" not in str(attribute):
                    if hasattr(attribute, "error_message"):
                        # If expected to fail, should have a classmethod called "error_message"
                        # that contains the error message raised
                        exp_exception, msg = getattr(attribute, "error_message")()
                        test = exception_test_generator(
                            "testing_archive." + name, exp_exception, msg
                        )
                    else:  # No exception expected - this is a positive test
                        test = experiment_test_generator("testing_archive." + name)
                    # setattr makes a properly named test method within TestExperimentArchive which
                    # can be run by unittest.main()
                    setattr(TestExperimentArchive, name, test)
                    break


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
            redirect_to_devnull(ehmain, experiment_name=module_name)

    return test


def build_experiment_tests(experiments=None, kwargs=None):
    """
    Create individual unit tests for all experiments within the base borealis_experiments/
    directory. All experiments are run to ensure no exceptions are thrown when they are built
    """
    from experiment_prototype.experiment_prototype import ExperimentPrototype

    experiment_package = "borealis_experiments"
    experiment_path = f"{BOREALISPATH}/src/{experiment_package}/"
    if not os.path.exists(experiment_path):
        raise OSError(f"Error: experiment path {experiment_path} is invalid")

    # parse kwargs and pass to experiment
    kwargs_dict = {}
    if kwargs:
        for element in kwargs:
            if element == "":
                continue
            kwarg = element.split("=")
            if len(kwarg) == 2:
                kwargs_dict[kwarg[0]] = kwarg[1]
            else:
                raise ValueError(f"Bad kwarg: {element}")

    def add_experiment_test(exp_name: str):
        """Add a unit test for a given experiment"""
        imported_module = import_module("." + exp_name, package=experiment_package)
        # Loop through all attributes of each found module
        for i in dir(imported_module):
            attribute = getattr(imported_module, i)
            # To verify that an attribute is a runnable experiment, check that the attribute is
            # a class and inherits from ExperimentPrototype
            if inspect.isclass(attribute) and issubclass(
                attribute, ExperimentPrototype
            ):
                # Only create a test if the current attribute is the experiment itself
                if "ExperimentPrototype" not in str(attribute):
                    test = experiment_test_generator(exp_name, **kwargs_dict)
                    # setattr make the "test" function a method within TestActiveExperiments called
                    # "test_[exp_name]" which can be run via unittest.main()
                    setattr(TestActiveExperiments, f"test_{exp_name}", test)

    # Grab the experiments specified
    if experiments is not None:
        for name in experiments:
            spec = importlib.util.find_spec("." + name, package=experiment_package)
            if spec is None:
                # Add in a failing test for this experiment name
                setattr(
                    TestActiveExperiments,
                    f"test_{name}",
                    lambda self: self.fail("Experiment not found"),
                )
            else:
                add_experiment_test(name)

    else:
        # Iterate through all modules in the borealis_experiments directory
        for _, name, _ in pkgutil.iter_modules([experiment_path]):
            add_experiment_test(name)


def experiment_test_generator(module_name, **kwargs):
    """
    Generate a single test for a given experiment name. The test will try to run the experiment,
    and if any exceptions are thrown (i.e. the experiment is built incorrectly) the test will fail.

    :param module_name: Experiment module name (i.e. 'normalscan')
    :type module_name: str
    """

    def test(self):
        try:
            redirect_to_devnull(ehmain, experiment_name=module_name, **kwargs)
        except Exception as err:
            self.fail(err)

    return test


def run_tests(raw_args=None, buffer=True, print_results=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--site_id",
        required=False,
        default="sas",
        choices=["sas", "pgr", "inv", "rkn", "cly", "lab"],
        help="Site ID of site to test experiments as. Defaults to sas.",
    )
    parser.add_argument(
        "--experiments",
        required=False,
        nargs="+",
        default=None,
        help="Only run the experiments specified after this option. Experiments \
                            specified must exist within the top-level Borealis experiments directory.",
    )
    parser.add_argument(
        "--kwargs",
        required=False,
        nargs="+",
        default=list(),
        help="Keyword arguments to pass to the experiments. Note that kwargs are passed to all "
        "experiments specified.",
    )
    parser.add_argument(
        "--module",
        required=False,
        default="__main__",
        help="If calling from another python file, this should be set to "
        "'experiment_unittests' in order to properly work.",
    )
    args = parser.parse_args(raw_args)

    os.environ["RADAR_ID"] = args.site_id

    # Read in config.ini file for current site to make necessary directories
    path = (
        f'{os.environ["BOREALISPATH"]}/config/'
        f'{os.environ["RADAR_ID"]}/'
        f'{os.environ["RADAR_ID"]}_config.ini'
    )
    try:
        with open(path, "r") as data:
            raw_config = json.load(data)
    except OSError:
        errmsg = f"Cannot open config file at {path}"
        raise ValueError(errmsg)

    # These directories are required for ExperimentHandler to run
    data_directory = raw_config["data_directory"]
    log_directory = raw_config["log_handlers"]["logfile"]["directory"]
    hdw_path = raw_config["hdw_path"]
    hdw_dat_file = f'{hdw_path}/hdw.dat.{os.environ["RADAR_ID"]}'
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    if not os.path.exists(hdw_path):
        os.makedirs(hdw_path)
    if not os.path.exists(hdw_dat_file):
        open(hdw_dat_file, "w")

    experiments = args.experiments
    if experiments is None:  # Run all unit tests and experiment tests
        print("Running tests on all experiments")
        build_unit_tests()
        build_experiment_tests()
        argv = [sys.argv[0]]
    else:  # Only test specified experiments
        print(f"Running tests on experiments {experiments}")
        build_experiment_tests(experiments, args.kwargs)
        exp_tests = []
        for exp in experiments:
            # Check experiment exists
            if hasattr(TestActiveExperiments, f"test_{exp}"):
                # Create correct string to test the experiment with unittest
                exp_tests.append(f"TestActiveExperiments.test_{exp}")
            else:
                print(f"Could not find experiment {exp}. Exiting...")
                exit(1)
        argv = [parser.prog] + exp_tests
    if print_results:
        result = unittest.main(module=args.module, argv=argv, exit=False, buffer=buffer)
    else:
        result = redirect_to_devnull(
            unittest.main, module=args.module, argv=argv, exit=False, buffer=buffer
        )

    # Clean up the directories/files we created
    try:
        os.removedirs(data_directory)
    except OSError:  # If directories not empty, this will fail. That is fine.
        pass
    try:
        os.removedirs(log_directory)
    except OSError:  # If directories not empty, this will fail. That is fine.
        pass
    if os.path.getsize(hdw_dat_file) == 0:
        try:
            os.remove(hdw_dat_file)
        except OSError:  # Path is a directory
            os.removedirs(hdw_dat_file)
        else:  # File removed, now clean up directories
            try:
                os.removedirs(hdw_path)
            except OSError:  # If directories not empty, this will fail. That is fine.
                pass

    return result


if __name__ == "__main__":
    from utils import log_config

    log = log_config.log(console=False, logfile=False, aggregator=False)

    run_tests(sys.argv[1:])
