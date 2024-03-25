"""
Test module for the realtime code.
Run via: 'python3 test_realtime.py'
:copyright: 2022 SuperDARN Canada
:author: Kevin Krieger
"""
import unittest
import os
import sys

if not os.environ['BOREALISPATH']:
    BOREALISPATH = f"{os.environ['HOME']}/PycharmProjects/borealis/"
else:
    BOREALISPATH = os.environ['BOREALISPATH']

sys.path.append(BOREALISPATH)

from src import realtime
from src.utils.options import realtime_options as rto


class TestRealtimeOptions(unittest.TestCase):
    """
    unittest class to test the realtime options module.
    All test methods must begin with the word 'test' to be run
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

    def test_no_borealispath(self):
        """
        Test no environment variable BOREALISPATH, should raise ValueError
        """
        with self.assertRaises(ValueError):
            pass

    def test_no_radar_code(self):
        """
        Test no environment variable RADAR_CODE, should raise ValueError
        """
        with self.assertRaises(ValueError):
            pass

    def test_bad_config_path(self):
        """
        Test not being able to open the config file path for read-only
        Should raise IOError
        """
        with self.assertRaises(IOError):
            pass

    def test_no_rt_to_dw_identity(self):
        """
        Test no 'rt_to_dw_identity' in the json of the config file
        Should raise KeyError
        """
        with self.assertRaises(KeyError):
            pass

    def test_no_dw_to_rt_identity(self):
        """
        Test no 'dw_to_rt_identity' in the json of the config file
        Should raise KeyError
        """
        with self.assertRaises(KeyError):
            pass

    def test_no_realtime_address(self):
        """
        Test no 'realtime_address' in the json of the config file
        Should raise KeyError
        """
        with self.assertRaises(KeyError):
            pass

    def test_no_router_address(self):
        """
        Test no 'router_address' in the json of the config file
        Should raise KeyError
        """
        with self.assertRaises(KeyError):
            pass


class TestRealtime(unittest.TestCase):
    """
    unittest class to test the realtime module.
    All test methods must begin with the word 'test' to be run
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)

# read_and_convert_file_to_fitacf tests:
    def test_bad_filename(self):
        """
        Test passing a bad filename to the rt module's read_and_convert_file_to_fitacf
        It expects format YYYYMMDD.HHMM.ss.uuuuuu.[rad].[slice_id].rawacf.hdf5
        Should return None
        """
        pass


if __name__ == "__main__":
    unittest.main()