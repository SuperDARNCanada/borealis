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


class TestRealtime(unittest.TestCase):
    """
    unittest class to test the realtime module. All test methods must begin with the word 'test' to be run
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setUp(self):
        """
        Called before every test_* method
        """
        print("Method: ", self._testMethodName)


if __name__ == "__main__":
    unittest.main()
