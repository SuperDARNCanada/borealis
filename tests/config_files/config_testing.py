"""
config_testing.py
~~~~~~~~~~~~~~~~~

This script is designed to test that various configurations of N200 specification in a config file are properly
handled. The script modifies the directory ``$BOREALISPATH/config/test``, and will remove any existing files in
that directory. This script should be run to verify that the ``src.utils.options.Options`` class can properly
parse the N200 channel/antenna specifications of a config file.

If run without any arguments, this script will run through a series of unit tests for the ``Options`` class.
Alternatively, any number of paths to config files for testing can be passed to the script, and this will then
test all the config files to ensure that they have valid fields. The following invocation tests all config files
in the Borealis config file directory::

    python3 $BOREALISPATH/tests/config_files/config_testing.py $BOREALISPATH/config/*/*_config.ini

"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import unittest

# Need the path append to import within this file
BOREALISPATH = os.environ["BOREALISPATH"]
sys.path.append(f"{BOREALISPATH}/src")

from utils.options import Options


class MockOptions(Options):
    """
    This class overrides the `__post_init__` method of `Options` to ensure that valid RADAR_ID values are set, so Options
    can correctly read in `restrict.dat` and `hdw.dat` files.
    """

    def __post_init__(self):
        """Override the `__post_init__` method to hack the paths to files"""
        if not os.environ["BOREALISPATH"]:
            raise ValueError("BOREALISPATH env variable not set")

        os.environ["RADAR_ID"] = "test"  # For testing a new config file
        self.parse_config()  # Parse info from config file

        os.environ["RADAR_ID"] = (
            "sas"  # Use SAS to ensure that valid hdw and restrict files are loaded.
        )
        self.parse_hdw()
        self.parse_restrict()

        os.environ["RADAR_ID"] = (
            self.site_id
        )  # Match the value from the config file, to hack a test in verify_options
        self.verify_options()  # Check that all parsed values are valid


class TestConfigFile(unittest.TestCase):
    """Tests a config file to ensure the N200 specifications are correct."""

    def setUp(self):
        """Create a new directory `$BOREALISPATH/config/test/`."""
        if not os.path.exists(f'{os.environ["BOREALISPATH"]}/config/test'):
            os.mkdir(f'{os.environ["BOREALISPATH"]}/config/test')

    def tearDown(self):
        """Delete the `$BOREALISPATH/config/test directory and all contained files."""
        for f in os.listdir(f'{os.environ["BOREALISPATH"]}/config/test/'):
            os.remove(f'{os.environ["BOREALISPATH"]}/config/test/{f}')
        os.rmdir(f'{os.environ["BOREALISPATH"]}/config/test/')


class TestConfig(unittest.TestCase):
    """This class modifies fields of `base_config.ini` to ensure that config file parsing is handled correctly."""

    def setUp(self):
        """Create a new directory `$BOREALISPATH/config/test/`."""
        if not os.path.exists(f'{os.environ["BOREALISPATH"]}/config/test'):
            os.mkdir(f'{os.environ["BOREALISPATH"]}/config/test')
        for f in os.listdir(f'{os.environ["BOREALISPATH"]}/config/test/'):
            os.remove(f'{os.environ["BOREALISPATH"]}/config/test/{f}')

    def tearDown(self):
        """Delete the `$BOREALISPATH/config/test directory and all contained files."""
        for f in os.listdir(f'{os.environ["BOREALISPATH"]}/config/test/'):
            os.remove(f'{os.environ["BOREALISPATH"]}/config/test/{f}')
        os.rmdir(f'{os.environ["BOREALISPATH"]}/config/test/')

    def testBaseConfig(self):
        """Test the parameters of the base_config.ini file"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        options = MockOptions()
        self.assertEqual(options.n200_count, 16)
        self.assertEqual(len(options.n200_addrs), 16)
        self.assertEqual(len(options.rx_main_antennas), 16)
        self.assertEqual(len(options.rx_intf_antennas), 4)
        self.assertEqual(len(options.tx_main_antennas), 16)

    def testNoN200s(self):
        """No N200s specified in config file"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = []
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        options = MockOptions()
        self.assertEqual(options.n200_count, 0)
        self.assertEqual(len(options.n200_addrs), 0)
        self.assertEqual(len(options.rx_main_antennas), 0)
        self.assertEqual(len(options.rx_intf_antennas), 0)
        self.assertEqual(len(options.tx_main_antennas), 0)

    def testSingleN200NoConnections(self):
        """No N200s connected to antennas"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = [config["n200s"][0]]  # Only keep the first
        n200 = config["n200s"][0]
        n200["rx_channel_0"] = ""
        n200["rx_channel_1"] = ""
        n200["tx_channel_0"] = ""
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        options = MockOptions()
        self.assertEqual(options.n200_count, 0)
        self.assertEqual(len(options.n200_addrs), 0)
        self.assertEqual(len(options.rx_main_antennas), 0)
        self.assertEqual(len(options.rx_intf_antennas), 0)
        self.assertEqual(len(options.tx_main_antennas), 0)

    def testSingleN200Connected(self):
        """Single N200, connected to main antenna and intf antenna"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = [config["n200s"][0]]  # Only keep the first
        n200 = config["n200s"][0]
        n200["rx_channel_0"] = "m0"
        n200["rx_channel_1"] = "i0"
        n200["tx_channel_0"] = "m0"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        options = MockOptions()
        self.assertEqual(options.n200_count, 1)
        self.assertEqual(options.n200_addrs, [n200["addr"]])
        self.assertEqual(len(options.rx_main_antennas), 1)
        self.assertEqual(len(options.rx_intf_antennas), 1)
        self.assertEqual(len(options.tx_main_antennas), 1)
        self.assertEqual(options.rx_main_antennas, [0])
        self.assertEqual(options.rx_intf_antennas, [0])
        self.assertEqual(options.tx_main_antennas, [0])

    def testSingleN200BadMain(self):
        """Single N200, connected to invalid main antenna (main_antenna_count = 16)"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = [config["n200s"][0]]  # Only keep the first
        n200 = config["n200s"][0]
        n200["rx_channel_0"] = "m100"  # main_antenna_count = 16, this is too large
        n200["rx_channel_1"] = ""
        n200["tx_channel_0"] = ""
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "rx_main_antennas and main_antenna_count are not consistent"
        ):
            MockOptions()

    def testSingleN200BadIntf(self):
        """Single N200, connected to invalid intf antenna (intf_antenna_count = 4)"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = [config["n200s"][0]]  # Only keep the first
        n200 = config["n200s"][0]
        n200["rx_channel_0"] = ""
        n200["rx_channel_1"] = "i4"  # intf_antenna_count = 4, this is too large
        n200["tx_channel_0"] = ""
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "rx_intf_antennas and intf_antenna_count are not consistent"
        ):
            MockOptions()

    def testSingleN200BadTxMain(self):
        """Single N200, connected to invalid main antenna for TX (main_antenna_count = 16)"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = [config["n200s"][0]]  # Only keep the first
        n200 = config["n200s"][0]
        n200["rx_channel_0"] = ""
        n200["rx_channel_1"] = ""
        n200["tx_channel_0"] = "m100"  # main_antenna_count = 16, this is too large
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "tx_main_antennas and main_antenna_count are not consistent"
        ):
            MockOptions()

    def testSingleN200TxIntf(self):
        """Single N200, connected to an intf antenna for TX"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = [config["n200s"][0]]  # Only keep the first
        n200 = config["n200s"][0]
        n200["rx_channel_0"] = ""
        n200["rx_channel_1"] = ""
        n200["tx_channel_0"] = "i0"  # cannot connect to intf antenna
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "Cannot connect tx channel to interferometer array"
        ):
            MockOptions()

    def testSingleN200TxRxDifferentAntennas(self):
        """Single N200, connected to different antennas for TX and RX (which is allowed)"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = [config["n200s"][0]]  # Only keep the first
        n200 = config["n200s"][0]
        n200["rx_channel_0"] = "m0"
        n200["rx_channel_1"] = ""
        n200["tx_channel_0"] = "m1"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        options = MockOptions()
        self.assertEqual(options.n200_count, 1)
        self.assertEqual(options.n200_addrs, [n200["addr"]])
        self.assertEqual(len(options.rx_main_antennas), 1)
        self.assertEqual(len(options.rx_intf_antennas), 0)
        self.assertEqual(len(options.tx_main_antennas), 1)
        self.assertEqual(options.rx_main_antennas, [0])
        self.assertEqual(options.tx_main_antennas, [1])

    def testTwoN200sOutOfOrder(self):
        """Two N200s, the first connected to a higher-index antenna than the second"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = config["n200s"][:2]  # Only keep the first two
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "m1"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = "m1"
        n200_1 = config["n200s"][1]
        n200_1["rx_channel_0"] = "m0"
        n200_1["rx_channel_1"] = ""
        n200_1["tx_channel_0"] = "m0"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        options = MockOptions()
        self.assertEqual(options.n200_count, 2)
        self.assertEqual(options.n200_addrs, [n200_0["addr"], n200_1["addr"]])
        self.assertEqual(len(options.rx_main_antennas), 2)
        self.assertEqual(len(options.rx_intf_antennas), 0)
        self.assertEqual(len(options.tx_main_antennas), 2)
        self.assertEqual(options.rx_main_antennas, [0, 1])
        self.assertEqual(options.tx_main_antennas, [0, 1])

    def testTooManyIntfAntennas(self):
        """Connected to all intf antennas, plus one invalid intf antenna (intf_antenna_count = 4)"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)
        config["n200s"] = config["n200s"][:3]  # Only keep the first three
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "i0"
        n200_0["rx_channel_1"] = "i1"
        n200_0["tx_channel_0"] = ""
        n200_1 = config["n200s"][1]
        n200_1["rx_channel_0"] = "i2"
        n200_1["rx_channel_1"] = "i3"
        n200_1["tx_channel_0"] = ""
        n200_2 = config["n200s"][2]
        n200_2["rx_channel_0"] = "i4"
        n200_2["rx_channel_1"] = ""
        n200_2["tx_channel_0"] = ""
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "rx_intf_antennas and intf_antenna_count are not consistent"
        ):
            MockOptions()

    def testTooManyMainAntennas(self):
        """Connected to all main antennas, plus one invalid main antenna (main_antenna_count = 16)"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        n200_16 = config["n200s"][16]
        n200_16["rx_channel_0"] = "m16"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "rx_main_antennas and main_antenna_count are not consistent"
        ):
            MockOptions()

    def testDuplicateMainAntennaRx(self):
        """Connected to a main antenna twice for RX"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200 = config["n200s"][0]
        n200["rx_channel_0"] = "m0"
        n200["rx_channel_1"] = "m0"
        n200["tx_channel_0"] = ""
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "rx_main_antennas has duplicate values"
        ):
            MockOptions()

    def testDuplicateIntfAntennaRx(self):
        """Connected to an intf antenna twice for RX"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200 = config["n200s"][0]
        n200["rx_channel_0"] = "i0"
        n200["rx_channel_1"] = "i0"
        n200["tx_channel_0"] = ""
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "rx_intf_antennas has duplicate values"
        ):
            MockOptions()

    def testDuplicateMainAntennaTx(self):
        """Connected to a main antenna twice for TX"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:2]  # Only keep the first two n200s
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "m0"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = "m0"
        n200_1 = config["n200s"][1]
        n200_1["rx_channel_0"] = "m1"
        n200_1["rx_channel_1"] = ""
        n200_1["tx_channel_0"] = "m0"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "tx_main_antennas has duplicate values"
        ):
            MockOptions()

    def testInvalidArraySpecifierRx(self):
        """Invalid array specifier for RX channel"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "a0"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = ""
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "channel must start with either 'm' or 'i'"
        ):
            MockOptions()

    def testInvalidArraySpecifierTx(self):
        """Invalid array specifier for TX channel"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "m0"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = "z0"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "channel must start with either 'm' or 'i'"
        ):
            MockOptions()

    def testInvalidMainAntennaSpecifierRx(self):
        """Invalid main antenna index specifier for RX channel"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "mo"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = ""
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(ValueError, "channel\[1:\] must be an integer"):
            MockOptions()

    def testInvalidIntfAntennaSpecifierRx(self):
        """Invalid intf antenna index specifier for RX channel"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "io"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = ""
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(ValueError, "channel\[1:\] must be an integer"):
            MockOptions()

    def testInvalidMainAntennaSpecifierTx(self):
        """Invalid main antenna index specifier for TX channel"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "m0"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = "mo"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(ValueError, "channel\[1:\] must be an integer"):
            MockOptions()

    def testDuplicateN200(self):
        """Two N200s have the same IP address"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "m0"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = ""
        n200_1 = copy.deepcopy(n200_0)
        n200_1["rx_channel_0"] = "m1"
        config["n200s"].append(n200_1)
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        with self.assertRaisesRegex(
            ValueError, "Two or more n200s have identical IP addresses"
        ):
            MockOptions()

    def testUnconnectedMainAntennaInExperimentSlice(self):
        """ExperimentSlice tries to use a main antenna for RX that is not connected to an N200"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "m0"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = "m0"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        # Do this to avoid logging from ExperimentSlice
        from utils import log_config

        log = log_config.log(console=False, logfile=False, aggregator=False)

        # Set up a slice to test with ExperimentSlice
        from borealis_experiments import superdarn_common_fields as scf

        slice_dict = {
            "slice_id": 0,  # arbitrary
            "cpid": 0,  # arbitrary
            "tx_bandwidth": 5e6,  # default
            "rx_bandwidth": 5e6,  # default
            "output_rx_rate": 10e3 / 3,  # default
            "transition_bandwidth": 750e3,  # default
            "pulse_sequence": scf.SEQUENCE_7P,  # default
            "tau_spacing": scf.TAU_SPACING_7P,  # default
            "pulse_len": scf.PULSE_LEN_45KM,  # default
            "num_ranges": 75,  # default
            "first_range": scf.STD_FIRST_RANGE,  # default
            "intt": scf.INTT_7P,  # default
            "beam_angle": scf.STD_16_BEAM_ANGLE,  # default
            "rx_beam_order": [0],  # smallest base case
            "tx_beam_order": [0],  # smallest base case
            "rx_main_antennas": [
                0,
                1,
            ],  # This should fail since antenna 1 not specified in config above
            "freq": scf.COMMON_MODE_FREQ_1,  # default
        }

        # Test that the ExperimentSlice object will raise an exception
        from experiment_prototype import experiment_slice

        experiment_slice.options = MockOptions()
        with self.assertRaisesRegex(
            ValueError, "RX main antenna 1 not specified in config file"
        ):
            experiment_slice.ExperimentSlice(**slice_dict)

    def testUnconnectedIntfAntennaInExperimentSlice(self):
        """ExperimentSlice tries to use an intf antenna for RX that is not connected to an N200"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "m0"
        n200_0["rx_channel_1"] = "i0"
        n200_0["tx_channel_0"] = "m0"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        # Do this to avoid logging from ExperimentSlice
        from utils import log_config

        log = log_config.log(console=False, logfile=False, aggregator=False)

        # Set up a slice to test with ExperimentSlice
        from borealis_experiments import superdarn_common_fields as scf

        slice_dict = {
            "slice_id": 0,  # arbitrary
            "cpid": 0,  # arbitrary
            "tx_bandwidth": 5e6,  # default
            "rx_bandwidth": 5e6,  # default
            "output_rx_rate": 10e3 / 3,  # default
            "transition_bandwidth": 750e3,  # default
            "pulse_sequence": scf.SEQUENCE_7P,  # default
            "tau_spacing": scf.TAU_SPACING_7P,  # default
            "pulse_len": scf.PULSE_LEN_45KM,  # default
            "num_ranges": 75,  # default
            "first_range": scf.STD_FIRST_RANGE,  # default
            "intt": scf.INTT_7P,  # default
            "beam_angle": scf.STD_16_BEAM_ANGLE,  # default
            "rx_beam_order": [0],  # smallest base case
            "tx_beam_order": [0],  # smallest base case
            "rx_intf_antennas": [
                0,
                1,
            ],  # This should fail since antenna 1 not specified in config above
            "freq": scf.COMMON_MODE_FREQ_1,  # default
        }

        # Test that the ExperimentSlice object will raise an exception
        from experiment_prototype import experiment_slice

        experiment_slice.options = MockOptions()
        with self.assertRaisesRegex(
            ValueError, "RX intf antenna 1 not specified in config file"
        ):
            experiment_slice.ExperimentSlice(**slice_dict)

    def testUnconnectedTxAntennaInExperimentSlice(self):
        """ExperimentSlice tries to use a main antenna for TX that is not connected to an N200"""
        with open(Path(__file__).with_name("base_config.ini"), "r") as f:
            config = json.load(f)

        config["n200s"] = config["n200s"][:1]  # Only keep the first n200
        n200_0 = config["n200s"][0]
        n200_0["rx_channel_0"] = "m0"
        n200_0["rx_channel_1"] = ""
        n200_0["tx_channel_0"] = "m0"
        with open(
            f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
        ) as f:
            json.dump(config, f)

        # Do this to avoid logging from ExperimentSlice
        from utils import log_config

        log = log_config.log(console=False, logfile=False, aggregator=False)

        # Set up a slice to test with ExperimentSlice
        from borealis_experiments import superdarn_common_fields as scf

        slice_dict = {
            "slice_id": 0,  # arbitrary
            "cpid": 0,  # arbitrary
            "tx_bandwidth": 5e6,  # default
            "rx_bandwidth": 5e6,  # default
            "output_rx_rate": 10e3 / 3,  # default
            "transition_bandwidth": 750e3,  # default
            "pulse_sequence": scf.SEQUENCE_7P,  # default
            "tau_spacing": scf.TAU_SPACING_7P,  # default
            "pulse_len": scf.PULSE_LEN_45KM,  # default
            "num_ranges": 75,  # default
            "first_range": scf.STD_FIRST_RANGE,  # default
            "intt": scf.INTT_7P,  # default
            "beam_angle": scf.STD_16_BEAM_ANGLE,  # default
            "rx_beam_order": [0],  # smallest base case
            "tx_beam_order": [0],  # smallest base case
            "tx_antennas": [
                0,
                1,
            ],  # This should fail since antenna 1 not specified in config above
            "freq": scf.COMMON_MODE_FREQ_1,  # default
        }

        # Test that the ExperimentSlice object will raise an exception
        from experiment_prototype import experiment_slice

        experiment_slice.options = MockOptions()
        with self.assertRaisesRegex(
            ValueError, "TX antenna 1 not specified in config file"
        ):
            experiment_slice.ExperimentSlice(**slice_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("configs", help="List of config files to check", nargs="*")
    args = parser.parse_args()

    if len(args.configs) == 0:
        unittest.main()  # No files specified, run the test
    else:

        def test_generator(config):
            def test_method(self):
                with open(config, "r") as f:
                    config_data = json.load(f)
                if not os.path.exists(f'{os.environ["BOREALISPATH"]}/config/test'):
                    os.mkdir(f'{os.environ["BOREALISPATH"]}/config/test')
                with open(
                    f'{os.environ["BOREALISPATH"]}/config/test/test_config.ini', "w"
                ) as f:
                    json.dump(config_data, f)
                try:
                    MockOptions()
                except Exception as err:
                    self.fail(err)

            return test_method

        for config_file in args.configs:
            # Get just the file name, stripping off the extension and replacing periods with underscores
            basename = "_".join(os.path.basename(config_file).split(".")[:-1])
            test = test_generator(config_file)
            setattr(TestConfigFile, f"test_{basename}", test)

        unittest.main(defaultTest="TestConfigFile", argv=[sys.argv[0]])
