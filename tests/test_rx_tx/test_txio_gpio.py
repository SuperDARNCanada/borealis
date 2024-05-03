import uhd
import sys
import time
from itertools import chain


class gpio_testing(object):
    """
    Class for handling gpio tests
    """

    def __init__(self, address, bank):
        """
        Constructor for gpio testing class
        :param address: Device address for linked USRP
        """

        # Create USRP object
        self._bank = bank
        self.PULSE_TIME = 1
        self._usrp = uhd.usrp.MultiUSRP(address)
        print("USRP has", self._usrp.get_num_mboards(), "motherboards on board")
        pinout = input("Set pinout: 'user' for user or enter for default ")
        # user input pinout
        self.signal_pins = {}
        self.input_pins = {}
        if pinout == "user":
            self.signal_pins["RXO"] = int(
                input("Specify pin connected to the RXO signal ")
            )
            self.signal_pins["TXO"] = int(
                input("Specify pin connected to the TXO signal ")
            )
            self.signal_pins["TR"] = int(
                input("Specify pin connected to the T/R signal ")
            )
            self.signal_pins["IDLE"] = int(
                input("Specify pin connected to the IDLE signal ")
            )
            self.signal_pins["TM"] = int(
                input("Specify pin connected to the TEST_MODE signal ")
            )
            self.input_pins["LP"] = int(
                input("Specify pin connected to the LOW_PWR input ")
            )
            self.input_pins["AGC"] = int(
                input("Specify pin connected to the AGC_STATUS input ")
            )

            print("Would you like to set up alternate pins for these signals?")
            alternate = input("Enter Y for yes, or enter for no ")
            if alternate == "Y":
                self.signal_pins["RXO_alt"] = int(
                    input("Specify alternate pin connected to the RXO signal ")
                )
                self.signal_pins["TXO_alt"] = int(
                    input("Specify alternate pin connected to the TXO signal ")
                )
                self.signal_pins["TR_alt"] = int(
                    input("Specify alternate pin connected to the T/R signal ")
                )
                self.signal_pins["IDLE_alt"] = int(
                    input("Specify alternate pin connected to the IDLE signal ")
                )
                self.signal_pins["TM_alt"] = int(
                    input("Specify alternate pin connected to the TEST_MODE signal ")
                )
                self.input_pins["LP_alt"] = int(
                    input("Specify alternate pin connected to the LOW_PWR input ")
                )
                self.input_pins["AGC_alt"] = int(
                    input("Specify alternate pin connected to the AGC_STATUS input ")
                )
        # default pins
        else:
            self.signal_pins = {
                "RXO": 1,  # "RXO_alt": 2,
                "TXO": 3,  # "TXO_alt": 4,
                "TR": 5,  # "TR_alt":  6,
                "IDLE": 7,  # "IDLE_alt":8,
                "TM": 13,
                "TM_alt": 14,
            }

            self.input_pins = {"LP": 9, "AGC": 11}  # "LP_alt":  10,  # ,"AGC_alt": 12}

        # Make sure all pins are unique in the dictionaries. See geeksforgeeks.org
        rev_dict = {}
        for key, value in self.signal_pins.items():
            rev_dict.setdefault(value, set()).add(key)
        for key, value in self.input_pins.items():
            rev_dict.setdefault(value, set()).add(key)

        result = set(
            chain.from_iterable(
                values for key, values in rev_dict.items() if len(values) > 1
            )
        )

        if len(result) > 0:
            print(
                "Warning: You've selected multiple signals to be routed to one pin. Please double check your entries! Exiting"
            )
            sys.exit(1)

    def set_all_low(self):
        """
        Set all GPIO pins on gpio bank low
        :param bank: String identifying the GPIO bank
        """
        # for i in arange(self._usrp.get_num_mboards()):
        print("Setting all pins on", self._bank, "as low outputs.")
        self._usrp.set_gpio_attr(self._bank, "CTRL", 0x0000, 0b11111111111111111)
        self._usrp.set_gpio_attr(self._bank, "DDR", 0xFFFF, 0b11111111111111111)
        self._usrp.set_gpio_attr(self._bank, "OUT", 0x0000, 0b11111111111111111)

    def set_pulse_time(self, pt):
        """
        Sets the length of the pulse for testing
        :param time: The pulse length in seconds
        """
        self.PULSE_TIME = pt

    def get_pin_mask(self, pin):
        """
        Converts a pin number to a pin mask usable by usrp methods
        :param pin: GPIO pin number
        """
        mask = 2**pin
        return mask

    def query_user(self):
        """
        Gets user input for continuing or aborting testing
        """
        user = input("\nContinue? [y/n] ")
        if user == "y":
            print("Going to next test...\n")
            return "y"
        elif user == "n":
            print("Ending tests...\n")
            return "n"
        else:
            print("Invalid input\n")
            return self.query_user()

    def run_single_signals_test(self):
        """
        Handles the single ended output signals test
        """
        # Select test
        # signals = [("RXO", self._rxo_pin), ("TXO", self._txo_pin), ("T/R", self._tr_pin), ("IDLE", self._idle_pin), ("TEST_MODE", self._tm_pin)]
        # signals2 = [("RXO2", self._rxo_pin2), ("TXO2", self._txo_pin2), ("T/R2", self._tr_pin2), ("IDLE2", self._idle_pin2), ("TEST_MODE2", self._tm_pin2)]

        for (
            signal,
            pin,
        ) in self.signal_pins.items():
            mask = self.get_pin_mask(pin)
            self.set_all_low()
            print("Testing", signal)
            try:
                # run current test
                while True:
                    self._usrp.set_gpio_attr(self._bank, "OUT", 0xFFFF, mask)
                    # print(self._usrp.get_gpio_attr(self._bank, "READBACK"))
                    time.sleep(self.PULSE_TIME)
                    self._usrp.set_gpio_attr(self._bank, "OUT", 0x0000, mask)
                    # print(self._usrp.get_gpio_attr(self._bank, "READBACK"))
                    time.sleep(self.PULSE_TIME)
            except KeyboardInterrupt:
                # ask user whether they want to continue the test sequence
                user = self.query_user()
                if user == "y":
                    continue
                else:
                    return

    def run_differential_signal_test(self):
        """
        Handles the loop-back differential signals test.
        This test requires a loopback cable to be plugged into the back of the N200 DE9 port.
        The cable needs to connect pins in the following way:

            AGC+ connects to TR+
            TM+ connects to  LP+
            AGC- connects to TR-
            TM- connects to LP-

        AGC+ --->   6   1   <--- AGC-
             |                  |
        TR+  --->   7   2   <--- TR-
        TM+  --->   8   3   <--- TM-
             |                  |
        LP+  --->   9   4   <--- LP-
                        5   [NC]

        This directly connects the output differential signals (TM and TR) to
        the input differential signals (LP and AGC). When the outputs are set high,
        the inputs should read high, and visa versa.
        """
        lp_mask = self.get_pin_mask(self.input_pins["LP"])
        agc_mask = self.get_pin_mask(self.input_pins["AGC"])

        def set_lp_agc_inputs():
            """
            sets pins connected to low power and AGC status signals as inputs
            """
            self._usrp.set_gpio_attr(self._bank, "DDR", 0x0000, lp_mask)
            self._usrp.set_gpio_attr(self._bank, "DDR", 0x0000, agc_mask)

        for pin in [self.signal_pins["TR"], self.signal_pins["TM"]]:
            # configure GPIO for testing
            self.set_all_low()
            print(
                "Initial Status:", hex(self._usrp.get_gpio_attr(self._bank, "READBACK"))
            )
            set_lp_agc_inputs()
            print(
                "Status after setting inputs:",
                hex(self._usrp.get_gpio_attr(self._bank, "READBACK")),
            )
            time.sleep(1)
            mask = self.get_pin_mask(pin)
            try:
                # run current test
                # State name of current test
                if pin == self.signal_pins["TR"]:
                    print("Testing TR, AGC Loopback")
                    read_mask = agc_mask
                else:
                    print("Testing TM, LP Loopback")
                    read_mask = lp_mask

                # make sure that we are starting in an all pins 0 state
                while True:
                    self._usrp.set_gpio_attr(self._bank, "OUT", 0xFFFF, mask)
                    print("State with output pin high:")
                    print(hex(self._usrp.get_gpio_attr(self._bank, "READBACK")))
                    # Alt: print(self._usrp.get_gpio_attr(self._bank, "READBACK") & read_mask)
                    time.sleep(self.PULSE_TIME)
                    self._usrp.set_gpio_attr(self._bank, "OUT", 0x0000, mask)
                    print("State with output pin low:")
                    print(hex(self._usrp.get_gpio_attr(self._bank, "READBACK")))
                    # Alt: print(self._usrp.get_gpio_attr(self._bank, "READBACK") & read_mask)
                    time.sleep(self.PULSE_TIME)
            except KeyboardInterrupt:
                # ask user whether they want to continue the test sequence
                user = self.query_user()
                if user == "y":
                    continue
                else:
                    return


if __name__ == "__main__":
    # Get command line arguments
    # script should be called as: test_txio_gpio.py <device_address>
    argv = sys.argv
    ADDR = argv[1]

    tests = gpio_testing(
        "num_recv_frames=512,num_send_frames=256,send_buff_size=2304000,addr=" + ADDR,
        "RXA",
    )

    print("Beginning tests")
    tests.run_single_signals_test()
    tests.run_differential_signal_test()
    print("Done!")
