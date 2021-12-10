"""
Copyright SuperDARN Canada 2021
Original Auth: Keith Kotyk
Modified: Dec. 9, 2021 by Remington Rohel

This file was adapted into Python in December 2021 by Remington Rohel, based
off of the file usrp_driver.cpp written by Keith Kotyk.
"""
import time
import numpy as np
import uhd

import utils.shared_macros.shared_macros as sm
from utils.driver_options.driveroptions import DriverOptions

pprint = sm.MODULE_PRINT("n200 driver", "yellow")


class USRP(object):
    """ Contains an abstract wrapper for the USRP object. """
    def __init__(self, driver_options: DriverOptions, tx_rate: float, rx_rate: float):
        """Creates the multiUSRP abstraction with the options from the config file.

        :param  driver_options:  The driver options parsed from config
        :param  tx_rate:         The transmit rate in Sps  (samples per second, Hz).
        :param  rx_rate:         The receive rate in Sps (samples per second, Hz).
        """
        # A string representing what GPIO bank to use on the USRPs for active high signals.
        self._gpio_bank_high = driver_options.gpio_bank_high

        # A string representing what GPIO bank to use on the USRPs for active low signals.
        self._gpio_bank_low = driver_options.gpio_bank_low

        # The bitmask to use for the scope sync GPIO.
        self._scope_sync_mask = None

        # The bitmask to use for the attenuator GPIO.
        self._atten_mask = None

        # The bitmask to use for the TR GPIO.
        self._tr_mask = None

        # Bitmask used for the rx only ATR.
        self._atr_rx = driver_options.atr_rx

        # Bitmask used for the tx only ATR.
        self._atr_tx = driver_options.atr_tx

        # Bitmask used for the full duplex ATR.
        self._atr_xx = driver_options.atr_xx

        # Bitmask used for the idle ATR.
        self._atr_0x = driver_options.atr_0x

        # Bitmask used for the AGC signal.
        self._agc_st = driver_options.agc_st

        # Bitmask used for the test mode signal.
        self._test_mode = driver_options.test_mode

        # Bitmask used for the lo pwr signal.
        self._lo_pwr = driver_options.lo_pwr

        # The tx rate in Hz.
        self._tx_rate = tx_rate

        # The rx rate in Hz.
        self._rx_rate = rx_rate

        # Reference to a new multi-USRP device.
        self._usrp = uhd.usrp.MultiUSRP(driver_options.devices)

        # Reference to a new multi-USRP-clock device.
        self._gps_clock = None

        # Reference to the multi-USRP transmit stream.
        self._tx_stream = None

        # Reference to the multi-USRP receive stream.
        self._rx_stream = None

        # Set up the multiUSRP object with the driver options
        self.set_usrp_clock_source(driver_options.ref)
        self.set_tx_subdev(driver_options.tx_subdev)
        self.set_main_rx_subdev(driver_options.main_rx_subdev)
        self.set_intf_rx_subdev(driver_options.intf_rx_subdev, driver_options.intf_antenna_count)
        self.set_time_source(driver_options.pps, driver_options.clk_addr)
        self.check_ref_locked()
        self._set_atr_gpios()
        self._set_output_gpios()
        self._set_input_gpios()

        self.set_tx_rate(driver_options.transmit_channels)
        self.set_rx_rate(driver_options.receive_channels)

        self.create_usrp_tx_stream(driver_options.cpu, driver_options.otw, driver_options.transmit_channels)
        self.create_usrp_rx_stream(driver_options.cpu, driver_options.otw, driver_options.receive_channels)

        # Number of USRP devices in the multi-USRP object
        self._num_mboards = self._usrp.get_num_mboards()

    def set_usrp_clock_source(self, source: str):
        """Sets the USRP clock source.

        :param  source: A string for a valid USRP clock source.
        :type   source: str
        """
        self._usrp.set_clock_source(source)

    def set_tx_subdev(self, tx_subdev: str):
        """Sets the USRP transmit subdev specification.

        :param  tx_subdev:  A string for a valid transmit subdev.
        :type   tx_subdev:  str
        """
        self._usrp.set_tx_subdev_spec(tx_subdev)

    def set_tx_rate(self, channels: list) -> float:
        """Sets the transmit sample rate.

        :param  channels: A list of USRP channels to transmit on.
        :type   channels: list(int)

        :return tx_rate_actual: The actual set tx rate.
        """
        if self._tx_rate <= 0.0:
            # TODO: Handle error
            return -1.0

        self._usrp.set_tx_rate(self._tx_rate)

        # Check that the rates were set correctly and identically
        rate_1 = self._usrp.get_tx_rate(channels[0])
        for channel in channels:
            actual_rate = self._usrp.get_tx_rate(channel)

            if actual_rate != rate_1:
                # TODO: Error
                return -2.0

            if actual_rate != self._tx_rate:
                # TODO: Error - fail because experiment will assume and we will transmit different than expected.
                return -3.0

        return rate_1

    def get_tx_rate(self, channel: int) -> float:
        """Gets the USRP transmit sample rate.

        :return:    The transmit sample rate is Sps.
        """
        return self._usrp.get_tx_rate(channel)

    def set_tx_center_freq(self, freq: float, channels: list, tune_delay: uhd.types.TimeSpec) -> float:
        """Sets the transmit center frequency.

        The USRP uses a numbered channel mapping system to identify which data streams come from which
        USRP and its daughterboard frontends. With the daughtboard frontends connected to the
        transmitters, controlling what USRP channels are selected will control what antennas are
        used and what order they are in. To synchronize tuning of all boxes, timed commands are used so
        that everything is done at once.

        :param  freq:       The center frequency in Hz.
        :type   freq:       float
        :param  channels:   A list of which USRP channels to set a center frequency.
        :type   channels:   list(int)
        :param  tune_delay: The amount of time in the future to tune the devices.
        :type   tune_delay: uhd.types.TimeSpec

        :return:    The actual set tx center frequency for the USRPs.
        """
        tune_request = uhd.types.TuneRequest(freq)

        self.set_command_time(self.get_current_usrp_time() + tune_delay)
        for channel in channels:
            self._usrp.set_tx_freq(tune_request, channel)    # TODO: Test tune request.

        self.clear_command_time()

        duration = tune_delay.get_real_secs()
        time.sleep(duration)

        # check for varying USRPs
        freq_1 = self._usrp.get_tx_freq(channels[0])
        for channel in channels:
            actual_freq = self._usrp.get_tx_freq(channel)

            if actual_freq != freq:
                # TODO: throw error.
                pass

            elif actual_freq != freq_1:
                # TODO: throw error.
                pass

        return freq_1

    def get_tx_center_freq(self, channel: int) -> float:
        """Gets the transmit center frequency for the specified channel.

        :param  channel:    The USRP channel to probe.
        :type   channel:    int

        :return:    The actual center frequency that the USRP channel is tuned to.
        """
        return self._usrp.get_tx_freq(channel)

    def set_main_rx_subdev(self, main_subdev: str):
        """Sets the receive subdev for the main array antennas.

        Will set all boxes to receive from first USRP channel of all mboards for main array.

        :param  main_subdev:    A valid receive subdev.
        :type   main_subdev:    str
        """
        self._usrp.set_rx_subdev_spec(main_subdev)

# REVIEW #43 It would be best if we could have in the config file a map of direct antenna to USRP box/subdev/channel so you can change the interferometer to a different set of boxes for example. Also if a rx daughterboard stopped working and you needed to move both main and int to a totally different box for receive, then you could do that. This would be useful for both rx and tx channels.
# REPLY OKAY, but maybe we should leave it for now. That's easier said than done.

    def set_intf_rx_subdev(self, intf_subdev: str, intf_antenna_count: int):
        """Sets the interferometer receive subdev.

        Override the subdev spec of the first mboards to receive on a second channel for the interferometer.

        :param  intf_subdev:        A valid receive subdev.
        :type   intf_subdev:        str
        :param  intf_antenna_count: Interferometer antenna count.
        :type   intf_antenna_count: int
        """
        for i in range(intf_antenna_count):
            self._usrp.set_rx_subdev_spec(intf_subdev, i)

    def set_rx_rate(self, rx_channels: list) -> float:
        """Sets the receive sample rate.

        :param  rx_channels:    The USRP channels to receive on.
        :type   rx_channels:    list(int)

        :return:    The actual rx rate set.
        """
        if self._rx_rate <= 0.0:
            # TODO: Handle error
            return -1.0

        self._usrp.set_rx_rate(self._rx_rate)

        rate_1 = self._usrp.get_rx_rate(rx_channels[0])
        for channel in rx_channels:
            actual_rate = self._usrp.get_rx_rate(channel)

            if actual_rate != rate_1:
                # TODO: Throw error
                return -2.0

            if actual_rate != self._rx_rate:
                # TODO: Throw error. Fail because we will be receiving unknown frequencies.
                return -3.0

        return rate_1

    def get_rx_rate(self, channel: int) -> float:
        """Gets the USRP receive sample rate for the specified channel.

        :param  channel:    Channel to check the rate of.
        :type   channel:    int

        :return:    The receive sample rate in Samples per second.
        """
        return self._usrp.get_rx_rate(channel)

    def set_rx_center_freq(self, freq: float, channels: list, tune_delay: uhd.types.TimeSpec) -> float:
        """Sets the receive center frequency.

        The USRP uses a numbered channel mapping system to identify which data streams come from which
        USRP and its daughterboard frontends. With the daughtboard frontends connected to the
        transmitters, controlling what USRP channels are selected will control what antennas are
        used and what order they are in. To simplify data processing, all antenna mapped channels are
        used. To synchronize tuning of all boxes, timed commands are used so that everything is done at
        once.

        :param  freq:       The frequency in Hz.
        :type   freq:       float
        :param  channels:   A list of which USRP channels to set a center frequency.
        :type   channels:   list(int)
        :param  tune_delay: The amount of time in future to tune the devices.
        :type   tune_delay: uhd.type.TimeSpec

        :return:     The actual center frequency that the USRPs are tuned to.
        """
        tune_request = uhd.types.TuneRequest(freq)

        self.set_command_time(self.get_current_usrp_time() + tune_delay)

        for channel in channels:
            self._usrp.set_rx_freq(tune_request, channel)

        self.clear_command_time()

        duration = tune_delay.get_real_secs()
        time.sleep(duration)

        # Check for varying USRPs
        freq_1 = self._usrp.get_rx_freq(channels[0])
        for channel in channels:
            actual_freq = self._usrp.get_rx_freq(channel)

            if actual_freq != freq:
                # TODO: Throw error
                pass

            if actual_freq != freq_1:
                # TODO: Throw error
                pass

        return freq_1

    def get_rx_center_freq(self, channel: int) -> float:
        """Gets the receive center frequency.

        :param  channel:    USRP receive channel.
        :type   channel:    int

        :return:    The actual center frequency that the specified USRP channel is tuned to.
        """
        return self._usrp.get_rx_freq(channel)

    def set_time_source(self, source: str, clk_addr: str):
        """Sets the USRP time source.

        Uses the method Ettus suggests for setting time on the x300.
        https://files.ettus.com/manual/page_gpsdo_x3x0.html
        Falls back to Juha Vierinen's method of latching to the current time by making sure the clock
        time is in a stable place past the second if no gps is available.
        The USRP is then set to this time.

        :param  source:     Time source the USRP will use.
        :type   source:     str
        :param  clk_addr:   IP address of the octoclock for gps timing.
        :type   clk_addr:   str
        """
        # TODO: Figure out how to do this well in Python
        # tt = time.perf_counter_ns()
        # tt_sc = duration_cast(tt.time_since_epoch())
        # while tt_sc.count() - floor(tt_sc.count()) < 0.2 or tt_sc.count() - math.floor(tt_sc.count()) > 0.3:
        #     tt = high_resolution_clock.now()
        #     tt_sc = duration_cast(tt.time_since_epoch())
        #     usleep(10000)

        if source == 'external':
            # TODO: Find out where the MultiUSRPClock class is
            # self._gps_clock = uhd.usrp_clock.MultiUSRPClock(clk_addr)

            # Make sure clock configuration is correct
            if self._gps_clock.get_sensor("gps_detected").value == "false":
                raise RuntimeError("No GPSDO detected on clock.")

            if self._gps_clock.get_sensor("using_ref").value != "internal":
                msg = "Clock must be using an internal reference. Using {}" \
                      "".format(self._gps_clock.get_sensor("using_ref").value)
                raise RuntimeError(msg)

            while not self._gps_clock.get_sensor("gps_locked").to_bool():
                time.sleep(2)
                pprint("Waiting for gps lock...")

            self._usrp.set_time_source(source)

            def wait_for_update():
                last_pulse = self._usrp.get_time_last_pps()
                next_pulse = self._usrp.get_time_last_pps()
                while next_pulse == last_pulse:
                    time.sleep(0.05)
                    last_pulse = next_pulse
                    next_pulse = self._usrp.get_time_last_pps()

                time.sleep(0.2)

            wait_for_update()

            self._usrp.set_time_next_pps(self._gps_clock.get_time() + 1)

            wait_for_update()

            clock_time = self._gps_clock.get_time()

            for board in range(self._num_mboards):
                usrp_time = self._usrp.get_time_last_pps(board)
                time_diff = clock_time - usrp_time

                pprint("Time difference between USRPs and gps clock for board {}: {}"
                       "".format(board, time_diff.get_real_secs()))

        else:
            # TODO: throw error
            # self._usrp.set_time_now(math.ceil(tt_sc.count()))
            pass

    def check_ref_locked(self):
        """Makes a quick check that each USRP is locked to a reference frequency."""
        for board in range(self._num_mboards):
            sensor_names = self._usrp.get_mboard_sensor_names(board)

            if "ref_locked" not in sensor_names:
                ref_locked = self._usrp.get_mboard_sensor("ref_locked", board)
                # TODO: Something like this
                #   UHD_ASSERT_THROW(ref_locked.to_bool())
            else:
                # TODO: Get an else statement and do something if there's no ref_locked sensor found.
                pass

    def set_command_time(self, cmd_time: uhd.types.TimeSpec):
        """Sets the command time.

        :param  cmd_time:   The command time to run a timed command.
        :type   cmd_time:   uhd.types.TimeSpec
        """
        self._usrp.set_command_time(cmd_time)

    def clear_command_time(self):
        """Clears any timed USRP commands."""
        self._usrp.clear_command_time()

    def _set_atr_gpios(self):
        """Sets the USRP automatic transmit/receive states on GPIO for the given daughtercard bank."""
        output_pins = 0
        output_pins |= self._atr_xx | self._atr_rx | self._atr_tx | self._atr_0x

        for board in range(self._num_mboards):
            self._usrp.set_gpio_attr(self._gpio_bank_high, "CTRL", 0xFFFF, output_pins, board)
            self._usrp.set_gpio_attr(self._gpio_bank_high, "DDR", 0xFFFF, output_pins, board)

            self._usrp.set_gpio_attr(self._gpio_bank_low, "CTRL", 0xFFFF, output_pins, board)
            self._usrp.set_gpio_attr(self._gpio_bank_low, "DDR", 0xFFFF, output_pins, board)

            # XX is the actual TR signal
            self._usrp.set_gpio_attr(self._gpio_bank_high, "ATR_XX", self._atr_xx, 0xFFFF, board)
            self._usrp.set_gpio_attr(self._gpio_bank_high, "ATR_RX", self._atr_rx, 0xFFFF, board)
            self._usrp.set_gpio_attr(self._gpio_bank_high, "ATR_TX", self._atr_tx, 0xFFFF, board)
            self._usrp.set_gpio_attr(self._gpio_bank_high, "ATR_0X", self._atr_0x, 0xFFFF, board)

            self._usrp.set_gpio_attr(self._gpio_bank_low, "ATR_XX", ~self._atr_xx, 0xFFFF, board)
            self._usrp.set_gpio_attr(self._gpio_bank_low, "ATR_RX", ~self._atr_rx, 0xFFFF, board)
            self._usrp.set_gpio_attr(self._gpio_bank_low, "ATR_TX", ~self._atr_tx, 0xFFFF, board)
            self._usrp.set_gpio_attr(self._gpio_bank_low, "ATR_0X", ~self._atr_0x, 0xFFFF, board)

    def _set_output_gpios(self):
        """Sets the pins mapping the test mode signals as GPIO outputs."""
        for board in range(self._num_mboards):
            # CTRL 0 sets the pins in gpio mode, DDR 1 sets them as outputs
            self._usrp.set_gpio_attr(self._gpio_bank_high, "CTRL", 0x0000, self._test_mode, board)

            self._usrp.set_gpio_attr(self._gpio_bank_high, "DDR", 0xFFFF, self._test_mode, board)

            self._usrp.set_gpio_attr(self._gpio_bank_low, "CTRL", 0x0000, self._test_mode, board)

            self._usrp.set_gpio_attr(self._gpio_bank_low, "DDR", 0xFFFF, self._test_mode, board)

    def _set_input_gpios(self):
        """Sets the pins mapping the AGC and low power signals as GPIO inputs."""
        for board in range(self._num_mboards):
            # CTRL 0 sets the pins in gpio mode, DDR 1 sets them as outputs
            self._usrp.set_gpio_attr(self._gpio_bank_high, "CTRL", 0x0000, self._agc_st, board)
            self._usrp.set_gpio_attr(self._gpio_bank_high, "CTRL", 0x0000, self._lo_pwr, board)

            self._usrp.set_gpio_attr(self._gpio_bank_high, "DDR", 0x0000, self._agc_st, board)
            self._usrp.set_gpio_attr(self._gpio_bank_high, "DDR", 0x0000, self._lo_pwr, board)

            self._usrp.set_gpio_attr(self._gpio_bank_low, "CTRL", 0x0000, self._agc_st, board)
            self._usrp.set_gpio_attr(self._gpio_bank_low, "CTRL", 0x0000, self._lo_pwr, board)

            self._usrp.set_gpio_attr(self._gpio_bank_low, "DDR", 0x0000, self._agc_st, board)
            self._usrp.set_gpio_attr(self._gpio_bank_low, "DDR", 0x0000, self._lo_pwr, board)

    def invert_test_mode(self, mboard: int):
        """Inverts the current test mode signal. Useful for testing.

        :param  mboard: The USRP to invert test mode on.
        :type   mboard: int
        """
        tm_value = self._usrp.get_gpio_attr(self._gpio_bank_high, "OUT", mboard)
        self._usrp.set_gpio_attr(self._gpio_bank_high, "OUT", self._test_mode, ~tm_value, mboard)
        self._usrp.set_gpio_attr(self._gpio_bank_low, "OUT", self._test_mode, ~tm_value, mboard)

    def set_test_mode(self, mboard: int):
        """Sets the current test mode signal HIGH.

        :param  mboard: The USRP to set test mode HIGH on.
        :type   mboard: int
        """
        self._usrp.set_gpio_attr(self._gpio_bank_high, "OUT", self._test_mode, 0xFFFF, mboard)
        self._usrp.set_gpio_attr(self._gpio_bank_low, "OUT", self._test_mode, 0x0000, mboard)

    def clear_test_mode(self, mboard: int):
        """Clears the current test mode signal LOW.

        :param  mboard: The USRP to clear test mode LOW on.
        :type   mboard: int
        """
        self._usrp.set_gpio_attr(self._gpio_bank_high, "OUT", self._test_mode, 0x0000, mboard)
        self._usrp.set_gpio_attr(self._gpio_bank_low, "OUT", self._test_mode, 0xFFFF, mboard)

    def get_gpio_bank_high_state(self) -> np.array:
        """Gets the state of the GPIO bank represented as a decimal number.

        :return:    State of the GPIO bank, in base-10.
        """
        readback_values = np.empty(self._num_mboards, dtype=np.uint32)
        for board in range(self._num_mboards):
            readback_values[board] = self._usrp.get_gpio_attr(self._gpio_bank_high, "READBACK", board)

        return readback_values

    def get_gpio_bank_low_state(self) -> np.array:
        """Gets the state of the GPIO bank represented as a decimal number.

        :return:    State of the GPIO bank, in base-10.
        """
        readback_values = np.empty(self._num_mboards, dtype=np.uint32)
        for board in range(self._num_mboards):
            readback_values[board] = self._usrp.get_gpio_attr(self._gpio_bank_low, "READBACK", board)

        return readback_values

    def gps_locked(self) -> bool:
        """Gets the current status of the GPS fix (locked or unlocked).

        :return:    True if the GPS has a lock.
        """
        # This takes on the order of a few microseconds
        if self._gps_clock is None:
            return False
        else:
            return self._gps_clock.get_sensor("gps_locked").to_bool()

    def get_agc_status_bank_h(self) -> int:
        """
        Gets the status of all of the active-high AGC fault signals as a single binary number.
        The bits represent each motherboard/USRP device, with bit index mapped to mboard num.
        """
        agc_status = 0b0
        for board in range(self._num_mboards):
            if self._usrp.get_gpio_attr(self._gpio_bank_high, "READBACK", board) & self._agc_st:
                agc_status = agc_status | 1 << board

        return agc_status

    def get_lp_status_bank_h(self) -> int:
        """
        Gets the status of all of the active-high Low power signals as a single binary number.
        The bits represent each motherboard/USRP device, with bit index mapped to mboard num.
        """
        low_power_status = 0b0
        for board in range(self._num_mboards):
            if self._usrp.get_gpio_attr(self._gpio_bank_high, "READBACK", board) & self._lo_pwr:
                low_power_status = low_power_status | 1 << board

        return low_power_status

    def get_agc_status_bank_l(self) -> int:
        """
        Gets the status of all of the active-low AGC fault signals as a single binary number.
        The bits represent each motherboard/USRP device, with bit index mapped to mboard num.
        """
        agc_status = 0b0
        for board in range(self._num_mboards):
            if self._usrp.get_gpio_attr(self._gpio_bank_low, "READBACK", board) & self._agc_st:
                agc_status = agc_status | 1 << board

        return agc_status

    def get_lp_status_bank_l(self) -> int:
        """
        Gets the status of all of the active-low Low power signals as a single binary number.
        The bits represent each motherboard/USRP device, with bit index mapped to mboard num.
        """
        low_power_status = 0b0
        for board in range(self._num_mboards):
            if self._usrp.get_gpio_attr(self._gpio_bank_low, "READBACK", board) & self._lo_pwr:
                low_power_status = low_power_status | 1 << board

        return low_power_status

    def get_current_usrp_time(self) -> uhd.types.TimeSpec:
        """Gets the current USRP time.

        :return:    The current USRP time.
        """
        return self._usrp.get_time_now()

    def create_usrp_rx_stream(self, cpu_fmt: str, otw_fmt: str, channels: list):
        """Creates a USRP receive stream.

        :param  cpu_fmt:    The cpu format for the rx stream. Described in UHD docs.
        :type   cpu_fmt:    str
        :param  otw_fmt:    The otw format for the rx stream. Described in UHD docs.
        :type   otw_fmt:    str
        :param  channels:   USRP channels to receive on.
        :type   channels:   list(int)
        """
        stream_args = uhd.usrp.StreamArgs(cpu_fmt, otw_fmt)
        stream_args.channels = channels
        self._rx_stream = self._usrp.get_rx_stream(stream_args)

    @property
    def get_usrp_rx_stream(self) -> uhd.usrp.RXStreamer:
        """Gets a reference to the USRP rx stream.

        :return:    The USRP rx stream.
        """
        return self._rx_stream

    def create_usrp_tx_stream(self, cpu_fmt: str, otw_fmt: str, channels: list):
        """Creates a USRP transmit stream.

        :param  cpu_fmt:    The cpu format for the tx stream. Described in UHD docs.
        :type   cpu_fmt:    str
        :param  otw_fmt:    The otw format for the tx stream. Described in UHD docs.
        :type   otw_fmt:    str
        :param  channels:   USRP channels to transmit on.
        :type   channels:   list(int)
        """
        stream_args = uhd.usrp.StreamArgs(cpu_fmt, otw_fmt)
        stream_args.channels = channels
        self._tx_stream = self._usrp.get_tx_stream(stream_args)

    @property
    def get_usrp(self) -> uhd.usrp.MultiUSRP:
        """Gets the usrp.

        :return:    The multi-USRP shared reference.
        """
        return self._usrp

    @property
    def get_usrp_tx_stream(self) -> uhd.usrp.TXStreamer:
        """Gets a reference to the USRP tx stream.

        :return:    The USRP tx stream.
        """
        return self._tx_stream

    def to_string(self, tx_channels: list, rx_channels: list) -> str:
        """Returns a string representation of the USRP parameters.

        :param  tx_channels:    USRP TX channels for which to generate info.
        :type   tx_channels:    list(int)
        :param  rx_channels:    USRP RX channels for which to generate info.
        :type   rx_channels:    list(int)

        :return:    String representation of the USRP parameters.
        """
        # Printable summary of the device
        device_str = "Using device {}\n".format(self._usrp.get_pp_string()) + \
                     "TX Rate {} Msps\n".format(self._usrp.get_tx_rate() / 1e6) + \
                     "RX Rate {} Msps\n".format(self._usrp.get_rx_rate() / 1e6)

        for channel in tx_channels:
            device_str += "TX Channel {} freq {} MHz\n".format(channel, self._usrp.get_tx_freq(channel))

        for channel in rx_channels:
            device_str += "RX Channel {} freq {} MHz\n".format(channel, self._usrp.get_rx_freq(channel))

        return device_str


class TXMetadata(object):
    """ Wrapper for the TX metadata object.

    Used to hold and initialize a new tx_metadata object. Creates getters and setters to access properties.
    """
    def __init__(self):
        """Constructs a blank USRP TX metadata object."""
        self._md = uhd.types.TXMetadata()
        self._md.start_of_burst = False
        self._md.end_of_burst = False
        self._md.has_time_spec = False
        self._md.time_spec = 0.0

    def get_md(self) -> uhd.types.TXMetadata:
        """Gets the TX metadata object that can be sent to the USRPs.

        :return:    The USRP TX metadata.
        """
        return self._md

    def set_start_of_burst(self, start_of_burst: bool):
        """Sets whether this data is the start of a burst.

        :param  start_of_burst: The start of burst flag.
        :type   start_of_burst: bool
        """
        self._md.start_of_burst = start_of_burst

    def set_end_of_burst(self, end_of_burst: bool):
        """Sets whether this data is the end of the burst.

        :param  end_of_burst: The end of burst flag.
        :type   end_of_burst: bool
        """
        self._md.end_of_burst = end_of_burst

    def set_has_time_spec(self, has_time_spec: bool):
        """Sets whether this data will have a particular timing.

        :param  has_time_spec:  Indicates if this metadata will have a time specifier.
        :type   has_time_spec:  bool
        """
        self._md.has_time_spec = has_time_spec

    def set_time_spec(self, time_spec: uhd.types.TimeSpec):
        """Sets the timing in the future for this metadata.

        :param  time_spec:  The time specifier for this metadata.
        :type   time_spec:  float
        """
        self._md.time_spec = time_spec


class RXMetadata(object):
    """ Wrapper for the USRP RX metadata object.

    Used to hold and initialize a new rx_metadata object. Creates getters and setters to access properties.
    """
    def __init__(self):
        """Initializes the fields of this RXMetadata object."""
        self._md = uhd.types.RXMetadata()

    def get_md(self) -> uhd.types.RXMetadata:
        """Gets the RX metadata object that will be retrieved on receiving.

        :return:    The USRP metadata object.
        """
        return self._md

    def get_end_of_burst(self) -> bool:
        """Gets the end of burst.

        :return:    The end of burst.
        """
        return self._md.end_of_burst

    def get_error_code(self) -> uhd.types.RXMetadataErrorCode:
        """Gets the error code from the metadata on receive.

        :return:    The error code.
        """
        return self._md.error_code

    def get_fragment_offset(self) -> int:
        """Gets the fragment offset. The fragment offset is the sample number at start of buffer.

        :return:    The fragment offset.
        """
        return self._md.fragment_offset

    def get_has_time_spec(self) -> bool:
        """Gets the 'has time specifier' status.

        :return:    The 'has time specifier' boolean.
        """
        return self._md.has_time_spec

    def get_out_of_sequence(self) -> bool:
        """Gets out of sequence status. Queries whether a packet is dropped or out of order.

        :return:    The out of sequence boolean.
        """
        return self._md.out_of_sequence

    def get_start_of_burst(self) -> bool:
        """Gets the start of burst status.

        :return:    The start of burst.
        """
        return self._md.start_of_burst

    def get_time_spec(self) -> uhd.types.TimeSpec:
        """Gets the time specifier of the packet.

        :return:    The time specifier.
        """
        return self._md.time_spec
