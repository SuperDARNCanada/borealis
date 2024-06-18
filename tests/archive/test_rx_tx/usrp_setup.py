"""
Class USRPSetup is intended to do the work of configuring a USRP device using
options from a configuration file, as well as user defined tx/rx frequencies
and channels. The class also provides setting methods for all configuration
options in case the user wishes to change the configuration on-the-fly.
"""

import setup_options
import numpy as np
import uhd
import time


class USRPSetup(object):
    """
    Class for handling the setup of a USRP based on
    config file options
    """

    def __init__(self, config_file, tx_freq, rx_freq, tx_chans, rx_chans):
        """
        Initializes txio board setup

        :param config_file: path to config file with options
        :param tx_freq: transmission center frequency
        :param rx_freq: recieving center frequency
        :param tx_chans: transmission channels
        :param rx_chans: receiver channels
        """
        self.options = setup_options.SetupOptions(config_file)
        self._tx_freq = tx_freq
        self._rx_freq = tx_freq
        self._rx_chans = rx_chans
        self._tx_chans = tx_chans

        # create usrp device
        self.usrp = uhd.usrp.MultiUSRP(self.options.get_devices)

    def set_usrp_clock_source(self, source):
        """
        Sets the clock source on the usrp
        :param source: string representing the clock source
        """
        self.usrp.set_clock_source(source)

    def set_time_source(self, source):
        """
        Sets the pps time source
        :param source: String representing the source
        """
        clk_addr = self.options.get_clk_addr()
        tt = time.time()
        while (tt - np.floor(tt)) < 0.2 or (tt - np.floor(tt)) > 0.3:
            tt = time.time()
            time.sleep(0.01)

        # TODO: Figure out how to set this to an external source
        self.usrp.set_time_source("none")

        curr_time = uhd.types.TimeSpec(tt)
        self.usrp.set_time_now(curr_time)

    def set_tx_subdev(self, tx_subdev_str):
        """
        Sets the subdevice for handling transmissions
        :param tx_subdev_str: A string specifying the subdevice
        """
        tx_subdev = uhd.usrp.SubdevSpec(tx_subdev_str)
        self.usrp.set_tx_subdev_spec(tx_subdev)

    def set_tx_rate(self, tx_rate):
        """
        Sets the transmission rate for specified transmission channels
        :param tx_rate: The desired data rate for transmission
        """
        self.usrp.set_tx_rate(tx_rate)

    def get_tx_rate(self, channel):
        """
        Gets the actual tx rate being used on a transmission channel
        :param channel: an integer representing the channel number
        """
        return self.usrp.get_tx_rate(np.uint32(channel))

    def set_tx_center_freq(self, freq, chans):
        """
        Tunes the usrp to the desired center frequency
        :param freq: The desired frequency in Hz
        :param chans: The channels to be tuned
        """
        tx_tune_request = uhd.types.TuneRequest(freq)
        for channel in chans:
            self.usrp.set_tx_freq(tx_tune_request, channel)
            actual_freq = self.usrp.get_tx_freq(channel)
            if not (actual_freq == freq):
                print(
                    "Requested tx center frequency:",
                    freq,
                    "actual frequency:",
                    actual_freq,
                    "\n",
                )

    def get_tx_center_freq(self, chan):
        """
        Gets the center frequency for a specified channel
        :param chan: The channel at which to retrieve the center frequency
        """
        return self.usrp.get_tx_freq(chan)

    def create_tx_stream(self, cpu, otw, chans):
        """
        Sets up the tx streamer object based on given streaming options
        :param cpu: The host cpu format as a string
        :param otw: The over the wire format as a string
        :param chans: Desired transmission channels
        """
        tx_stream_args = uhd.usrp.StreamArgs(cpu, otw)
        tx_stream_args.channels = chans
        tx_stream = self.usrp.get_tx_stream(tx_stream_args)
        return tx_stream

    def set_main_rx_subdev(self, main_subdev):
        """
        Sets up the subdevice for the main receiver
        :param main_subdev: String representing the subdevice(s) for the main receiver
        """
        rx_subdev = uhd.usrp.SubdevSpec(main_subdev)
        self.usrp.set_rx_subdev_spec(rx_subdev)

    def set_rx_rate(self, rx_rate):
        """
        Sets the data rate for the reciever
        :param rx_rate: The reciever data rate
        """
        self.usrp.set_rx_rate(rx_rate)

    def get_rx_rate(self, channel):
        """
        Gets the reciever rate on a specified channel
        :param channel: The desired channel
        """
        return self.usrp.get_rx_rate(np.uint32(channel))

    def set_rx_center_freq(self, freq, chans):
        """
        Tunes the reciever to a desired frequency
        :param freq: The desired reciever center frequency
        :param chans: The channels to tune to freq
        """
        rx_tune_request = uhd.types.TuneRequest(freq)
        for channel in chans:
            self.usrp.set_rx_freq(rx_tune_request, channel)
            actual_freq = self.usrp.get_rx_freq(channel)
            if not (actual_freq == freq):
                print(
                    "Requested rx center frequency:",
                    freq,
                    "actual frequency:",
                    actual_freq,
                    "\n",
                )

    def get_rx_center_freq(self, chan):
        """
        Gets the center frequency for a specified channel
        :param chan: The channel at which to retrieve the center frequency
        """
        return self.usrp.get_rx_freq(chan)

    def create_rx_stream(self, cpu, otw, chans):
        """
        Sets up an rx streaming object based on given options
        :par
        :param cpu: The host cpu format as a string
        :param otw: The over the wire format as a string
        :param chans: Desired receiving channels
        """
        rx_stream_args = uhd.usrp.StreamArgs(cpu, otw)
        rx_stream_args.channels = chans
        rx_stream = self.usrp.get_rx_stream(rx_stream_args)
        return rx_stream

    def setup_gpio(self, gpio_bank):
        """
        Configures the given gpio bank for the txio
        :param gpio_bank: String representing the gpio bank
        """

        def mask_map(attr, mask):
            """
            Applies a mask to the attribute for the given gpio bank

            :param attr: The gpio attribute to set
            :param mask: The bitmask that specifies the bits to set
                                     the attribute on
            """
            for i in self.usrp.get_num_mboards():
                self.usrp.set_gpio_attr(gpio_bank, attr, 0xFFFF, mask, i)

        def mask_unpack(gpio_tup):
            """
            Unpacks a gpio attribute, bitmask tuple for use in mask_map
            """
            mask_map(*gpio_tup)

        # Setup control and data direction register
        for i in np.arange(self.usrp.get_num_mboards()):
            self.usrp.set_gpio_attr(gpio_bank, "CTRL", 0xFFFF, 0b11111111, i)
            self.usrp.set_gpio_attr(gpio_bank, "DDR", 0xFFFF, 0b11111111, i)

        # setup GPIO attributes
        gpio_tups = [
            ("ATR_RX", self.options.get_atr_rx),
            ("ATR_TX", self.options.get_atr_tx),
            ("ATR_XX", self.options.get_atr_xx),
            ("ATR_0X", self.options.get_atr_0x),
        ]
        map(mask_unpack, gpio_tups)

    def setup(self):
        """
        main method for handling board setup based on given config
        file, frequencies, and channels
        """
        # Configure USRP clock and gpio bank
        self.set_usrp_clock_source(self.options.get_ref)
        self.setup_gpio(self.options.get_gpio_bank)

        # Configure RX subdevice
        self.set_main_rx_subdev(self.options.get_main_rx_subdev)
        self.set_rx_rate(self.options.get_rx_sample_rate)
        self.set_rx_center_freq(self._rx_freq, self._rx_chans)

        # Configure TX subdevice
        self.set_tx_subdev(self.options.get_tx_subdev)
        self.set_tx_rate(self.options.get_tx_sample_rate)
        self.set_tx_center_freq(self._tx_freq, self._tx_chans)

        # Create streams
        self.rx_stream = self.create_rx_stream(
            self.options.get_cpu, self.options.get_otw, self._rx_chans
        )
        self.tx_stream = self.create_tx_stream(
            self.options.get_cpu, self.options.get_otw, self._tx_chans
        )
