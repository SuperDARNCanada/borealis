# Copyright 2021 SuperDARN Canada
import json
import os
import numpy as np

borealis_path = os.environ['BOREALISPATH']
config_file = borealis_path + '/config.ini'


class DriverOptions(object):
    """
    Extracts the relevant driver options from the config into class variables.
    """
    def __init__(self):

        try:
            with open(config_file) as config_data:
                config = json.load(config_data)
        except IOError:
            errmsg = 'Cannot open config file at {}'.format(config_file)
            raise Exception(errmsg)
        try:
            # All un-commented fields are used by driver_options.cpp
            # All fields from config.ini are listed here, in order

            # self._site_id = config['site_id']

            self._gps_octoclock_addr = str(config['gsp_octoclock_addr'])
            self._devices = str(config['devices'])

            self._main_antenna_count = int(config['main_antenna_count'])
            self._intf_antenna_count = int(config['interferometer_antenna_count'])
            self._main_antenna_usrp_rx_channels = [int(x) for x in config['main_antenna_usrp_rx_channels'].split(',')]
            self._intf_antenna_usrp_rx_channels = [int(x) for x in config['intf_antenna_usrp_rx_channels'].split(',')]
            self._total_rx_channels = self._main_antenna_usrp_rx_channels + self._intf_antenna_usrp_rx_channels
            self._main_antenna_usrp_tx_channels = [int(x) for x in config['main_antenna_usrp_tx_channels'].split(',')]

            # self._main_antenna_spacing = float(config['main_antenna_spacing'])
            # self._intf_antenna_spacing = float(config['interferometer_antenna_spacing'])
            #
            # self._min_freq = float(config['min_freq'])  # Hz
            # self._max_freq = float(config['max_freq'])  # Hz
            # self._min_pulse_length = float(config['minimum_pulse_length'])  # us
            # self._min_tau_spacing_length = float(config['minimum_tau_spacing_length'])  # us
            # # Minimum pulse separation is the minimum before the experiment treats it as a single
            # # pulse (transmitting zeroes or no receiving between the pulses)
            # # 125 us is approx two TX/RX times
            # self._minimum_pulse_separation = float(config['minimum_pulse_separation'])  # us

            self._tx_subdev = str(config['tx_subdev'])

            # self._max_tx_sample_rate = float(config['max_tx_sample_rate'])

            self._main_rx_subdev = str(config['main_rx_subdev'])
            self._intf_rx_subdev = str(config['interferometer_rx_subdev'])

            # self._max_rx_sample_rate = float(config['max_rx_sample_rate'])

            self._pps = str(config['pps'])
            self._ref = str(config['ref'])
            self._over_the_wire = str(config['overthewire'])
            self._cpu = str(config['cpu'])
            self._gpio_bank_high = str(config['gpio_bank_high'])
            self._gpio_bank_low = str(config['gpio_bank_low'])

            # Addresses of registers for reading/writing
            self._atr_rx = np.uint32(config['atr_rx'])
            self._atr_tx = np.uint32(config['atr_tx'])
            self._atr_xx = np.uint32(config['atr_xx'])
            self._atr_0x = np.uint32(config['atr_0x'])
            self._tst_md = np.uint32(config['txt_md'])
            self._lo_pwr = np.uint32(config['lo_pwr'])
            self._agc_st = np.uint32(config['agc_st'])

            # self._max_usrp_dac_amplitude = float(config['max_usrp_dac_amplitude'])
            # self._pulse_ramp_time = float(config['pulse_ramp_time'])  # in seconds
            # self._gpio_bank = config['gpio_bank']

            self._tr_window_time = float(config['tr_window_time'])
            self._agc_signal_read_delay = float(config['agc_signal_read_delay'])

            # self._usrp_master_clock_rate = float(config['usrp_master_clock_rate'])  # Hz
            #
            # # should use to check iqdata samples when adjusting the experiment during operations.
            # self._max_output_sample_rate = float(config['max_output_sample_rate'])
            # self._max_number_of_filtering_stages = int(config['max_number_of_filtering_stages'])
            # self._max_number_of_filter_taps_per_stage = int(config['max_number_of_filter_taps_per_stage'])

            self._router_address = str(config['router_address'])

            # self._realtime_address = config['realtime_address']
            # self._radctrl_to_exphan_identity = str(config["radctrl_to_exphan_identity"])
            # self._radctrl_to_dsp_identity = str(config["radctrl_to_dsp_identity"])

            self._radctrl_to_driver_identity = str(config["radctrl_to_driver_identity"])

            # self._radctrl_to_brian_identity = str(config["radctrl_to_brian_identity"])
            # self._radctrl_to_dw_identity = str(config["radctrl_to_dw_identity"])

            self._driver_to_radctrl_identity = str(config["driver_to_radctrl_identity"])
            self._driver_to_dsp_identity = str(config["driver_to_dsp_identity"])
            self._driver_to_brian_identity = str(config["driver_to_brian_identity"])

            # self._driver_to_mainaffinity_identity = str(config["driver_to_mainaffinity_identity"])
            # self._driver_to_txaffinity_identity = str(config["driver_to_txaffinity_identity"])
            # self._driver_to_rxaffinity_identity = str(config["driver_to_rxaffinity_identity"])
            # self._mainaffinity_to_driver_identity = str(config["mainaffinity_to_driver_identity"])
            # self._txaffinity_to_driver_identity = str(config["txaffinity_to_driver_identity"])
            # self._rxaffinity_to_driver_identity = str(config["rxaffinity_to_driver_identity"])
            # self._exphan_to_radctrl_identity = str(config["exphan_to_radctrl_identity"])
            # self._exphan_to_dsp_identity = str(config["exphan_to_dsp_identity"])
            # self._dsp_to_radctrl_identity = str(config["dsp_to_radctrl_identity"])

            self._dsp_to_driver_identity = str(config["dsp_to_driver_identity"])

            # self._dsp_to_exphan_identity = str(config["dsp_to_exphan_identity"])
            # self._dsp_to_dw_identity = str(config["dsp_to_dw_identity"])
            # self._dspbegin_to_brian_identity = str(config["dspbegin_to_brian_identity"])
            # self._dspend_to_brian_identity = str(config["dspend_to_brian_identity"])
            # self._dw_to_dsp_identity = str(config["dw_to_dsp_identity"])
            # self._dw_to_radctrl_identity = str(config["dw_to_radctrl_identity"])
            # self._dw_to_rt_identity = str(config["dw_to_rt_identity"])
            # self._rt_to_dw_identity = str(config["rt_to_dw_identity"])
            # self._brian_to_radctrl_identity = str(config["brian_to_radctrl_identity"])

            self._brian_to_driver_identity = str(config["brian_to_driver_identity"])

            # self._brian_to_dspbegin_identity = str(config["brian_to_dspbegin_identity"])
            # self._brian_to_dspend_identity = str(config["brian_to_dspend_identity"])

            self._ringbuffer_name = str(config["ringbuffer_name"])
            self._ringbuffer_size_bytes = float(config["ringbuffer_size_bytes"])

            # self._data_directory = str(config["data_directory"])
            # self._log_directory = str(config["log_directory"])

        except ValueError as e:
            # TODO: error
            raise e

    @property
    def devices(self):
        """Gets the device arguments."""
        return self._devices

    @property
    def clk_addr(self):
        """Get the clock address."""
        return self._gps_octoclock_addr

    @property
    def tx_subdev(self):
        """Gets the USRP subdev for transmit bank."""
        return self._tx_subdev

    @property
    def main_rx_subdev(self):
        """Gets the USRP receive subdev for the main antenna bank."""
        return self._main_rx_subdev

    @property
    def intf_rx_subdev(self):
        """Gets the USRP receive subdev for the interferometer antenna bank."""
        return self._intf_rx_subdev

    @property
    def pps(self):
        """Gets the pps source."""
        return self._pps

    @property
    def ref(self):
        """Gets the 10 MHz reference source."""
        return self._ref

    @property
    def cpu(self):
        """Gets the USRP cpu data type."""
        return self._cpu

    @property
    def otw(self):
        """Gets the USRP over-the-wire format."""
        return self._over_the_wire

    @property
    def gpio_bank_high(self):
        """Gets the active high gpio bank."""
        return self._gpio_bank_high

    @property
    def gpio_bank_low(self):
        """Gets the active low gpio bank."""
        return self._gpio_bank_low

    @property
    def atr_rx(self):
        """Gets the RX atr bank."""
        return self._atr_rx

    @property
    def atr_tx(self):
        """Gets the TX atr bank."""
        return self._atr_tx

    @property
    def atr_xx(self):
        """Gets the duplex atr bank."""
        return self._atr_xx

    @property
    def atr_0x(self):
        """Gets the idle atr bank."""
        return self._atr_rx

    @property
    def lo_pwr(self):
        """Gets the low power input bank."""
        return self._lo_pwr

    @property
    def agc_st(self):
        """Gets the agc status input bank."""
        return self._agc_st

    @property
    def test_mode(self):
        """Gets the test mode input bank."""
        return self._tst_md

    @property
    def tr_window_time(self):
        """Gets the tr window time."""
        return self._tr_window_time

    @property
    def agc_signal_read_delay(self):
        """Gets the agc status signal read delay."""
        return self._agc_signal_read_delay

    @property
    def main_antenna_count(self):
        """Gets the main antenna count."""
        return self._main_antenna_count

    @property
    def intf_antenna_count(self):
        """Gets the interferometer antenna count."""
        return self._intf_antenna_count

    @property
    def receive_channels(self):
        """Gets all the USRP receive channels."""
        return self._total_rx_channels

    @property
    def transmit_channels(self):
        """Gets all the USRP transmit channels."""
        return self._main_antenna_usrp_tx_channels

    @property
    def router_address(self):
        """Gets the router address."""
        return self._router_address

    @property
    def driver_to_radctrl_identity(self):
        """Gets the driver to radctrl identity."""
        return self._driver_to_radctrl_identity

    @property
    def driver_to_dsp_identity(self):
        """Gets the driver to dsp identity."""
        return self._driver_to_dsp_identity

    @property
    def driver_to_brian_identity(self):
        """Gets the driver to brian identity."""
        return self._driver_to_brian_identity

    @property
    def radctrl_to_driver_identity(self):
        """Gets the radctrl to driver identity."""
        return self._radctrl_to_driver_identity

    @property
    def dsp_to_driver_identity(self):
        """Gets the dsp to driver identity."""
        return self._dsp_to_driver_identity

    @property
    def brian_to_driver_identity(self):
        """Gets the brian to driver identity."""
        return self._brian_to_driver_identity

    @property
    def ringbuffer_size(self):
        """Gets the ringbuffer size."""
        return self._ringbuffer_size_bytes

    @property
    def ringbuffer_name(self):
        """Gets the ringbuffer name."""
        return self._ringbuffer_name
