# Copyright 2021 SuperDARN Canada
import json
import os

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

            self._gps_octoclock_addr = config['gsp_octoclock_addr']
            self._devices = config['devices']

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

            self._tx_subdev = config['tx_subdev']

            # self._max_tx_sample_rate = float(config['max_tx_sample_rate'])

            self._main_rx_subdev = config['main_rx_subdev']
            self._intf_rx_subdev = config['interferometer_rx_subdev']

            # self._max_rx_sample_rate = float(config['max_rx_sample_rate'])

            self._pps = config['pps']
            self._ref = config['ref']
            self._over_the_wire = config['overthewire']
            self._cpu = config['cpu']
            self._gpio_bank_high = config['gpio_bank_high']
            self._gpio_bank_low = config['gpio_bank_low']

            # Addresses of registers for reading/writing
            self._atr_rx = config['atr_rx']
            self._atr_tx = config['atr_tx']
            self._atr_xx = config['atr_xx']
            self._atr_0x = config['atr_0x']
            self._tst_md = config['txt_md']
            self._lo_pwr = config['lo_pwr']
            self._agc_set = config['agc_set']

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

            self._router_address = config['router_address']

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

# #include <boost/property_tree/ini_parser.hpp>
# #include <boost/range/algorithm_ext/erase.hpp>
# #include <boost/algorithm/string.hpp>
# #include <boost/lexical_cast.hpp>
# #include <string>
# #include <iostream>
# #include <sstream>
# #include "utils/options/options.hpp"
# #include "utils/driver_options/driveroptions.hpp"
#
#
# /**
#  * @brief      Extracts the relevant driver options from the config into class variables.
#  */
# DriverOptions::DriverOptions() {
#     Options::parse_config_file();
#
#     devices_ = config_pt.get<std::string>("devices");
#     clk_addr_ = config_pt.get<std::string>("gps_octoclock_addr");
#     /*Remove whitespace/new lines from device list*/
#     boost::remove_erase_if (devices_, boost::is_any_of(" \n\f\t\v"));
#
#     tx_subdev_ = config_pt.get<std::string>("tx_subdev");
#     main_rx_subdev_ = config_pt.get<std::string>("main_rx_subdev");
#     interferometer_rx_subdev_ = config_pt.get<std::string>("interferometer_rx_subdev");
#     pps_ = config_pt.get<std::string>("pps");
#     ref_ = config_pt.get<std::string>("ref");
#     cpu_ = config_pt.get<std::string>("cpu");
#     otw_ = config_pt.get<std::string>("overthewire");
#     gpio_bank_high_ = config_pt.get<std::string>("gpio_bank_high");
#     gpio_bank_low_ = config_pt.get<std::string>("gpio_bank_low");
#
#     std::stringstream ss;
#
#     ss << std::hex << config_pt.get<std::string>("atr_rx");
#     ss >> atr_rx_;
#     ss.clear();
#
#     ss << std::hex << config_pt.get<std::string>("atr_tx");
#     ss >> atr_tx_;
#     ss.clear();
#
#     ss << std::hex << config_pt.get<std::string>("atr_xx");
#     ss >> atr_xx_;
#     ss.clear();
#
#     ss << std::hex << config_pt.get<std::string>("atr_0x");
#     ss >> atr_0x_;
#     ss.clear();
#
#     ss << std::hex << config_pt.get<std::string>("lo_pwr");
#     ss >> lo_pwr_;
#     ss.clear();
#
#     ss << std::hex << config_pt.get<std::string>("agc_st");
#     ss >> agc_st_;
#     ss.clear();
#
#     ss << std::hex << config_pt.get<std::string>("tst_md");
#     ss >> test_mode_;
#     ss.clear();
#
#     tr_window_time_ = boost::lexical_cast<double>(
#                                 config_pt.get<std::string>("tr_window_time"));
#     agc_signal_read_delay_ = boost::lexical_cast<double>(
#                                 config_pt.get<std::string>("agc_signal_read_delay"));
#     main_antenna_count_ = boost::lexical_cast<uint32_t>(
#                                 config_pt.get<std::string>("main_antenna_count"));
#     interferometer_antenna_count_ = boost::lexical_cast<uint32_t>(
#                                 config_pt.get<std::string>("interferometer_antenna_count"));
#
#     auto make_channels = [&](std::string chs){
#
#         std::stringstream ss(chs);
#
#         std::vector<size_t> channels;
#         while (ss.good()) {
#             std::string s;
#             std::getline(ss, s, ',');
#             channels.push_back(boost::lexical_cast<size_t>(s));
#         }
#
#         return channels;
#     };
#
#     auto ma_recv_str = config_pt.get<std::string>("main_antenna_usrp_rx_channels");
#     auto ia_recv_str = config_pt.get<std::string>("interferometer_antenna_usrp_rx_channels");
#     auto total_recv_chs_str = ma_recv_str + "," + ia_recv_str;
#
#     auto ma_tx_str = config_pt.get<std::string>("main_antenna_usrp_tx_channels");
#
#     receive_channels_ = make_channels(total_recv_chs_str);
#     transmit_channels_ = make_channels(ma_tx_str);
#
#     router_address_ = config_pt.get<std::string>("router_address");
#     driver_to_radctrl_identity_ = config_pt.get<std::string>("driver_to_radctrl_identity");
#     driver_to_dsp_identity_ = config_pt.get<std::string>("driver_to_dsp_identity");
#     driver_to_brian_identity_ = config_pt.get<std::string>("driver_to_brian_identity");
#     radctrl_to_driver_identity_ = config_pt.get<std::string>("radctrl_to_driver_identity");
#     dsp_to_driver_identity_ = config_pt.get<std::string>("dsp_to_driver_identity");
#     brian_to_driver_identity_ = config_pt.get<std::string>("brian_to_driver_identity");
#     ringbuffer_name_ = config_pt.get<std::string>("ringbuffer_name");
#     ringbuffer_size_bytes_ = boost::lexical_cast<double>(
#                                     config_pt.get<std::string>("ringbuffer_size_bytes"));
# }
#
# /**
#  * @brief      Gets the device arguments.
#  *
#  * @return     The device arguments.
#  */
# std::string DriverOptions::get_device_args() const
# {
#     return devices_;
# }
#
# /**
#  * @brief      Gets the clock address.
#  *
#  * @return     The clock address.
#  */
# std::string DriverOptions::get_clk_addr() const
# {
#     return clk_addr_;
# }
#
# /**
#  * @brief      Gets the USRP subdev for transmit bank.
#  *
#  * @return     The transmit subdev.
#  */
# std::string DriverOptions::get_tx_subdev() const
# {
#     return tx_subdev_;
# }
#
# /**
#  * @brief      Gets the USRP receive subdev for main antenna bank.
#  *
#  * @return     The main receive subdev.
#  */
# std::string DriverOptions::get_main_rx_subdev() const
# {
#     return main_rx_subdev_;
# }
#
# /**
#  * @brief      Gets the USRP receive subdev for interferometer antenna bank.
#  *
#  * @return     The interferometer receive subdev.
#  */
# std::string DriverOptions::get_interferometer_rx_subdev() const
# {
#     return interferometer_rx_subdev_;
# }
#
# /**
#  * @brief      Gets the pps source.
#  *
#  * @return     The pps source.
#  */
# std::string DriverOptions::get_pps() const
# {
#     return pps_;
# }
#
# /**
#  * @brief      Gets the 10 MHz reference source.
#  *
#  * @return     The 10 MHz reference source.
#  */
# std::string DriverOptions::get_ref() const
# {
#     return ref_;
# }
#
# /**
#  * @brief      Gets the USRP cpu data type.
#  *
#  * @return     The cpu data type.
#  */
# std::string DriverOptions::get_cpu() const
# {
#     return cpu_;
# }
#
# /**
#  * @brief      Gets the USRP otw format.
#  *
#  * @return     The USRP otw format.
#  */
# std::string DriverOptions::get_otw() const
# {
#     return otw_;
# }
#
# /**
#  * @brief      Gets the active high gpio bank.
#  *
#  * @return     The active high gpio bank.
#  */
# std::string DriverOptions::get_gpio_bank_high() const
# {
#     return gpio_bank_high_;
# }
#
# /**
#  * @brief      Gets the active low gpio bank.
#  *
#  * @return     The active low gpio bank.
#  */
# std::string DriverOptions::get_gpio_bank_low() const
# {
#     return gpio_bank_low_;
# }
#
# /**
#  * @brief      Gets the RX atr bank.
#  *
#  * @return     The RX atr bank.
#  */
# uint32_t DriverOptions::get_atr_rx() const
# {
#     return atr_rx_;
# }
#
# /**
#  * @brief      Gets the TX atr bank.
#  *
#  * @return     The TX atr bank.
#  */
# uint32_t DriverOptions::get_atr_tx() const
# {
#     return atr_tx_;
# }
#
# /**
#  * @brief      Gets the duplex atr bank.
#  *
#  * @return     The duplex atr bank.
#  */
# uint32_t DriverOptions::get_atr_xx() const
# {
#     return atr_xx_;
# }
#
# /**
#  * @brief      Gets the idle atr bank.
#  *
#  * @return     The idle atr bank.
#  */
# uint32_t DriverOptions::get_atr_0x() const
# {
#     return atr_0x_;
#
# }
#
# /**
#  * @brief      Gets the low power input bank.
#  *
#  * @return     The low power input bank.
#  */
# uint32_t DriverOptions::get_lo_pwr() const
# {
#     return lo_pwr_;
# }
#
#
# /**
#  * @brief      Gets the agc status input bank.
#  *
#  * @return     The agc status bank.
#  */
# uint32_t DriverOptions::get_agc_st() const
# {
#     return agc_st_;
# }
#
# /**
#  * @brief      Gets the test mode input bank.
#  *
#  * @return     The test mode input bank.
#  */
# uint32_t DriverOptions::get_test_mode() const
# {
#     return test_mode_;
# }
#
# /**
#  * @brief      Gets the tr window time.
#  *
#  * @return     The tr window time.
#  */
# double DriverOptions::get_tr_window_time() const
# {
#     return tr_window_time_;
# }
#
# /**
#  * @brief      Gets the agc status signal read delay.
#  *
#  * @return     The agc status signal read delay.
#  */
# double DriverOptions::get_agc_signal_read_delay() const
# {
#     return agc_signal_read_delay_;
# }
#
# /**
#  * @brief      Gets the main antenna count.
#  *
#  * @return     The main antenna count.
#  */
# uint32_t DriverOptions::get_main_antenna_count() const
# {
#     return main_antenna_count_;
# }
#
# /**
#  * @brief      Gets the interferometer antenna count.
#  *
#  * @return     The interferometer antenna count.
#  */
# uint32_t DriverOptions::get_interferometer_antenna_count() const
# {
#     return interferometer_antenna_count_;
# }
#
# /**
#  * @brief      Gets the ringbuffer size.
#  *
#  * @return     The ringbuffer size.
#  */
# double DriverOptions::get_ringbuffer_size() const
# {
#     return ringbuffer_size_bytes_;
# }
#
# /**
#  * @brief      Gets the all USRP receive channels.
#  *
#  * @return     The USRP receive channels.
#  */
# std::vector<size_t> DriverOptions::get_receive_channels() const
# {
#     return receive_channels_;
# }
#
# /**
#  * @brief      Gets the USRP transmit channels.
#  *
#  * @return     The USRP transmit channels.
#  */
# std::vector<size_t> DriverOptions::get_transmit_channels() const
# {
#     return transmit_channels_;
# }
#
# /**
#  * @brief      Gets the driver to radctrl identity.
#  *
#  * @return     The driver to radctrl identity.
#  */
# std::string DriverOptions::get_driver_to_radctrl_identity() const
# {
#     return driver_to_radctrl_identity_;
# }
#
# /**
#  * @brief      Gets the driver to dsp identity.
#  *
#  * @return     The driver to dsp identity.
#  */
# std::string DriverOptions::get_driver_to_dsp_identity() const
# {
#     return driver_to_dsp_identity_;
# }
#
# /**
#  * @brief      Gets the driver to brian identity.
#  *
#  * @return     The driver to brian identity.
#  */
# std::string DriverOptions::get_driver_to_brian_identity() const
# {
#     return driver_to_brian_identity_;
# }
#
# /**
#  * @brief      Gets the router address.
#  *
#  * @return     The router address.
#  */
# std::string DriverOptions::get_router_address() const
# {
#     return router_address_;
# }
#
# /**
#  * @brief      Gets the radctrl to driver identity.
#  *
#  * @return     The radctrl to driver identity.
#  */
# std::string DriverOptions::get_radctrl_to_driver_identity() const
# {
#     return radctrl_to_driver_identity_;
# }
#
#
# /**
#  * @brief      Gets the dsp to driver identity.
#  *
#  * @return     The dsp to driver identity.
#  */
# std::string DriverOptions::get_dsp_to_driver_identity() const
# {
#     return dsp_to_driver_identity_;
# }
#
#
# /**
#  * @brief      Gets the brian to driver identity.
#  *
#  * @return     The brian to driver identity.
#  */
# std::string DriverOptions::get_brian_to_driver_identity() const
# {
#     return brian_to_driver_identity_;
# }
#
# /**
#  * @brief      Gets the ringbuffer name.
#  *
#  * @return     The ringbuffer name.
#  */
# std::string DriverOptions::get_ringbuffer_name() const
# {
#     return ringbuffer_name_;
# }
