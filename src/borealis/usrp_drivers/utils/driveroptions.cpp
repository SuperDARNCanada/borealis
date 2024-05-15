/*Copyright 2016 SuperDARN*/
#include "driveroptions.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "./options.hpp"

/**
 * @brief   Parses a channel specifier into the correct
 * antenna-to-channel-number map
 */
void processChannel(
    const std::string& channel, uint32_t channel_num,
    uint32_t main_antenna_count, uint32_t intf_antenna_count,
    std::map<uint32_t, uint32_t>& rx_main_antenna_to_channel_map,
    std::map<uint32_t, uint32_t>& rx_intf_antenna_to_channel_map,
    std::map<uint32_t, uint32_t>& tx_antenna_to_channel_map,
    bool rx_if_true_or_tx_if_false) {
  if (rx_if_true_or_tx_if_false) {
    if (!channel.empty()) {
      uint32_t antenna_num =
          boost::lexical_cast<uint32_t>(channel.substr(1, std::string::npos));
      if (channel.substr(0, 1) == "m") {
        if (antenna_num >= main_antenna_count) {
          throw std::invalid_argument(
              "Invalid antenna number for RX main array");
        }
        rx_main_antenna_to_channel_map[antenna_num] = channel_num;
      } else if (channel.substr(0, 1) == "i") {
        if (antenna_num >= intf_antenna_count) {
          throw std::invalid_argument(
              "Invalid antenna number for RX intf array");
        }
        rx_intf_antenna_to_channel_map[antenna_num] = channel_num;
      } else {
        throw std::invalid_argument("Cannot parse array identifier");
      }
    }
  } else {
    if (!channel.empty()) {
      uint32_t antenna_num =
          boost::lexical_cast<uint32_t>(channel.substr(1, std::string::npos));
      if (channel.substr(0, 1) == "m") {
        if (antenna_num >= main_antenna_count) {
          throw std::invalid_argument(
              "Invalid antenna number for TX main array");
        }
        tx_antenna_to_channel_map[antenna_num] = channel_num;
      } else {
        throw std::invalid_argument("Cannot connect TX channel to intf array");
      }
    }
  }
}

/**
 * @brief      Extracts the relevant driver options from the config into class
 * variables.
 */
DriverOptions::DriverOptions() {
  Options::parse_config_file();

  devices_ = config_pt.get<std::string>("device_options");

  auto n200_list = config_pt.get_child("n200s");

  std::map<uint32_t, uint32_t>
      rx_main_antenna_to_channel_map;  // Maps rx main antenna number to channel
                                       // number
  std::map<uint32_t, uint32_t>
      rx_intf_antenna_to_channel_map;  // Maps rx intf antenna number to channel
                                       // number
  std::map<uint32_t, uint32_t>
      tx_main_antenna_to_channel_map;  // Maps tx main antenna number to channel
                                       // number

  // Get number of physical antennas
  auto main_antenna_count = boost::lexical_cast<uint32_t>(
      config_pt.get<std::string>("main_antenna_count"));
  auto intf_antenna_count = boost::lexical_cast<uint32_t>(
      config_pt.get<std::string>("intf_antenna_count"));

  uint32_t device_num = 0;  // Active device counter

  // Iterate through all N200s in the json array
  for (auto n200 = n200_list.begin(); n200 != n200_list.end(); n200++) {
    std::string addr = "";
    std::string rx_channel_0 = "";
    std::string rx_channel_1 = "";
    std::string tx_channel_0 = "";

    // Iterate through all N200 parameters and store them in variables
    for (auto iter = n200->second.begin(); iter != n200->second.end(); iter++) {
      auto param = iter->first;
      if (param.compare("addr") == 0) {
        addr = iter->second.data();
      } else if (param.compare("rx_channel_0") == 0) {
        rx_channel_0 = iter->second.data();
      } else if (param.compare("rx_channel_1") == 0) {
        rx_channel_1 = iter->second.data();
      } else if (param.compare("tx_channel_0") == 0) {
        tx_channel_0 = iter->second.data();
      } else {
        throw std::invalid_argument("Invalid N200 parameter in config file");
      }
    }

    // If current N200 is connected to an antenna on any channel, add to
    // devices_
    if (!rx_channel_0.empty() || !rx_channel_1.empty() ||
        !tx_channel_0.empty()) {
      devices_ = devices_ + ",addr" + std::to_string(device_num) + "=" + addr;

      // Parse the antennas connected to each channel.
      // Each N200 has 2 RX channels and 1 TX channel, so the channel number
      // with respect to the multi-USRP object is calculated differently for
      // each channel of a given N200.
      processChannel(rx_channel_0, device_num * 2, main_antenna_count,
                     intf_antenna_count, rx_main_antenna_to_channel_map,
                     rx_intf_antenna_to_channel_map,
                     tx_main_antenna_to_channel_map, true);
      processChannel(rx_channel_1, device_num * 2 + 1, main_antenna_count,
                     intf_antenna_count, rx_main_antenna_to_channel_map,
                     rx_intf_antenna_to_channel_map,
                     tx_main_antenna_to_channel_map, true);
      processChannel(tx_channel_0, device_num, main_antenna_count,
                     intf_antenna_count, rx_main_antenna_to_channel_map,
                     rx_intf_antenna_to_channel_map,
                     tx_main_antenna_to_channel_map, false);

      // Increment the active device counter
      device_num++;
    }
  }

  // Main rx antennas
  std::string ma_recv_str = "";  // Main array receive channel string
  for (auto element : rx_main_antenna_to_channel_map) {
    auto channel_num = element.second;
    ma_recv_str = ma_recv_str + std::to_string(channel_num) + ",";
  }

  // Interferometer rx antennas
  std::string ia_recv_str = "";  // Interferometer array receive channel string
  for (auto element : rx_intf_antenna_to_channel_map) {
    auto channel_num = element.second;
    ia_recv_str = ia_recv_str + std::to_string(channel_num) + ",";
  }

  // Main tx antennas
  std::string ma_tx_str = "";  // Main array transmit channel string
  for (auto element : tx_main_antenna_to_channel_map) {
    auto channel_num = element.second;
    ma_tx_str = ma_tx_str + std::to_string(channel_num) + ",";
  }

  // Remove trailing comma from channel strings
  if (!ma_recv_str.empty()) ma_recv_str.pop_back();
  if (!ma_tx_str.empty()) ma_tx_str.pop_back();
  if (!ia_recv_str.empty()) ia_recv_str.pop_back();

  std::string total_recv_chs_str;
  if (ma_recv_str.empty())
    total_recv_chs_str = ia_recv_str;
  else if (ia_recv_str.empty())
    total_recv_chs_str = ma_recv_str;
  else
    total_recv_chs_str = ma_recv_str + "," + ia_recv_str;

  clk_addr_ = config_pt.get<std::string>("gps_octoclock_addr");

  tx_subdev_ = config_pt.get<std::string>("tx_subdev");
  if (tx_subdev_.compare("A:A") != 0) {
    throw std::invalid_argument("Invalid tx_subdev spec: Only 'A:A' supported");
  }
  main_rx_subdev_ = config_pt.get<std::string>("main_rx_subdev");
  if (main_rx_subdev_.compare("A:A A:B") != 0) {
    throw std::invalid_argument(
        "Invalid main_rx_subdev spec: Only 'A:A A:B' supported");
  }
  intf_rx_subdev_ = config_pt.get<std::string>("intf_rx_subdev");
  if (intf_rx_subdev_.compare("A:A A:B") != 0) {
    throw std::invalid_argument(
        "Invalid intf_rx_subdev spec: Only 'A:A A:B' supported");
  }
  pps_ = config_pt.get<std::string>("pps");
  ref_ = config_pt.get<std::string>("ref");
  cpu_ = config_pt.get<std::string>("cpu");
  otw_ = config_pt.get<std::string>("overthewire");
  gpio_bank_high_ = config_pt.get<std::string>("gpio_bank_high");
  gpio_bank_low_ = config_pt.get<std::string>("gpio_bank_low");

  std::stringstream ss;

  ss << std::hex << config_pt.get<std::string>("atr_rx");
  ss >> atr_rx_;
  ss.clear();

  ss << std::hex << config_pt.get<std::string>("atr_tx");
  ss >> atr_tx_;
  ss.clear();

  ss << std::hex << config_pt.get<std::string>("atr_xx");
  ss >> atr_xx_;
  ss.clear();

  ss << std::hex << config_pt.get<std::string>("atr_0x");
  ss >> atr_0x_;
  ss.clear();

  ss << std::hex << config_pt.get<std::string>("lo_pwr");
  ss >> lo_pwr_;
  ss.clear();

  ss << std::hex << config_pt.get<std::string>("agc_st");
  ss >> agc_st_;
  ss.clear();

  ss << std::hex << config_pt.get<std::string>("tst_md");
  ss >> test_mode_;
  ss.clear();

  tr_window_time_ =
      boost::lexical_cast<double>(config_pt.get<std::string>("tr_window_time"));
  agc_signal_read_delay_ = boost::lexical_cast<double>(
      config_pt.get<std::string>("agc_signal_read_delay"));

  auto make_channels = [&](std::string chs) {
    std::stringstream ss(chs);

    std::vector<size_t> channels;
    while (ss.good()) {
      std::string s;
      std::getline(ss, s, ',');
      if (s.empty()) {
        break;
      }
      channels.push_back(boost::lexical_cast<size_t>(s));
    }

    return channels;
  };

  receive_channels_ = make_channels(total_recv_chs_str);
  transmit_channels_ = make_channels(ma_tx_str);

  router_address_ = config_pt.get<std::string>("router_address");
  ringbuffer_name_ = config_pt.get<std::string>("ringbuffer_name");
  ringbuffer_size_bytes_ = boost::lexical_cast<double>(
      config_pt.get<std::string>("ringbuffer_size_bytes"));

  driver_to_radctrl_identity_ = "DRIVER_RADCTRL_IDEN";
  driver_to_dsp_identity_ = "DRIVER_DSP_IDEN";
  driver_to_brian_identity_ = "DRIVER_BRIAN_IDEN";
  radctrl_to_driver_identity_ = "RADCTRL_DRIVER_IDEN";
  dsp_to_driver_identity_ = "DSP_DRIVER_IDEN";
  brian_to_driver_identity_ = "BRIAN_DRIVER_IDEN";
}

/**
 * @brief      Gets the device arguments.
 *
 * @return     The device arguments.
 */
std::string DriverOptions::get_device_args() const { return devices_; }

/**
 * @brief      Gets the clock address.
 *
 * @return     The clock address.
 */
std::string DriverOptions::get_clk_addr() const { return clk_addr_; }

/**
 * @brief      Gets the USRP subdev for transmit bank.
 *
 * @return     The transmit subdev.
 */
std::string DriverOptions::get_tx_subdev() const { return tx_subdev_; }

/**
 * @brief      Gets the USRP receive subdev for main antenna bank.
 *
 * @return     The main receive subdev.
 */
std::string DriverOptions::get_main_rx_subdev() const {
  return main_rx_subdev_;
}

/**
 * @brief      Gets the USRP receive subdev for interferometer antenna bank.
 *
 * @return     The interferometer receive subdev.
 */
std::string DriverOptions::get_interferometer_rx_subdev() const {
  return intf_rx_subdev_;
}

/**
 * @brief      Gets the pps source.
 *
 * @return     The pps source.
 */
std::string DriverOptions::get_pps() const { return pps_; }

/**
 * @brief      Gets the 10 MHz reference source.
 *
 * @return     The 10 MHz reference source.
 */
std::string DriverOptions::get_ref() const { return ref_; }

/**
 * @brief      Gets the USRP cpu data type.
 *
 * @return     The cpu data type.
 */
std::string DriverOptions::get_cpu() const { return cpu_; }

/**
 * @brief      Gets the USRP otw format.
 *
 * @return     The USRP otw format.
 */
std::string DriverOptions::get_otw() const { return otw_; }

/**
 * @brief      Gets the active high gpio bank.
 *
 * @return     The active high gpio bank.
 */
std::string DriverOptions::get_gpio_bank_high() const {
  return gpio_bank_high_;
}

/**
 * @brief      Gets the active low gpio bank.
 *
 * @return     The active low gpio bank.
 */
std::string DriverOptions::get_gpio_bank_low() const { return gpio_bank_low_; }

/**
 * @brief      Gets the RX atr bank.
 *
 * @return     The RX atr bank.
 */
uint32_t DriverOptions::get_atr_rx() const { return atr_rx_; }

/**
 * @brief      Gets the TX atr bank.
 *
 * @return     The TX atr bank.
 */
uint32_t DriverOptions::get_atr_tx() const { return atr_tx_; }

/**
 * @brief      Gets the duplex atr bank.
 *
 * @return     The duplex atr bank.
 */
uint32_t DriverOptions::get_atr_xx() const { return atr_xx_; }

/**
 * @brief      Gets the idle atr bank.
 *
 * @return     The idle atr bank.
 */
uint32_t DriverOptions::get_atr_0x() const { return atr_0x_; }

/**
 * @brief      Gets the low power input bank.
 *
 * @return     The low power input bank.
 */
uint32_t DriverOptions::get_lo_pwr() const { return lo_pwr_; }

/**
 * @brief      Gets the agc status input bank.
 *
 * @return     The agc status bank.
 */
uint32_t DriverOptions::get_agc_st() const { return agc_st_; }

/**
 * @brief      Gets the test mode input bank.
 *
 * @return     The test mode input bank.
 */
uint32_t DriverOptions::get_test_mode() const { return test_mode_; }

/**
 * @brief      Gets the tr window time.
 *
 * @return     The tr window time.
 */
double DriverOptions::get_tr_window_time() const { return tr_window_time_; }

/**
 * @brief      Gets the agc status signal read delay.
 *
 * @return     The agc status signal read delay.
 */
double DriverOptions::get_agc_signal_read_delay() const {
  return agc_signal_read_delay_;
}

/**
 * @brief      Gets the ringbuffer size.
 *
 * @return     The ringbuffer size.
 */
double DriverOptions::get_ringbuffer_size() const {
  return ringbuffer_size_bytes_;
}

/**
 * @brief      Gets the all USRP receive channels.
 *
 * @return     The USRP receive channels.
 */
std::vector<size_t> DriverOptions::get_receive_channels() const {
  return receive_channels_;
}

/**
 * @brief      Gets the USRP transmit channels.
 *
 * @return     The USRP transmit channels.
 */
std::vector<size_t> DriverOptions::get_transmit_channels() const {
  return transmit_channels_;
}

/**
 * @brief      Gets the driver to radctrl identity.
 *
 * @return     The driver to radctrl identity.
 */
std::string DriverOptions::get_driver_to_radctrl_identity() const {
  return driver_to_radctrl_identity_;
}

/**
 * @brief      Gets the driver to dsp identity.
 *
 * @return     The driver to dsp identity.
 */
std::string DriverOptions::get_driver_to_dsp_identity() const {
  return driver_to_dsp_identity_;
}

/**
 * @brief      Gets the driver to brian identity.
 *
 * @return     The driver to brian identity.
 */
std::string DriverOptions::get_driver_to_brian_identity() const {
  return driver_to_brian_identity_;
}

/**
 * @brief      Gets the router address.
 *
 * @return     The router address.
 */
std::string DriverOptions::get_router_address() const {
  return router_address_;
}

/**
 * @brief      Gets the radctrl to driver identity.
 *
 * @return     The radctrl to driver identity.
 */
std::string DriverOptions::get_radctrl_to_driver_identity() const {
  return radctrl_to_driver_identity_;
}

/**
 * @brief      Gets the dsp to driver identity.
 *
 * @return     The dsp to driver identity.
 */
std::string DriverOptions::get_dsp_to_driver_identity() const {
  return dsp_to_driver_identity_;
}

/**
 * @brief      Gets the brian to driver identity.
 *
 * @return     The brian to driver identity.
 */
std::string DriverOptions::get_brian_to_driver_identity() const {
  return brian_to_driver_identity_;
}

/**
 * @brief      Gets the ringbuffer name.
 *
 * @return     The ringbuffer name.
 */
std::string DriverOptions::get_ringbuffer_name() const {
  return ringbuffer_name_;
}
