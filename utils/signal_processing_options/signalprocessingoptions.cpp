/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <vector>
#include <regex>
#include <iostream>
#include "utils/options/options.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"

std::vector<uint32_t> split(const std::string str, const std::string regex_str)
{
    std::regex re(regex_str);
    std::vector<std::string> str_list(
    std::sregex_token_iterator(str.begin(), str.end(), re, -1),
    std::sregex_token_iterator()
  );

  std::vector<uint32_t> int_list;
  for (auto& item: str_list) {
    int_list.push_back(boost::lexical_cast<uint32_t>(item));
  }

  return int_list;
}

SignalProcessingOptions::SignalProcessingOptions() {
  Options::parse_config_file();

  main_antenna_count_ = boost::lexical_cast<uint32_t>(
                              config_pt.get<std::string>("main_antenna_count"));
  interferometer_antenna_count_ = boost::lexical_cast<uint32_t>(
                              config_pt.get<std::string>("interferometer_antenna_count"));

  // Parse N200 list and determine which antennas are active
  std::vector<uint32_t> main_antenna_vec;
  std::vector<uint32_t> intf_antenna_vec;
  auto n200_list = config_pt.get_child("n200s");
  for (auto n200 = n200_list.begin(); n200 != n200_list.end(); n200++) {
    std::string addr = "";
    bool rx = false;
    bool tx = false;
    bool rx_int = false;
    std::string main_antenna = "";
    std::string interferometer_antenna = "";
    // Iterate through all N200 parameters and store them in variables
    for (auto iter = n200->second.begin(); iter != n200->second.end(); iter++)
    {
      auto param = iter->first;
      if (param.compare("addr") == 0) {
        addr = iter->second.data();
      }
      else if (param.compare("rx") == 0) {
        rx = (iter->second.data().compare("true") == 0);
      }
      else if (param.compare("tx") == 0) {
        tx = (iter->second.data().compare("true") == 0);
      }
      else if (param.compare("rx_int") == 0) {
        rx_int = (iter->second.data().compare("true") == 0);
      }
      else if (param.compare("main_antenna") == 0) {
        main_antenna = iter->second.data();
      }
      else if (param.compare("interferometer_antenna") == 0) {
        interferometer_antenna = iter->second.data();
      }
      else {
        throw std::invalid_argument("Invalid N200 parameter in config file");
      }
    }
    if (rx || tx) {
      auto main_antenna_num = boost::lexical_cast<uint32_t>(main_antenna);
      main_antenna_vec.push_back(main_antenna_num);
    }
    if (rx_int) {
      auto int_antenna_num = boost::lexical_cast<uint32_t>(interferometer_antenna);
      intf_antenna_vec.push_back(int_antenna_num);
    }
  }
  std::sort(main_antenna_vec.begin(), main_antenna_vec.end());
  std::sort(intf_antenna_vec.begin(), intf_antenna_vec.end());
  std::string main_antenna_list = "";
  std::string interferometer_antenna_list = "";
  for (auto antenna_num : main_antenna_vec) {  // TODO: Create the attribute directly from the config file without the secondary function
    main_antenna_list = main_antenna_list + std::to_string(antenna_num) + ",";
  }
  for (auto intf_num : intf_antenna_vec) {
    interferometer_antenna_list = interferometer_antenna_list + std::to_string(intf_num) + ",";
  }
  main_antenna_list.pop_back();
  interferometer_antenna_list.pop_back();

  main_antennas = split(main_antenna_list, ",");
  interferometer_antennas = split(interferometer_antenna_list, ",");

  router_address = config_pt.get<std::string>("router_address");
  dsp_to_radctrl_identity = config_pt.get<std::string>("dsp_to_radctrl_identity");
  dsp_driver_identity = config_pt.get<std::string>("dsp_to_driver_identity");
  dsp_exphan_identity = config_pt.get<std::string>("dsp_to_exphan_identity");
  dsp_dw_identity = config_pt.get<std::string>("dsp_to_dw_identity");
  dspbegin_brian_identity = config_pt.get<std::string>("dspbegin_to_brian_identity");
  dspend_brian_identity = config_pt.get<std::string>("dspend_to_brian_identity");
  radctrl_dsp_identity = config_pt.get<std::string>("radctrl_to_dsp_identity");
  driver_dsp_identity = config_pt.get<std::string>("driver_to_dsp_identity");
  brian_dspbegin_identity = config_pt.get<std::string>("brian_to_dspbegin_identity");
  brian_dspend_identity = config_pt.get<std::string>("brian_to_dspend_identity");
  exphan_dsp_identity = config_pt.get<std::string>("exphan_to_dsp_identity");
  dw_dsp_identity = config_pt.get<std::string>("dw_to_dsp_identity");
  ringbuffer_name = config_pt.get<std::string>("ringbuffer_name");
}

std::vector<uint32_t> SignalProcessingOptions::get_main_antennas() const
{
  return main_antennas;
}

uint32_t SignalProcessingOptions::get_main_antenna_count() const
{
  return main_antenna_count;
}

std::vector<uint32_t> SignalProcessingOptions::get_interferometer_antennas() const
{
  return interferometer_antennas;
}

uint32_t SignalProcessingOptions::get_interferometer_antenna_count() const
{
  return interferometer_antenna_count;
}

std::string SignalProcessingOptions::get_router_address() const
{
  return router_address;
}

std::string SignalProcessingOptions::get_dsp_radctrl_identity() const
{
  return dsp_to_radctrl_identity;
}

std::string SignalProcessingOptions::get_dsp_driver_identity() const
{
  return dsp_driver_identity;
}

std::string SignalProcessingOptions::get_dsp_exphan_identity() const
{
  return dsp_exphan_identity;
}

std::string SignalProcessingOptions::get_dsp_dw_identity() const
{
  return dsp_dw_identity;
}

std::string SignalProcessingOptions::get_dspbegin_brian_identity() const
{
  return   dspbegin_brian_identity;
}

std::string SignalProcessingOptions::get_dspend_brian_identity() const
{
  return   dspend_brian_identity;
}

std::string SignalProcessingOptions::get_radctrl_dsp_identity() const
{
  return radctrl_dsp_identity;
}

std::string SignalProcessingOptions::get_driver_dsp_identity() const
{
  return driver_dsp_identity;
}

std::string SignalProcessingOptions::get_brian_dspbegin_identity() const
{
  return brian_dspbegin_identity;
}

std::string SignalProcessingOptions::get_brian_dspend_identity() const
{
  return brian_dspend_identity;
}

std::string SignalProcessingOptions::get_exphan_dsp_identity() const
{
  return exphan_dsp_identity;
}

std::string SignalProcessingOptions::get_dw_dsp_identity() const
{
  return dw_dsp_identity;
}

std::string SignalProcessingOptions::get_ringbuffer_name() const
{
  return ringbuffer_name;
}
