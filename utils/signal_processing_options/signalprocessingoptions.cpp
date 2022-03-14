/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <regex>
#include "utils/options/options.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"

std::vector<uint32_t> split(const std::string str, const std::string regex_str)
{
  std::vector<std::string> str_list{
    std::sregex_token_iterator(str.begin(), str.end(), std::regex(regex_str), -1),
    std::sregex_toke_iterator()
  };

  std::vector<uint32_t> int_list;
  for (auto& item: str_list) {
    int_list.push_back(boost::lexical_cast<uint32_t>(item));
  }

  return int_list;
}

SignalProcessingOptions::SignalProcessingOptions() {
  Options::parse_config_file();

  std::string main_antenna_list = config_pt.get<std::string>("main_antennas");
  main_antennas = split(main_antenna_list, ",");
  main_antenna_count = boost::lexical_cast<uint32_t>(
                config_pt.get<std::string>("main_antenna_count"));
  std::string interferometer_antenna_list = config_pt.get<std::string>("interferometer_antennas");
  interferometer_antennas = split(interferometer_antenna_list, ",");
  interferometer_antenna_count = boost::lexical_cast<uint32_t>(
                config_pt.get<std::string>("interferometer_antenna_count"));
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
