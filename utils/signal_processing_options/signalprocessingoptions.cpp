/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include "utils/options/options.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"


SignalProcessingOptions::SignalProcessingOptions() {
  Options::parse_config_file();

  // Parse N200 array and calculate main and intf antenna count
  main_antenna_count = 0;
  interferometer_antenna_count = 0;
  auto n200_list = config_pt.get_child("n200s");
  for (auto n200 = n200_list.begin(); n200 != n200_list.end(); n200++) {
    // Start iterator on first item (addr)
    auto iter = n200->second.begin();

    // Get rx, tx, and rx_int flags
    iter++;
    bool rx = (iter->second.data().compare("true") == 0);
    iter++;
    bool tx = (iter->second.data().compare("true") == 0);
    iter++;
    bool rx_int = (iter->second.data().compare("true") == 0);

    if (rx || tx) {
      main_antenna_count += 1;
    }
    if (rx_int) {
      interferometer_antenna_count += 1;
    }
  }

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


uint32_t SignalProcessingOptions::get_main_antenna_count() const
{
  return main_antenna_count;
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
