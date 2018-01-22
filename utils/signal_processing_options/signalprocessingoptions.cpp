/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include "utils/options/options.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"


SignalProcessingOptions::SignalProcessingOptions() {
  Options::parse_config_file();

  first_stage_sample_rate = boost::lexical_cast<double>(
                config_pt.get<std::string>("first_stage_sample_rate"));
  second_stage_sample_rate = boost::lexical_cast<double>(
                config_pt.get<std::string>("second_stage_sample_rate"));
  third_stage_sample_rate = boost::lexical_cast<double>(
                config_pt.get<std::string>("third_stage_sample_rate"));
  first_stage_filter_cutoff = boost::lexical_cast<double>(
                config_pt.get<std::string>("first_stage_filter_cutoff"));
  first_stage_filter_transition = boost::lexical_cast<double>(
                config_pt.get<std::string>("first_stage_filter_transition"));
  second_stage_filter_cutoff = boost::lexical_cast<double>(
                config_pt.get<std::string>("second_stage_filter_cutoff"));
  second_stage_filter_transition = boost::lexical_cast<double>(
                config_pt.get<std::string>("second_stage_filter_transition"));
  third_stage_filter_cutoff = boost::lexical_cast<double>(
                config_pt.get<std::string>("third_stage_filter_cutoff"));
  third_stage_filter_transition = boost::lexical_cast<double>(
                config_pt.get<std::string>("third_stage_filter_transition"));
  main_antenna_count = boost::lexical_cast<uint32_t>(
                config_pt.get<std::string>("main_antenna_count"));
  interferometer_antenna_count = boost::lexical_cast<uint32_t>(
                config_pt.get<std::string>("interferometer_antenna_count"));
  driver_socket_address = config_pt.get<std::string>("driver_to_rx_dsp_address");
  radar_control_socket_address =  config_pt.get<std::string>("radar_control_to_rx_dsp_address");
  ack_socket_address = config_pt.get<std::string>("rx_dsp_to_radar_control_ack_address");
  timing_socket_address = config_pt.get<std::string>("rx_dsp_to_radar_control_timing_address");
  data_write_address = config_pt.get<std::string>("rx_dsp_to_data_write_address");
}

uint32_t SignalProcessingOptions::get_main_antenna_count() const
{
  return main_antenna_count;
}

uint32_t SignalProcessingOptions::get_interferometer_antenna_count() const
{
  return interferometer_antenna_count;
}
double SignalProcessingOptions::get_first_stage_sample_rate() const
{
  return first_stage_sample_rate;
}

double SignalProcessingOptions::get_second_stage_sample_rate() const
{
  return second_stage_sample_rate;
}

double SignalProcessingOptions::get_third_stage_sample_rate() const
{
  return third_stage_sample_rate;
}

double SignalProcessingOptions::get_first_stage_filter_cutoff() const
{
  return first_stage_filter_cutoff;
}

double SignalProcessingOptions::get_first_stage_filter_transition() const
{
  return first_stage_filter_transition;
}

double SignalProcessingOptions::get_second_stage_filter_cutoff() const
{
  return second_stage_filter_cutoff;
}

double SignalProcessingOptions::get_second_stage_filter_transition() const
{
  return second_stage_filter_transition;
}

double SignalProcessingOptions::get_third_stage_filter_cutoff() const
{
  return third_stage_filter_cutoff;
}

double SignalProcessingOptions::get_third_stage_filter_transition() const
{
  return third_stage_filter_transition;
}

std::string SignalProcessingOptions::get_driver_socket_address() const
{
  return driver_socket_address;
}

std::string SignalProcessingOptions::get_radar_control_socket_address() const
{
  return radar_control_socket_address;
}

std::string SignalProcessingOptions::get_ack_socket_address() const
{
  return ack_socket_address;
}

std::string SignalProcessingOptions::get_timing_socket_address() const
{
  return timing_socket_address;
}

std::string SignalProcessingOptions::get_data_write_address() const
{
  return data_write_address;
}
