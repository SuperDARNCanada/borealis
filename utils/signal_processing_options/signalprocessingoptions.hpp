/*Copyright 2016 SuperDARN*/
#ifndef SIGNALPROCESSINGOPTIONS_H
#define SIGNALPROCESSINGOPTIONS_H

#include <stdint.h>
#include <string>
#include "utils/options/options.hpp"

class SignalProcessingOptions: public Options {
 public:
  explicit SignalProcessingOptions();
  double get_first_stage_sample_rate() const;
  double get_second_stage_sample_rate() const;
  double get_third_stage_sample_rate() const;
  double get_first_stage_filter_cutoff() const;
  double get_first_stage_filter_transition() const;
  double get_second_stage_filter_cutoff() const;
  double get_second_stage_filter_transition() const;
  double get_third_stage_filter_cutoff() const;
  double get_third_stage_filter_transition() const;
  uint32_t get_main_antenna_count() const;
  uint32_t get_interferometer_antenna_count() const;
  std::string get_driver_socket_address() const;
  std::string get_radar_control_socket_address() const;
  std::string get_ack_socket_address() const;
  std::string get_timing_socket_address() const;
  std::string get_data_write_address() const;

 private:
  uint32_t main_antenna_count;
  uint32_t interferometer_antenna_count;
  double first_stage_sample_rate;
  double second_stage_sample_rate;
  double third_stage_sample_rate;
  double first_stage_filter_cutoff;
  double first_stage_filter_transition;
  double second_stage_filter_cutoff;
  double second_stage_filter_transition;
  double third_stage_filter_cutoff;
  double third_stage_filter_transition;
  std::string driver_socket_address;
  std::string radar_control_socket_address;
  std::string ack_socket_address;
  std::string timing_socket_address;
  std::string data_write_address;


};

#endif
