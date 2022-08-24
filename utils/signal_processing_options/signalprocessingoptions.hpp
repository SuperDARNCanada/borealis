/*Copyright 2016 SuperDARN*/
#ifndef SIGNALPROCESSINGOPTIONS_H
#define SIGNALPROCESSINGOPTIONS_H

#include <stdint.h>
#include <string>
#include "utils/options/options.hpp"

class SignalProcessingOptions: public Options {
 public:
  explicit SignalProcessingOptions();
  std::vector<uint32_t> get_main_antennas() const;
  uint32_t get_main_antenna_count() const;
  std::vector<uint32_t> get_interferometer_antennas() const;
  uint32_t get_interferometer_antenna_count() const;

  std::string get_router_address() const;
  std::string get_dsp_radctrl_identity() const;
  std::string get_dsp_driver_identity() const;
  std::string get_dsp_exphan_identity() const;
  std::string get_dsp_dw_identity() const;
  std::string get_dspbegin_brian_identity() const;
  std::string get_dspend_brian_identity() const;
  std::string get_radctrl_dsp_identity() const;
  std::string get_driver_dsp_identity() const;
  std::string get_brian_dspbegin_identity() const;
  std::string get_brian_dspend_identity() const;
  std::string get_exphan_dsp_identity() const;
  std::string get_dw_dsp_identity() const;
  std::string get_ringbuffer_name() const;

 private:
  std::vector<uint32_t> main_antennas;
  uint32_t main_antenna_count;
  std::vector<uint32_t> interferometer_antennas;
  uint32_t interferometer_antenna_count;
  std::string router_address;
  std::string dsp_to_radctrl_identity;
  std::string dsp_driver_identity;
  std::string dsp_exphan_identity;
  std::string dsp_dw_identity;
  std::string dspbegin_brian_identity;
  std::string dspend_brian_identity;
  std::string radctrl_dsp_identity;
  std::string driver_dsp_identity;
  std::string brian_dspbegin_identity;
  std::string brian_dspend_identity;
  std::string exphan_dsp_identity;
  std::string dw_dsp_identity;
  std::string ringbuffer_name;


};

#endif
