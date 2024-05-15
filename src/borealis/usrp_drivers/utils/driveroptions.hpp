/*Copyright 2016 SuperDARN*/
#ifndef SRC_USRP_DRIVERS_UTILS_DRIVEROPTIONS_HPP_
#define SRC_USRP_DRIVERS_UTILS_DRIVEROPTIONS_HPP_

#include <stdint.h>

#include <map>
#include <string>
#include <vector>

#include "./options.hpp"

void processChannel(
    const std::string& channel, uint32_t channel_num,
    uint32_t main_antenna_count, uint32_t intf_antenna_count,
    std::map<uint32_t, uint32_t>& rx_main_antenna_to_channel_map,
    std::map<uint32_t, uint32_t>& rx_intf_antenna_to_channel_map,
    std::map<uint32_t, uint32_t>& tx_antenna_to_channel_map,
    bool rx_if_true_or_tx_if_false);

class DriverOptions : public Options {
 public:
  DriverOptions();
  std::string get_device_args() const;
  std::string get_clk_addr() const;
  std::string get_tx_subdev() const;
  std::string get_main_rx_subdev() const;
  std::string get_interferometer_rx_subdev() const;
  std::string get_pps() const;
  std::string get_ref() const;
  std::string get_cpu() const;
  std::string get_otw() const;
  std::string get_gpio_bank_high() const;
  std::string get_gpio_bank_low() const;
  uint32_t get_atr_rx() const;
  uint32_t get_atr_tx() const;
  uint32_t get_atr_xx() const;
  uint32_t get_atr_0x() const;
  uint32_t get_lo_pwr() const;
  uint32_t get_agc_st() const;
  uint32_t get_test_mode() const;
  double get_tr_window_time() const;
  double get_agc_signal_read_delay() const;
  double get_ringbuffer_size() const;
  std::vector<size_t> get_receive_channels() const;
  std::vector<size_t> get_transmit_channels() const;
  std::string get_driver_to_radctrl_identity() const;
  std::string get_driver_to_dsp_identity() const;
  std::string get_driver_to_brian_identity() const;
  std::string get_router_address() const;
  std::string get_radctrl_to_driver_identity() const;
  std::string get_dsp_to_driver_identity() const;
  std::string get_brian_to_driver_identity() const;
  std::string get_ringbuffer_name() const;

 private:
  std::string devices_;
  std::string clk_addr_;
  std::string tx_subdev_;
  std::vector<size_t> receive_channels_;
  std::vector<size_t> transmit_channels_;
  std::string main_rx_subdev_;
  std::string intf_rx_subdev_;
  std::string pps_;
  std::string ref_;
  std::string cpu_;
  std::string otw_;
  std::string gpio_bank_high_;
  std::string gpio_bank_low_;
  double tr_window_time_;
  double agc_signal_read_delay_;
  double ringbuffer_size_bytes_;
  uint32_t atr_rx_;
  uint32_t atr_tx_;
  uint32_t atr_xx_;
  uint32_t atr_0x_;
  uint32_t agc_st_;
  uint32_t lo_pwr_;
  uint32_t test_mode_;
  std::string router_address_;
  std::string driver_to_radctrl_identity_;
  std::string driver_to_dsp_identity_;
  std::string driver_to_brian_identity_;
  std::string radctrl_to_driver_identity_;
  std::string dsp_to_driver_identity_;
  std::string brian_to_driver_identity_;
  std::string ringbuffer_name_;
};

#endif  // SRC_USRP_DRIVERS_UTILS_DRIVEROPTIONS_HPP_
