/*Copyright 2016 SuperDARN*/
#ifndef DRIVEROPTIONS_H
#define DRIVEROPTIONS_H

#include <stdint.h>
#include <string>
#include "utils/options/options.hpp"

class DriverOptions: public Options {
 public:
        explicit DriverOptions();
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
        uint32_t get_main_antenna_count() const;
        uint32_t get_interferometer_antenna_count() const;
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
        std::string interferometer_rx_subdev_;
        std::string pps_;
        std::string ref_;
        std::string cpu_;
        std::string otw_;
        std::string gpio_bank_high_;
        std::string gpio_bank_low_;
        double tr_window_time_;
        double agc_signal_read_delay_;
        uint32_t main_antenna_count_;
        uint32_t interferometer_antenna_count_;
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

#endif
