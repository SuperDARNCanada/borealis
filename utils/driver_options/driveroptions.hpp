/*Copyright 2016 SuperDARN*/
#ifndef DRIVEROPTIONS_H
#define DRIVEROPTIONS_H

#include <stdint.h>
#include <string>
#include "utils/options/options.hpp"

class DriverOptions: public Options {
 public:
        explicit DriverOptions();
        double get_tx_rate() const;
        double get_rx_rate() const;
        std::string get_device_args() const;
        std::string get_tx_subdev() const;
        std::string get_main_rx_subdev() const;
        std::string get_interferometer_rx_subdev() const;
        std::string get_pps() const;
        std::string get_ref() const;
        std::string get_cpu() const;
        std::string get_otw() const;
        std::string get_gpio_bank() const;
        uint32_t get_scope_sync_mask() const;
        uint32_t get_atten_mask() const;
        uint32_t get_tr_mask() const;
        double get_atten_window_time_start() const;
        double get_atten_window_time_end() const;
        double get_tr_window_time() const;
        uint32_t get_main_antenna_count() const;
        uint32_t get_interferometer_antenna_count() const;
        double get_ringbuffer_size() const;
        std::vector<size_t> get_receive_channels() const;
        std::string get_driver_to_radctrl_identity() const;
        std::string get_driver_to_dsp_identity() const;
        std::string get_driver_to_brian_identity() const;
        std::string get_router_address() const;
        std::string get_radctrl_to_driver_identity() const;
        std::string get_dsp_to_driver_identity() const;
        std::string get_brian_to_driver_identity() const;
        std::string get_ringbuffer_name() const;
        std::string get_driver_to_mainaffinity_identity() const;
        std::string get_driver_to_txaffinity_identity() const;
        std::string get_driver_to_rxaffinity_identity() const;
        std::string get_mainaffinity_to_driver_identity() const;
        std::string get_txaffinity_to_driver_identity() const;
        std::string get_rxaffinity_to_driver_identity() const;
 private:

        std::string devices_;
        std::string tx_subdev_;
        std::vector<size_t> receive_channels_;
        std::string main_rx_subdev_;
        std::string interferometer_rx_subdev_;
        std::string pps_;
        std::string ref_;
        double tx_sample_rate_;
        double rx_sample_rate_;
        std::string cpu_;
        std::string otw_;
        std::string gpio_bank_;
        uint32_t scope_sync_mask_;
        uint32_t atten_mask_;
        uint32_t tr_mask_;
        double atten_window_time_start_;
        double atten_window_time_end_;
        double tr_window_time_;
        uint32_t main_antenna_count_;
        uint32_t interferometer_antenna_count_;
        double ringbuffer_size_bytes_;
        std::string router_address_;
        std::string driver_to_radctrl_identity_;
        std::string driver_to_dsp_identity_;
        std::string driver_to_brian_identity_;
        std::string radctrl_to_driver_identity_;
        std::string dsp_to_driver_identity_;
        std::string brian_to_driver_identity_;
        std::string ringbuffer_name_;
        std::string driver_to_mainaffinity_identity_;
        std::string driver_to_txaffinity_identity_;
        std::string driver_to_rxaffinity_identity_;
        std::string mainaffinity_to_driver_identity_;
        std::string txaffinity_to_driver_identity_;
        std::string rxaffinity_to_driver_identity_;


};

#endif
