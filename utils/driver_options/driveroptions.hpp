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

 private:
        std::string devices;
        std::string tx_subdev;
        std::string main_rx_subdev;
        std::string interferometer_rx_subdev;
        std::string pps;
        std::string ref;
        double tx_sample_rate;
        double rx_sample_rate;
        std::string cpu;
        std::string otw;
        std::string gpio_bank;
        uint32_t scope_sync_mask;
        uint32_t atten_mask;
        uint32_t tr_mask;
        double atten_window_time_start;
        double atten_window_time_end;
        double tr_window_time;
        uint32_t main_antenna_count;
        uint32_t interferometer_antenna_count;
};

#endif
