/*Copyright 2016 SuperDARN*/
#ifndef DRIVEROPTIONS_H
#define DRIVEROPTIONS_H

#include <stdint.h>
#include <string>
#include "utils/options/options.hpp"

class DriverOptions: public Options {
 public:
        std::string devices;
        std::string tx_subdev;
        std::string rx_subdev;
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
        std::vector<uint32_t> receive_channels;

        DriverOptions();
        double get_tx_rate();
        double get_rx_rate();
        std::string get_device_args();
        std::string get_tx_subdev();
        std::string get_rx_subdev();
        std::string get_pps();
        std::string get_ref();
        std::string get_cpu();
        std::string get_otw();
        std::string get_gpio_bank();
        uint32_t get_scope_sync_mask();
        uint32_t get_atten_mask();
        uint32_t get_tr_mask();
        double get_atten_window_time_start();
        double get_atten_window_time_end();
        double get_tr_window_time();
        std::vector<uint32_t> get_receive_channels();
};

#endif
