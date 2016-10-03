/*Copyright 2016 SuperDARN*/
#ifndef DRIVEROPTIONS_H
#define DRIVEROPTIONS_H

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
};

#endif
