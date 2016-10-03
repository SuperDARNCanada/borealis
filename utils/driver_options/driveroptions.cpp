/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <string>
#include "utils/options/options.hpp"
#include "utils/driver_options/driveroptions.hpp"


DriverOptions::DriverOptions() {
    Options::parse_config_file();

    devices = config_pt.get<std::string>("devices");
    /*Remove whitespace/new lines from device list*/
    boost::remove_erase_if (devices, boost::is_any_of(" \n"));

    tx_subdev = config_pt.get<std::string>("tx_subdev");
    rx_subdev = config_pt.get<std::string>("rx_subdev");
    pps = config_pt.get<std::string>("pps");
    ref = config_pt.get<std::string>("ref");
    tx_sample_rate = config_pt.get<double>("tx_sample_rate");
    rx_sample_rate = config_pt.get<double>("rx_sample_rate");
    cpu = config_pt.get<std::string>("cpu");
    otw = config_pt.get<std::string>("overthewire"); 
}

double DriverOptions::get_tx_rate() {
    return tx_sample_rate;
}

double DriverOptions::get_rx_rate() {
    return rx_sample_rate;
}

std::string DriverOptions::get_device_args() {
    return devices;
}

std::string DriverOptions::get_tx_subdev() {
    return tx_subdev;
}

std::string DriverOptions::get_rx_subdev() {
    return rx_subdev;
}

std::string DriverOptions::get_pps() {
    return pps;
}

std::string DriverOptions::get_ref() {
    return ref;
}

std::string DriverOptions::get_cpu() {
    return cpu;
}

std::string DriverOptions::get_otw() {
    return otw;
}
