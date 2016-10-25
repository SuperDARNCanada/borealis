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
    gpio_bank = config_pt.get<std::string>("gpio_bank");
    scope_sync_mask = config_pt.get<uint32_t>("scope_sync_mask");
    atten_mask = config_pt.get<uint32_t>("atten_mask");
    tr_mask = config_pt.get<uint32_t>("tr_mask");
    atten_window_time_start = config_pt.get<double>("atten_window_time_start");
    atten_window_time_end = config_pt.get<double>("atten_window_time_end");
    tr_window_time = config_pt.get<double>("tr_window_time");
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

std::string DriverOptions::get_gpio_bank() {
    return gpio_bank;
}

uint32_t DriverOptions::get_scope_sync_mask() {
    return scope_sync_mask;
}

uint32_t DriverOptions::get_atten_mask() {
    return atten_mask;
}

uint32_t DriverOptions::get_tr_mask() {
    return tr_mask;
}

double DriverOptions::get_atten_window_time_start() {
    return atten_window_time_start;
}

double DriverOptions::get_atten_window_time_end() {
    return atten_window_time_end;
}

double DriverOptions::get_tr_window_time() {
    return tr_window_time;
}