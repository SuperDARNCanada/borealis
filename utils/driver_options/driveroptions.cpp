/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include "utils/options/options.hpp"
#include "utils/driver_options/driveroptions.hpp"


DriverOptions::DriverOptions() {
    Options::parse_config_file();

    devices = config_pt.get<std::string>("devices");
    /*Remove whitespace/new lines from device list*/
    boost::remove_erase_if (devices, boost::is_any_of(" \n"));

    tx_subdev = config_pt.get<std::string>("tx_subdev");
    main_rx_subdev = config_pt.get<std::string>("main_rx_subdev");
    interferometer_rx_subdev = config_pt.get<std::string>("interferometer_rx_subdev");
    pps = config_pt.get<std::string>("pps");
    ref = config_pt.get<std::string>("ref");
    cpu = config_pt.get<std::string>("cpu");
    otw = config_pt.get<std::string>("overthewire");
    gpio_bank = config_pt.get<std::string>("gpio_bank");
    rx_sample_rate = boost::lexical_cast<double>(
                                config_pt.get<std::string>("rx_sample_rate"));
    tx_sample_rate = boost::lexical_cast<double>(
                                config_pt.get<std::string>("tx_sample_rate"));
    scope_sync_mask = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("scope_sync_mask"));
    atten_mask = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("atten_mask"));
    tr_mask = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("tr_mask"));
    atten_window_time_start = boost::lexical_cast<double>(
                                config_pt.get<std::string>("atten_window_time_start"));
    atten_window_time_end = boost::lexical_cast<double>(
                                config_pt.get<std::string>("atten_window_time_end"));
    tr_window_time = boost::lexical_cast<double>(
                                config_pt.get<std::string>("tr_window_time"));
    main_antenna_count = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("main_antenna_count"));
    interferometer_antenna_count = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("interferometer_antenna_count"));
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

std::string DriverOptions::get_main_rx_subdev() {
    return main_rx_subdev;
}

std::string DriverOptions::get_interferometer_rx_subdev() {
    return interferometer_rx_subdev;
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

uint32_t DriverOptions::get_main_antenna_count() {
    return main_antenna_count;
}

uint32_t DriverOptions::get_interferometer_antenna_count() {
    return interferometer_antenna_count;
}