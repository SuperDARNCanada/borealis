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
    boost::remove_erase_if (devices, boost::is_any_of(" \n"));// REVIEW #0 Do you need to also remove \r \f \t \v for example?

    tx_subdev = config_pt.get<std::string>("tx_subdev");
    main_rx_subdev = config_pt.get<std::string>("main_rx_subdev");// REVIEW #7 Talk about the subdevs/ etc in documentation since the USRP documentation is not very straightforward.
    interferometer_rx_subdev = config_pt.get<std::string>("interferometer_rx_subdev");
    pps = config_pt.get<std::string>("pps");// REVIEW #7 Document all the options available to user in config.ini file - with examples
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

double DriverOptions::get_tx_rate() const
{
    return tx_sample_rate;
}

double DriverOptions::get_rx_rate() const
{
    return rx_sample_rate;
}

std::string DriverOptions::get_device_args() const
{
    return devices;
}

std::string DriverOptions::get_tx_subdev() const
{
    return tx_subdev;
}

std::string DriverOptions::get_main_rx_subdev() const
{
    return main_rx_subdev;
}

std::string DriverOptions::get_interferometer_rx_subdev() const
{
    return interferometer_rx_subdev;
}

std::string DriverOptions::get_pps() const
{
    return pps;
}

std::string DriverOptions::get_ref() const
{
    return ref;
}

std::string DriverOptions::get_cpu() const
{
    return cpu;
}

std::string DriverOptions::get_otw() const
{
    return otw;
}

std::string DriverOptions::get_gpio_bank() const
{
    return gpio_bank;
}

uint32_t DriverOptions::get_scope_sync_mask() const
{
    return scope_sync_mask;
}

uint32_t DriverOptions::get_atten_mask() const
{
    return atten_mask;
}

uint32_t DriverOptions::get_tr_mask() const
{
    return tr_mask;
}

double DriverOptions::get_atten_window_time_start() const
{
    return atten_window_time_start;
}

double DriverOptions::get_atten_window_time_end() const
{
    return atten_window_time_end;
}

double DriverOptions::get_tr_window_time() const
{
    return tr_window_time;
}

uint32_t DriverOptions::get_main_antenna_count() const
{
    return main_antenna_count;
}

uint32_t DriverOptions::get_interferometer_antenna_count() const
{
    return interferometer_antenna_count;
}
