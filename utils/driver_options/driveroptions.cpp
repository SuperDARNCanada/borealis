/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include "utils/options/options.hpp"
#include "utils/driver_options/driveroptions.hpp"


DriverOptions::DriverOptions() {
    Options::parse_config_file();

    devices_ = config_pt.get<std::string>("devices");
    /*Remove whitespace/new lines from device list*/
    boost::remove_erase_if (devices_, boost::is_any_of(" \n\f\t\v"));

    tx_subdev_ = config_pt.get<std::string>("tx_subdev");
    main_rx_subdev_ = config_pt.get<std::string>("main_rx_subdev");
    interferometer_rx_subdev_ = config_pt.get<std::string>("interferometer_rx_subdev");
    pps_ = config_pt.get<std::string>("pps");
    ref_ = config_pt.get<std::string>("ref");
    cpu_ = config_pt.get<std::string>("cpu");
    otw_ = config_pt.get<std::string>("overthewire");
    gpio_bank_ = config_pt.get<std::string>("gpio_bank");
    rx_sample_rate_ = boost::lexical_cast<double>(
                                config_pt.get<std::string>("rx_sample_rate"));
    tx_sample_rate_ = boost::lexical_cast<double>(
                                config_pt.get<std::string>("tx_sample_rate"));
    scope_sync_mask_ = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("scope_sync_mask"));
    atten_mask_ = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("atten_mask"));
    tr_mask_ = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("tr_mask"));
    atten_window_time_start_ = boost::lexical_cast<double>(
                                config_pt.get<std::string>("atten_window_time_start"));
    atten_window_time_end_ = boost::lexical_cast<double>(
                                config_pt.get<std::string>("atten_window_time_end"));
    tr_window_time_ = boost::lexical_cast<double>(
                                config_pt.get<std::string>("tr_window_time"));
    main_antenna_count_ = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("main_antenna_count"));
    interferometer_antenna_count_ = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("interferometer_antenna_count"));

    receive_channels_ = [&](){
        auto ma_str = config_pt.get<std::string>("main_antenna_usrp_channels");
        auto ia_str = config_pt.get<std::string>("interferometer_antenna_usrp_channels");
        auto total_chs_str = ma_str + "," + ia_str;

        std::stringstream ss(total_chs_str);

        std::vector<size_t> receive_channels;
        while (ss.good()) {
            std::string s;
            std::getline(ss, s, ',');
            receive_channels.push_back(boost::lexical_cast<size_t>(s));
        }

        return receive_channels;
    }();

    radar_control_to_driver_address_ = config_pt.get<std::string>
                                            ("radar_control_to_driver_address");
    driver_to_rx_dsp_address_ = config_pt.get<std::string>("driver_to_rx_dsp_address");

}

double DriverOptions::get_tx_rate() const
{
    return tx_sample_rate_;
}

double DriverOptions::get_rx_rate() const
{
    return rx_sample_rate_;
}

std::string DriverOptions::get_device_args() const
{
    return devices_;
}

std::string DriverOptions::get_tx_subdev() const
{
    return tx_subdev_;
}

std::string DriverOptions::get_main_rx_subdev() const
{
    return main_rx_subdev_;
}

std::string DriverOptions::get_interferometer_rx_subdev() const
{
    return interferometer_rx_subdev_;
}

std::string DriverOptions::get_pps() const
{
    return pps_;
}

std::string DriverOptions::get_ref() const
{
    return ref_;
}

std::string DriverOptions::get_cpu() const
{
    return cpu_;
}

std::string DriverOptions::get_otw() const
{
    return otw_;
}

std::string DriverOptions::get_gpio_bank() const
{
    return gpio_bank_;
}

uint32_t DriverOptions::get_scope_sync_mask() const
{
    return scope_sync_mask_;
}

uint32_t DriverOptions::get_atten_mask() const
{
    return atten_mask_;
}

uint32_t DriverOptions::get_tr_mask() const
{
    return tr_mask_;
}

double DriverOptions::get_atten_window_time_start() const
{
    return atten_window_time_start_;
}

double DriverOptions::get_atten_window_time_end() const
{
    return atten_window_time_end_;
}

double DriverOptions::get_tr_window_time() const
{
    return tr_window_time_;
}

uint32_t DriverOptions::get_main_antenna_count() const
{
    return main_antenna_count_;
}

uint32_t DriverOptions::get_interferometer_antenna_count() const
{
    return interferometer_antenna_count_;
}

std::vector<size_t> DriverOptions::get_receive_channels() const
{
    return receive_channels_;
}

std::string DriverOptions::get_radar_control_to_driver_address() const
{
    return radar_control_to_driver_address_;
}

std::string DriverOptions::get_driver_to_rx_dsp_address() const
{
    return driver_to_rx_dsp_address_;
}
