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
    clk_addr_ = config_pt.get<std::string>("gps_octoclock_addr");
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

    std::stringstream ss;

    ss << std::hex << config_pt.get<std::string>("atr_rx");
    ss >> atr_rx_;
    ss.clear();

    ss << std::hex << config_pt.get<std::string>("atr_tx");
    ss >> atr_tx_;
    ss.clear();

    ss << std::hex << config_pt.get<std::string>("atr_xx");
    ss >> atr_xx_;
    ss.clear();

    ss << std::hex << config_pt.get<std::string>("atr_0x");
    ss >> atr_0x_;
    ss.clear();

    ss << std::hex << config_pt.get<std::string>("lo_pwr");
    ss >> lo_pwr_;
    ss.clear();

    ss << std::hex << config_pt.get<std::string>("agc_st");
    ss >> agc_st_;
    ss.clear();


    tr_window_time_ = boost::lexical_cast<double>(
                                config_pt.get<std::string>("tr_window_time"));
    agc_signal_read_delay_ = boost::lexical_cast<double>(
                                config_pt.get<std::string>("agc_signal_read_delay"));
    main_antenna_count_ = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("main_antenna_count"));
    interferometer_antenna_count_ = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("interferometer_antenna_count"));

    auto make_channels = [&](std::string chs){

        std::stringstream ss(chs);

        std::vector<size_t> channels;
        while (ss.good()) {
            std::string s;
            std::getline(ss, s, ',');
            channels.push_back(boost::lexical_cast<size_t>(s));
        }

        return channels;
    };

    auto ma_recv_str = config_pt.get<std::string>("main_antenna_usrp_rx_channels");
    auto ia_recv_str = config_pt.get<std::string>("interferometer_antenna_usrp_rx_channels");
    auto total_recv_chs_str = ma_recv_str + "," + ia_recv_str;

    auto ma_tx_str = config_pt.get<std::string>("main_antenna_usrp_tx_channels");

    receive_channels_ = make_channels(total_recv_chs_str);
    transmit_channels_ = make_channels(ma_tx_str);

    router_address_ = config_pt.get<std::string>("router_address");
    driver_to_radctrl_identity_ = config_pt.get<std::string>("driver_to_radctrl_identity");
    driver_to_dsp_identity_ = config_pt.get<std::string>("driver_to_dsp_identity");
    driver_to_brian_identity_ = config_pt.get<std::string>("driver_to_brian_identity");
    driver_to_mainaffinity_identity_ = config_pt.get<std::string>("driver_to_mainaffinity_identity");
    driver_to_txaffinity_identity_ = config_pt.get<std::string>("driver_to_txaffinity_identity");
    driver_to_rxaffinity_identity_ = config_pt.get<std::string>("driver_to_rxaffinity_identity");
    radctrl_to_driver_identity_ = config_pt.get<std::string>("radctrl_to_driver_identity");
    dsp_to_driver_identity_ = config_pt.get<std::string>("dsp_to_driver_identity");
    brian_to_driver_identity_ = config_pt.get<std::string>("brian_to_driver_identity");
    mainaffinity_to_driver_identity_ = config_pt.get<std::string>("mainaffinity_to_driver_identity");
    txaffinity_to_driver_identity_ = config_pt.get<std::string>("txaffinity_to_driver_identity");
    rxaffinity_to_driver_identity_ = config_pt.get<std::string>("rxaffinity_to_driver_identity");
    ringbuffer_name_ = config_pt.get<std::string>("ringbuffer_name");
    ringbuffer_size_bytes_ = boost::lexical_cast<double>(
                                    config_pt.get<std::string>("ringbuffer_size_bytes"));
}

std::string DriverOptions::get_device_args() const
{
    return devices_;
}

std::string DriverOptions::get_clk_addr() const
{
    return clk_addr_;
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

uint32_t DriverOptions::get_atr_rx() const
{
    return atr_rx_;
}

uint32_t DriverOptions::get_atr_tx() const
{
    return atr_tx_;
}

uint32_t DriverOptions::get_atr_xx() const
{
    return atr_xx_;
}

uint32_t DriverOptions::get_atr_0x() const
{
    return atr_0x_;

}

uint32_t DriverOptions::get_lo_pwr() const
{
    return lo_pwr_;
}

uint32_t DriverOptions::get_agc_st() const
{
    return agc_st_;
}

double DriverOptions::get_tr_window_time() const
{
    return tr_window_time_;
}

double DriverOptions::get_agc_signal_read_delay() const
{
    return agc_signal_read_delay_;
}

uint32_t DriverOptions::get_main_antenna_count() const
{
    return main_antenna_count_;
}

uint32_t DriverOptions::get_interferometer_antenna_count() const
{
    return interferometer_antenna_count_;
}

double DriverOptions::get_ringbuffer_size() const
{
    return ringbuffer_size_bytes_;
}

std::vector<size_t> DriverOptions::get_receive_channels() const
{
    return receive_channels_;
}

std::vector<size_t> DriverOptions::get_transmit_channels() const
{
    return transmit_channels_;
}

std::string DriverOptions::get_driver_to_radctrl_identity() const
{
    return driver_to_radctrl_identity_;
}

std::string DriverOptions::get_driver_to_dsp_identity() const
{
    return driver_to_dsp_identity_;
}

std::string DriverOptions::get_driver_to_brian_identity() const
{
    return driver_to_brian_identity_;
}

std::string DriverOptions::get_router_address() const
{
    return router_address_;
}

std::string DriverOptions::get_radctrl_to_driver_identity() const
{
    return radctrl_to_driver_identity_;
}


std::string DriverOptions::get_dsp_to_driver_identity() const
{
    return dsp_to_driver_identity_;
}


std::string DriverOptions::get_brian_to_driver_identity() const
{
    return brian_to_driver_identity_;
}

std::string DriverOptions::get_ringbuffer_name() const
{
    return ringbuffer_name_;
}

std::string DriverOptions::get_driver_to_mainaffinity_identity() const
{
    return driver_to_mainaffinity_identity_;
}

std::string DriverOptions::get_driver_to_txaffinity_identity() const
{
    return driver_to_txaffinity_identity_;
}

std::string DriverOptions::get_driver_to_rxaffinity_identity() const
{
    return driver_to_rxaffinity_identity_;
}

std::string DriverOptions::get_mainaffinity_to_driver_identity() const
{
    return mainaffinity_to_driver_identity_;
}

std::string DriverOptions::get_txaffinity_to_driver_identity() const
{
    return txaffinity_to_driver_identity_;
}

std::string DriverOptions::get_rxaffinity_to_driver_identity() const
{
    return rxaffinity_to_driver_identity_;
}

