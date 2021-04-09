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


/**
 * @brief      Extracts the relevant driver options from the config into class variables.
 */
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
    gpio_bank_high_ = config_pt.get<std::string>("gpio_bank_high");
    gpio_bank_low_ = config_pt.get<std::string>("gpio_bank_low");

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

    ss << std::hex << config_pt.get<std::string>("tst_md");
    ss >> test_mode_;
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
    radctrl_to_driver_identity_ = config_pt.get<std::string>("radctrl_to_driver_identity");
    dsp_to_driver_identity_ = config_pt.get<std::string>("dsp_to_driver_identity");
    brian_to_driver_identity_ = config_pt.get<std::string>("brian_to_driver_identity");
    ringbuffer_name_ = config_pt.get<std::string>("ringbuffer_name");
    ringbuffer_size_bytes_ = boost::lexical_cast<double>(
                                    config_pt.get<std::string>("ringbuffer_size_bytes"));
}

/**
 * @brief      Gets the device arguments.
 *
 * @return     The device arguments.
 */
std::string DriverOptions::get_device_args() const
{
    return devices_;
}

/**
 * @brief      Gets the clock address.
 *
 * @return     The clock address.
 */
std::string DriverOptions::get_clk_addr() const
{
    return clk_addr_;
}

/**
 * @brief      Gets the USRP subdev for transmit bank.
 *
 * @return     The transmit subdev.
 */
std::string DriverOptions::get_tx_subdev() const
{
    return tx_subdev_;
}

/**
 * @brief      Gets the USRP receive subdev for main antenna bank.
 *
 * @return     The main receive subdev.
 */
std::string DriverOptions::get_main_rx_subdev() const
{
    return main_rx_subdev_;
}

/**
 * @brief      Gets the USRP receive subdev for interferometer antenna bank.
 *
 * @return     The interferometer receive subdev.
 */
std::string DriverOptions::get_interferometer_rx_subdev() const
{
    return interferometer_rx_subdev_;
}

/**
 * @brief      Gets the pps source.
 *
 * @return     The pps source.
 */
std::string DriverOptions::get_pps() const
{
    return pps_;
}

/**
 * @brief      Gets the 10 MHz reference source.
 *
 * @return     The 10 MHz reference source.
 */
std::string DriverOptions::get_ref() const
{
    return ref_;
}

/**
 * @brief      Gets the USRP cpu data type.
 *
 * @return     The cpu data type.
 */
std::string DriverOptions::get_cpu() const
{
    return cpu_;
}

/**
 * @brief      Gets the USRP otw format.
 *
 * @return     The USRP otw format.
 */
std::string DriverOptions::get_otw() const
{
    return otw_;
}

/**
 * @brief      Gets the active high gpio bank.
 *
 * @return     The active high gpio bank.
 */
std::string DriverOptions::get_gpio_bank_high() const
{
    return gpio_bank_high_;
}

/**
 * @brief      Gets the active low gpio bank.
 *
 * @return     The active low gpio bank.
 */
std::string DriverOptions::get_gpio_bank_low() const
{
    return gpio_bank_low_;
}

/**
 * @brief      Gets the RX atr bank.
 *
 * @return     The RX atr bank.
 */
uint32_t DriverOptions::get_atr_rx() const
{
    return atr_rx_;
}

/**
 * @brief      Gets the TX atr bank.
 *
 * @return     The TX atr bank.
 */
uint32_t DriverOptions::get_atr_tx() const
{
    return atr_tx_;
}

/**
 * @brief      Gets the duplex atr bank.
 *
 * @return     The duplex atr bank.
 */
uint32_t DriverOptions::get_atr_xx() const
{
    return atr_xx_;
}

/**
 * @brief      Gets the idle atr bank.
 *
 * @return     The idle atr bank.
 */
uint32_t DriverOptions::get_atr_0x() const
{
    return atr_0x_;

}

/**
 * @brief      Gets the low power input bank.
 *
 * @return     The low power input bank.
 */
uint32_t DriverOptions::get_lo_pwr() const
{
    return lo_pwr_;
}


/**
 * @brief      Gets the agc status input bank.
 *
 * @return     The agc status bank.
 */
uint32_t DriverOptions::get_agc_st() const
{
    return agc_st_;
}

/**
 * @brief      Gets the test mode input bank.
 *
 * @return     The test mode input bank.
 */
uint32_t DriverOptions::get_test_mode() const
{
    return test_mode_;
}

/**
 * @brief      Gets the tr window time.
 *
 * @return     The tr window time.
 */
double DriverOptions::get_tr_window_time() const
{
    return tr_window_time_;
}

/**
 * @brief      Gets the agc status signal read delay.
 *
 * @return     The agc status signal read delay.
 */
double DriverOptions::get_agc_signal_read_delay() const
{
    return agc_signal_read_delay_;
}

/**
 * @brief      Gets the main antenna count.
 *
 * @return     The main antenna count.
 */
uint32_t DriverOptions::get_main_antenna_count() const
{
    return main_antenna_count_;
}

/**
 * @brief      Gets the interferometer antenna count.
 *
 * @return     The interferometer antenna count.
 */
uint32_t DriverOptions::get_interferometer_antenna_count() const
{
    return interferometer_antenna_count_;
}

/**
 * @brief      Gets the ringbuffer size.
 *
 * @return     The ringbuffer size.
 */
double DriverOptions::get_ringbuffer_size() const
{
    return ringbuffer_size_bytes_;
}

/**
 * @brief      Gets the all USRP receive channels.
 *
 * @return     The USRP receive channels.
 */
std::vector<size_t> DriverOptions::get_receive_channels() const
{
    return receive_channels_;
}

/**
 * @brief      Gets the USRP transmit channels.
 *
 * @return     The USRP transmit channels.
 */
std::vector<size_t> DriverOptions::get_transmit_channels() const
{
    return transmit_channels_;
}

/**
 * @brief      Gets the driver to radctrl identity.
 *
 * @return     The driver to radctrl identity.
 */
std::string DriverOptions::get_driver_to_radctrl_identity() const
{
    return driver_to_radctrl_identity_;
}

/**
 * @brief      Gets the driver to dsp identity.
 *
 * @return     The driver to dsp identity.
 */
std::string DriverOptions::get_driver_to_dsp_identity() const
{
    return driver_to_dsp_identity_;
}

/**
 * @brief      Gets the driver to brian identity.
 *
 * @return     The driver to brian identity.
 */
std::string DriverOptions::get_driver_to_brian_identity() const
{
    return driver_to_brian_identity_;
}

/**
 * @brief      Gets the router address.
 *
 * @return     The router address.
 */
std::string DriverOptions::get_router_address() const
{
    return router_address_;
}

/**
 * @brief      Gets the radctrl to driver identity.
 *
 * @return     The radctrl to driver identity.
 */
std::string DriverOptions::get_radctrl_to_driver_identity() const
{
    return radctrl_to_driver_identity_;
}


/**
 * @brief      Gets the dsp to driver identity.
 *
 * @return     The dsp to driver identity.
 */
std::string DriverOptions::get_dsp_to_driver_identity() const
{
    return dsp_to_driver_identity_;
}


/**
 * @brief      Gets the brian to driver identity.
 *
 * @return     The brian to driver identity.
 */
std::string DriverOptions::get_brian_to_driver_identity() const
{
    return brian_to_driver_identity_;
}

/**
 * @brief      Gets the ringbuffer name.
 *
 * @return     The ringbuffer name.
 */
std::string DriverOptions::get_ringbuffer_name() const
{
    return ringbuffer_name_;
}
