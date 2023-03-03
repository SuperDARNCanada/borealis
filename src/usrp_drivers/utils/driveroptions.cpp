/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include "options.hpp"
#include "driveroptions.hpp"

/**
 * @brief      Extracts the relevant driver options from the config into class variables.
 */
DriverOptions::DriverOptions() {
    Options::parse_config_file();

    devices_ = config_pt.get<std::string>("device_options");

    auto n200_list = config_pt.get_child("n200s");
    // These maps are sorted by their keys (device number / int antenna number)
    std::map<uint32_t, std::string> devices_map;    // Maps device number to IP address
    std::map<uint32_t, bool> rx_map;                // Maps device number to rx flag
    std::map<uint32_t, bool> tx_map;                // Maps device number to tx flag
    std::map<uint32_t, uint32_t> int_antenna_map;   // Maps interferometer antenna number to device number

    // Get number of physical antennas
    main_antenna_count_ = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("main_antenna_count"));
    interferometer_antenna_count_ = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("interferometer_antenna_count"));

    // Iterate through all N200s in the json array
    for (auto n200 = n200_list.begin(); n200 != n200_list.end(); n200++)
    {
        std::string addr = "";
        bool rx = false;
        bool tx = false;
        bool rx_int = false;
        std::string main_antenna = "";
        std::string interferometer_antenna = "";
        // Iterate through all N200 parameters and store them in variables
        for (auto iter = n200->second.begin(); iter != n200->second.end(); iter++)
        {
            auto param = iter->first;
            if (param.compare("addr") == 0) {
                addr = iter->second.data();
            }
            else if (param.compare("rx") == 0) {
                rx = (iter->second.data().compare("true") == 0);
            }
            else if (param.compare("tx") == 0) {
                tx = (iter->second.data().compare("true") == 0);
            }
            else if (param.compare("rx_int") == 0) {
                rx_int = (iter->second.data().compare("true") == 0);
            }
            else if (param.compare("main_antenna") == 0) {
                main_antenna = iter->second.data();
            }
            else if (param.compare("interferometer_antenna") == 0) {
                interferometer_antenna = iter->second.data();
            }
            else {
                throw std::invalid_argument("Invalid N200 parameter in config file");
            }
        }

        // If current n200 is transmitting, receiving, or receiving from interferometer, add to devices
        if (tx || rx || rx_int)
        {
            // Get device number. Devices are sorted by the main antenna they are connected to
            auto device_num = boost::lexical_cast<uint32_t>(main_antenna);

            // Add the address, tx flag, and rx flag to the respective dictionaries keyed with the device num
            devices_map[device_num] = addr;
            tx_map[device_num] = tx;
            rx_map[device_num] = rx;

            // If N200 has interferometer, map device number to interferometer antenna number
            if (rx_int) {
                auto int_antenna_num = boost::lexical_cast<uint32_t>(interferometer_antenna);
                int_antenna_map[int_antenna_num] = device_num;
            }
        }
    }

    // To ensure device numbers follow UHD conventions (0 to N in steps of 1),
    // the addr_idx must be mapped to the device number to get the correct
    // addr_idx for the interferometer antennas.
    auto addr_idx = 0;
    std::map<uint32_t,uint32_t> device_num_to_addr_idx;
    // Loop through sorted list of N200s and create devices_ string
    std::string ma_recv_str = "";
    std::string ma_tx_str = "";
    std::string ma_channel_str = "";
    for (auto element : devices_map) {
        auto device_num = element.first;
        device_num_to_addr_idx[device_num] = addr_idx;  // Store conversion for interferometer use

        devices_ = devices_ + ",addr" + std::to_string(addr_idx) + "=" + devices_map[device_num];
        if (rx_map[device_num]) {
            ma_recv_str = ma_recv_str + std::to_string(addr_idx*2) + ",";
        }
        if (tx_map[device_num]) {
            ma_tx_str = ma_tx_str + std::to_string(addr_idx) + ",";
        }
        if (rx_map[device_num] || tx_map[device_num]) {
            ma_channel_str = ma_channel_str + std::to_string(device_num) + ",";
        }
        addr_idx++;
    }

    // Interferometer antenna
    std::string ia_recv_str = "";       // Interferometer receive channel string
    std::string ia_channel_str = "";    // Interferometer antenna string
    for (auto element : int_antenna_map) {
        auto intf_antenna_num = element.first;
        auto device_num = element.second;
        auto addr_idx = device_num_to_addr_idx[device_num]; // Get correct address index
        ia_recv_str = ia_recv_str + std::to_string(2*addr_idx + 1) + ",";
        ia_channel_str = ia_channel_str + std::to_string(intf_antenna_num) + ",";
    }

    // Remove trailing comma from channel strings
    ma_recv_str.pop_back();
    ma_tx_str.pop_back();
    ma_channel_str.pop_back();
    ia_recv_str.pop_back();
    ia_channel_str.pop_back();

    auto total_recv_chs_str = ma_recv_str + "," + ia_recv_str;

    clk_addr_ = config_pt.get<std::string>("gps_octoclock_addr");

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

    receive_channels_ = make_channels(total_recv_chs_str);
    transmit_channels_ = make_channels(ma_tx_str);
    main_antennas_ = make_channels(ma_channel_str);
    interferometer_antennas_ = make_channels(ia_channel_str);

    router_address_ = config_pt.get<std::string>("router_address");
    ringbuffer_name_ = config_pt.get<std::string>("ringbuffer_name");
    ringbuffer_size_bytes_ = boost::lexical_cast<double>(
                                    config_pt.get<std::string>("ringbuffer_size_bytes"));

    std::string driver_to_radctrl_identity_ = "DRIVER_RADCTRL_IDEN";
    std::string driver_to_dsp_identity_ = "DRIVER_DSP_IDEN";
    std::string driver_to_brian_identity_ = "DRIVER_BRIAN_IDEN";
    std::string radctrl_to_driver_identity_ = "RADCTRL_DRIVER_IDEN";
    std::string dsp_to_driver_identity_ = "DSP_DRIVER_IDEN";
    std::string brian_to_driver_identity_ = "BRIAN_DRIVER_IDEN";
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
 * @brief      Gets all antennas connected to N200s
 *
 * @return     The list of antennas connected to N200s
 */
std::vector<size_t> DriverOptions::get_main_antennas() const
{
    return main_antennas_;
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
 * @brief      Gets all interferometer antennas connected to N200s
 *
 * @return     The list of interferometer antennas connected to N200s
 */
std::vector<size_t> DriverOptions::get_interferometer_antennas() const
{
    return interferometer_antennas_;
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
