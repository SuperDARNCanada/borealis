/*
Copyright 2016 SuperDARN

See LICENSE for details.

  @file usrp.hpp
  This file contains class declarations for ease of use of USRP related features.

*/
#ifndef USRP_H
#define USRP_H

#include <uhd/usrp/multi_usrp.hpp>
#include "utils/driver_options/driveroptions.hpp"


/**
 * @brief      Contains an abstract wrapper for the USRP object.
 */
class USRP{
  public:
    explicit USRP(const DriverOptions& driver_options);
    void set_usrp_clock_source(std::string source);
    void set_tx_subdev(std::string tx_subdev);
    void set_tx_rate(double tx_rate);
    double get_tx_rate();
    void set_tx_center_freq(double freq, std::vector<size_t> chs);
    void set_main_rx_subdev(std::string main_subdev);
    void set_interferometer_rx_subdev(std::string interferometer_subdev,
                                        uint32_t interferometer_antenna_count);
    void set_rx_rate(double rx_rate);
    void set_rx_center_freq(double freq, std::vector<size_t> chs);
    void set_time_source(std::string source);
    void check_ref_locked();
    void set_gpio(uint32_t mask, std::string gpio_bank, size_t mboard);
    void set_gpio(uint32_t mask);
    void set_scope_sync();
    void set_atten();
    void set_tr();
    void clear_gpio(uint32_t mask, std::string gpio_bank, size_t mboard);
    void clear_gpio(uint32_t mask);
    void clear_scope_sync();
    void clear_atten();
    void clear_tr();
    std::vector<size_t> get_receive_channels();
    uhd::usrp::multi_usrp::sptr get_usrp();
    std::string to_string(std::vector<size_t> chs);

  private:
    //! A shared pointer to a new multi-USRP device.
    uhd::usrp::multi_usrp::sptr usrp_;

    //! A string representing what GPIO bank to use on the USRPs.
    std::string gpio_bank_;

    //! The motherboard for which to use GPIOs for high speed I/O.
    uint32_t mboard_;

    //! The bitmask to use for the scope sync GPIO.
    uint32_t scope_sync_mask_;

    //! The bitmask to use for the attenuator GPIO.
    uint32_t atten_mask_;

    //! The bitmask to use for the TR GPIO.
    uint32_t tr_mask_;

    //! This is the reordered USRP receive channels.
    std::vector<size_t> receive_channels;

    std::vector<size_t> create_receive_channels(uint32_t main_antenna_count,
                uint32_t interferometer_antenna_count);

};

/**
 * @brief      Wrapper for the USRP TX metadata object.
 */
class TXMetadata{
  public:
    TXMetadata();
    uhd::tx_metadata_t get_md();
    void set_start_of_burst(bool start_of_burst);
    void set_end_of_burst(bool end_of_burst);
    void set_has_time_spec(bool has_time_spec);
    void set_time_spec(uhd::time_spec_t time_spec);

  private:
    //! A raw USRP TX metadata object.
    uhd::tx_metadata_t md_;

};

/**
 * @brief      Wrapper for the USRP RX metadata object. // REVIEW #1 what is this used for .. more explanation
 */
class RXMetadata{
  public:
    RXMetadata() = default;  // REVIEW #1 what does this do?
    uhd::rx_metadata_t& get_md();
    bool get_end_of_burst();
    uhd::rx_metadata_t::error_code_t get_error_code();
    size_t get_fragment_offset();
    bool get_has_time_spec();
    bool get_out_of_sequence();
    bool get_start_of_burst();
    uhd::time_spec_t get_time_spec(); // REVIEW #6 TODO: add getter for more_fragments boolean

  private:
    //! A raw USRP RX metadata object.
    uhd::rx_metadata_t md_;
};

#endif

