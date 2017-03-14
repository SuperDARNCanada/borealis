/*Copyright 2016 SuperDARN*/
#ifndef USRP_H
#define USRP_H

#include <uhd/usrp/multi_usrp.hpp>
#include "utils/driver_options/driveroptions.hpp"

class USRP{
 public:
        explicit USRP(std::shared_ptr<DriverOptions> driver_options);
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
        uhd::usrp::multi_usrp::sptr usrp_;
        std::string gpio_bank_;
        uint32_t mboard_;
        uint32_t scope_sync_mask_;
        uint32_t atten_mask_;
        uint32_t tr_mask_;
        std::vector<size_t> receive_channels;

        std::vector<size_t> create_receive_channels(uint32_t main_antenna_count,
                                                        uint32_t interferometer_antenna_count);

};

class TXMetadata{
 public:
        uhd::tx_metadata_t md_;

        TXMetadata();
        uhd::tx_metadata_t get_md();
        void set_start_of_burst(bool start_of_burst);
        void set_end_of_burst(bool end_of_burst);
        void set_has_time_spec(bool has_time_spec);
        void set_time_spec(uhd::time_spec_t time_spec);

};

class RXMetadata{
 public:
        uhd::rx_metadata_t md_;

        RXMetadata() = default;
        uhd::rx_metadata_t& get_md();
        bool get_end_of_burst();
        uhd::rx_metadata_t::error_code_t get_error_code();
        size_t get_fragment_offset();
        bool get_has_time_spec();
        bool get_out_of_sequence();
        bool get_start_of_burst();
        uhd::time_spec_t get_time_spec();
};

#endif

