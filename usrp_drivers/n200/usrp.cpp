/*Copyright 2016 SuperDARN*/
#include <uhd/usrp/multi_usrp.hpp>
#include <memory>
#include <string>
#include <vector>
#include "usrp_drivers/n200/usrp.hpp"
#include "utils/driver_options/driveroptions.hpp"

USRP::USRP(std::shared_ptr<DriverOptions> driver_options) {
    mboard_ = 0;
    gpio_bank_ = driver_options->get_gpio_bank();
    scope_sync_mask_ = driver_options->get_scope_sync_mask();
    atten_mask_ = driver_options->get_atten_mask();
    tr_mask_ = driver_options->get_tr_mask();

    usrp_ = uhd::usrp::multi_usrp::make(driver_options->get_device_args());
    // Set first four GPIO on gpio_bank_ to output, the rest are input
    usrp_->set_gpio_attr(gpio_bank_, "DDR", 0x000F, 0xFFFF);
    set_usrp_clock_source(driver_options->get_ref());
    set_tx_subdev(driver_options->get_tx_subdev());
    //set_tx_rate(driver_options->get_tx_rate());
    set_rx_subdev(driver_options->get_rx_subdev());
    //set_rx_rate(driver_options->get_rx_rate());

    set_time_source(driver_options->get_pps());
    check_ref_locked();

}

void USRP::set_usrp_clock_source(std::string source) {
    usrp_->set_clock_source(source);
}

void USRP::set_tx_subdev(std::string tx_subdev) {
    usrp_->set_tx_subdev_spec(tx_subdev);
}

void USRP::set_tx_rate(double tx_rate) {
    usrp_->set_tx_rate(tx_rate);

    double actual_rate = usrp_->get_tx_rate();

    if (actual_rate != tx_rate) {
        /*TODO: something*/
    }
}

void USRP::set_tx_center_freq(double freq, std::vector<size_t> chs) {
    uhd::tune_request_t tune_request(freq);

    for(auto &channel : chs) {
        usrp_->set_tx_freq(tune_request, channel);

        double actual_freq = usrp_->get_tx_freq(channel);
        if (actual_freq != freq) {
            /*TODO: something*/
        }

    }

    /*boost::this_thread::sleep(boost::posix_time::seconds(1)); */
}

void USRP::set_rx_subdev(std::string rx_subdev) {
    usrp_->set_rx_subdev_spec(rx_subdev);
}

void USRP::set_rx_rate(double rx_rate) {
    usrp_->set_rx_rate(rx_rate);

    double actual_rate = usrp_->get_rx_rate();

    if (actual_rate != rx_rate) {
        /*TODO: something*/
    }
}

void USRP::set_rx_center_freq(double freq, std::vector<size_t> chs) {
    uhd::tune_request_t tune_request(freq);

    for(auto &channel : chs) {
        usrp_->set_rx_freq(tune_request, channel);

        double actual_freq = usrp_->get_rx_freq(channel);
        if (actual_freq != freq) {
            /*TODO: something*/
        }

    }

    /*boost::this_thread::sleep(boost::posix_time::seconds(1)); */
}        

void USRP::set_time_source(std::string source) {
    usrp_->set_time_source(source);
    usrp_->set_time_unknown_pps(uhd::time_spec_t(0.0));
}

void USRP::check_ref_locked() {
    size_t num_boards = usrp_->get_num_mboards();

    for(size_t i = 0; i < num_boards; i++) {
        std::vector<std::string> sensor_names;
        sensor_names = usrp_->get_mboard_sensor_names(i);
        if ((std::find(sensor_names.begin(), sensor_names.end(), "ref_locked") != sensor_names.end())) {
            uhd::sensor_value_t ref_locked = usrp_->get_mboard_sensor("ref_locked", i);
            /*
            TODO: something like this
            UHD_ASSERT_THROW(ref_locked.to_bool());
            */
        }

    }
}

void USRP::set_gpio(uint32_t mask, std::string gpio_bank, size_t mboard) {
  usrp_->set_gpio_attr(gpio_bank, "OUT", 0xFFFF, mask, mboard);
}

void USRP::set_gpio(uint32_t mask) {
  set_gpio(mask, gpio_bank_, mboard_);
}

void USRP::set_scope_sync() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0xFFFF, scope_sync_mask_, mboard_);
}

void USRP::set_atten() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0xFFFF, atten_mask_, mboard_);
}

void USRP::set_tr() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0xFFFF, tr_mask_, mboard_);
}

void USRP::clear_gpio(uint32_t mask, std::string gpio_bank, size_t mboard) {
  usrp_->set_gpio_attr(gpio_bank, "OUT", 0x0000, mask, mboard);
}

void USRP::clear_gpio(uint32_t mask) {
  set_gpio(mask, gpio_bank_, mboard_);
}

void USRP::clear_scope_sync() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0x0000, scope_sync_mask_, mboard_);
}

void USRP::clear_atten() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0x0000, atten_mask_, mboard_);
}

void USRP::clear_tr() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0x0000, tr_mask_, mboard_);
}

uhd::usrp::multi_usrp::sptr USRP::get_usrp(){
    return usrp_;
}

std::string USRP::to_string(std::vector<size_t> chs) {
    std::stringstream device_str;

    device_str << "Using device " << usrp_->get_pp_string() << std::endl
               << "TX rate " << usrp_->get_tx_rate()/1e6 << " Msps" << std::endl
               << "RX rate " << usrp_->get_rx_rate()/1e6 << " Msps" << std::endl;
                 

    for(auto &channel : chs) {
        device_str << "TX channel " << channel << " freq " 
                   << usrp_->get_tx_freq(channel) << " MHz" << std::endl;
    }

    for(auto &channel : chs) {
        device_str << "RX channel " << channel << " freq " 
                   << usrp_->get_tx_freq(channel) << " MHz" << std::endl;
    }

    return device_str.str();
                  
}

TXMetadata::TXMetadata() {
    md_.start_of_burst = false;
    md_.end_of_burst = false;
    md_.has_time_spec = false;
    md_.time_spec = uhd::time_spec_t(0.0);

}

uhd::tx_metadata_t TXMetadata::get_md() {
    return md_;
}

void TXMetadata::set_start_of_burst(bool start_of_burst) {
    md_.start_of_burst = start_of_burst;

}

void TXMetadata::set_end_of_burst(bool end_of_burst) {
    md_.end_of_burst = end_of_burst;
}

void TXMetadata::set_has_time_spec(bool has_time_spec) {
    md_.has_time_spec = has_time_spec;
}

void TXMetadata::set_time_spec(uhd::time_spec_t time_spec) {
    md_.time_spec = time_spec;
}

uhd::rx_metadata_t RXMetadata::get_md() {
    return md_;
}

bool RXMetadata::get_end_of_burst() {
    return md_.end_of_burst;
}

uhd::rx_metadata_t::error_code_t RXMetadata::get_error_code() {
    return md_.error_code;
}

size_t RXMetadata::get_fragment_offset() {
    return md_.fragment_offset;
}

bool RXMetadata::get_has_time_spec() {
    return md_.has_time_spec;
}

bool RXMetadata::get_out_of_sequence() {
    return md_.out_of_sequence;
}

bool RXMetadata::get_start_of_burst() {
    return md_.start_of_burst;


}

uhd::time_spec_t RXMetadata::get_time_spec() {
    return md_.time_spec;
}
