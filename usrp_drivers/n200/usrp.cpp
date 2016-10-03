/*Copyright 2016 SuperDARN*/
#include <uhd/usrp/multi_usrp.hpp>
#include <memory>
#include <string>
#include <vector>
#include "usrp.hpp"
#include "utils/driver_options/driveroptions.hpp"

USRP::USRP(std::shared_ptr<DriverOptions> driver_options) {
    usrp = uhd::usrp::multi_usrp::make(driver_options->get_device_args());
    set_usrp_clock_source(driver_options->get_ref());
    set_tx_subdev(driver_options->get_tx_subdev());
    set_tx_rate(driver_options->get_tx_rate());
    set_rx_subdev(driver_options->get_rx_subdev());
    set_rx_rate(driver_options->get_rx_rate());

    set_time_source(driver_options->get_pps());
    check_ref_locked();

}

void USRP::set_usrp_clock_source(std::string source) {
    usrp->set_clock_source(source);
}

void USRP::set_tx_subdev(std::string tx_subdev) {
    usrp->set_tx_subdev_spec(tx_subdev);
}

void USRP::set_tx_rate(double tx_rate) {
    usrp->set_tx_rate(tx_rate);

    double actual_rate = usrp->get_tx_rate();

    if (actual_rate != tx_rate) {
        /*TODO: something*/
    }
}

void USRP::set_tx_center_freq(double freq, std::vector<size_t> chs) {
    uhd::tune_request_t tune_request(freq);

    for(auto &channel : chs) {
        usrp->set_tx_freq(tune_request, channel);

        double actual_freq = usrp->get_tx_freq(channel);
        if (actual_freq != freq) {
            /*TODO: something*/
        }

    }

    /*boost::this_thread::sleep(boost::posix_time::seconds(1)); */
}

void USRP::set_rx_subdev(std::string rx_subdev) {
    usrp->set_rx_subdev_spec(rx_subdev);
}

void USRP::set_rx_rate(double rx_rate) {
    usrp->set_rx_rate(rx_rate);

    double actual_rate = usrp->get_rx_rate();

    if (actual_rate != rx_rate) {
        /*TODO: something*/
    }
}

void USRP::set_rx_center_freq(double freq, std::vector<size_t> chs) {
    uhd::tune_request_t tune_request(freq);

    for(auto &channel : chs) {
        usrp->set_rx_freq(tune_request, channel);

        double actual_freq = usrp->get_rx_freq(channel);
        if (actual_freq != freq) {
            /*TODO: something*/
        }

    }

    /*boost::this_thread::sleep(boost::posix_time::seconds(1)); */
}        

void USRP::set_time_source(std::string source) {
    usrp->set_time_source(source);
    usrp->set_time_unknown_pps(uhd::time_spec_t(0.0));
}

void USRP::check_ref_locked() {
    size_t num_boards = usrp->get_num_mboards();

    for(size_t i = 0; i < num_boards; i++) {
        std::vector<std::string> sensor_names;
        sensor_names = usrp->get_mboard_sensor_names(i);
        if ((std::find(sensor_names.begin(), sensor_names.end(), "ref_locked") != sensor_names.end())) {
            uhd::sensor_value_t ref_locked = usrp->get_mboard_sensor("ref_locked", i);
            /*
            TODO: something like this
            UHD_ASSERT_THROW(ref_locked.to_bool());
            */
        }

    }
}

std::string USRP::to_string(std::vector<size_t> chs) {
    std::stringstream device_str;

    device_str << "Using device " << usrp->get_pp_string() << std::endl
               << "TX rate " << usrp->get_tx_rate()/1e6 << " Msps" << std::endl
               << "RX rate " << usrp->get_rx_rate()/1e6 << " Msps" << std::endl;
                 

    for(auto &channel : chs) {
        device_str << "TX channel " << channel << " freq " 
                   << usrp->get_tx_freq(channel) << " MHz" << std::endl;
    }

    for(auto &channel : chs) {
        device_str << "RX channel " << channel << " freq " 
                   << usrp->get_tx_freq(channel) << " MHz" << std::endl;
    }

    return device_str.str();
                  
}

TXMetadata::TXMetadata(bool start_of_burst,bool end_of_burst,
           bool has_time_spec,uhd::time_spec_t time_spec) {

    md.start_of_burst = start_of_burst;
    md.end_of_burst = end_of_burst;
    md.has_time_spec = has_time_spec;
    md.time_spec = time_spec;

}

uhd::tx_metadata_t TXMetadata::get_md() {
    return md;
}

void TXMetadata::set_start_of_burst(bool start_of_burst) {
    md.start_of_burst = start_of_burst;

}

void TXMetadata::set_end_of_burst(bool end_of_burst) {
    md.end_of_burst = end_of_burst;
}

void TXMetadata::set_has_time_spec(bool has_time_spec) {
    md.has_time_spec = has_time_spec;
}

void TXMetadata::set_time_spec(uhd::time_spec_t time_spec) {
    md.time_spec = time_spec;
}

uhd::rx_metadata_t RXMetadata::get_md() {
    return md;
}

bool RXMetadata::get_end_of_burst() {
    return md.end_of_burst;
}

uhd::rx_metadata_t::error_code_t RXMetadata::get_error_code() {
    return md.error_code;
}

size_t RXMetadata::get_fragment_offset() {
    return md.fragment_offset;
}

bool RXMetadata::get_has_time_spec() {
    return md.has_time_spec;
}

bool RXMetadata::get_out_of_sequence() {
    return md.out_of_sequence;
}

bool RXMetadata::get_start_of_burst() {
    return md.start_of_burst;
}

uhd::time_spec_t RXMetadata::get_time_spec() {
    return md.time_spec;
}