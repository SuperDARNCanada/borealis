/*Copyright 2016 SuperDARN*/
#include <uhd/usrp/multi_usrp.hpp>
#include <memory>
#include <string>
#include <vector>
#include "usrp_drivers/n200/usrp.hpp"
#include "utils/driver_options/driveroptions.hpp"


/**
 * @brief      Creates the multiUSRP abstraction with the options from the config file.
 *
 * @param[in]  driver_options  The driver options
 */
USRP::USRP(const DriverOptions& driver_options) {
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
    set_main_rx_subdev(driver_options->get_main_rx_subdev());
    set_interferometer_rx_subdev(driver_options->get_interferometer_rx_subdev(),
                                    driver_options->get_interferometer_antenna_count());
    receive_channels = create_receive_channels(driver_options->get_main_antenna_count(),
                                    driver_options->get_interferometer_antenna_count());
    set_time_source(driver_options->get_pps());
    check_ref_locked();

}




/**
 * @brief      Sets the USRP clock source.
 *
 * @param[in]  source A string for a valid USRP clock source.
 */
void USRP::set_usrp_clock_source(std::string source) {
    usrp_->set_clock_source(source);
}

/**
 * @brief      Sets the USRP transmit subdev specification.
 *
 * @param[in]  tx_subdev  A string for a valid transmit subdev.
 */
void USRP::set_tx_subdev(std::string tx_subdev) {
    usrp_->set_tx_subdev_spec(tx_subdev);
}

/**
 * @brief      Sets the transmit sample rate.
 *
 * @param[in]  tx_rate  The transmit sample rate in Sps.
 */
void USRP::set_tx_rate(double tx_rate) {
    usrp_->set_tx_rate(tx_rate);

    double actual_rate = usrp_->get_tx_rate();

    if (actual_rate != tx_rate) {
        /*TODO: something*/
    }
}

/**
 * @brief      Gets the USRP transmit sample rate.
 *
 * @return     The transmit sample rate in Sps.
 */
double USRP::get_tx_rate(){
    return usrp_->get_tx_rate();
}

/**
 * @brief      Sets the transmit center frequency.
 *
 * @param[in]  freq  The frequency in Hz.
 * @param[in]  chs   A vector of which USRP channels to set a center frequency.
 *
 * The USRP uses a numbered channel mapping system to identify which data streams come from which
 * USRP and its daughterboard frontends. With the daughtboard frontends connected to the
 * transmitters, controlling what USRP channels are selected will control what antennas are
 * used and what order they are in.
 */
void USRP::set_tx_center_freq(double freq, std::vector<size_t> chs) {
    uhd::tune_request_t tune_request(freq);

    for(auto &channel : chs) {
        usrp_->set_tx_freq(tune_request, channel);

        double actual_freq = usrp_->get_tx_freq(channel);
        if (actual_freq != freq) {
            /*TODO(Keith): something*/
        }

    }
}

/**
 * @brief      Sets the receive subdev for the main array antennas.
 *
 * @param[in]  main_subdev  A string for a valid receive subdev.
 *
 * Will set all boxes to receive from first USRP channel of all mboards for main array.
 *
 */
void USRP::set_main_rx_subdev(std::string main_subdev) {
    usrp_->set_rx_subdev_spec(main_subdev);
}


/**
 * @brief      Sets the interferometer receive subdev.
 *
 * @param[in]  interferometer_subdev         A string for a valid receive subdev.
 * @param[in]  interferometer_antenna_count  The interferometer antenna count.
 *
 * Override the subdev spec of the first mboards to receive on a second channel for
 * the interferometer.
 */
void USRP::set_interferometer_rx_subdev(std::string interferometer_subdev,
                                         uint32_t interferometer_antenna_count) {
    for(uint32_t i=0; i<interferometer_antenna_count; i++) {
        usrp_->set_rx_subdev_spec(interferometer_subdev, i);
    }

}

/**
 * @brief      Creates the order of USRP receiver channels.
 *
 * @param[in]  main_antenna_count            The main antenna count.
 * @param[in]  interferometer_antenna_count  The interferometer antenna count.
 *
 * @return     A vector with the USRP channel ordering.
 *
 * The USRP default channel mapping will cause the main antenna and interferometer
 * antenna data to be interleaved buffer by buffer. When setting up the receive streamer from the
 * USRP, it is possible to reorder the host side data buffers. This algorithm reorders the channels
 * so that the main array buffers are in order, followed by the interferometers buffers. This
 * ordering is easiest to work with and is what most people would assume the data layout looks like.
 */
std::vector<size_t> USRP::create_receive_channels(uint32_t main_antenna_count,
                                                uint32_t interferometer_antenna_count) {

    std::vector<size_t> channels;

    //Add the main array channels from the mboards that will also be receiving the
    //interferometer samples
    for(uint32_t i=0; i<interferometer_antenna_count; i++) {
        channels.push_back(2*i);
    }

    //starts at 2 * i_count since those channels are interleaved
    auto start = 2 * interferometer_antenna_count;
    auto end = main_antenna_count + interferometer_antenna_count;

    //Add the rest of main array channels
    for(uint32_t i=start; i<end; i++){
        channels.push_back(i);
    }

    //Now add the interferometer channels
    for(uint32_t i=0; i<interferometer_antenna_count; i++) {
        channels.push_back(2*i + 1);
    }

    return channels;

}

/**
 * @brief      Gets the receive channels.
 *
 * @return     A vector of the receive channel ordering.
 */
std::vector<size_t> USRP::get_receive_channels() {
    return receive_channels;
}

/**
 * @brief      Sets the receive sample rate.
 *
 * @param[in]  rx_rate  The receive rate in Sps.
 */
void USRP::set_rx_rate(double rx_rate) {
    usrp_->set_rx_rate(rx_rate);

    double actual_rate = usrp_->get_rx_rate();

    if (actual_rate != rx_rate) {
        /*TODO: something*/
    }
}

/**
 * @brief      Sets the receive center frequency.
 *
 * @param[in]  freq  The frequency in Hz.
 * @param[in]  chs   A vector of which USRP channels to set a center frequency.
 *
 * The USRP uses a numbered channel mapping system to identify which data streams come from which
 * USRP and its daughterboard frontends. With the daughtboard frontends connected to the
 * transmitters, controlling what USRP channels are selected will control what antennas are
 * used and what order they are in. To simplify data processing, all receive channels are used.
 */
void USRP::set_rx_center_freq(double freq, std::vector<size_t> chs) {
    uhd::tune_request_t tune_request(freq);

    for(auto &channel : chs) {
        usrp_->set_rx_freq(tune_request, channel);

        double actual_freq = usrp_->get_rx_freq(channel);
        if (actual_freq != freq) {
            /*TODO: something*/
        }

    }

}

/**
 * @brief      Sets the USRP time source.
 *
 * @param[in]  source   A string with the time source the USRP will use.
 */
void USRP::set_time_source(std::string source) {
    if (source == "pps"){
        usrp_->set_time_source(source);
        usrp_->set_time_unknown_pps(uhd::time_spec_t(0.0));
    }
    else {
        usrp_->set_time_now(0.0);
    }
}

/**
 * @brief      Makes a quick check that each USRP is locked to a reference frequency.
 */
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


/**
 * @brief      Sets the scope sync GPIO to high.
 */
void USRP::set_scope_sync() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0xFFFF, scope_sync_mask_, mboard_);
}

/**
 * @brief      Sets the attenuator GPIO to high.
 */
void USRP::set_atten() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0xFFFF, atten_mask_, mboard_);
}

/**
 * @brief      Sets the t/r GPIO to high.
 */
void USRP::set_tr() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0xFFFF, tr_mask_, mboard_);
}

void USRP::clear_gpio(uint32_t mask, std::string gpio_bank, size_t mboard) {
  usrp_->set_gpio_attr(gpio_bank, "OUT", 0x0000, mask, mboard);
}

void USRP::clear_gpio(uint32_t mask) {
  set_gpio(mask, gpio_bank_, mboard_);
}

/**
 * @brief      Clears the scope sync GPIO to low.
 */
void USRP::clear_scope_sync() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0x0000, scope_sync_mask_, mboard_);
}

/**
 * @brief      Clears the attenuator GPIO to low.
 */
void USRP::clear_atten() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0x0000, atten_mask_, mboard_);
}

/**
 * @brief      Clears the t/r GPIO to low.
 */
void USRP::clear_tr() {
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0x0000, tr_mask_, mboard_);
}

/**
 * @brief      Gets the usrp.
 *
 * @return     The usrp.
 */
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

    for(auto &channel : receive_channels) {
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


uhd::rx_metadata_t& RXMetadata::get_md() {
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
