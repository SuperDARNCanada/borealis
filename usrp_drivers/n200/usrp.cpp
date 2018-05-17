/*
Copyright 2016 SuperDARN

See LICENSE for details.

  @file usrp.cpp
  This file contains the implementations for USRP related classes.

*/
#include <uhd/usrp/multi_usrp.hpp>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include "usrp_drivers/n200/usrp.hpp"
#include "utils/driver_options/driveroptions.hpp"


/**
 * @brief      Creates the multiUSRP abstraction with the options from the config file.
 *
 * @param[in]  driver_options  The driver options parsed from config
 */
USRP::USRP(const DriverOptions& driver_options)
{
  mboard_ = 0;
  gpio_bank_ = driver_options.get_gpio_bank();
  scope_sync_mask_ = driver_options.get_scope_sync_mask();
  atten_mask_ = driver_options.get_atten_mask();
  tr_mask_ = driver_options.get_tr_mask();
  usrp_ = uhd::usrp::multi_usrp::make(driver_options.get_device_args());
  // Set first four GPIO on gpio_bank_ to output, the rest are input
  usrp_->set_gpio_attr(gpio_bank_, "DDR", 0x000F, 0xFFFF);
  set_usrp_clock_source(driver_options.get_ref());
  set_tx_subdev(driver_options.get_tx_subdev());
  set_main_rx_subdev(driver_options.get_main_rx_subdev());
  set_interferometer_rx_subdev(driver_options.get_interferometer_rx_subdev(),
                                driver_options.get_interferometer_antenna_count());
/*  receive_channels = create_receive_channels(driver_options.get_main_antenna_count(),
                                              driver_options.get_interferometer_antenna_count());*/
  set_time_source(driver_options.get_pps());
  check_ref_locked();
  set_atr_gpios();

}




/**
 * @brief      Sets the USRP clock source.
 *
 * @param[in]  source A string for a valid USRP clock source.
 */
void USRP::set_usrp_clock_source(std::string source)
{
  usrp_->set_clock_source(source);
}

/**
 * @brief      Sets the USRP transmit subdev specification.
 *
 * @param[in]  tx_subdev  A string for a valid transmit subdev.
 */
void USRP::set_tx_subdev(std::string tx_subdev)
{
  usrp_->set_tx_subdev_spec(tx_subdev);
}

/**
 * @brief      Sets the transmit sample rate.
 *
 * @param[in]  tx_rate  The transmit sample rate in Sps.
 *
 * @return     Actual set tx rate.
 */
double USRP::set_tx_rate(double tx_rate, std::vector<size_t> chs)
{
  if (tx_rate <= 0.0) {
    //todo(keith): handle error
  }

  usrp_->set_tx_rate(tx_rate);
  for (auto& ch : chs)
  {
    double actual_rate = usrp_->get_tx_rate(ch);
    double rate_1 = usrp_->get_tx_rate(chs[0]);

    if (actual_rate != rate_1) {
      /*TODO(keith): error*/
    }
  }

  return usrp_->get_tx_rate(chs[0]);

}

/**
 * @brief      Gets the USRP transmit sample rate.
 *
 * @return     The transmit sample rate in Sps.
 */
double USRP::get_tx_rate(uint32_t channel)
{
  return usrp_->get_tx_rate(channel);
}

/**
 * @brief      Sets the transmit center frequency.
 *
 * @param[in]  freq  The frequency in Hz.
 * @param[in]  chs   A vector of which USRP channels to set a center frequency.
 *
 * @return     The actual set tx center frequency for the USRPs
 *
 * The USRP uses a numbered channel mapping system to identify which data streams come from which
 * USRP and its daughterboard frontends. With the daughtboard frontends connected to the
 * transmitters, controlling what USRP channels are selected will control what antennas are
 * used and what order they are in.
 */
double USRP::set_tx_center_freq(double freq, std::vector<size_t> chs)
{
  uhd::tune_request_t tune_request(freq);

  for(auto &channel : chs) {
    usrp_->set_tx_freq(tune_request, channel); //TODO(keith): test tune request.

    double actual_freq = usrp_->get_tx_freq(channel);
    if (actual_freq != freq) {
      /*TODO(Keith): something*/
    }

  }

  //check for varying USRPs
  for (auto &channel : chs) {
    auto actual_freq = usrp_->get_tx_freq(channel);
    auto freq_1 = usrp_->get_tx_freq(chs[0]);

    if (actual_freq != freq_1) {
        //TODO(keith): throw error.
    }
  }

  return usrp_->get_tx_freq(chs[0]);
}

/**
 * @brief      Sets the receive subdev for the main array antennas.
 *
 * @param[in]  main_subdev  A string for a valid receive subdev.
 *
 * Will set all boxes to receive from first USRP channel of all mboards for main array.
 *
 */
void USRP::set_main_rx_subdev(std::string main_subdev)
{
  usrp_->set_rx_subdev_spec(main_subdev);
}

// REVIEW #43 It would be best if we could have in the config file a map of direct antenna to USRP box/subdev/channel so you can change the interferometer to a different set of boxes for example. Also if a rx daughterboard stopped working and you needed to move both main and int to a totally different box for receive, then you could do that. This would be useful for both rx and tx channels.
// REPLY OKAY, but maybe we should leave it for now. That's easier said than done.
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
                                          uint32_t interferometer_antenna_count)
{

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
/*std::vector<size_t> USRP::create_receive_channels(uint32_t main_antenna_count,
                        uint32_t interferometer_antenna_count) //TODO(keith): add channel mapping
{

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
*/
/**
 * @brief      Gets the receive channels.
 *
 * @return     A vector of the receive channel ordering.
 */
/*std::vector<size_t> USRP::get_receive_channels()
{
  return receive_channels;
}
*/
/**
 * @brief      Sets the receive sample rate.
 *
 * @param[in]  rx_rate  The receive rate in Sps.
 */
double USRP::set_rx_rate(double rx_rate, std::vector<size_t> rx_chs)
{
  usrp_->set_rx_rate(rx_rate);

  //check for varying USRPs
  for (auto &channel : rx_chs) {
    auto actual_freq = usrp_->get_rx_freq(channel);
    auto freq_1 = usrp_->get_rx_freq(rx_chs[0]);

    if (actual_freq != freq_1) {
        //TODO(keith): throw error.
    }
  }

  return usrp_->get_rx_freq(rx_chs[0]);
}

/**
 * @brief      Sets the receive center frequency.
 *
 * @param[in]  freq  The frequency in Hz.
 * @param[in]  chs   A vector of which USRP channels to set a center frequency.
 *
 * @return     The actual center frequency that the USRPs are tuned to.
 *
 * The USRP uses a numbered channel mapping system to identify which data streams come from which
 * USRP and its daughterboard frontends. With the daughtboard frontends connected to the
 * transmitters, controlling what USRP channels are selected will control what antennas are
 * used and what order they are in. To simplify data processing, all antenna mapped channels are
 * used.
 */
double USRP::set_rx_center_freq(double freq, std::vector<size_t> chs)
{
  uhd::tune_request_t tune_request(freq);

  for(auto &channel : chs) {
    usrp_->set_rx_freq(tune_request, channel);
    double actual_freq = usrp_->get_rx_freq(channel);
    if (actual_freq != freq) {
      /*TODO: something*/
    }

  }

  //return will assume that the set frequency of one USRP is the same as all of them.
  return usrp_->get_rx_freq(chs[0]);
}

/**
 * @brief      Sets the USRP time source.
 *
 * @param[in]  source   A string with the time source the USRP will use.
 *
 * This uses Juha Vierinen's method of latching to the current time by making sure the clock time
 * is in a stable place past the second. The USRP is then set to this time.
 */
void USRP::set_time_source(std::string source)
{
  auto tt =  std::chrono::high_resolution_clock::now();
  auto tt_sc = std::chrono::duration_cast<std::chrono::duration<double>>(tt.time_since_epoch());
  while (tt_sc.count()-std::floor(tt_sc.count()) < 0.2 or
         tt_sc.count()-std::floor(tt_sc.count()) > 0.3)
  {
    tt =  std::chrono::high_resolution_clock::now();
    tt_sc = std::chrono::duration_cast<std::chrono::duration<double>>(tt.time_since_epoch());
    usleep(10000);
  }
  if (source == "external"){
    usrp_->set_time_source(source);
    usrp_->set_time_unknown_pps(std::ceil(tt_sc.count()) + 1);
  }
  else {
    //TODO(keith): throw error
    usrp_->set_time_now(tt_sc.count());
  }
}

/**
 * @brief      Makes a quick check that each USRP is locked to a reference frequency.
 */
void USRP::check_ref_locked()
{
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
    else {
      //TODO(Keith): Get an else statement and do something if there's no ref_locked sensor found
    }

  }
}

/**
 * @brief      Sets the scope sync GPIO to high.
 *
 * @param[in]  scope_high  The timespec for when in the future to set the scope sync pin.
 */
void USRP::set_scope_sync(uhd::time_spec_t scope_high)
{
  usrp_->set_command_time(scope_high);
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0xFFFF, scope_sync_mask_, mboard_);
}

/**
 * @brief      Sets the attenuator GPIO to high.
 *
 * @param[in]  atten_high  The timespec for when in the future to set the attenuator pin.
 */
void USRP::set_atten(uhd::time_spec_t atten_high)
{
  usrp_->set_command_time(atten_high);
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0xFFFF, atten_mask_, mboard_);
}

/**
 * @brief      Sets the t/r GPIO to high.
 *
 * @param[in]  tr_high  The timespec for when in the future to set the TR pin.
 */
void USRP::set_tr(uhd::time_spec_t tr_high)
{
  usrp_->set_command_time(tr_high);
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0xFFFF, tr_mask_, mboard_);
}

/**
 * @brief      Clears the scope sync GPIO to low.
 *
 * @param[in]  scope_low  The timespec for when in the future to clear the scope sync pin.
 */
void USRP::clear_scope_sync(uhd::time_spec_t scope_low)
{
  usrp_->set_command_time(scope_low);
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0x0000, scope_sync_mask_, mboard_);
}

/**
 * @brief      Clears the attenuator GPIO to low.
 *
 * @param[in]  atten_low  The timespec for when in the future to clear the attenuator pin.
 */
void USRP::clear_atten(uhd::time_spec_t atten_low)
{
  usrp_->set_command_time(atten_low);
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0x0000, atten_mask_, mboard_);
}

/**
 * @brief      Clears the t/r GPIO to low.
 *
 * @param[in]  tr_low  The timespec for when in the future to clear the TR pin.
 */
void USRP::clear_tr(uhd::time_spec_t tr_low)
{
  usrp_->set_command_time(tr_low);
  usrp_->set_gpio_attr(gpio_bank_, "OUT", 0x0000, tr_mask_, mboard_);
}

/**
 * @brief      Clears any timed USRP commands.
 */
void USRP::clear_command_times()
{
  usrp_->clear_command_time();
}

void USRP::set_atr_gpios()
{
  for (int i=0; i<usrp_->get_num_mboards(); i++){
    usrp_->set_gpio_attr(gpio_bank_, "CTRL", 0xFFFF, 0b11111111, i);
    usrp_->set_gpio_attr(gpio_bank_, "DDR", 0xFFFF, 0b11111111, i);

    //Mirror pins along bank for easier scoping.
    usrp_->set_gpio_attr(gpio_bank_, "ATR_RX", 0xFFFF, 0b000000010, i);
    usrp_->set_gpio_attr(gpio_bank_, "ATR_RX", 0xFFFF, 0b000000100, i);

    usrp_->set_gpio_attr(gpio_bank_, "ATR_TX", 0xFFFF, 0b000001000, i);
    usrp_->set_gpio_attr(gpio_bank_, "ATR_TX", 0xFFFF, 0b000010000, i);

    //XX is the actual TR signal
    usrp_->set_gpio_attr(gpio_bank_, "ATR_XX", 0xFFFF, 0b000100000, i);
    usrp_->set_gpio_attr(gpio_bank_, "ATR_XX", 0xFFFF, 0b001000000, i);

    //0X acts as 'scope sync'
    usrp_->set_gpio_attr(gpio_bank_, "ATR_0X", 0xFFFF, 0b010000000, i);
    usrp_->set_gpio_attr(gpio_bank_, "ATR_0X", 0xFFFF, 0b100000000, i);
  }


}

/**
 * @brief      Gets the current USRP time.
 *
 * @return     The current USRP time.
 */
uhd::time_spec_t USRP::get_current_usrp_time()
{
  return usrp_->get_time_now();
}

/**
 * @brief      Gets a pointer to the USRP rx stream.
 *
 * @param[in]  stream_args  The arguments for the type of stream. Described in UHD docs.
 *
 * @return     The USRP rx stream.
 */
uhd::rx_streamer::sptr USRP::get_usrp_rx_stream(uhd::stream_args_t stream_args)
{
  return usrp_->get_rx_stream(stream_args);
}

/**
 * @brief      Gets a pointer to the USRP tx stream.
 *
 * @param[in]  stream_args  The arguments for the type of stream. Described in UHD docs.
 *
 * @return     The USRP tx stream.
 */
uhd::tx_streamer::sptr USRP::get_usrp_tx_stream(uhd::stream_args_t stream_args)
{
  return usrp_->get_tx_stream(stream_args);
}

// REVIEW #6 TODO create a set and clear for new method of timing that is a hybrid between atten and t/r (we are only using one pin for both and breaking off on separate board)

/**
 * @brief      Gets the usrp.
 *
 * @return     The multi-USRP shared pointer.
 */
uhd::usrp::multi_usrp::sptr USRP::get_usrp()
{
  return usrp_;
}


/**
 * @brief      Returns a string representation of the USRP parameters.
 *
 * @param[in]  tx_chs  USRP TX channels for which to generate info for.
 * @param[in]  rx_chs  USRP RX channels for which to generate info for.
 *
 * @return     String representation of the USRP parameters.
 */
std::string USRP::to_string(std::vector<size_t> tx_chs, std::vector<size_t> rx_chs)
{
  std::stringstream device_str;

  //printable summary of the device.
  device_str << "Using device " << usrp_->get_pp_string() << std::endl
         << "TX rate " << usrp_->get_tx_rate()/1e6 << " Msps" << std::endl
         << "RX rate " << usrp_->get_rx_rate()/1e6 << " Msps" << std::endl;


  for(auto &channel : tx_chs) {
    device_str << "TX channel " << channel << " freq "
           << usrp_->get_tx_freq(channel) << " MHz" << std::endl;
  }

  for(auto &channel : rx_chs) {
    device_str << "RX channel " << channel << " freq "
           << usrp_->get_rx_freq(channel) << " MHz" << std::endl;
  }

  return device_str.str();

}

/**
 * @brief      Constructs a blank USRP TX metadata object.
 */
TXMetadata::TXMetadata()
{
  md_.start_of_burst = false;
  md_.end_of_burst = false;
  md_.has_time_spec = false;
  md_.time_spec = uhd::time_spec_t(0.0);
}

/**
 * @brief      Gets the TX metadata oject that can be sent the USRPs.
 *
 * @return     The USRP TX metadata.
 */
uhd::tx_metadata_t TXMetadata::get_md()
{
  return md_;
}

/**
 * @brief      Sets whether this data is the start of a burst.
 *
 * @param[in]  start_of_burst  The start of burst boolean.
 */
void TXMetadata::set_start_of_burst(bool start_of_burst)
{
  md_.start_of_burst = start_of_burst;
}

/**
 * @brief      Sets whether this data is the end of the burst.
 *
 * @param[in]  end_of_burst  The end of burst boolean.
 */
void TXMetadata::set_end_of_burst(bool end_of_burst)
{
  md_.end_of_burst = end_of_burst;
}

/**
 * @brief      Sets whether this data will have a particular timing.
 *
 * @param[in]  has_time_spec  Indicates if this metadata will have a time specifier.
 */
void TXMetadata::set_has_time_spec(bool has_time_spec)
{
  md_.has_time_spec = has_time_spec;
}

/**
 * @brief      Sets the timing in the future for this metadata.
 *
 * @param[in]  time_spec  The time specifier for this metadata.
 */
void TXMetadata::set_time_spec(uhd::time_spec_t time_spec)
{
  md_.time_spec = time_spec;
}


/**
 * @brief      Gets the RX metadata object that will be retrieved on receiving.
 *
 * @return     The USRP RX metadata object.
 */
uhd::rx_metadata_t& RXMetadata::get_md()
{
  return md_;
}

/**
 * @brief      Gets the end of burst.
 *
 * @return     The end of burst.
 */
bool RXMetadata::get_end_of_burst()
{
  return md_.end_of_burst;
}

/**
 * @brief      Gets the error code from the metadata on receive.
 *
 * @return     The error code.
 */
uhd::rx_metadata_t::error_code_t RXMetadata::get_error_code()
{
  return md_.error_code;
}

/**
 * @brief      Gets the fragment offset. The fragment offset is the sample number at start of
 *             buffer.
 *
 * @return     The fragment offset.
 */
size_t RXMetadata::get_fragment_offset()
{
  return md_.fragment_offset;
}

/**
 * @brief      Gets the has time specifier status.
 *
 * @return     The has time specifier boolean.
 */
bool RXMetadata::get_has_time_spec()
{
  return md_.has_time_spec;
}

/**
 * @brief      Gets out of sequence status. Queries whether a packet is dropped or out of order.
 *
 * @return     The out of sequence boolean.
 */
bool RXMetadata::get_out_of_sequence()
{
  return md_.out_of_sequence;
}

/**
 * @brief      Gets the start of burst status.
 *
 * @return     The start of burst.
 */
bool RXMetadata::get_start_of_burst()
{
  return md_.start_of_burst;
}

/**
 * @brief      Gets the time specifier of the packet.
 *
 * @return     The time specifier.
 */
uhd::time_spec_t RXMetadata::get_time_spec()
{
  return md_.time_spec;
}
