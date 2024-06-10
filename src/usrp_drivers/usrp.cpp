/*
Copyright 2016 SuperDARN

See LICENSE for details.

  @file usrp.cpp
  This file contains the implementations for USRP related classes.

*/
#include "./usrp.hpp"

#include <chrono>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <uhd/usrp/multi_usrp.hpp>
#include <vector>

#include "utils/driveroptions.hpp"

/**
 * @brief      Creates the multiUSRP abstraction with the options from the
 * config file.
 *
 * @param[in]  driver_options  The driver options parsed from config
 * @param[in]  tx_rate         The transmit rate in Sps  (samples per second,
 * Hz).
 * @param[in]  rx_rate         The receive rate in Sps (samples per second, Hz).
 */
USRP::USRP(const DriverOptions &driver_options, float tx_rate, float rx_rate) {
  gpio_bank_high_ = driver_options.get_gpio_bank_high();
  gpio_bank_low_ = driver_options.get_gpio_bank_low();
  atr_rx_ = driver_options.get_atr_rx();
  atr_tx_ = driver_options.get_atr_tx();
  atr_xx_ = driver_options.get_atr_xx();
  atr_0x_ = driver_options.get_atr_0x();
  agc_st_ = driver_options.get_agc_st();
  test_mode_ = driver_options.get_test_mode();
  lo_pwr_ = driver_options.get_lo_pwr();
  tx_rate_ = tx_rate;
  rx_rate_ = rx_rate;

  usrp_ = uhd::usrp::multi_usrp::make(driver_options.get_device_args());

  set_usrp_clock_source(driver_options.get_ref());
  set_tx_subdev(driver_options.get_tx_subdev());
  set_main_rx_subdev(driver_options.get_main_rx_subdev());
  //  set_interferometer_rx_subdev(driver_options.get_interferometer_rx_subdev(),
  //                               driver_options.get_interferometer_antennas().size());
  set_time_source(driver_options.get_pps(), driver_options.get_clk_addr());
  check_ref_locked();
  set_atr_gpios();
  set_output_gpios();
  set_input_gpios();

  set_tx_rate(driver_options.get_transmit_channels());
  set_rx_rate(driver_options.get_receive_channels());

  create_usrp_tx_stream(driver_options.get_cpu(), driver_options.get_otw(),
                        driver_options.get_transmit_channels());
  create_usrp_rx_stream(driver_options.get_cpu(), driver_options.get_otw(),
                        driver_options.get_receive_channels());
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
 * @param[in]  chs   A vector of USRP channels to tx on.
 *
 * @return     Actual set tx rate.
 */
double USRP::set_tx_rate(std::vector<size_t> chs) {
  if (tx_rate_ <= 0.0) {
    // todo(keith): handle error
  }

  usrp_->set_tx_rate(tx_rate_);
  for (auto &ch : chs) {
    double actual_rate = usrp_->get_tx_rate(ch);
    double rate_1 = usrp_->get_tx_rate(chs[0]);

    if (actual_rate != rate_1) {
      /*TODO(keith): error*/
    }

    if (actual_rate != tx_rate_) {
      /*TODO(keith): error - fail because experiment will assume and we will
       * transmit different than expected*/
    }
  }

  return usrp_->get_tx_rate(chs[0]);
}

/**
 * @brief      Gets the USRP transmit sample rate.
 *
 * @return     The transmit sample rate in Sps.
 */
double USRP::get_tx_rate(uint32_t channel) {
  return usrp_->get_tx_rate(channel);
}

/**
 * @brief      Sets the transmit center frequency.
 *
 * @param[in]  freq         The frequency in Hz.
 * @param[in]  chs          A vector of which USRP channels to set a center
 * frequency.
 * @param[in]  tune_delay   The amount of time in future to tune the devices.
 *
 * @return     The actual set tx center frequency for the USRPs
 *
 * The USRP uses a numbered channel mapping system to identify which data
 * streams come from which USRP and its daughterboard frontends. With the
 * daughtboard frontends connected to the transmitters, controlling what USRP
 * channels are selected will control what antennas are used and what order they
 * are in. To synchronize tuning of all boxes, timed commands are used so that
 * everything is done at once.
 */
double USRP::set_tx_center_freq(double freq, std::vector<size_t> chs,
                                uhd::time_spec_t tune_delay) {
  uhd::tune_request_t tune_request(freq);

  set_command_time(get_current_usrp_time() + tune_delay);
  for (auto &channel : chs) {
    usrp_->set_tx_freq(tune_request,
                       channel);  // TODO(keith): test tune request.
  }
  clear_command_time();

  auto duration = std::chrono::duration<double>(tune_delay.get_real_secs());
  std::this_thread::sleep_for(duration);

  // check for varying USRPs
  for (auto &channel : chs) {
    auto actual_freq = usrp_->get_tx_freq(channel);
    auto freq_1 = usrp_->get_tx_freq(chs[0]);

    if (actual_freq != freq) {
      // TODO(keith): throw error.
    } else if (actual_freq != freq_1) {
      // TODO(keith): throw error.
    }
  }

  return usrp_->get_tx_freq(chs[0]);
}

/**
 * @brief      Gets the transmit center frequency.
 *
 * @return     The actual center frequency that the USRPs are tuned to.
 */
double USRP::get_tx_center_freq(uint32_t channel) {
  return usrp_->get_tx_freq(channel);
}

/**
 * @brief      Sets the receive subdev for the main array antennas.
 *
 * @param[in]  main_subdev  A string for a valid receive subdev.
 *
 * Will set all boxes to receive from first USRP channel of all mboards for main
 * array.
 *
 */
void USRP::set_main_rx_subdev(std::string main_subdev) {
  usrp_->set_rx_subdev_spec(main_subdev);
}

// TODO(remington): REVIEW #43 It would be best if we could have in the config
// file a map of direct antenna to USRP
//       box/subdev/channel so you can change the interferometer to a different
//       set of boxes for example. Also if a rx daughterboard stopped working
//       and you needed to move both main and int to a totally different box for
//       receive, then you could do that. This would be useful for both rx and
//       tx channels.
// REPLY OKAY, but maybe we should leave it for now. That's easier said than
// done. Comment April 2024: Config file format now maps N200 channels to
// antennas. This function is however incorrect, as the
//       assumption that the interferometer antennas are connected to the first
//       N N200s is no longer valid. All config files currently set both the
//       main and intf rx subdevs to "A:A A:B", which simply means to use both
//       LFRX daughterboard channels as the two rx channels for the device. If
//       the N200 connections are configured differently, then the subdev spec
//       would need to be changed, which would change the channel numbers of the
//       multi-USRP object and thus all the hard-coded channel number
//       configuration calculated in DriverOptions.
///**
// * @brief      Sets the interferometer receive subdev.
// *
// * @param[in]  interferometer_subdev         A string for a valid receive
// subdev.
// * @param[in]  interferometer_antenna_count  The interferometer antenna count.
// *
// * Override the subdev spec of the first mboards to receive on a second
// channel for
// * the interferometer.
// */
// void USRP::set_interferometer_rx_subdev(std::string interferometer_subdev,
//                                          uint32_t
//                                          interferometer_antenna_count)
//{
//
//  for(uint32_t i=0; i<interferometer_antenna_count; i++) {
//    usrp_->set_rx_subdev_spec(interferometer_subdev, i);
//  }
//
//}

/**
 * @brief      Sets the receive sample rate.
 *
 * @param[in]  rx_chs  The USRP channels to rx on.
 *
 * @return     The actual rate set.
 */
double USRP::set_rx_rate(std::vector<size_t> rx_chs) {
  if (rx_rate_ <= 0.0) {
    // todo(keith): handle error
  }
  usrp_->set_rx_rate(rx_rate_);

  // check for varying USRPs
  for (auto &channel : rx_chs) {
    auto actual_rate = usrp_->get_rx_rate(channel);
    auto rate_1 = usrp_->get_rx_rate(rx_chs[0]);

    if (actual_rate != rate_1) {
      // TODO(keith): throw error.
    }

    if (actual_rate != rx_rate_) {
      // TODO(keith): throw error. Fail because will be receiving unknown freqs.
    }
  }

  return usrp_->get_rx_rate(rx_chs[0]);
}

/**
 * @brief      Gets the USRP transmit sample rate.
 *
 * @return     The transmit sample rate in Sps.
 */
double USRP::get_rx_rate(uint32_t channel) {
  return usrp_->get_rx_rate(channel);
}
/**
 * @brief      Sets the receive center frequency.
 *
 * @param[in]  freq         The frequency in Hz.
 * @param[in]  chs          A vector of which USRP channels to set a center
 * frequency.
 * @param[in]  tune_delay   The amount of time in future to tune the devices.
 *
 * @return     The actual center frequency that the USRPs are tuned to.
 *
 * The USRP uses a numbered channel mapping system to identify which data
 * streams come from which USRP and its daughterboard frontends. With the
 * daughtboard frontends connected to the transmitters, controlling what USRP
 * channels are selected will control what antennas are used and what order they
 * are in. To simplify data processing, all antenna mapped channels are used. To
 * synchronize tuning of all boxes, timed commands are used so that everything
 * is done at once.
 */
double USRP::set_rx_center_freq(double freq, std::vector<size_t> chs,
                                uhd::time_spec_t tune_delay) {
  uhd::tune_request_t tune_request(freq);

  set_command_time(get_current_usrp_time() + tune_delay);
  for (auto &channel : chs) {
    usrp_->set_rx_freq(tune_request,
                       channel);  // TODO(keith): test tune request.
  }
  clear_command_time();

  auto duration = std::chrono::duration<double>(tune_delay.get_real_secs());
  std::this_thread::sleep_for(duration);

  // check for varying USRPs
  for (auto &channel : chs) {
    auto actual_freq = usrp_->get_rx_freq(channel);
    auto freq_1 = usrp_->get_rx_freq(chs[0]);

    if (actual_freq != freq) {
      // TODO(keith): throw error.
    } else if (actual_freq != freq_1) {
      // TODO(keith): throw error.
    }
  }

  return usrp_->get_rx_freq(chs[0]);
}
/**
 * @brief      Gets the receive center frequency.
 *
 * @return     The actual center frequency that the USRPs are tuned to.
 */
double USRP::get_rx_center_freq(uint32_t channel) {
  return usrp_->get_rx_freq(channel);
}

/**
 * @brief      Sets the USRP time source.
 *
 * @param[in]  source    A string with the time source the USRP will use.
 * @param[in]  clk_addr  IP address of the octoclock for gps timing.
 *
 * Uses the method Ettus suggests for setting time on the x300.
 * https://files.ettus.com/manual/page_gpsdo_x3x0.html
 * Falls back to Juha Vierinen's method of latching to the current time by
 * making sure the clock time is in a stable place past the second if no gps is
 * available. The USRP is then set to this time.
 */
void USRP::set_time_source(std::string source, std::string clk_addr) {
  auto tt = std::chrono::high_resolution_clock::now();
  auto tt_sc = std::chrono::duration_cast<std::chrono::duration<double>>(
      tt.time_since_epoch());
  while (tt_sc.count() - std::floor(tt_sc.count()) < 0.2 ||
         tt_sc.count() - std::floor(tt_sc.count()) > 0.3) {
    tt = std::chrono::high_resolution_clock::now();
    tt_sc = std::chrono::duration_cast<std::chrono::duration<double>>(
        tt.time_since_epoch());
    usleep(10000);
  }
  if (source == "external") {
    gps_clock_ =
        uhd::usrp_clock::multi_usrp_clock::make(uhd::device_addr_t(clk_addr));

    // Make sure Clock configuration is correct
    if (gps_clock_->get_sensor("gps_detected").value == "false") {
      throw uhd::runtime_error("No GPSDO detected on Clock.");
    }
    if (gps_clock_->get_sensor("using_ref").value != "internal") {
      std::ostringstream msg;
      msg << "Clock must be using an internal reference. Using "
          << gps_clock_->get_sensor("using_ref").value;
      throw uhd::runtime_error(msg.str());
    }

    while (!(gps_clock_->get_sensor("gps_locked").to_bool())) {
      std::this_thread::sleep_for(std::chrono::seconds(2));
      RUNTIME_MSG("Waiting for gps lock...");
    }
    usrp_->set_time_source(source);

    auto wait_for_update = [&]() {
      uhd::time_spec_t last = usrp_->get_time_last_pps();
      uhd::time_spec_t next = usrp_->get_time_last_pps();
      while (next == last) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        last = next;
        next = usrp_->get_time_last_pps();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    };

    wait_for_update();

    usrp_->set_time_next_pps(
        uhd::time_spec_t(static_cast<double>(gps_clock_->get_time() + 1)));

    wait_for_update();

    auto clock_time =
        uhd::time_spec_t(static_cast<double>(gps_clock_->get_time()));

    for (uint32_t board = 0; board < usrp_->get_num_mboards(); board++) {
      auto usrp_time = usrp_->get_time_last_pps(board);
      auto time_diff = clock_time - usrp_time;

      RUNTIME_MSG("Time difference between USRPs and gps clock for board "
                  << board << " " << time_diff.get_real_secs());
    }

  } else {
    // TODO(keith): throw error
    usrp_->set_time_now(uhd::time_spec_t(std::ceil(tt_sc.count())));
  }
}

/**
 * @brief      Makes a quick check that each USRP is locked to a reference
 * frequency.
 */
void USRP::check_ref_locked() {
  size_t num_boards = usrp_->get_num_mboards();

  for (size_t i = 0; i < num_boards; i++) {
    std::vector<std::string> sensor_names;
    sensor_names = usrp_->get_mboard_sensor_names(i);
    if ((std::find(sensor_names.begin(), sensor_names.end(), "ref_locked") !=
         sensor_names.end())) {
      uhd::sensor_value_t ref_locked =
          usrp_->get_mboard_sensor("ref_locked", i);
      /*
      TODO: something like this
      UHD_ASSERT_THROW(ref_locked.to_bool());
      */
    } else {
      // TODO(Keith): Get an else statement and do something if there's no
      // ref_locked sensor found
    }
  }
}

/**
 * @brief      Sets the command time.
 *
 * @param[in]  cmd_time  The command time to run a timed command.
 */
void USRP::set_command_time(uhd::time_spec_t cmd_time) {
  usrp_->set_command_time(cmd_time);
}

/**
 * @brief      Clears any timed USRP commands.
 */
void USRP::clear_command_time() { usrp_->clear_command_time(); }

/**
 * @brief      Sets the USRP automatic transmit/receive states on GPIO for the
 * given daughtercard bank.
 */
void USRP::set_atr_gpios() {
  auto output_pins = 0;
  output_pins |= atr_xx_ | atr_rx_ | atr_tx_ | atr_0x_;

  for (uint32_t i = 0; i < usrp_->get_num_mboards(); i++) {
    usrp_->set_gpio_attr(gpio_bank_high_, "CTRL", 0xFFFF, output_pins, i);
    usrp_->set_gpio_attr(gpio_bank_high_, "DDR", 0xFFFF, output_pins, i);

    usrp_->set_gpio_attr(gpio_bank_low_, "CTRL", 0xFFFF, output_pins, i);
    usrp_->set_gpio_attr(gpio_bank_low_, "DDR", 0xFFFF, output_pins, i);

    // XX is the actual TR signal
    usrp_->set_gpio_attr(gpio_bank_high_, "ATR_XX", atr_xx_, 0xFFFF, i);
    usrp_->set_gpio_attr(gpio_bank_high_, "ATR_RX", atr_rx_, 0xFFFF, i);
    usrp_->set_gpio_attr(gpio_bank_high_, "ATR_TX", atr_tx_, 0xFFFF, i);
    usrp_->set_gpio_attr(gpio_bank_high_, "ATR_0X", atr_0x_, 0xFFFF, i);

    usrp_->set_gpio_attr(gpio_bank_low_, "ATR_XX", ~atr_xx_, 0xFFFF, i);
    usrp_->set_gpio_attr(gpio_bank_low_, "ATR_RX", ~atr_rx_, 0xFFFF, i);
    usrp_->set_gpio_attr(gpio_bank_low_, "ATR_TX", ~atr_tx_, 0xFFFF, i);
    usrp_->set_gpio_attr(gpio_bank_low_, "ATR_0X", ~atr_0x_, 0xFFFF, i);
  }
}

/**
 * @brief      Sets the pins mapping the test mode signals as GPIO
 *             outputs.
 */
void USRP::set_output_gpios() {
  for (uint32_t i = 0; i < usrp_->get_num_mboards(); i++) {
    // CTRL 0 sets the pins in gpio mode, DDR 1 sets them as outputs
    usrp_->set_gpio_attr(gpio_bank_high_, "CTRL", 0x0000, test_mode_, i);

    usrp_->set_gpio_attr(gpio_bank_high_, "DDR", 0xFFFF, test_mode_, i);

    usrp_->set_gpio_attr(gpio_bank_low_, "CTRL", 0x0000, test_mode_, i);

    usrp_->set_gpio_attr(gpio_bank_low_, "DDR", 0xFFFF, test_mode_, i);
  }
}

/**
 * @brief      Sets the pins mapping the AGC and low power signals as GPIO
 *             inputs.
 */
void USRP::set_input_gpios() {
  for (uint32_t i = 0; i < usrp_->get_num_mboards(); i++) {
    // CTRL 0 sets the pins in gpio mode, DDR 0 sets them as inputs
    usrp_->set_gpio_attr(gpio_bank_high_, "CTRL", 0x0000, agc_st_, i);
    usrp_->set_gpio_attr(gpio_bank_high_, "CTRL", 0x0000, lo_pwr_, i);

    usrp_->set_gpio_attr(gpio_bank_high_, "DDR", 0x0000, agc_st_, i);
    usrp_->set_gpio_attr(gpio_bank_high_, "DDR", 0x0000, lo_pwr_, i);

    usrp_->set_gpio_attr(gpio_bank_low_, "CTRL", 0x0000, agc_st_, i);
    usrp_->set_gpio_attr(gpio_bank_low_, "CTRL", 0x0000, lo_pwr_, i);

    usrp_->set_gpio_attr(gpio_bank_low_, "DDR", 0x0000, agc_st_, i);
    usrp_->set_gpio_attr(gpio_bank_low_, "DDR", 0x0000, lo_pwr_, i);
  }
}

/**
 * @brief   Inverts the current test mode signal. Useful for testing
 *
 * @param[in]   mboard  The USRP to invert test mode on. Default 0.
 */
void USRP::invert_test_mode(uint32_t mboard /* =0 */) {
  uint32_t tm_value = usrp_->get_gpio_attr(gpio_bank_high_, "OUT", mboard);
  usrp_->set_gpio_attr(gpio_bank_high_, "OUT", test_mode_, ~tm_value, mboard);
  usrp_->set_gpio_attr(gpio_bank_low_, "OUT", test_mode_, tm_value, mboard);
}

/**
 * @brief   Sets the current test mode signal HIGH.
 *
 * @param[in]   mboard  The USRP to set test mode HIGH on. Default 0.
 */
void USRP::set_test_mode(uint32_t mboard /* =0 */) {
  usrp_->set_gpio_attr(gpio_bank_high_, "OUT", test_mode_, 0xFFFF, mboard);
  usrp_->set_gpio_attr(gpio_bank_low_, "OUT", test_mode_, 0x0000, mboard);
}

/**
 * @brief   Clears the current test mode signal LOW.
 *
 * @param[in]   mboard  The USRP to clear test mode LOW on. Default 0.
 */
void USRP::clear_test_mode(uint32_t mboard /* =0 */) {
  usrp_->set_gpio_attr(gpio_bank_high_, "OUT", test_mode_, 0x0000, mboard);
  usrp_->set_gpio_attr(gpio_bank_low_, "OUT", test_mode_, 0xFFFF, mboard);
}

/**
 * @brief      Gets the state of the GPIO bank represented as a decimal number
 */
std::vector<uint32_t> USRP::get_gpio_bank_high_state() {
  std::vector<uint32_t> readback_values;
  for (uint32_t i = 0; i < usrp_->get_num_mboards(); i++) {
    readback_values.push_back(
        usrp_->get_gpio_attr(gpio_bank_high_, "READBACK", i));
  }
  return readback_values;
}

/**
 * @brief      Gets the state of the GPIO bank represented as a decimal number
 */
std::vector<uint32_t> USRP::get_gpio_bank_low_state() {
  std::vector<uint32_t> readback_values;
  for (uint32_t i = 0; i < usrp_->get_num_mboards(); i++) {
    readback_values.push_back(
        usrp_->get_gpio_attr(gpio_bank_low_, "READBACK", i));
  }
  return readback_values;
}

/**
 * @brief      Gets the current status of the GPS fix (locked or unlocked).
 *
 * @return     True if the GPS has a lock.
 */
bool USRP::gps_locked() {
  // This takes on the order of a few microseconds
  if (gps_clock_ == nullptr) {
    return false;
  } else {
    return gps_clock_->get_sensor("gps_locked").to_bool();
  }
}

/**
 * @brief      Gets the status of all of the active-high AGC fault signals as a
 * single binary number. The bits represent each motherboard/USRP device, with
 * bit index mapped to mboard num.
 */
uint32_t USRP::get_agc_status_bank_h() {
  uint32_t agc_status = 0b0;
  for (uint32_t i = 0; i < usrp_->get_num_mboards(); i++) {
    if (usrp_->get_gpio_attr(gpio_bank_high_, "READBACK", i) & agc_st_) {
      agc_status = agc_status | 1 << i;
    }
  }
  return agc_status;
}

/**
 * @brief      Gets the status of all of the active-high Low power signals as a
 * single binary number. The bits represent each motherboard/USRP device, with
 * bit index mapped to mboard num.
 */
uint32_t USRP::get_lp_status_bank_h() {
  uint32_t low_power_status = 0b0;
  for (uint32_t i = 0; i < usrp_->get_num_mboards(); i++) {
    if (usrp_->get_gpio_attr(gpio_bank_high_, "READBACK", i) & lo_pwr_) {
      low_power_status = low_power_status | 1 << i;
    }
  }
  return low_power_status;
}

/**
 * @brief      Gets the status of all of the active-low AGC fault signals as a
 * single binary number. The bits represent each motherboard/USRP device, with
 * bit index mapped to mboard num.
 */
uint32_t USRP::get_agc_status_bank_l() {
  uint32_t agc_status = 0b0;
  for (uint32_t i = 0; i < usrp_->get_num_mboards(); i++) {
    if (usrp_->get_gpio_attr(gpio_bank_low_, "READBACK", i) & agc_st_) {
      agc_status = agc_status | 1 << i;
    }
  }
  return agc_status;
}

/**
 * @brief      Gets the status of all of the active-low Low power signals as a
 * single binary number. The bits represent each motherboard/USRP device, with
 * bit index mapped to mboard num.
 */
uint32_t USRP::get_lp_status_bank_l() {
  uint32_t low_power_status = 0b0;
  for (uint32_t i = 0; i < usrp_->get_num_mboards(); i++) {
    if (usrp_->get_gpio_attr(gpio_bank_low_, "READBACK", i) & lo_pwr_) {
      low_power_status = low_power_status | 1 << i;
    }
  }
  return low_power_status;
}

/**
 * @brief      Gets the current USRP time.
 *
 * @return     The current USRP time.
 */
uhd::time_spec_t USRP::get_current_usrp_time() { return usrp_->get_time_now(); }

/**
 * @brief      Creates an USRP receive stream.
 *
 * @param[in]  cpu_fmt  The cpu format for the tx stream. Described in UHD docs.
 * @param[in]  otw_fmt  The otw format for the tx stream. Described in UHD docs.
 * @param[in]  chs      A vector of which USRP channels to receive on.
 */
void USRP::create_usrp_rx_stream(std::string cpu_fmt, std::string otw_fmt,
                                 std::vector<size_t> chs) {
  uhd::stream_args_t stream_args(cpu_fmt, otw_fmt);
  stream_args.channels = chs;
  rx_stream_ = usrp_->get_rx_stream(stream_args);
}

/**
 * @brief      Gets a pointer to the USRP rx stream.
 *
 * @return     The USRP rx stream.
 */
uhd::rx_streamer::sptr USRP::get_usrp_rx_stream() { return rx_stream_; }

/**
 * @brief      Creates an USRP transmit stream.
 *
 * @param[in]  cpu_fmt  The cpu format for the tx stream. Described in UHD docs.
 * @param[in]  otw_fmt  The otw format for the tx stream. Described in UHD docs.
 * @param[in]  chs      A vector of which USRP channels to transmit on.
 */
void USRP::create_usrp_tx_stream(std::string cpu_fmt, std::string otw_fmt,
                                 std::vector<size_t> chs) {
  uhd::stream_args_t stream_args(cpu_fmt, otw_fmt);
  stream_args.channels = chs;
  tx_stream_ = usrp_->get_tx_stream(stream_args);
}

/**
 * @brief      Gets the usrp.
 *
 * @return     The multi-USRP shared pointer.
 */
uhd::usrp::multi_usrp::sptr USRP::get_usrp() { return usrp_; }

/**
 * @brief      Gets a pointer to the USRP tx stream.
 *
 * @return     The USRP tx stream.
 */
uhd::tx_streamer::sptr USRP::get_usrp_tx_stream() { return tx_stream_; }

/**
 * @brief      Returns a string representation of the USRP parameters.
 *
 * @param[in]  tx_chs  USRP TX channels for which to generate info for.
 * @param[in]  rx_chs  USRP RX channels for which to generate info for.
 *
 * @return     String representation of the USRP parameters.
 */
std::string USRP::to_string(std::vector<size_t> tx_chs,
                            std::vector<size_t> rx_chs) {
  std::stringstream device_str;

  // printable summary of the device.
  device_str << "Using device " << usrp_->get_pp_string() << std::endl
             << "TX rate " << usrp_->get_tx_rate() / 1e6 << " Msps" << std::endl
             << "RX rate " << usrp_->get_rx_rate() / 1e6 << " Msps"
             << std::endl;

  for (auto &channel : tx_chs) {
    device_str << "TX channel " << channel << " freq "
               << usrp_->get_tx_freq(channel) << " MHz" << std::endl;
  }

  for (auto &channel : rx_chs) {
    device_str << "RX channel " << channel << " freq "
               << usrp_->get_rx_freq(channel) << " MHz" << std::endl;
  }

  return device_str.str();
}

/**
 * @brief      Constructs a blank USRP TX metadata object.
 */
TXMetadata::TXMetadata() {
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
uhd::tx_metadata_t TXMetadata::get_md() { return md_; }

/**
 * @brief      Sets whether this data is the start of a burst.
 *
 * @param[in]  start_of_burst  The start of burst boolean.
 */
void TXMetadata::set_start_of_burst(bool start_of_burst) {
  md_.start_of_burst = start_of_burst;
}

/**
 * @brief      Sets whether this data is the end of the burst.
 *
 * @param[in]  end_of_burst  The end of burst boolean.
 */
void TXMetadata::set_end_of_burst(bool end_of_burst) {
  md_.end_of_burst = end_of_burst;
}

/**
 * @brief      Sets whether this data will have a particular timing.
 *
 * @param[in]  has_time_spec  Indicates if this metadata will have a time
 * specifier.
 */
void TXMetadata::set_has_time_spec(bool has_time_spec) {
  md_.has_time_spec = has_time_spec;
}

/**
 * @brief      Sets the timing in the future for this metadata.
 *
 * @param[in]  time_spec  The time specifier for this metadata.
 */
void TXMetadata::set_time_spec(uhd::time_spec_t time_spec) {
  md_.time_spec = time_spec;
}

/**
 * @brief      Gets the RX metadata object that will be retrieved on receiving.
 *
 * @return     The USRP RX metadata object.
 */
uhd::rx_metadata_t &RXMetadata::get_md() { return md_; }

/**
 * @brief      Gets the end of burst.
 *
 * @return     The end of burst.
 */
bool RXMetadata::get_end_of_burst() { return md_.end_of_burst; }

/**
 * @brief      Gets the error code from the metadata on receive.
 *
 * @return     The error code.
 */
uhd::rx_metadata_t::error_code_t RXMetadata::get_error_code() {
  return md_.error_code;
}

/**
 * @brief      Gets the fragment offset. The fragment offset is the sample
 * number at start of buffer.
 *
 * @return     The fragment offset.
 */
size_t RXMetadata::get_fragment_offset() { return md_.fragment_offset; }

/**
 * @brief      Gets the has time specifier status.
 *
 * @return     The has time specifier boolean.
 */
bool RXMetadata::get_has_time_spec() { return md_.has_time_spec; }

/**
 * @brief      Gets out of sequence status. Queries whether a packet is dropped
 * or out of order.
 *
 * @return     The out of sequence boolean.
 */
bool RXMetadata::get_out_of_sequence() { return md_.out_of_sequence; }

/**
 * @brief      Gets the start of burst status.
 *
 * @return     The start of burst.
 */
bool RXMetadata::get_start_of_burst() { return md_.start_of_burst; }

/**
 * @brief      Gets the time specifier of the packet.
 *
 * @return     The time specifier.
 */
uhd::time_spec_t RXMetadata::get_time_spec() { return md_.time_spec; }
