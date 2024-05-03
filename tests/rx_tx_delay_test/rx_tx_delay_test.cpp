//
// Copyright 2010-2012 Ettus Research LLC
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

// part of this file was created by Max Scharrenbroich, use it as you like...

#include <math.h>
#include <stdlib.h>

#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <complex>
#include <csignal>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <string>
#include <uhd/exception.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/utils/thread.hpp>
#include <uhd/version.hpp>
#include <vector>

typedef std::complex<float> cfloat;

namespace po = boost::program_options;

size_t count = 0;

/***********************************************************************
 * Signal handlers
 **********************************************************************/
static bool stop_signal_called = false;
void sig_int_handler(int) { stop_signal_called = true; }

void do_tx(uhd::usrp::multi_usrp::sptr usrp, const std::string& wire_format,
           size_t samps_per_buff, double rate, double ampl, double start_time);

void do_rx(uhd::usrp::multi_usrp::sptr usrp, const std::string& wire_format,
           size_t samps_per_buff, int rx_channel, double rate,
           double start_time, double thresh);

bool do_leading_edge_detection(cfloat* samples, size_t nsamp,
                               size_t noise_est_window_len, float det_thresh_db,
                               size_t* led_idx, float* led_val, float* noise_mu,
                               float* noise_sig, float* max_val) {
  float mu = 0;
  float sig = 0;
  size_t led_temp = 0;
  float led_val_temp = 0;
  float max_val_temp = 0;
  bool do_once = true;

  float det_thresh = pow(10.0, det_thresh_db / 10.0);

  // estimate the initial mu and sigma //
  for (size_t ii = 0; ii < noise_est_window_len; ii++) {
    float val = std::abs(samples[ii]);
    mu += val;
  }

  mu /= static_cast<float>(noise_est_window_len);

  for (size_t ii = 0; ii < noise_est_window_len; ii++) {
    float val = pow((std::abs(samples[ii]) - mu), 2.0);
    sig += val;
  }

  sig /= static_cast<float>(noise_est_window_len);
  sig = sqrt(sig);

  *noise_mu = mu;
  *noise_sig = sig;

  for (size_t ii = noise_est_window_len; ii < nsamp; ii++) {
    float val_to_check = std::abs(samples[ii]);

    if (val_to_check > det_thresh * sig && do_once) {
      do_once = false;
      led_temp = ii;
      led_val_temp = val_to_check;
    }

    if (val_to_check > max_val_temp) max_val_temp = val_to_check;
  }

  *led_val = led_val_temp;
  *led_idx = led_temp;
  *max_val = max_val_temp;
  return !do_once;
}

/***********************************************************************
 * Main function
 **********************************************************************/
int UHD_SAFE_MAIN(int argc, char* argv[]) {
  uhd::set_thread_priority_safe();

  // transmit variables to be set by po
  std::string tx_addr, tx_ant, tx_subdev, tx_ref, otw;
  double tx_rate, tx_freq, tx_gain, tx_bw;
  float ampl;
  std::string args;
  bool sync_to_pps = false;

  // receive variables to be set by po
  std::string rx_addr, rx_ant, rx_subdev, rx_ref;
  size_t spb;
  double rx_rate, rx_freq, rx_gain, rx_bw;

  size_t iterations;
  double thresh = 0;

  // setup the program options
  po::options_description desc("Allowed options");
  desc.add_options()("help", "help message")(
      "tx-addr", po::value<std::string>(&tx_addr)->default_value(""),
      "uhd transmit device address")(
      "rx-addr", po::value<std::string>(&rx_addr)->default_value(""),
      "uhd receive device address")(
      "args", po::value<std::string>(&args)->default_value(""),
      "uhd transmit and receive args")(
      "spb", po::value<size_t>(&spb)->default_value(0),
      "samples per buffer, 0 for default")("tx-rate",
                                           po::value<double>(&tx_rate),
                                           "rate of transmit outgoing samples")(
      "rx-rate", po::value<double>(&rx_rate),
      "rate of receive incoming samples")("tx-freq",
                                          po::value<double>(&tx_freq),
                                          "transmit RF center frequency in Hz")(
      "rx-freq", po::value<double>(&rx_freq),
      "receive RF center frequency in Hz")(
      "ampl", po::value<float>(&ampl)->default_value(float(0.5)),
      "amplitude of the waveform [0 to 0.7]")(
      "tx-gain", po::value<double>(&tx_gain)->default_value(0),
      "gain (dB) for the transmit RF chain")(
      "rx-gain", po::value<double>(&rx_gain)->default_value(0),
      "gain (dB) for the receive RF chain")(
      "tx-ant", po::value<std::string>(&tx_ant),
      "daughterboard transmit antenna selection")(
      "rx-ant", po::value<std::string>(&rx_ant),
      "daughterboard receive antenna selection")(
      "tx-subdev", po::value<std::string>(&tx_subdev),
      "daughterboard transmit subdevice specification")(
      "rx-subdev", po::value<std::string>(&rx_subdev),
      "daughterboard receive subdevice specification")(
      "tx-bw", po::value<double>(&tx_bw),
      "daughterboard transmit IF filter bandwidth in Hz")(
      "rx-bw", po::value<double>(&rx_bw),
      "daughterboard receive IF filter bandwidth in Hz")(
      "tx-ref", po::value<std::string>(&tx_ref)->default_value("internal"),
      "clock reference (internal, external, mimo)")(
      "rx-ref", po::value<std::string>(&rx_ref)->default_value("internal"),
      "clock reference (internal, external, mimo)")(
      "otw", po::value<std::string>(&otw)->default_value("sc16"),
      "specify the over-the-wire sample mode")(
      "pps-sync",
      "if this switch is included then the usrp times will be synchronized "
      "using PPS")("thresh", po::value<double>(&thresh)->default_value(20),
                   "detection threshold (dB) above noise sigma")(
      "iterations", po::value<size_t>(&iterations),
      "How many iterations to run the LED code");
  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  po::notify(vm);

  std::cout << "ABI String: " << uhd::get_abi_string() << std::endl;
  std::cout << "Version String: " << uhd::get_version_string() << std::endl;

  // print the help message
  if (vm.count("help")) {
    std::cout << boost::format("UHD Timing Offset Test %s") % desc << std::endl;

    std::cout
        << "Example for WBX: " << std::endl
        << argv[0] << " --tx-addr=192.168.10.2 --rx-addr=192.168.10.3 "
        << "--args=recv_frame_size=4095,send_frame_size=4095,recv_buff_size="
           "50e6,send_buff_size=50e6 "
        << "--tx-rate=25000000 --rx-rate=25000000 --tx-freq=915000000 "
           "--rx-freq=915000000 "
        << "--tx-ant=TX/RX --rx-ant=RX2 --tx-ref=gpsdo --rx-ref=mimo --pps-sync"
        << std::endl;

    return ~0;
  }

  if (vm.count("iterations")) {
    std::cout << "Doing " << iterations << " iterations" << std::endl;
  }

  size_t samps_per_buff = 1e4;

  // create a usrp device
  std::string usrp_args = "";
  int rx_channel = 0;

  if (tx_addr == rx_addr) {
    usrp_args += "addr=" + tx_addr;
    rx_channel = 0;
  } else {
    usrp_args += "addr0=" + tx_addr;
    usrp_args += ",addr1=" + rx_addr;
    rx_channel = 1;
  }

  if (args != "") usrp_args += "," + args;

  std::cout << std::endl;
  std::cout << boost::format("Creating the dual usrp device with: %s...") %
                   usrp_args
            << std::endl;
  uhd::usrp::multi_usrp::sptr dual_usrp =
      uhd::usrp::multi_usrp::make(usrp_args);

  uhd::usrp::multi_usrp::sptr tx_usrp = dual_usrp;
  uhd::usrp::multi_usrp::sptr rx_usrp = dual_usrp;

  // always select the subdevice first, the channel mapping affects the other
  // settings
  if (vm.count("tx-subdev")) tx_usrp->set_tx_subdev_spec(tx_subdev, 0);
  if (vm.count("rx-subdev")) rx_usrp->set_rx_subdev_spec(rx_subdev, rx_channel);

  std::cout << boost::format("Using Device: %s") % dual_usrp->get_pp_string()
            << std::endl;

  // Lock mboard clocks
  tx_usrp->set_clock_source(tx_ref, 0);

  if (tx_ref != "internal") tx_usrp->set_time_source(tx_ref, 0);

  if (rx_channel != 0) {
    rx_usrp->set_clock_source(rx_ref, rx_channel);

    if (tx_ref != "internal") rx_usrp->set_time_source(rx_ref, rx_channel);
  }

  if (vm.count("pps-sync")) {
    sync_to_pps = true;
  }

  // set the transmit sample rate
  if (!vm.count("tx-rate")) {
    std::cerr << "Please specify the transmit sample rate with --tx-rate"
              << std::endl;
    return ~0;
  }
  std::cout << boost::format("Setting TX Rate: %f Msps...") % (tx_rate / 1e6)
            << std::endl;
  tx_usrp->set_tx_rate(tx_rate);
  std::cout << boost::format("Actual TX Rate: %f Msps...") %
                   (tx_usrp->get_tx_rate() / 1e6)
            << std::endl
            << std::endl;

  // set the receive sample rate
  if (!vm.count("rx-rate")) {
    std::cerr << "Please specify the sample rate with --rx-rate" << std::endl;
    return ~0;
  }
  std::cout << boost::format("Setting RX Rate: %f Msps...") % (rx_rate / 1e6)
            << std::endl;
  rx_usrp->set_rx_rate(rx_rate, rx_channel);
  std::cout << boost::format("Actual RX Rate: %f Msps...") %
                   (rx_usrp->get_rx_rate(rx_channel) / 1e6)
            << std::endl
            << std::endl;

  // set the transmit center frequency
  if (!vm.count("tx-freq")) {
    std::cerr << "Please specify the transmit center frequency with --tx-freq"
              << std::endl;
    return ~0;
  }

  std::cout << boost::format("Setting TX Freq: %f MHz...") % (tx_freq / 1e6)
            << std::endl;
  tx_usrp->set_tx_freq(tx_freq);
  std::cout << boost::format("Actual TX Freq: %f MHz...") %
                   (tx_usrp->get_tx_freq() / 1e6)
            << std::endl
            << std::endl;

  // set the rf gain
  if (vm.count("tx-gain")) {
    std::cout << boost::format("Setting TX Gain: %f dB...") % tx_gain
              << std::endl;
    tx_usrp->set_tx_gain(tx_gain);
    std::cout << boost::format("Actual TX Gain: %f dB...") %
                     tx_usrp->get_tx_gain()
              << std::endl
              << std::endl;
  }

  // set the IF filter bandwidth
  if (vm.count("tx-bw")) {
    std::cout << boost::format("Setting TX Bandwidth: %f MHz...") %
                     (tx_bw / 1e6)
              << std::endl;
    tx_usrp->set_tx_bandwidth(tx_bw);
    std::cout << boost::format("Actual TX Bandwidth: %f MHz...") %
                     (tx_usrp->get_tx_bandwidth() / 1e6)
              << std::endl
              << std::endl;
  }

  // set the antenna
  if (vm.count("tx-ant")) {
    std::cout << boost::format("Setting TX Antenna: %s ...") % rx_ant
              << std::endl;
    rx_usrp->set_tx_antenna(tx_ant);
    std::cout << boost::format("Actual TX Antenna: %s ...") %
                     (rx_usrp->get_tx_antenna())
              << std::endl
              << std::endl;
  }

  // set the receive center frequency
  if (!vm.count("rx-freq")) {
    std::cerr << "Please specify the center frequency with --rx-freq"
              << std::endl;
    return ~0;
  }
  std::cout << boost::format("Setting RX Freq: %f MHz...") % (rx_freq / 1e6)
            << std::endl;
  rx_usrp->set_rx_freq(rx_freq, rx_channel);
  std::cout << boost::format("Actual RX Freq: %f MHz...") %
                   (rx_usrp->get_rx_freq(rx_channel) / 1e6)
            << std::endl
            << std::endl;

  // set the receive rf gain
  if (vm.count("rx-gain")) {
    std::cout << boost::format("Setting RX Gain: %f dB...") % rx_gain
              << std::endl;
    rx_usrp->set_rx_gain(rx_gain, rx_channel);
    std::cout << boost::format("Actual RX Gain: %f dB...") %
                     rx_usrp->get_rx_gain(rx_channel)
              << std::endl
              << std::endl;
  }

  // set the receive IF filter bandwidth
  if (vm.count("rx-bw")) {
    std::cout << boost::format("Setting RX Bandwidth: %f MHz...") %
                     (rx_bw / 1e6)
              << std::endl;
    rx_usrp->set_rx_bandwidth(rx_bw, rx_channel);
    std::cout << boost::format("Actual RX Bandwidth: %f MHz...") %
                     (rx_usrp->get_rx_bandwidth(rx_channel) / 1e6)
              << std::endl
              << std::endl;
  }

  if (vm.count("rx-ant")) {
    std::cout << boost::format("Setting RX Antenna: %s ...") % rx_ant
              << std::endl;
    rx_usrp->set_rx_antenna(rx_ant, rx_channel);
    std::cout << boost::format("Actual RX Antenna: %s ...") %
                     (rx_usrp->get_rx_antenna(rx_channel))
              << std::endl
              << std::endl;
  }

  // sleep to let things settle
  boost::this_thread::sleep(boost::posix_time::milliseconds(100));

  // Check Ref and LO Lock detect
  std::vector<std::string> tx_sensor_names, rx_sensor_names;
  tx_sensor_names = tx_usrp->get_tx_sensor_names(0);
  if (std::find(tx_sensor_names.begin(), tx_sensor_names.end(), "lo_locked") !=
      tx_sensor_names.end()) {
    uhd::sensor_value_t lo_locked = tx_usrp->get_tx_sensor("lo_locked", 0);
    std::cout << boost::format("Checking TX: %s ...") % lo_locked.to_pp_string()
              << std::endl;
    UHD_ASSERT_THROW(lo_locked.to_bool());
  }
  rx_sensor_names = rx_usrp->get_rx_sensor_names(rx_channel);
  if (std::find(rx_sensor_names.begin(), rx_sensor_names.end(), "lo_locked") !=
      rx_sensor_names.end()) {
    uhd::sensor_value_t lo_locked =
        rx_usrp->get_rx_sensor("lo_locked", rx_channel);
    std::cout << boost::format("Checking RX: %s ...") % lo_locked.to_pp_string()
              << std::endl;
    UHD_ASSERT_THROW(lo_locked.to_bool());
  }

  tx_sensor_names = tx_usrp->get_mboard_sensor_names(0);
  if ((tx_ref == "mimo") &&
      (std::find(tx_sensor_names.begin(), tx_sensor_names.end(),
                 "mimo_locked") != tx_sensor_names.end())) {
    uhd::sensor_value_t mimo_locked =
        tx_usrp->get_mboard_sensor("mimo_locked", 0);
    std::cout << boost::format("Checking TX: %s ...") %
                     mimo_locked.to_pp_string()
              << std::endl;
    UHD_ASSERT_THROW(mimo_locked.to_bool());
  }
  if ((tx_ref == "gpsdo") &&
      (std::find(tx_sensor_names.begin(), tx_sensor_names.end(),
                 "gps_locked") != tx_sensor_names.end())) {
    uhd::sensor_value_t gpsdo_locked =
        tx_usrp->get_mboard_sensor("gps_locked", 0);
    std::cout << boost::format("Checking TX: %s ...") %
                     gpsdo_locked.to_pp_string()
              << std::endl;
    // UHD_ASSERT_THROW(gpsdo_locked.to_bool());
    if (!gpsdo_locked.to_bool()) {
      std::cout << "Warning!  GPS is not locked, continuing anyway..."
                << std::endl;
    }
  }
  if ((tx_ref == "external" || tx_ref == "internal") &&
      (std::find(tx_sensor_names.begin(), tx_sensor_names.end(),
                 "ref_locked") != tx_sensor_names.end())) {
    uhd::sensor_value_t ref_locked =
        tx_usrp->get_mboard_sensor("ref_locked", 0);
    std::cout << boost::format("Checking TX: %s ...") %
                     ref_locked.to_pp_string()
              << std::endl;
    UHD_ASSERT_THROW(ref_locked.to_bool());
  }

  rx_sensor_names = rx_usrp->get_mboard_sensor_names(rx_channel);
  if ((rx_ref == "mimo") &&
      (std::find(rx_sensor_names.begin(), rx_sensor_names.end(),
                 "mimo_locked") != rx_sensor_names.end())) {
    uhd::sensor_value_t mimo_locked =
        rx_usrp->get_mboard_sensor("mimo_locked", rx_channel);
    std::cout << boost::format("Checking RX: %s ...") %
                     mimo_locked.to_pp_string()
              << std::endl;
    UHD_ASSERT_THROW(mimo_locked.to_bool());
  }
  if ((rx_ref == "gpsdo") &&
      (std::find(rx_sensor_names.begin(), rx_sensor_names.end(),
                 "gps_locked") != rx_sensor_names.end())) {
    uhd::sensor_value_t gpsdo_locked =
        rx_usrp->get_mboard_sensor("gps_locked", rx_channel);
    std::cout << boost::format("Checking TX: %s ...") %
                     gpsdo_locked.to_pp_string()
              << std::endl;
    // UHD_ASSERT_THROW(gpsdo_locked.to_bool());
    if (!gpsdo_locked.to_bool()) {
      std::cout << "Warning!  GPS is not locked, continuing anyway..."
                << std::endl;
    }
  }
  if ((rx_ref == "external" || rx_ref == "internal") &&
      (std::find(rx_sensor_names.begin(), rx_sensor_names.end(),
                 "ref_locked") != rx_sensor_names.end())) {
    uhd::sensor_value_t ref_locked =
        rx_usrp->get_mboard_sensor("ref_locked", rx_channel);
    std::cout << boost::format("Checking RX: %s ...") %
                     ref_locked.to_pp_string()
              << std::endl;
    UHD_ASSERT_THROW(ref_locked.to_bool());
  }

  std::signal(SIGINT, &sig_int_handler);
  std::cout << "Press Ctrl + C to stop the test..." << std::endl;

  // reset usrp time to prepare for transmit/receive
  std::cout << "Synchronizing device times..." << std::flush;

  if (sync_to_pps) {
    std::cout << "to PPS..." << std::endl;
    dual_usrp->set_time_next_pps(uhd::time_spec_t(0.0));
    boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
  } else {
    std::cout << "to NOW..." << std::endl;
    dual_usrp->set_time_now(uhd::time_spec_t(0.0));
  }

  double check_time0 = dual_usrp->get_time_now(0).get_real_secs();
  double check_time1 = dual_usrp->get_time_now(rx_channel).get_real_secs();

  std::cout << "Check Time Channel 0: "
            << boost::format("%.3f sec") % check_time0 << std::endl;
  std::cout << "Check Time Channel 1: "
            << boost::format("%.3f sec") % check_time1 << std::endl;
  std::cout << "Difference: "
            << boost::format("%.4f usec") % ((check_time1 - check_time0) * 1e6)
            << std::endl;

  double start_time = 7;

  boost::thread tx_thread(boost::bind(&do_tx, dual_usrp, otw, samps_per_buff,
                                      tx_rate, ampl, start_time));

  boost::thread rx_thread(boost::bind(&do_rx, dual_usrp, otw, samps_per_buff,
                                      rx_channel, rx_rate, start_time, thresh));

  // do_rx(dual_usrp, otw, 1e4, rx_channel, rx_rate, start_time);
  while (!stop_signal_called) {
    if (vm.count("iterations") && (count >= iterations)) {
      break;
    }
    //    std::cout << "Iterations: " << count << std::endl;
    boost::this_thread::sleep(boost::posix_time::milliseconds(1000));
  }

  tx_thread.interrupt();
  rx_thread.interrupt();

  tx_thread.join();
  rx_thread.join();

  // finished
  std::cout << std::endl << "Done!" << std::endl << std::endl;
  return EXIT_SUCCESS;
}

void do_rx(uhd::usrp::multi_usrp::sptr usrp, const std::string& wire_format,
           size_t samps_per_buff, int rx_channel, double rate,
           double start_time, double thresh) {
  // uber prio //
  uhd::set_thread_priority_safe(.8, false);
  std::complex<float>* rx_buff_ptr = NULL;

  try {
    // create a receive streamer //
    uhd::stream_args_t stream_args("fc32", wire_format);
    stream_args.channels = std::vector<size_t>(1, 0);
    stream_args.channels[0] = rx_channel;
    uhd::rx_streamer::sptr rx_stream = usrp->get_rx_stream(stream_args);

    // padding for receive //
    size_t rx_padding = 10;

    // create the receive buffer //
    size_t burst_length = samps_per_buff * (2 * rx_padding + 1);

    rx_buff_ptr =
        (std::complex<float>*)calloc(burst_length, sizeof(std::complex<float>));

    // cout << "RX here!" << endl;

    // get the sample rate //
    double fs = rate;

    // start the receive so the tx burst occurs near the middle of the buffer //
    double start_ahead = rx_padding * static_cast<double>(samps_per_buff) / fs;

    std::cout << boost::format("Starting RX Ahead: %.4f msec") %
                     (start_ahead * 1e3)
              << std::endl;

    // set t0 //
    double event_time = start_time - start_ahead;

    // rx meta data structure //
    uhd::rx_metadata_t md;

    uhd::stream_cmd_t stream_cmd(
        uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.num_samps = burst_length;
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = uhd::time_spec_t(event_time);

    // double-check the timing of the event //
    double usrp_time_now = usrp->get_time_now(rx_channel).get_real_secs();
    double seconds_in_future = (event_time - usrp_time_now);
    double timeout = seconds_in_future;
    double packet_timeout = 0;

    size_t num_acc_samps = 0;  // number of accumulated samples
    size_t num_samps_rxd = 0;  // number of samples actually received
    size_t to_rx = 0;          // number of samples to receive

    // monitor overflows and errors //
    bool overflow = false;
    size_t overflow_counts = 0;
    bool error_in_burst = false;

    // issue the stream command //
    usrp->issue_stream_cmd(stream_cmd, rx_channel);

    bool do_once = true;
    size_t reps = 0;

    while (!boost::this_thread::interruption_requested()) {
      overflow_counts = 0;
      error_in_burst = false;
      num_acc_samps = 0;

      do_once = true;
      reps = 0;

      while (!boost::this_thread::interruption_requested() &&
             num_acc_samps < burst_length) {
        reps++;

        // how many samples to rx this time //
        to_rx = burst_length - num_acc_samps;
        // set the timeout condition //
        packet_timeout = static_cast<double>(to_rx) / fs + 0.1;
        timeout = seconds_in_future + packet_timeout;

        std::cout << "packettimeout: " << packet_timeout
                  << " timeout: " << timeout
                  << " Seconds in future: " << seconds_in_future << std::endl;

        // get the data //
        num_samps_rxd =
            rx_stream->recv((rx_buff_ptr + num_acc_samps), to_rx, md, timeout);

        // increment the number of received samples //
        num_acc_samps += num_samps_rxd;

        // the future is now!!! //
        seconds_in_future = 0;

        // handle the error code
        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
          std::cout << boost::format(
                           "Timeout while streaming, received: %d out of "
                           "samples %d (%.2f pct).") %
                           num_acc_samps % burst_length %
                           (static_cast<double>(num_acc_samps) /
                            static_cast<double>(burst_length) * 100)
                    << std::endl;

          std::cout << "Tried To RX: " << to_rx << ", Loop Reps: " << reps
                    << std::endl;

          error_in_burst = true;
          break;
        }

        if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
          if (!overflow) {
            overflow = true;
            error_in_burst = true;
          }

          overflow_counts++;
          continue;
        }

        if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
          std::cout << "Error Code: " << md.error_code << std::endl;
          error_in_burst = true;
          break;
        }

        if (do_once) {
          do_once = false;
        }
      }

      if (error_in_burst) {
        std::cout << "***** Error in RX Burst! Code: " << md.error_code
                  << std::endl;
        if (overflow)
          std::cout << "\tRX Burst had Overflows: " << overflow_counts
                    << std::endl;
      }

      // std::cout << "RX Burst Complete! " <<  num_acc_samps << " Samples
      // Acquired." << std::endl;

      // get time now
      usrp_time_now = usrp->get_time_now(rx_channel).get_real_secs();

      size_t led_idx = 0;
      float led_val = 0;
      float noise_mu = 0;
      float noise_sig = 0;
      float max_val = 0;

      // Do the edge detection //
      do_leading_edge_detection(rx_buff_ptr, burst_length, 10 * samps_per_buff,
                                thresh, &led_idx, &led_val, &noise_mu,
                                &noise_sig, &max_val);

      double tdiff = static_cast<double>(led_idx) / fs - start_ahead;

      std::cout << boost::format(
                       "Noise Mu: %.2f dB, Noise Sigma: %.2f dB, Max Value = "
                       "%.2f dB") %
                       (10 * log10(noise_mu)) % (10 * log10(noise_sig)) %
                       (10 * log10(max_val))
                << std::endl;

      std::cout << boost::format(
                       "Detected at %d (of %d), LED Value = %.2f dB, TX/RX "
                       "Timing Offset: %.4f usec, # Samples @ %.3f MHz: %d") %
                       led_idx % burst_length % (10 * log10(led_val)) %
                       (tdiff * 1e6) % (fs / 1000000) % ceil(tdiff * fs)
                << std::endl;

      count += 1;
      std::cout << "Count: " << count << std::endl;

      // double led_proc_time = usrp->get_time_now(rx_channel).get_real_secs() -
      // usrp_time_now;

      // zero the receive buffer //
      memset(rx_buff_ptr, 0, sizeof(std::complex<float>) * burst_length);

      // std::cout << boost::format("Leading-Edge-Detection Processing Time:
      // %.3f msec\n") % (led_proc_time*1e3) << std::flush;

      // increment the event time //
      event_time += 1.0;

      // get time now
      usrp_time_now = usrp->get_time_now(rx_channel).get_real_secs();
      seconds_in_future = (event_time - usrp_time_now);

      // issue the stream command //
      stream_cmd.time_spec = uhd::time_spec_t(event_time);

      usrp->issue_stream_cmd(stream_cmd, rx_channel);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Error: unknown exception" << std::endl;
  }

  if (rx_buff_ptr != NULL) free(rx_buff_ptr);

  // finished
  std::cout << "RX Done!" << std::endl;
}

// transmit worker thread //
void do_tx(uhd::usrp::multi_usrp::sptr usrp, const std::string& wire_format,
           size_t samps_per_buff, double rate, double ampl, double start_time) {
  // uber prio //
  uhd::set_thread_priority_safe(0.8, false);

  try {
    // create a transmit streamer //
    uhd::stream_args_t stream_args("fc32", wire_format);
    uhd::tx_streamer::sptr tx_stream = usrp->get_tx_stream(stream_args);

    // create the transmit buffer //
    size_t burst_length = samps_per_buff;

    std::vector<std::complex<float> > send_buffer(burst_length, ampl);
    std::complex<float>* tx_buff_ptr =
        (std::complex<float>*)(send_buffer.data());

    // cout << "TX here!" << endl;

    // get the sample rate //
    double fs = rate;

    // set t0 //
    double event_time = start_time;

    // tx meta data structure //
    uhd::tx_metadata_t md;
    md.start_of_burst = true;
    md.end_of_burst = false;
    md.has_time_spec = true;
    md.time_spec = uhd::time_spec_t(event_time);

    // double-check the timing of the event //
    double usrp_time_now = usrp->get_time_now().get_real_secs();
    double seconds_in_future = (event_time - usrp_time_now);
    double timeout = seconds_in_future;
    double packet_timeout = 0;

    size_t num_acc_samps = 0;  // number of accumulated samples
    size_t num_samps_txd = 0;  // number of samples actually transmitted
    size_t to_tx = 0;          // number of samples to transmit

    uhd::async_metadata_t async_md;

    while (!boost::this_thread::interruption_requested()) {
      num_acc_samps = 0;

      while (!boost::this_thread::interruption_requested() &&
             num_acc_samps < burst_length) {
        // how many samples to tx this time //
        to_tx = burst_length - num_acc_samps;

        // set the timeout condition //
        packet_timeout = static_cast<double>(to_tx) / fs + 0.1;
        timeout = seconds_in_future + packet_timeout;

        // send the data //
        num_samps_txd =
            tx_stream->send((tx_buff_ptr + num_acc_samps), to_tx, md, timeout);

        // increment the number of transmitted samples //
        num_acc_samps += num_samps_txd;

        // do not use time spec for subsequent packets //
        md.has_time_spec = false;
        md.start_of_burst = false;

        // the future is now!!! //
        seconds_in_future = 0;
      }

      // send a mini EOB packet //
      md.end_of_burst = true;
      tx_stream->send("", 0, md);

      bool error_in_burst = false;
      size_t last_error_code = uhd::async_metadata_t::EVENT_CODE_BURST_ACK;

      // loop through all messages for the ACK packet (may have underflow
      // messages in queue)
      while (!boost::this_thread::interruption_requested() &&
             usrp->get_device()->recv_async_msg(async_md)) {
        if (async_md.event_code ==
            uhd::async_metadata_t::EVENT_CODE_BURST_ACK) {
          break;
        } else if (!error_in_burst) {
          error_in_burst = true;
          last_error_code = async_md.event_code;
        }
      }

      if (error_in_burst) {
        std::cout << "***** Error in TX Burst!  Last Error Code: "
                  << last_error_code << std::endl;
      }

      // std::cout << "TX Burst Complete!" << std::endl;

      // increment the event time //
      event_time += 1.0;

      md.start_of_burst = true;
      md.end_of_burst = false;
      md.has_time_spec = true;
      md.time_spec = uhd::time_spec_t(event_time);

      usrp_time_now = usrp->get_time_now().get_real_secs();
      seconds_in_future = (event_time - usrp_time_now);
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Error: unknown exception" << std::endl;
  }

  // finished
  std::cout << "TX Done!" << std::endl;
}
