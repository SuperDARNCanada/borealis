/*Copyright 2016 SuperDARN*/
#include <stdint.h>
#include <sys/mman.h>
#include <unistd.h>

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <tuple>
#include <uhd/exception.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/utils/thread.hpp>
#include <utility>
#include <vector>
#include <zmq.hpp>

#include "src/utils/protobuf/driverpacket.pb.h"
#include "src/utils/protobuf/rxsamplesmetadata.pb.h"
#include "usrp.hpp"
#include "utils/driveroptions.hpp"
#include "utils/shared_macros.hpp"
#include "utils/shared_memory.hpp"
#include "utils/zmq_borealis_helpers.hpp"

// Delay needed for before any set_time_commands will work.
#define SET_TIME_COMMAND_DELAY 5e-3  // seconds
// Tuning delay time provides how long the USRP device will
// wait for the device to settle after sending a tuning request.
// If the device retunes a local oscillator, this should be on the
// order of 1-2 seconds. The LF daughter boards do not have a lo
// and therefore can have a much smaller tuning delay. The value
// chosen below was determined via trial and error, and has been
// set small enough that the delay is less than the time required
// to execute the tuning request.
#define TUNING_DELAY 1e-3  // seconds

// struct containing clocks: one for usrp_time (from the N200s, supplied by
// Octoclock-G) as well as one for the operating system time (by NTP). Updated
// upon recv of RX packet.
typedef struct {
  uhd::time_spec_t usrp_time;  // GPS clock variable.
  std::chrono::time_point<std::chrono::system_clock>
      system_time;  // Operating system clock variable.
} clocks_t;

static clocks_t borealis_clocks;

/**
 * @brief      Makes a set of vectors of the samples for each TX channel from
 * the driver packet.
 *
 * @param[in]  driver_packet    A received driver packet from radar_control.
 * @param[in]  driver_options   The parsed config options needed by the driver.
 *
 * @return     A set of vectors of TX samples for each USRP channel.
 *
 * Values in a protobuffer have no contiguous underlying storage so values need
 * to be parsed into a vector.
 */
std::vector<std::vector<std::complex<float>>> make_tx_samples(
    const driverpacket::DriverPacket &driver_packet,
    const DriverOptions &driver_options) {
  // channel_samples_size() will get you # of channels (protobuf in c++)
  std::vector<std::vector<std::complex<float>>> samples(
      driver_packet.channel_samples_size());

  for (int channel = 0; channel < driver_packet.channel_samples_size();
       channel++) {
    // Get the number of real samples in this particular channel (_size() is
    // from protobuf)
    auto num_samps = driver_packet.channel_samples(channel).real_size();
    std::vector<std::complex<float>> v(num_samps);
    // Type for smp? protobuf object, containing repeated double real and double
    // imag
    auto smp = driver_packet.channel_samples(channel);
    for (int smp_num = 0; smp_num < num_samps; smp_num++) {
      v[smp_num] = std::complex<float>(smp.real(smp_num), smp.imag(smp_num));
    }
    samples[channel] = v;
  }

  for (auto &s : samples) {
    if (s.size() != samples[0].size()) {
      // TODO(keith): Handle this error. Samples buffers are of different
      // lengths.
    }
  }

  return samples;
}

void transmit(zmq::context_t &driver_c, USRP &usrp_d,
              const DriverOptions &driver_options) {
  DEBUG_MSG("Enter transmit thread");
  uhd::set_thread_priority_safe(1.0, true);

  auto identities = {driver_options.get_driver_to_radctrl_identity(),
                     driver_options.get_driver_to_dsp_identity(),
                     driver_options.get_driver_to_brian_identity()};

  auto sockets_vector =
      create_sockets(driver_c, identities, driver_options.get_router_address());

  auto receive_channels = driver_options.get_receive_channels();

  zmq::socket_t &driver_to_radar_control = sockets_vector[0];
  zmq::socket_t &driver_to_dsp = sockets_vector[1];
  zmq::socket_t &driver_to_brian = sockets_vector[2];

  auto samples_set = false;

  auto rx_rate = usrp_d.get_rx_rate();

  driverpacket::DriverPacket driver_packet;

  zmq::socket_t start_trigger(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(start_trigger.connect("inproc://thread"))

  std::vector<size_t> tx_channels = driver_options.get_transmit_channels();
  auto tx_stream = usrp_d.get_usrp_tx_stream();

  std::vector<std::vector<std::vector<std::complex<float>>>> pulses;
  std::vector<std::vector<std::complex<float>>> last_pulse_sent;

  double tx_center_freq = usrp_d.get_tx_center_freq(tx_channels[0]);
  double rx_center_freq = usrp_d.get_rx_center_freq(receive_channels[0]);

  uint32_t sqn_num = 0;
  uint32_t expected_sqn_num = 0;

  uint32_t num_recv_samples = 0;

  size_t ringbuffer_size;

  uhd::time_spec_t sequence_start_time;
  uhd::time_spec_t initialization_time;

  double seqtime;

  double agc_signal_read_delay =
      driver_options.get_agc_signal_read_delay() * 1e-6;

  auto clocks = borealis_clocks;
  auto system_since_epoch =
      std::chrono::duration<double>(clocks.system_time.time_since_epoch());
  auto gps_to_system_time_diff =
      system_since_epoch.count() - clocks.usrp_time.get_real_secs();

  zmq::message_t request;

  start_trigger.recv(request, zmq::recv_flags::none);
  memcpy(&ringbuffer_size, static_cast<size_t *>(request.data()),
         request.size());

  start_trigger.recv(request, zmq::recv_flags::none);
  memcpy(&initialization_time, static_cast<uhd::time_spec_t *>(request.data()),
         request.size());

  auto driver_ready_msg = std::string("DRIVER_READY");
  SEND_REPLY(driver_to_radar_control,
             driver_options.get_radctrl_to_driver_identity(), driver_ready_msg);

  /*
   * This loop accepts pulse by pulse from the radar_control. It parses the
   * samples, configures the USRP, sets up the timing, and then sends
   * samples/timing to the USRPs.
   */
  while (1) {
    auto more_pulses = true;
    std::vector<double> time_to_send_samples;
    uint32_t agc_status_bank_h = 0b0;
    uint32_t lp_status_bank_h = 0b0;
    uint32_t agc_status_bank_l = 0b0;
    uint32_t lp_status_bank_l = 0b0;
    while (more_pulses) {
      auto pulse_data =
          recv_data(driver_to_radar_control,
                    driver_options.get_radctrl_to_driver_identity());

      // Here we accept our driver_packet from the radar_control. We use that
      // info in order to configure the USRP devices based on experiment
      // requirements.
      TIMEIT_IF_TRUE_OR_DEBUG(
          false, COLOR_BLUE("TRANSMIT") << " total setup time: ",
          [&]() {
            if (driver_packet.ParseFromString(pulse_data) == false) {
              // TODO(keith): handle error
            }

            sqn_num = driver_packet.sequence_num();
            seqtime = driver_packet.seqtime();
            if (sqn_num != expected_sqn_num) {
              DEBUG_MSG("SEQUENCE NUMBER MISMATCH: SQN "
                        << sqn_num << " EXPECTED: " << expected_sqn_num);
              // TODO(keith): handle error
            }

            DEBUG_MSG(COLOR_BLUE("TRANSMIT")
                      << " burst flags: SOB " << driver_packet.sob() << " EOB "
                      << driver_packet.eob());

            TIMEIT_IF_TRUE_OR_DEBUG(
                false, COLOR_BLUE("TRANSMIT") << " center freq ",
                [&]() {
                  // If there is new center frequency data, set TX center
                  // frequency for each USRP TX channel.
                  if (tx_center_freq != driver_packet.txcenterfreq()) {
                    if (driver_packet.txcenterfreq() > 0.0 &&
                        driver_packet.sob() == true) {
                      DEBUG_MSG(COLOR_BLUE("TRANSMIT")
                                << " setting tx center freq to "
                                << driver_packet.txcenterfreq());
                      tx_center_freq = usrp_d.set_tx_center_freq(
                          driver_packet.txcenterfreq(), tx_channels,
                          uhd::time_spec_t(TUNING_DELAY));
                    }
                  }

                  // rxcenterfreq() will return 0 if it hasn't changed, so check
                  // for changes here
                  if (rx_center_freq != driver_packet.rxcenterfreq()) {
                    if (driver_packet.rxcenterfreq() > 0.0 &&
                        driver_packet.sob() == true) {
                      DEBUG_MSG(COLOR_BLUE("TRANSMIT")
                                << " setting rx center freq to "
                                << driver_packet.rxcenterfreq());
                      rx_center_freq = usrp_d.set_rx_center_freq(
                          driver_packet.rxcenterfreq(), receive_channels,
                          uhd::time_spec_t(TUNING_DELAY));
                    }
                  }
                }()

            );

            TIMEIT_IF_TRUE_OR_DEBUG(
                false,
                COLOR_BLUE("TRANSMIT") << " sample unpack time: ", [&]() {
                  if (driver_packet.sob() == true) {
                    pulses.clear();
                  }
                  // Parse new samples from driver packet if they exist.
                  if (driver_packet.channel_samples_size() >
                      0) {  // ~700us to unpack 4x1600 samples
                    last_pulse_sent =
                        make_tx_samples(driver_packet, driver_options);
                    samples_set = true;
                  }
                  pulses.push_back(last_pulse_sent);
                }());
          }();

          time_to_send_samples.push_back(driver_packet.timetosendsamples());

          if (driver_packet.sob() == true) {
            num_recv_samples = driver_packet.numberofreceivesamples();
          }

          if (driver_packet.eob() == true) { more_pulses = false; });
    }

    // In order to transmit, these parameters need to be set at least once.
    if (samples_set == false) {
      // TODO(keith): throw error
      continue;
    }

    // If grabbing start of vector using samples[i] it doesn't work (samples are
    // firked) You need to grab the ptr to the vector using
    // samples[a][b].data(). See tx_waveforms for how to do this properly. Also
    // see uhd::tx_streamer::send(...) in the uhd docs see 'const buffs_type &''
    // argument to the send function, the description should read 'Typedef for a
    // pointer to a single, or a collection of pointers to send buffers'.
    std::vector<std::vector<std::complex<float> *>> pulse_ptrs(pulses.size());
    for (uint32_t i = 0; i < pulses.size(); i++) {
      std::vector<std::complex<float> *> ptrs(pulses[i].size());
      for (uint32_t j = 0; j < pulses[i].size(); j++) {
        ptrs[j] = pulses[i][j].data();
      }
      pulse_ptrs[i] = ptrs;
    }

    // Getting usrp box time to find out when to send samples. usrp_time
    // continuously being updated.
    auto delay = uhd::time_spec_t(SET_TIME_COMMAND_DELAY);
    auto time_now = borealis_clocks.usrp_time;
    // Earliest possible time to start sending samples
    auto sequence_start_time = time_now + delay;

    if (driver_packet.align_sequences() == true) {
      // Get the digit of the next tenth of a second after min_start_time
      double tenth_of_second =
          std::ceil(sequence_start_time.get_frac_secs() *
                    10);  // Result is integer in 1 through 10
      double fractional_second = tenth_of_second / 10;
      // this occurs if the current time is 0.9+ seconds, so the rounding takes
      // it up to 1.0 seconds. 0.95 chosen as fractional second will always be
      // 0.0, 0.1, ..., 1.0 so it falls in between 0.9 and 1.0
      if (fractional_second >= 0.95) {
        fractional_second = 0.0;
      }

      // Start the sequence at the next tenth of a second.
      // 0.05 chosen as fractional second will always be 0.0, 0.1, etc so it
      // falls in between.
      if (fractional_second < 0.05) {
        // this occurs if the fractional second is 0.0 because the second has
        // rolled over
        sequence_start_time = uhd::time_spec_t(
            sequence_start_time.get_full_secs() + 1, fractional_second);
      } else {
        sequence_start_time = uhd::time_spec_t(
            sequence_start_time.get_full_secs(), fractional_second);
      }
    }

    auto seqn_sampling_time = num_recv_samples / rx_rate;
    TIMEIT_IF_TRUE_OR_DEBUG(
        false, COLOR_BLUE("TRANSMIT") << " full usrp time stuff ",
        [&]() {
          // Here we are time-aligning our time_zero to the start of a sample.
          // Do this by recalculating time_zero using the calculated value of
          // start_sample.
          // TODO(someone): Account for offset btw TX/RX (seems to change with
          // sampling rate at least)

          auto time_diff = sequence_start_time - initialization_time;
          double future_start_sample =
              std::floor(time_diff.get_real_secs() * rx_rate);
          auto time_from_initialization =
              uhd::time_spec_t((future_start_sample / rx_rate));

          sequence_start_time = initialization_time + time_from_initialization;

          TIMEIT_IF_TRUE_OR_DEBUG(
              false,
              COLOR_BLUE("TRANSMIT") << " time to send all samples to USRP: ",
              [&]() {
                for (uint32_t i = 0; i < pulses.size(); i++) {
                  auto md = TXMetadata();
                  md.set_has_time_spec(true);
                  auto time = sequence_start_time +
                              uhd::time_spec_t(time_to_send_samples[i] / 1.0e6);
                  md.set_time_spec(time);
                  // The USRP tx_metadata start_of_burst and end_of_burst
                  // describe start and end of the pulse samples.
                  md.set_start_of_burst(true);
                  md.set_end_of_burst(false);

                  // This will loop until all samples are sent to the usrp. Send
                  // will block until all samples sent or timed out(too many
                  // samples to send within timeout period). Send has a default
                  // timing of 0.1 seconds.
                  auto samples_per_pulse = pulses[i][0].size();

                  TIMEIT_IF_TRUE_OR_DEBUG(
                      false,
                      COLOR_BLUE("TRANSMIT")
                          << " time to send pulse " << i << " to USRP: ",
                      [&]() {
                        uint64_t total_samps_sent = 0;
                        while (total_samps_sent < samples_per_pulse) {
                          auto num_samps_to_send =
                              samples_per_pulse - total_samps_sent;

                          auto num_samps_sent = tx_stream->send(
                              pulse_ptrs[i], num_samps_to_send,
                              md.get_md());  // TODO(keith): Determine timeout
                                             // properties.
                          DEBUG_MSG(COLOR_BLUE("TRANSMIT")
                                    << " Samples sent " << num_samps_sent);

                          total_samps_sent += num_samps_sent;
                          md.set_start_of_burst(false);
                          md.set_has_time_spec(false);
                        }
                        md.set_end_of_burst(true);
                        tx_stream->send("", 0, md.get_md());
                      }()  // pulse lambda
                  );       // pulse timeit macro
                }

                // Read AGC and Low Power signals, bitwise OR to catch any time
                // the signals are active during this sequence for each USRP
                // individually
                usrp_d.clear_command_time();
                auto read_time = sequence_start_time + (seqtime * 1e-6) +
                                 agc_signal_read_delay;
                usrp_d.set_command_time(read_time);
                agc_status_bank_h =
                    agc_status_bank_h | usrp_d.get_agc_status_bank_h();
                lp_status_bank_h =
                    lp_status_bank_h | usrp_d.get_lp_status_bank_h();
                agc_status_bank_l =
                    agc_status_bank_l | usrp_d.get_agc_status_bank_l();
                lp_status_bank_l =
                    lp_status_bank_l | usrp_d.get_lp_status_bank_l();
                usrp_d.clear_command_time();

                for (uint32_t i = 0; i < pulses.size(); i++) {
                  uhd::async_metadata_t async_md;
                  std::vector<size_t> acks(tx_channels.size(), 0);
                  std::vector<size_t> lates(tx_channels.size(), 0);
                  size_t channel_acks = 0;
                  size_t channel_lates = 0;
                  // loop through all messages for the ACK packets (may have
                  // underflow messages in queue)
                  while (channel_acks < tx_channels.size() and
                         tx_stream->recv_async_msg(async_md)) {
                    if (async_md.event_code ==
                        uhd::async_metadata_t::EVENT_CODE_BURST_ACK) {
                      channel_acks++;
                      acks[async_md.channel]++;
                    }

                    if (async_md.event_code ==
                        uhd::async_metadata_t::EVENT_CODE_TIME_ERROR) {
                      channel_lates++;
                      lates[async_md.channel]++;
                    }
                  }

                  for (uint32_t j = 0; j < lates.size(); j++) {
                    DEBUG_MSG(COLOR_BLUE("TRANSMIT")
                              << ": channel " << j << " got " << lates[j]
                              << " lates for pulse " << i);
                  }

                  DEBUG_MSG(COLOR_BLUE("TRANSMIT")
                            << ": Sequence " << sqn_num << " Got "
                            << channel_acks << " acks out of "
                            << tx_channels.size() << " channels for pulse "
                            << i);
                  DEBUG_MSG(COLOR_BLUE("TRANSMIT")
                            << ": Sequence " << sqn_num << " Got "
                            << channel_lates << " lates out of "
                            << tx_channels.size() << " channels for pulse "
                            << i);
                }
              }()  // all pulses lambda
          );       // all pulses timeit macro
        }()        // full usrp function lambda
    );             // full usrp function timeit macro

    rxsamplesmetadata::RxSamplesMetadata samples_metadata;

    clocks = borealis_clocks;
    system_since_epoch =
        std::chrono::duration<double>(clocks.system_time.time_since_epoch());
    // get_real_secs() may lose precision of the fractional seconds, but it's
    // close enough
    gps_to_system_time_diff =
        system_since_epoch.count() - clocks.usrp_time.get_real_secs();

    samples_metadata.set_gps_locked(usrp_d.gps_locked());
    samples_metadata.set_gps_to_system_time_diff(gps_to_system_time_diff);

    if (!usrp_d.gps_locked()) {
      RUNTIME_MSG("GPS UNLOCKED! time diff: "
                  << COLOR_RED(gps_to_system_time_diff * 1000.0) << "ms");
    }

    auto end_time = borealis_clocks.usrp_time;
    auto sleep_time = uhd::time_spec_t(seqn_sampling_time) -
                      (end_time - sequence_start_time) + delay;
    // sleep_time is how much longer we need to wait in tx thread before the end
    // of the sampling time

    DEBUG_MSG(COLOR_BLUE("TRANSMIT")
              << ": Sleep time " << sleep_time.get_real_secs() * 1e6 << " us");

    if (sleep_time.get_real_secs() > 0.0) {
      auto duration = std::chrono::duration<double>(sleep_time.get_real_secs());
      std::this_thread::sleep_for(duration);
    }

    samples_metadata.set_rx_rate(rx_rate);
    samples_metadata.set_initialization_time(
        initialization_time.get_real_secs());
    samples_metadata.set_sequence_start_time(
        sequence_start_time.get_real_secs());
    samples_metadata.set_ringbuffer_size(ringbuffer_size);
    samples_metadata.set_numberofreceivesamples(num_recv_samples);
    samples_metadata.set_sequence_num(sqn_num);
    auto actual_finish = borealis_clocks.usrp_time;
    samples_metadata.set_sequence_time(
        (actual_finish - time_now).get_real_secs());

    samples_metadata.set_agc_status_bank_h(agc_status_bank_h);
    samples_metadata.set_lp_status_bank_h(lp_status_bank_h);
    samples_metadata.set_agc_status_bank_l(agc_status_bank_l);
    samples_metadata.set_lp_status_bank_l(lp_status_bank_l);

    std::string samples_metadata_str;
    samples_metadata.SerializeToString(&samples_metadata_str);

    // Here we wait for a request from dsp for the samples metadata, then send
    // it, bro! https://www.youtube.com/watch?v=WIrWyr3HgXI
    auto request = RECV_REQUEST(driver_to_dsp,
                                driver_options.get_dsp_to_driver_identity());
    SEND_REPLY(driver_to_dsp, driver_options.get_dsp_to_driver_identity(),
               samples_metadata_str);

    // Here we wait for a request from brian for the samples metadata, then send
    // it
    request = RECV_REQUEST(driver_to_brian,
                           driver_options.get_brian_to_driver_identity());
    SEND_REPLY(driver_to_brian, driver_options.get_brian_to_driver_identity(),
               samples_metadata_str);

    expected_sqn_num++;
    DEBUG_MSG(std::endl << std::endl);
  }  // while(1)
}

/**
 * @brief      Runs in a seperate thread to control receiving from the USRPs.
 *
 * @param[in]  driver_c        The driver ZMQ context.
 * @param[in]  usrp_d          The multi-USRP SuperDARN wrapper object.
 * @param[in]  driver_options  The driver options parsed from config.
 */
void receive(zmq::context_t &driver_c, USRP &usrp_d,
             const DriverOptions &driver_options) {
  DEBUG_MSG("Enter receive thread");

  zmq::socket_t start_trigger(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(start_trigger.bind("inproc://thread"));

  auto receive_channels = driver_options.get_receive_channels();
  uhd::rx_streamer::sptr rx_stream = usrp_d.get_usrp_rx_stream();

  auto usrp_buffer_size = rx_stream->get_max_num_samps();

  /* The ringbuffer_size is calculated this way because it's first truncated
     (size_t) then rescaled by usrp_buffer_size */
  size_t ringbuffer_size =
      size_t(driver_options.get_ringbuffer_size() /
             sizeof(std::complex<float>) / usrp_buffer_size) *
      usrp_buffer_size;

  SharedMemoryHandler shrmem(driver_options.get_ringbuffer_name());

  auto total_rbuf_size =
      receive_channels.size() * ringbuffer_size * sizeof(std::complex<float>);
  shrmem.create_shr_mem(total_rbuf_size);
  mlock(shrmem.get_shrmem_addr(), total_rbuf_size);

  std::vector<std::complex<float> *> buffer_ptrs_start;

  for (uint32_t i = 0; i < receive_channels.size(); i++) {
    auto ptr = static_cast<std::complex<float> *>(shrmem.get_shrmem_addr()) +
               (i * ringbuffer_size);
    buffer_ptrs_start.push_back(ptr);
  }

  std::vector<std::complex<float> *> buffer_ptrs = buffer_ptrs_start;

  uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
  stream_cmd.stream_now = false;
  stream_cmd.num_samps = 0;
  stream_cmd.time_spec =
      usrp_d.get_current_usrp_time() + uhd::time_spec_t(SET_TIME_COMMAND_DELAY);

  rx_stream->issue_stream_cmd(stream_cmd);

  uhd::rx_metadata_t meta;

  uint32_t timeout_count = 0;
  uint32_t overflow_count = 0;
  uint32_t overflow_oos_count = 0;
  uint32_t late_count = 0;
  uint32_t bchain_count = 0;
  uint32_t align_count = 0;
  uint32_t badp_count = 0;

  auto rx_rate = usrp_d.get_rx_rate();

  zmq::message_t ring_size(sizeof(ringbuffer_size));
  memcpy(ring_size.data(), &ringbuffer_size, sizeof(ringbuffer_size));
  start_trigger.send(ring_size, zmq::send_flags::none);

  // This loop receives 1 pulse sequence worth of samples.
  auto first_time = true;
  while (1) {
    // 3.0 is the timeout in seconds for the recv call, arbitrary number
    rx_stream->recv(buffer_ptrs, usrp_buffer_size, meta, 3.0, true);
    if (first_time) {
      zmq::message_t start_time(sizeof(meta.time_spec));
      memcpy(start_time.data(), &meta.time_spec, sizeof(meta.time_spec));
      start_trigger.send(start_time, zmq::send_flags::none);
      first_time = false;
    }
    borealis_clocks.system_time = std::chrono::system_clock::now();
    borealis_clocks.usrp_time = meta.time_spec;
    auto error_code = meta.error_code;

    switch (error_code) {
      case uhd::rx_metadata_t::ERROR_CODE_NONE:
        break;
      case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT: {
        std::cout << "Timed out!" << std::endl;
        timeout_count++;
        break;
      }
      case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW: {
        std::cout << "Overflow!" << std::endl;
        std::cout << "OOS: " << meta.out_of_sequence << std::endl;
        if (meta.out_of_sequence == 1) overflow_oos_count++;
        overflow_count++;
        break;
      }
      case uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND: {
        std::cout << "LATE!" << std::endl;
        late_count++;
        break;
      }
      case uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN: {
        std::cout << "BROKEN CHAIN!" << std::endl;
        bchain_count++;
      }
      case uhd::rx_metadata_t::ERROR_CODE_ALIGNMENT: {
        std::cout << "ALIGNMENT!" << std::endl;
        align_count++;
      }
      case uhd::rx_metadata_t::ERROR_CODE_BAD_PACKET: {
        std::cout << "BAD PACKET!" << std::endl;
        badp_count++;
      }
      default:
        break;
    }

    auto rx_packet_time_diff =
        meta.time_spec.get_real_secs() - stream_cmd.time_spec.get_real_secs();
    auto diff_sample = rx_packet_time_diff * rx_rate;
    auto true_sample =
        (int64_t(diff_sample / usrp_buffer_size) + 1) * usrp_buffer_size;
    auto ringbuffer_idx = true_sample % ringbuffer_size;

    for (size_t buffer_idx = 0; buffer_idx < buffer_ptrs_start.size();
         buffer_idx++) {
      buffer_ptrs[buffer_idx] = buffer_ptrs_start[buffer_idx] + ringbuffer_idx;
    }
  }
}

/**
 * @brief      UHD wrapped main function to start threads.
 *
 * @return     EXIT_SUCCESS
 *
 * Creates a new multi-USRP object using parameters from config file. Starts
 * receive and transmit threads to operate on the multi-USRP object.
 */
int32_t UHD_SAFE_MAIN(int32_t argc, char *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  DriverOptions driver_options;

  DEBUG_MSG(driver_options.get_device_args());
  DEBUG_MSG(driver_options.get_pps());
  DEBUG_MSG(driver_options.get_ref());
  DEBUG_MSG(driver_options.get_tx_subdev());

  //  Prepare our context
  zmq::context_t driver_context(1);
  auto identities = {driver_options.get_driver_to_radctrl_identity()};

  auto sockets_vector = create_sockets(driver_context, identities,
                                       driver_options.get_router_address());

  zmq::socket_t &driver_to_radar_control = sockets_vector[0];

  // Begin setup process.
  // This exchange signals to radar control that the devices are ready to go so
  // that it can begin processing experiments without low averages in the first
  // integration period.

  auto setup_data = recv_data(driver_to_radar_control,
                              driver_options.get_radctrl_to_driver_identity());

  driverpacket::DriverPacket driver_packet;
  if (driver_packet.ParseFromString(setup_data) == false) {
    // TODO(keith): handle error
  }

  USRP usrp_d(driver_options, driver_packet.txrate(), driver_packet.rxrate());
  auto tune_delay = uhd::time_spec_t(TUNING_DELAY);
  usrp_d.set_tx_center_freq(driver_packet.txcenterfreq(),
                            driver_options.get_transmit_channels(), tune_delay);
  usrp_d.set_rx_center_freq(driver_packet.rxcenterfreq(),
                            driver_options.get_receive_channels(), tune_delay);

  driver_to_radar_control.close();
  std::vector<std::thread> threads;

  // std::ref http://stackoverflow.com/a/15530639/1793295
  std::thread receive_t(receive, std::ref(driver_context), std::ref(usrp_d),
                        std::ref(driver_options));

  std::thread transmit_t(transmit, std::ref(driver_context), std::ref(usrp_d),
                         std::ref(driver_options));

  threads.push_back(std::move(transmit_t));
  threads.push_back(std::move(receive_t));

  for (auto &th : threads) {
    th.join();
  }

  return EXIT_SUCCESS;
}
