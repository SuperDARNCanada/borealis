#include <cmath>
#include <complex>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <string>
#include <thread>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/utils/thread.hpp>
#include <vector>
#define ADDR                                                    \
  std::string(                                                  \
      "num_recv_frames=512,num_send_frames=256,send_buff_size=" \
      "2304000")
#define CLKSRC std::string("internal")

#define RXSUBDEV std::string("A:A A:B")
#define TXSUBDEV std::string("A:A")

#define TXCHAN \
  { 0 }
#define RXCHAN \
  { 0, 1 }

#define TXRATE (250.0e3)
#define RXRATE (250.0e3)
#define TXFREQ 11e6
#define RXFREQ 11e6
#define DELAY 10e-3

#define PULSETIMES \
  { 0.0, 13500e-6, 18000e-6, 30000e-6, 33000e-6, 39000e-6, 40500e-6 }
#define SAMPSPERCHAN(x) int(RXRATE * (x.back() + 23.5e-3))

bool start_tx = false;
uhd::time_spec_t start_time;
size_t ringbuffer_size = 10000;
char *test_mode = 0;

// MAKE RAMPED PULSES
std::vector<std::complex<float>> make_ramped_pulse(double tx_rate) {
  auto amp = 1.0 / sqrt(2.0);
  auto pulse_len = 300.0e-6;

  int tr_start_pad = 60, tr_end_pad = 60;
  int num_samps_per_antenna =
      std::ceil(pulse_len * (tx_rate + tr_start_pad + tr_end_pad));
  std::vector<double> tx_freqs = {1e6};

  auto default_v = std::complex<float>(0.0, 0.0);
  std::vector<std::complex<float>> samples(num_samps_per_antenna, default_v);

  for (auto j = tr_start_pad; j < num_samps_per_antenna - tr_end_pad; j++) {
    auto nco_point = std::complex<float>(0.0, 0.0);

    for (auto freq : tx_freqs) {
      auto sampling_freq = 2 * M_PI * freq / tx_rate;

      auto radians = fmod(sampling_freq * j, 2 * M_PI);
      auto I = amp * cos(radians);
      auto Q = amp * sin(radians);

      nco_point += std::complex<float>(I, Q);
    }
    samples[j] = nco_point;
  }

  auto ramp_size = int(10e-6 * tx_rate);

  for (auto j = tr_start_pad, k = 0; j < tr_start_pad + ramp_size; j++, k++) {
    auto a = ((k)*1.0) / ramp_size;
    samples[j] *= std::complex<float>(a, 0);
  }

  for (auto j = num_samps_per_antenna - tr_end_pad - 1, k = 0;
       j > num_samps_per_antenna - tr_end_pad - 1 - ramp_size; j--, k++) {
    auto a = ((k)*1.0) / ramp_size;
    samples[j] *= std::complex<float>(a, 0);
  }

  return samples;
}

// RX THREAD
void recv(uhd::usrp::multi_usrp::sptr &usrp_d, std::vector<size_t> &rx_chans) {
  uhd::stream_args_t rx_stream_args("fc32", "sc16");
  rx_stream_args.channels = rx_chans;
  uhd::rx_streamer::sptr rx_stream = usrp_d->get_rx_stream(rx_stream_args);

  auto usrp_buffer_size = 100 * rx_stream->get_max_num_samps();
  ringbuffer_size =
      (size_t(500.0e6) / sizeof(std::complex<float>) / usrp_buffer_size) *
      usrp_buffer_size;
  std::vector<std::complex<float>> buffer(rx_chans.size() * ringbuffer_size);
  std::vector<std::complex<float> *> buffer_ptrs_start;

  for (uint32_t i = 0; i < rx_chans.size(); i++) {
    auto ptr = static_cast<std::complex<float> *>(buffer.data() +
                                                  (i * ringbuffer_size));
    buffer_ptrs_start.push_back(ptr);
  }

  std::vector<std::complex<float> *> buffer_ptrs = buffer_ptrs_start;
  uhd::stream_cmd_t rx_stream_cmd(
      uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
  rx_stream_cmd.stream_now = false;
  rx_stream_cmd.num_samps = 0;
  rx_stream_cmd.time_spec = usrp_d->get_time_now() + uhd::time_spec_t(DELAY);

  rx_stream->issue_stream_cmd(rx_stream_cmd);

  uhd::rx_metadata_t meta;

  uint32_t buffer_inc = 0;
  uint32_t timeout_count = 0;
  uint32_t overflow_count = 0;
  uint32_t overflow_oos_count = 0;
  uint32_t late_count = 0;
  uint32_t bchain_count = 0;
  uint32_t align_count = 0;
  uint32_t badp_count = 0;
  auto first_time = true;

  int test_trials = 0;
  int test_while = 1;
  while (test_while) {
    size_t num_rx_samples =
        rx_stream->recv(buffer_ptrs, usrp_buffer_size, meta, 3.0);
    std::cout << "Recv " << num_rx_samples << " samples" << std::endl;
    std::cout << "On ringbuffer idx: " << usrp_buffer_size * buffer_inc
              << std::endl;
    // timeout = 0.5;
    auto error_code = meta.error_code;
    std::cout << "RX TIME: " << meta.time_spec.get_real_secs() << std::endl;
    if (first_time) {
      start_time = meta.time_spec;
      start_tx = true;
      first_time = false;
    }
    switch (error_code) {
      case uhd::rx_metadata_t::ERROR_CODE_NONE:
        break;
      case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT: {
        std::cout << "Timed out!" << std::endl;
        // exit(-1);
        timeout_count++;
        // kill_loop = true;
        break;
      }
      case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW: {
        std::cout << "Overflow!" << std::endl;
        std::cout << "OOS: " << meta.out_of_sequence << std::endl;
        // exit(-1);
        if (meta.out_of_sequence == 1) overflow_oos_count++;
        overflow_count++;
        // kill_loop = true;
        break;
      }
      case uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND: {
        std::cout << "LATE!" << std::endl;
        late_count++;
        // kill_loop = true;
        // exit(1);
        break;
      }
      case uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN: {
        std::cout << "BROKEN CHAIN!" << std::endl;
        bchain_count++;
        // kill_loop = true;
      }
      case uhd::rx_metadata_t::ERROR_CODE_ALIGNMENT: {
        std::cout << "ALIGNMENT!" << std::endl;
        align_count++;
        // kill_loop = true;
      }
      case uhd::rx_metadata_t::ERROR_CODE_BAD_PACKET: {
        std::cout << "BAD PACKET!" << std::endl;
        badp_count++;
        // kill_loop = true;
      }
      default:
        break;
    }
    if ((buffer_inc + 1) * usrp_buffer_size < ringbuffer_size) {
      for (auto &buffer_ptr : buffer_ptrs) {
        buffer_ptr += usrp_buffer_size;
      }
      buffer_inc++;
    } else {
      buffer_ptrs = buffer_ptrs_start;
      buffer_inc = 0;
    }

    std::cout << "Timeout count: " << timeout_count << std::endl;
    std::cout << "Overflow count: " << overflow_count << std::endl;
    std::cout << "Overflow oos count: " << overflow_oos_count << std::endl;
    std::cout << "Late count: " << late_count << std::endl;
    std::cout << "Broken chain count: " << bchain_count << std::endl;
    std::cout << "Alignment count: " << align_count << std::endl;
    std::cout << "Bad packet count: " << badp_count << std::endl;

    if (!std::strcmp(test_mode, "full")) {
      test_trials += 1;
    }
    if (test_trials == 10) {
      test_while = 0;
      test_trials = 0;
    }
  }
}

// TX THREAD
void tx(uhd::usrp::multi_usrp::sptr &usrp_d, std::vector<size_t> &tx_chans) {
  uhd::stream_args_t tx_stream_args("fc32", "sc16");
  tx_stream_args.channels = tx_chans;
  uhd::tx_streamer::sptr tx_stream = usrp_d->get_tx_stream(tx_stream_args);

  auto pulse = make_ramped_pulse(TXRATE);
  std::vector<std::vector<std::complex<float>>> tx_samples(tx_chans.size(),
                                                           pulse);

  std::vector<double> time_per_pulse = PULSETIMES;

  uint32_t count = 1;
  int test_trials = 0;
  int test_while = 1;
  while (test_while) {
    auto u_time_now = usrp_d->get_time_now();
    auto time_zero = u_time_now + uhd::time_spec_t(DELAY);

    std::cout << "Starting tx #" << count << std::endl;

    auto send = [&](double start_time) {
      uhd::tx_metadata_t meta;
      meta.has_time_spec = true;
      auto time_to_send_pulse = uhd::time_spec_t(start_time);
      auto pulse_start_time = time_zero + time_to_send_pulse;
      meta.time_spec = pulse_start_time;

      meta.start_of_burst = true;

      uint64_t num_samps_sent = 0;
      auto samples_per_buff = tx_samples[0].size();

      while (num_samps_sent < samples_per_buff) {
        auto num_samps_to_send = samples_per_buff - num_samps_sent;
        num_samps_sent = tx_stream->send(tx_samples, num_samps_to_send, meta);
        meta.start_of_burst = false;
        meta.has_time_spec = false;
      }

      meta.end_of_burst = true;
      tx_stream->send("", 0, meta);
    };

    for (uint32_t i = 0; i < time_per_pulse.size(); i++) {
      send(time_per_pulse[i]);
    }

    auto seq_time = (time_per_pulse.back() + 23.5e-3);
    auto start_sample =
        uint32_t((time_zero.get_real_secs() - start_time.get_real_secs()) *
                 RXRATE) %
        ringbuffer_size;

    usleep((seq_time + 2 * DELAY) * 1.0e6);
    if ((start_sample + (seq_time * RXRATE)) > ringbuffer_size) {
      auto end_sample =
          uint32_t(start_sample + (seq_time * RXRATE)) - ringbuffer_size;

      std::cout << "Tx #" << count << " needs sample " << start_sample << " to "
                << ringbuffer_size - 1 << " and 0 to " << end_sample
                << std::endl;
    } else {
      auto end_sample = uint32_t(start_sample + (seq_time * RXRATE));

      std::cout << "Tx #" << count << " needs sample " << start_sample << " to "
                << end_sample << std::endl;
    }

    usrp_d->clear_command_time();
    count++;

    if (!std::strcmp(test_mode, "full")) {
      test_trials += 1;
    }
    if (test_trials == 10) {
      test_while = 0;
      test_trials = 0;
    }
  }
}

// MAIN LOOP
int UHD_SAFE_MAIN(int argc, char *argv[]) {
  test_mode = argv[2];

  // Error for incomplete input arguments.
  if (argc != 3) {
    std::cout
        << "TXIO Board Tesing requires address and testing mode arguments."
        << std::endl;
    std::cout << "Test mode: txrx, txo, rxo, idle, full." << std::endl;
  }

  // Output heading information.
  std::cout << "" << std::endl;
  std::cout << "-----------------------------------------TXIO BOARD TESTING "
               "SCRIPT--------------------------------------------------"
            << std::endl;
  std::cout << "Version: " << argv[0] << std::endl;
  std::cout << "Unit IP Address: " << argv[1] << std::endl;
  std::cout << "Test mode: " << argv[2] << std::endl;
  std::cout << "" << std::endl;

  uhd::set_thread_priority_safe();
  auto usrp_d = uhd::usrp::multi_usrp::make(ADDR + "," + argv[1]);
  usrp_d->set_clock_source(CLKSRC);

  usrp_d->set_rx_subdev_spec(RXSUBDEV);

  usrp_d->set_rx_rate(RXRATE);

  std::vector<size_t> rx_chans = RXCHAN;

  uhd::tune_request_t rx_tune_request(RXFREQ);
  for (auto &channel : rx_chans) {
    usrp_d->set_rx_freq(rx_tune_request, channel);
    double actual_freq = usrp_d->get_rx_freq(channel);
    if (actual_freq != RXFREQ) {
      std::cout << "requested rx ctr freq " << RXFREQ << " actual freq "
                << actual_freq << std::endl;
    }
  }

  auto tt = std::chrono::high_resolution_clock::now();
  auto tt_sc = std::chrono::duration_cast<std::chrono::duration<double>>(
      tt.time_since_epoch());
  while (tt_sc.count() - std::floor(tt_sc.count()) < 0.2 or
         tt_sc.count() - std::floor(tt_sc.count()) > 0.3) {
    tt = std::chrono::high_resolution_clock::now();
    tt_sc = std::chrono::duration_cast<std::chrono::duration<double>>(
        tt.time_since_epoch());
    usleep(10000);
  }

  usrp_d->set_time_now(tt_sc.count());

  for (uint32_t i = 0; i < usrp_d->get_num_mboards(); i++) {
    usrp_d->set_gpio_attr("RXA", "CTRL", 0xFFFF, 0b11111111, i);
    usrp_d->set_gpio_attr("RXA", "DDR", 0xFFFF, 0b11111111, i);

    // Mirror pins along bank for easier scoping.
    usrp_d->set_gpio_attr("RXA", "ATR_RX", 0xFFFF, 0b000000010, i);
    usrp_d->set_gpio_attr("RXA", "ATR_RX", 0xFFFF, 0b000000100, i);

    usrp_d->set_gpio_attr("RXA", "ATR_TX", 0xFFFF, 0b000001000, i);
    usrp_d->set_gpio_attr("RXA", "ATR_TX", 0xFFFF, 0b000010000, i);

    // XX is the actual TR signal
    usrp_d->set_gpio_attr("RXA", "ATR_XX", 0xFFFF, 0b000100000, i);
    usrp_d->set_gpio_attr("RXA", "ATR_XX", 0xFFFF, 0b001000000, i);

    // 0X acts as 'scope sync'
    usrp_d->set_gpio_attr("RXA", "ATR_0X", 0xFFFF, 0b010000000, i);
    usrp_d->set_gpio_attr("RXA", "ATR_0X", 0xFFFF, 0b100000000, i);
  }

  // tx config
  std::vector<size_t> tx_chans = TXCHAN;

  usrp_d->set_tx_subdev_spec(TXSUBDEV);
  usrp_d->set_tx_rate(TXRATE);

  uhd::tune_request_t tx_tune_request(TXFREQ);
  for (auto &channel : tx_chans) {
    usrp_d->set_tx_freq(tx_tune_request, channel);
    double actual_freq = usrp_d->get_tx_freq(channel);
    if (actual_freq != RXFREQ) {
      std::cout << "requested tx ctr freq " << TXFREQ << " actual freq "
                << actual_freq << std::endl;
    }
  }

  // Select the testing sequence to be run.
  if (!std::strcmp(argv[2], "txrx")) {
    std::thread recv_t(recv, std::ref(usrp_d), std::ref(rx_chans));
    std::thread tx_t(tx, std::ref(usrp_d), std::ref(tx_chans));
    recv_t.join();
    tx_t.join();
  } else if (!std::strcmp(argv[2], "txo")) {
    std::thread tx_t(tx, std::ref(usrp_d), std::ref(tx_chans));
    tx_t.join();
  } else if (!std::strcmp(argv[2], "rxo")) {
    std::thread recv_t(recv, std::ref(usrp_d), std::ref(rx_chans));
    recv_t.join();
  } else if (!std::strcmp(argv[2], "idle")) {
    std::cout << "IDLE..." << std::endl;
    while (1) {
    }
  } else if (!std::strcmp(argv[2], "full")) {
    /*// TX Only
    std::cout << "Test mode: txo" << std::endl;
            std::thread tx_t(tx,std::ref(usrp_d), std::ref(tx_chans));
            tx_t.join();

            // TX/RX
    std::cout << "Test mode: txrx" << std::endl;
            std::thread recv_t(recv,std::ref(usrp_d), std::ref(rx_chans));
            recv_t.join();

    // RX Only
    std::cout << "Test mode: rxo" << std::endl;
            tx_t.detach();

            // Idle
    std::cout << "Test mode: idle" << std::endl;
    recv_t.detach();
            */
    std::cout << "Not yet implemented." << std::endl;
    // Error still occurs when switching from txrx to rxo. Need to terminate the
    // thread. Not sure how to do. Have tried ~Thread and .terminate().
    return 0;
  } else {
    // Error
    std::cout << "Invalid testing mode selected." << std::endl;
    return 0;
  }

  return 0;
}
