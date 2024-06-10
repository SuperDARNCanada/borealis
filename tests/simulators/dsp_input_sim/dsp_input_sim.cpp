#include <stdint.h>
#include <time.h>
#include <unistd.h>

#include <complex>
#include <fstream>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include <zmq.hpp>
#include <zmq_addon.hpp>

#include "utils/driver_options/driveroptions.hpp"
#include "utils/protobuf/rxsamplesmetadata.pb.h"
#include "utils/protobuf/sigprocpacket.pb.h"
#include "utils/shared_memory/shared_memory.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"
#include "utils/zmq_borealis_helpers/zmq_borealis_helpers.hpp"

/*std::string random_string( size_t length )
{
  auto randchar = []() -> char
  {
    const char charset[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[ rand() % max_index ];
  };
  std::string str(length,0);
  std::generate_n( str.begin(), length, randchar );
  return str;
}
*/
std::vector<std::complex<float>> simulate_samples(
    uint32_t num_antennas, uint32_t num_samps_per_antenna,
    std::vector<double> rx_freqs, double rx_rate, bool use_noise) {
  auto default_v = std::complex<float>(0.0, 0.0);
  auto total_samps = num_antennas * num_samps_per_antenna;
  std::vector<std::complex<float>> samples(total_samps, default_v);

  auto seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::normal_distribution<double> distribution(0.0, 1.0);

  auto amp = 1.0 / sqrt(2.0);
  for (auto i = 0; i < num_antennas; i++) {
    for (auto j = 0; j < num_samps_per_antenna; j++) {
      auto nco_point = std::complex<float>(0.0, 0.0);

      for (auto freq : rx_freqs) {
        auto sampling_freq = 2 * M_PI * freq / rx_rate;

        auto radians = fmod(sampling_freq * j, 2 * M_PI);
        auto noise = 0.0;
        if (use_noise == true) {
          noise = 0.1 * distribution(generator);
        }
        auto I = amp * cos(radians) + noise;
        auto Q = amp * sin(radians) + noise;

        nco_point += std::complex<float>(I, Q);
      }
      samples[(i * num_samps_per_antenna) + j] = nco_point;
    }
  }

  for (auto i = 0; i < num_antennas; i++) {
    auto start = int(0.3 * num_samps_per_antenna);
    auto end = int(0.6 * num_samps_per_antenna);
    auto ramp_size = int(0.1 * num_samps_per_antenna);

    for (auto j = 0; j < start; j++) {
      samples[(i * num_samps_per_antenna) + j] = std::complex<float>(0.0, 0.0);
    }

    for (auto j = start; j < start + ramp_size; j++) {
      auto a = ((j - start + 1) * 1.0) / ramp_size;
      samples[(i * num_samps_per_antenna) + j] *= std::complex<float>(a, 0);
    }

    for (auto j = end - ramp_size; j < end; j++) {
      auto a = (1 - (((j - end + 1) + ramp_size) * 1.0) / ramp_size);
      samples[(i * num_samps_per_antenna) + j] *= std::complex<float>(a, 0);
    }

    for (auto j = end; j < num_samps_per_antenna; j++) {
      samples[(i * num_samps_per_antenna) + j] = std::complex<float>(0.0, 0.0);
    }
  }

  std::ofstream samples_file("simulated_samples.dat",
                             std::ios::out | std::ios::binary);
  samples_file.write((char *)samples.data(),
                     samples.size() * sizeof(std::complex<float>));

  return samples;
}
void signals(zmq::context_t &context) {
  auto sig_options = SignalProcessingOptions();

  auto identities = {sig_options.get_radctrl_dsp_identity(),
                     sig_options.get_driver_dsp_identity(),
                     sig_options.get_exphan_dsp_identity(),
                     sig_options.get_dw_dsp_identity(),
                     sig_options.get_brian_dspbegin_identity(),
                     sig_options.get_brian_dspend_identity()};

  auto sockets_vector =
      create_sockets(context, identities, sig_options.get_router_address());

  zmq::socket_t &radar_control_to_dsp = sockets_vector[0];
  zmq::socket_t &driver_to_dsp = sockets_vector[1];
  zmq::socket_t &experiment_handler_to_dsp = sockets_vector[2];
  zmq::socket_t &data_write_to_dsp = sockets_vector[3];
  zmq::socket_t &brian_to_dspbegin = sockets_vector[4];
  zmq::socket_t &brian_to_dspend = sockets_vector[5];

  sigprocpacket::SigProcPacket sp;

  auto driver_options = DriverOptions();
  auto rx_rate = driver_options.get_rx_rate();

  std::vector<double> rx_freqs = {1.0e6, 2.0e6, 3.0e6};
  for (int i = 0; i < rx_freqs.size(); i++) {
    auto rxchan = sp.add_rxchannel();
    rxchan->set_rxfreq(rx_freqs[i]);
    rxchan->set_nrang(75);
    rxchan->set_frang(180);
  }

  auto num_antennas = driver_options.get_main_antenna_count() +
                      driver_options.get_interferometer_antenna_count();

  rxsamplesmetadata::RxSamplesMetadata samples_metadata;

  auto num_samples = uint32_t(rx_rate * 0.069);
  samples_metadata.set_numberofreceivesamples(num_samples);

  auto samples =
      simulate_samples(num_antennas, num_samples, rx_freqs, rx_rate, false);

  auto usrp_buffer_size = 363;
  /* The ringbuffer_size is calculated this way because it's first truncated
     (size_t) then rescaled by usrp_buffer_size */
  size_t ringbuffer_size =
      size_t(driver_options.get_ringbuffer_size() /
             sizeof(std::complex<float>) / usrp_buffer_size) *
      usrp_buffer_size;

  SharedMemoryHandler shrmem(driver_options.get_ringbuffer_name());

  shrmem.create_shr_mem(num_antennas * ringbuffer_size *
                        sizeof(std::complex<float>));

  std::vector<std::complex<float> *> buffer_ptrs_start;

  for (uint32_t i = 0; i < num_antennas; i++) {
    auto ptr = static_cast<std::complex<float> *>(shrmem.get_shrmem_addr()) +
               (i * ringbuffer_size);
    buffer_ptrs_start.push_back(ptr);
  }

  for (uint32_t i = 0; i < num_antennas; i++) {
    size_t counter = 0;
    while (counter < ringbuffer_size) {
      buffer_ptrs_start[i][counter] =
          samples[i * num_samples + counter % num_samples];
      counter++;
    }
  }

  auto sqn_num = 0;

  sp.set_sequence_num(sqn_num);
  samples_metadata.set_sequence_num(sqn_num);

  std::chrono::steady_clock::time_point timing_ack_start, timing_ack_end,
      total_time_start, total_time_end;
  std::chrono::milliseconds accum_time(0);

  auto seq_counter = 0;
  while (1) {
    std::string r_msg_str;
    sp.SerializeToString(&r_msg_str);

    // auto request = RECV_REQUEST(radar_control_to_dsp,
    // sig_options.get_dsp_radctrl_identity());
    SEND_REPLY(radar_control_to_dsp, sig_options.get_dsp_radctrl_identity(),
               r_msg_str);

    /*auto name_str = random_string(10);

    auto shr_start = std::chrono::steady_clock::now();
    SharedMemoryHandler shrmem(name_str);
    auto size = samples.size() * sizeof(std::complex<float>);
    shrmem.create_shr_mem(size);
    memcpy(shrmem.get_shrmem_addr(), samples.data(), size);
    auto shr_end = std::chrono::steady_clock::now();
    std::cout << "shrmem + memcpy for #" << sp.sequence_num()
      << " after "
      << std::chrono::duration_cast<std::chrono::milliseconds>(shr_end -
    shr_start).count()
      << "ms" << std::endl;
*/
    samples_metadata.set_initialization_time(0.0);
    samples_metadata.set_ringbuffer_size(ringbuffer_size);
    double start_time = (sqn_num + 1) * num_samples / rx_rate;
    samples_metadata.set_sequence_start_time(start_time);
    std::cout << "Sending data with sequence_num: " << sp.sequence_num()
              << std::endl;

    // samples_metadata.set_shrmemname(name_str.c_str());

    std::string samples_metadata_str;
    samples_metadata.SerializeToString(&samples_metadata_str);
    auto request =
        RECV_REQUEST(driver_to_dsp, sig_options.get_dsp_driver_identity());
    SEND_REPLY(driver_to_dsp, sig_options.get_dsp_driver_identity(),
               samples_metadata_str);

    total_time_start = std::chrono::steady_clock::now();
    timing_ack_start = std::chrono::steady_clock::now();

    request = std::string("Need ack");
    SEND_REQUEST(brian_to_dspbegin, sig_options.get_dspbegin_brian_identity(),
                 request);
    auto ack = RECV_REPLY(brian_to_dspbegin,
                          sig_options.get_dspbegin_brian_identity());
    sigprocpacket::SigProcPacket ack_from_dsp;
    ack_from_dsp.ParseFromString(ack);

    timing_ack_end = std::chrono::steady_clock::now();

    seq_counter++;
    auto seq_time = timing_ack_end - timing_ack_start;
    std::cout << "Received ack #" << ack_from_dsp.sequence_num()
              << " for sequence #" << sp.sequence_num() << " after "
              << std::chrono::duration_cast<std::chrono::milliseconds>(seq_time)
                     .count()
              << "ms" << std::endl;

    accum_time +=
        std::chrono::duration_cast<std::chrono::milliseconds>(seq_time);
    std::cout << "ACCUM_TIME " << accum_time.count() << std::endl;
    if (accum_time > std::chrono::milliseconds(3000)) {
      std::cout << "GETTING " << seq_counter << " SEQUENCES IN 3 SECONDS"
                << std::endl;
      seq_counter = 0;
      accum_time = std::chrono::milliseconds(0);
    }

    auto timing_timing_start = std::chrono::steady_clock::now();

    request = std::string("Need timing");
    SEND_REQUEST(brian_to_dspend, sig_options.get_dspend_brian_identity(),
                 request);
    auto timing =
        RECV_REPLY(brian_to_dspend, sig_options.get_dspend_brian_identity());

    sigprocpacket::SigProcPacket timing_from_dsp;
    timing_from_dsp.ParseFromString(timing);

    auto timing_timing_end = std::chrono::steady_clock::now();
    auto timing_time = timing_timing_end - timing_timing_start;
    std::cout << "Received timing for sequence #"
              << timing_from_dsp.sequence_num() << " after "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     timing_time)
                     .count()
              << "ms with decimation timing of " << timing_from_dsp.kerneltime()
              << "ms" << std::endl;

    request = std::string("Need data");
    SEND_REQUEST(data_write_to_dsp, sig_options.get_dsp_dw_identity(), request);
    auto data =
        RECV_REPLY(data_write_to_dsp, sig_options.get_dsp_dw_identity());

    total_time_end = std::chrono::steady_clock::now();
    auto total_time = total_time_end - total_time_start;
    std::cout << "Sequence_num #" << timing_from_dsp.sequence_num()
              << " total time "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     total_time)
                     .count()
              << "ms" << std::endl;

    sqn_num++;
    sp.set_sequence_num(sqn_num);
    samples_metadata.set_sequence_num(sqn_num);

#ifdef DEBUG
    usleep(0.1e6);
#endif
  }
}

int main(int argc, char **argv) {
  srand(time(NULL));
  zmq::context_t context(1);
  auto sig_options = SignalProcessingOptions();

  std::vector<std::thread> threads;
  // std::thread router_t(router,std::ref(context),
  // sig_options.get_router_address());
  std::thread signals_t(signals, std::ref(context));

  // threads.push_back(std::move(router_t));
  threads.push_back(std::move(signals_t));

  for (auto &th : threads) {
    th.join();
  }
}
