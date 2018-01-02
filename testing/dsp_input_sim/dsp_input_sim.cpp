#include <iostream>
#include <fstream>
#include <vector>
#include <complex>
#include <thread>
#include <unistd.h>
#include <stdint.h>
#include <zmq.hpp>
#include <random>
#include "utils/protobuf/rxsamplesmetadata.pb.h"
#include "utils/protobuf/sigprocpacket.pb.h"
#include "utils/shared_memory/shared_memory.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"
#include "utils/driver_options/driveroptions.hpp"
#include <time.h>

std::string random_string( size_t length )
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

void send_data(rxsamplesmetadata::RxSamplesMetadata& samples_metadata,
        sigprocpacket::SigProcPacket& sp, std::vector<std::complex<float>>& samples,
        zmq::socket_t& driver_socket, zmq::socket_t& radctrl_socket,
        std::chrono::steady_clock::time_point& timing_ack_start)
{
  std::string r_msg_str;
  sp.SerializeToString(&r_msg_str);
  zmq::message_t r_msg (r_msg_str.size());
  memcpy ((void *) r_msg.data (), r_msg_str.c_str(), r_msg_str.size());

  auto name_str = random_string(10);

  auto shr_start = std::chrono::steady_clock::now();
  SharedMemoryHandler shrmem(name_str);
  auto size = samples.size() * sizeof(std::complex<float>);
  shrmem.create_shr_mem(size);
  memcpy(shrmem.get_shrmem_addr(), samples.data(), size);
  auto shr_end = std::chrono::steady_clock::now();
  std::cout << "shrmem + memcpy for #" << sp.sequence_num()
    << " after "
    << std::chrono::duration_cast<std::chrono::milliseconds>(shr_end - shr_start).count()
    << "ms" << std::endl;

  std::cout << "Sending data with sequence_num: " << sp.sequence_num() << std::endl;

  radctrl_socket.send(r_msg);

  samples_metadata.set_shrmemname(name_str.c_str());

  std::string samples_metadata_str;
  samples_metadata.SerializeToString(&samples_metadata_str);
  zmq::message_t samples_metadata_msg (samples_metadata_str.size());
  memcpy ((void *) samples_metadata_msg.data (), samples_metadata_str.c_str(),
      samples_metadata_str.size());

  driver_socket.send(samples_metadata_msg);
}

void timing(zmq::context_t &context)
{
  auto sig_options = SignalProcessingOptions();
  zmq::socket_t timing_socket(context, ZMQ_PAIR);
  timing_socket.connect(sig_options.get_timing_socket_address());

  auto timing_timing_end = std::chrono::steady_clock::now();
  auto timing_timing_start = std::chrono::steady_clock::now();
  while(1) {
      sigprocpacket::SigProcPacket timing_from_dsp;
      zmq::message_t timing;
      timing_socket.recv(&timing);
      std::string s_msg_str2(static_cast<char*>(timing.data()), timing.size());
      timing_from_dsp.ParseFromString(s_msg_str2);

      timing_timing_end = std::chrono::steady_clock::now();
      auto seq_time = timing_timing_end - timing_timing_start;
      std::cout << "Received timing for sequence #" << timing_from_dsp.sequence_num()
        << " after " << std::chrono::duration_cast<std::chrono::milliseconds>(seq_time).count()
        << "ms with decimation timing of " << timing_from_dsp.kerneltime() << "ms" <<  std::endl;

      timing_timing_start = std::chrono::steady_clock::now();

    }
}

std::vector<std::complex<float>> simulate_samples(uint32_t num_antennas,
                                                    uint32_t num_samps_per_antenna,
                                                    std::vector<double> rx_freqs,double rx_rate,
                                                    bool use_noise)
{
    auto default_v = std::complex<float>(0.0,0.0);
    auto total_samps = num_antennas * num_samps_per_antenna;
    std::vector<std::complex<float>> samples(total_samps,default_v);

    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);

    std::normal_distribution<double> distribution (0.0,1.0);

    auto amp = 1.0/sqrt(2.0);
    for (auto i=0; i<num_antennas; i++) {
      for (auto j=0; j< num_samps_per_antenna; j++) {
        auto nco_point = std::complex<float>(0.0,0.0);

        for (auto freq : rx_freqs) {
          auto sampling_freq = 2 * M_PI * freq/rx_rate;

          auto radians = fmod(sampling_freq * j, 2 * M_PI);
          auto noise = 0.0;
          if (use_noise == true) {
            noise = 0.1 * distribution(generator);
          }
          auto I = amp * cos(radians) + noise;
          auto Q = amp * sin(radians) + noise;

          nco_point += std::complex<float>(I,Q);
        }
        samples[(i * num_samps_per_antenna) + j] = nco_point;
      }
    }


    for(auto i=0; i<num_antennas; i++) {
      auto start = int(0.3 * num_samps_per_antenna);
      auto end = int(0.6 * num_samps_per_antenna);
      auto ramp_size = int(0.1 * num_samps_per_antenna);

      for (auto j=0; j<start; j++){
        samples[(i * num_samps_per_antenna) + j] = std::complex<float>(0.0,0.0);
      }

      for (auto j=start; j<start+ramp_size; j++){
        auto a = ((j-start+1)*1.0)/ramp_size;
        samples[(i * num_samps_per_antenna) + j] *= std::complex<float>(a,0);
      }

      for (auto j=end-ramp_size;j<end;j++){
        auto a = (1 - (((j-end+1)+ramp_size)*1.0)/ramp_size);
        samples[(i * num_samps_per_antenna) + j] *= std::complex<float>(a,0);
      }

      for (auto j=end; j<num_samps_per_antenna;j++){
        samples[(i * num_samps_per_antenna) + j] = std::complex<float>(0.0,0.0);
      }

    }

    std::ofstream samples_file("simulated_samples.dat", std::ios::out | std::ios::binary);
    samples_file.write((char*)samples.data(), samples.size()*sizeof(std::complex<float>));

    return samples;
}
void signals(zmq::context_t &context)
{
  auto sig_options = SignalProcessingOptions();

  zmq::socket_t driver_socket(context, ZMQ_PAIR);
  driver_socket.connect(sig_options.get_driver_socket_address());

  zmq::socket_t radctrl_socket(context, ZMQ_PAIR);
  radctrl_socket.connect(sig_options.get_radar_control_socket_address());

  zmq::socket_t ack_socket(context, ZMQ_PAIR);
  ack_socket.connect(sig_options.get_ack_socket_address());

  sigprocpacket::SigProcPacket sp;

  auto driver_options = DriverOptions();
  auto rx_rate = driver_options.get_rx_rate();


  std::vector<double> rx_freqs = {3.0e6, 2.0e6, 1.0e6};
  for (int i=0; i<rx_freqs.size(); i++) {
    auto rxchan = sp.add_rxchannel();
    rxchan->set_rxfreq(rx_freqs[i]);
    rxchan->set_nrang(75);
    rxchan->set_frang(180);

  }

  auto num_antennas = driver_options.get_main_antenna_count() +
              driver_options.get_interferometer_antenna_count();


  rxsamplesmetadata::RxSamplesMetadata samples_metadata;

  auto num_samples = uint32_t(rx_rate* 0.1);
  samples_metadata.set_numberofreceivesamples(num_samples);

  auto samples = simulate_samples(num_antennas, num_samples, rx_freqs, rx_rate, true);

  auto sqn_num = 0;

  sp.set_sequence_num(sqn_num);
  samples_metadata.set_sequence_num(sqn_num);

  std::chrono::steady_clock::time_point timing_ack_start, timing_ack_end;
  std::chrono::milliseconds accum_time(0);

  send_data(samples_metadata, sp, samples, driver_socket, radctrl_socket, timing_ack_start);
  sqn_num += 1;

  auto seq_counter = 0;
  while(1) {
    sigprocpacket::SigProcPacket ack_from_dsp;
    zmq::message_t ack;
    ack_socket.recv(&ack);
    std::string s_msg_str1(static_cast<char*>(ack.data()), ack.size());
    ack_from_dsp.ParseFromString(s_msg_str1);

    timing_ack_end = std::chrono::steady_clock::now();
    seq_counter++;
    auto seq_time = timing_ack_end - timing_ack_start;
    std::cout << "Received ack #"<< ack_from_dsp.sequence_num() << " for sequence #"
      << sp.sequence_num() << " after "
      << std::chrono::duration_cast<std::chrono::milliseconds>(seq_time).count()
      << "ms" << std::endl;

    accum_time +=  std::chrono::duration_cast<std::chrono::milliseconds>(seq_time);
    std::cout << "ACCUM_TIME " <<accum_time.count() << std::endl;
    if (accum_time > std::chrono::milliseconds(3000)){
        std::cout << "GETTING " << seq_counter << " SEQUENCES IN 3 SECONDS" << std::endl;
        seq_counter = 0;
        accum_time = std::chrono::milliseconds(0);
    }

    timing_ack_start = std::chrono::steady_clock::now();
    sp.set_sequence_num(sqn_num);

    samples_metadata.set_sequence_num(sqn_num);

    send_data(samples_metadata,sp, samples,driver_socket, radctrl_socket, timing_ack_start);
    #ifdef DEBUG
      sleep(30);
    #endif
    sqn_num += 1;
  }
}

int main(int argc, char** argv){

  srand(time(NULL));
  zmq::context_t context(1);

  std::vector<std::thread> threads;
  std::thread timing_t(timing,std::ref(context));
  std::thread signals_t(signals, std::ref(context));

  threads.push_back(std::move(timing_t));
  threads.push_back(std::move(signals_t));

  for (auto& th : threads) {
    th.join();
  }

}
