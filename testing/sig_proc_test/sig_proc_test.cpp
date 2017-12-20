#include <iostream>
#include <vector>
#include <complex>
#include <thread>
#include <unistd.h>
#include <stdint.h>
#include <zmq.hpp>
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

  timing_ack_start = std::chrono::steady_clock::now();

  radctrl_socket.send(r_msg);

  samples_metadata.set_shrmemname(name_str.c_str());

  std::string samples_metadata_str;
  samples_metadata.SerializeToString(&samples_metadata_str);
  zmq::message_t samples_metadata_msg (samples_metadata_str.size());
  memcpy ((void *) samples_metadata_msg.data (), samples_metadata_str.c_str(),
      samples_metadata_str.size());

  driver_socket.send(samples_metadata_msg);
}

int main(int argc, char** argv){

  srand(time(NULL));
  zmq::context_t context(1);
  zmq::socket_t driver_socket(context, ZMQ_PAIR);
  driver_socket.connect("ipc:///tmp/feeds/1");

  zmq::socket_t radctrl_socket(context, ZMQ_PAIR);
  radctrl_socket.connect("ipc:///tmp/feeds/2");

  zmq::socket_t ack_socket(context, ZMQ_PAIR);
  ack_socket.connect("ipc:///tmp/feeds/3");

  zmq::socket_t timing_socket(context, ZMQ_PAIR);
  timing_socket.connect("ipc:///tmp/feeds/4");

  sigprocpacket::SigProcPacket sp;

  zmq::pollitem_t sockets[] = {
    {ack_socket,0,ZMQ_POLLIN,0},
    {timing_socket,0,ZMQ_POLLIN,0},
  };

  auto driver_options = DriverOptions();
  auto rx_rate = driver_options.get_rx_rate();


  std::vector<float> sample_buffer;
  enum shr_mem_switcher {FIRST, SECOND};

  std::vector<double> rxfreqs = {12.0e6,10.0e6,14.0e6};
  for (int i=0; i<rxfreqs.size(); i++) {
    auto rxchan = sp.add_rxchannel();
    rxchan->set_rxfreq(rxfreqs[i]);
    rxchan->set_nrang(75);
    rxchan->set_frang(180);

  }

  auto num_antennas = driver_options.get_main_antenna_count() +
              driver_options.get_interferometer_antenna_count();


  rxsamplesmetadata::RxSamplesMetadata samples_metadata;

  auto num_samples = uint32_t(rx_rate* 0.1);
  samples_metadata.set_numberofreceivesamples(num_samples);

  auto default_v = std::complex<float>(0.0,0.0);
  std::vector<std::complex<float>> samples(num_samples*num_antennas,default_v);


  for (int i=0; i<samples.size(); i++) {
    auto nco_point = std::complex<float>(0.0,0.0);
    for (auto freq : rxfreqs) {
      auto sampling_freq = 2 * M_PI * freq/rx_rate;

      auto radians = fmod(sampling_freq * i, 2 * M_PI);
      auto I = cos(radians);
      auto Q = sin(radians);

      nco_point += std::complex<float>(I,Q);
    }
    samples[i] = nco_point;
  }

  auto sqn_num = 0;

  sp.set_sequence_num(sqn_num);
  samples_metadata.set_sequence_num(sqn_num);

  std::chrono::steady_clock::time_point timing_ack_start, timing_ack_end, timing_timing_start,
                      timing_timing_end;

  send_data(samples_metadata, sp, samples,driver_socket, radctrl_socket, timing_ack_start);
  sqn_num += 1;

  while(1) {
    zmq::poll(&sockets[0],2,-1);

    sigprocpacket::SigProcPacket ack_from_dsp;
    if (sockets[0].revents & ZMQ_POLLIN)
    {
      zmq::message_t ack;
      ack_socket.recv(&ack);
      std::string s_msg_str1(static_cast<char*>(ack.data()), ack.size());
      ack_from_dsp.ParseFromString(s_msg_str1);

      timing_ack_end = std::chrono::steady_clock::now();

      std::cout << "Received ack #"<< ack_from_dsp.sequence_num() << " for sequence #"
        << sp.sequence_num() << " after "
        << std::chrono::duration_cast<std::chrono::milliseconds>(timing_ack_end -
                                                                  timing_ack_start).count()
        << "ms" << std::endl;

      sp.set_sequence_num(sqn_num);

      samples_metadata.set_sequence_num(sqn_num);

      send_data(samples_metadata,sp, samples,driver_socket, radctrl_socket, timing_ack_start);

      sqn_num += 1;
    }

    if (sockets[1].revents & ZMQ_POLLIN)
    {

      zmq::message_t timing;
      timing_socket.recv(&timing);
      std::string s_msg_str2(static_cast<char*>(timing.data()), timing.size());
      ack_from_dsp.ParseFromString(s_msg_str2);

      timing_timing_end = std::chrono::steady_clock::now();

      std::cout << "Received timing for sequence #" << ack_from_dsp.sequence_num()
        << " after "
        << std::chrono::duration_cast<std::chrono::milliseconds>(timing_timing_end -
                                  timing_timing_start).count()
        << "ms with decimation timing of " << ack_from_dsp.kerneltime() << "ms" <<  std::endl;

      timing_timing_start = std::chrono::steady_clock::now();

    }

  }


}
