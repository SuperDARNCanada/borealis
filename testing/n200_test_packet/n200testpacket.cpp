#include <zmq.hpp>
#include <thread>
#include <unistd.h>
#include <iostream>
#include <complex>
#include "utils/protobuf/driverpacket.pb.h"
#include "utils/driver_options/driveroptions.hpp"
#include "utils/shared_memory/shared_memory.hpp"
#include "utils/protobuf/rxsamplesmetadata.pb.h"
#include <cmath>

std::vector<std::complex<float>> make_pulse(DriverOptions &driver_options){
  auto amp = 1.0/sqrt(2.0);
  auto pulse_len = 300.0 * 1e-6;
  auto tx_rate = driver_options.get_tx_rate();
  int num_samps_per_antenna = std::ceil(pulse_len * tx_rate);
  std::vector<double> tx_freqs = {1e6};

  auto default_v = std::complex<float>(0.0,0.0);
  std::vector<std::complex<float>> samples(num_samps_per_antenna,default_v);


  for (auto j=0; j< num_samps_per_antenna; j++) {
    auto nco_point = std::complex<float>(0.0,0.0);

    for (auto freq : tx_freqs) {
      auto sampling_freq = 2 * M_PI * freq/tx_rate;

      auto radians = fmod(sampling_freq * j, 2 * M_PI);
      auto I = amp * cos(radians);
      auto Q = amp * sin(radians);

      nco_point += std::complex<float>(I,Q);
    }
    samples[j] = nco_point;
  }

    auto ramp_size = int(10e-6 * tx_rate);

    for (auto j=0; j<ramp_size; j++){
      auto a = ((j)*1.0)/ramp_size;
      samples[j] *= std::complex<float>(a,0);
    }

    for (auto j=num_samps_per_antenna-1, k=0;j>num_samps_per_antenna-1-ramp_size;j--,k++){
      auto a = ((k)*1.0)/ramp_size;
      samples[j] *= std::complex<float>(a,0);
    }


  return samples;
}

int main(int argc, char *argv[]){

  DriverOptions driver_options;

  driverpacket::DriverPacket dp;
  zmq::context_t context(1);
  zmq::socket_t rad_socket(context, ZMQ_PAIR);
  zmq::socket_t dsp_socket(context, ZMQ_PAIR);
  rad_socket.connect(driver_options.get_radar_control_to_driver_address());
  dsp_socket.connect(driver_options.get_driver_to_rx_dsp_address());


  auto pulse_samples = make_pulse(driver_options);
  for (int j=0; j<driver_options.get_main_antenna_count(); j++){
    dp.add_channels(j);
    auto samples = dp.add_channel_samples();

  int count=0;
    for (auto &sm : pulse_samples){
      samples->add_real(sm.real());
      samples->add_imag(sm.imag());
      count++;
    }
  }

  bool SOB, EOB = false;
  std::vector<int> pulse_seq ={0,9,12,20,22,26,27};//{0,3,15,41,66,95,97,106,142,152,220,221,225,242,295,330,338,354,382,388,402,415,486,504,523,546,553};//{0,1,4,11,26,32,56,68,76,115,117,134,150,163,168,177};//,{0,9,12,20,22,26,10000};

  auto first_time = true;
  auto seq_num = 0;
  while (1){
    for (auto &pulse : pulse_seq){
      std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

      if (pulse == pulse_seq.front()){
        SOB = true;
      }
      else{
        SOB = false;
      }

      if (pulse == pulse_seq.back()){
        EOB = true;
      }
      else{
        EOB = false;
      }
      std::cout << SOB << " " << EOB <<std::endl;
      dp.set_sob(SOB);
      dp.set_eob(EOB);
      dp.set_txrate(driver_options.get_tx_rate());
      dp.set_timetosendsamples(pulse * 1500);
      dp.set_txcenterfreq(12e6);
      dp.set_rxcenterfreq(14e6);
      dp.set_sequence_num(seq_num++);

      auto mpinc = 1500 * 1e-6;
      auto num_recv_samps = pulse_seq.back() * mpinc * driver_options.get_rx_rate();
      dp.set_numberofreceivesamples(num_recv_samps);

      std::string msg_str;
      dp.SerializeToString(&msg_str);
      zmq::message_t request (msg_str.size());
      memcpy ((void *) request.data (), msg_str.c_str(), msg_str.size());
      std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
      std::cout << "Time difference to serialize(us) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
      std::cout << "Time difference to serialize(ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() <<std::endl;

      begin = std::chrono::steady_clock::now();
      rad_socket.send (request);
      end= std::chrono::steady_clock::now();

      std::cout << "send time(us) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
      std::cout << "send time(ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() <<std::endl;

      if (first_time == true) {
          dp.clear_channels();
          dp.clear_channel_samples();
          first_time = false;
      }

    }

    zmq::message_t dsp_msg;
    dsp_socket.recv(&dsp_msg);
    std::string dsp_msg_str(static_cast<char*>(dsp_msg.data()), dsp_msg.size());

    rxsamplesmetadata::RxSamplesMetadata rx_metadata;
    rx_metadata.ParseFromString(dsp_msg_str);

    SharedMemoryHandler shr_mem(rx_metadata.shrmemname());
    shr_mem.remove_shr_mem();


    zmq::message_t ack_msg;
    rad_socket.recv(&ack_msg);
    std::string ack_msg_str(static_cast<char*>(ack_msg.data()), ack_msg.size());

    driverpacket::DriverPacket ack;
    ack.ParseFromString(ack_msg_str);

    std::cout << std::endl << std::endl<<"Got ack #" <<ack.sequence_num()<< " for seq #"
      << dp.sequence_num() <<std::endl << std::endl;

  }

}
