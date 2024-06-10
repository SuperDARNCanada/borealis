#include <unistd.h>

#include <cmath>
#include <complex>
#include <iostream>
#include <thread>
#include <zmq.hpp>

#include "utils/driver_options/driveroptions.hpp"
#include "utils/protobuf/driverpacket.pb.h"
#include "utils/protobuf/rxsamplesmetadata.pb.h"
#include "utils/shared_memory/shared_memory.hpp"
#include "utils/zmq_borealis_helpers/zmq_borealis_helpers.hpp"

#define PULSE7 \
  { 0, 9, 12, 20, 22, 26, 27 }
#define PULSE27                                                            \
  {                                                                        \
    0, 3, 15, 41, 66, 95, 97, 106, 142, 152, 220, 221, 225, 242, 295, 330, \
        338, 354, 382, 388, 402, 415, 486, 504, 523, 546, 553              \
  }
#define PULSE16 \
  { 0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150, 163, 168, 177 }
#define longdelay \
  { 0, 9, 12, 20, 22, 26, 10000 }

std::vector<std::complex<float>> make_pulse(DriverOptions &driver_options) {
  auto amp = 1.0 / sqrt(2.0);
  auto pulse_len = 300.0 * 1e-6;
  auto tx_rate = driver_options.get_tx_rate();
  // int tr_start_pad = std::ceil(tx_rate *
  // (driver_options.get_atten_window_time_start() +
  // driver_options.get_tr_window_time())); int tr_end_pad = tx_rate *
  // 10e-6;//std::ceil(tx_rate * (driver_options.get_atten_window_time_end() +
  // driver_options.get_tr_window_time()));
  int tr_start_pad = 0, tr_end_pad = 0;
  int num_samps_per_antenna =
      std::ceil(pulse_len * tx_rate) + tr_start_pad + tr_end_pad;
  std::cout << num_samps_per_antenna << std::endl;
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

int main(int argc, char *argv[]) {
  DriverOptions driver_options;

  driverpacket::DriverPacket dp;
  zmq::context_t context(1);

  // std::thread router_t(router,std::ref(context),
  // driver_options.get_router_address());

  /*  zmq::socket_t rad_socket(context, ZMQ_PAIR);
    zmq::socket_t dsp_socket(context, ZMQ_PAIR);
    rad_socket.connect(driver_options.get_radar_control_to_driver_address());
    dsp_socket.bind(driver_options.get_driver_to_rx_dsp_address());
  */
  auto identities = {driver_options.get_dsp_to_driver_identity(),
                     driver_options.get_brian_to_driver_identity(),
                     driver_options.get_radctrl_to_driver_identity()};

  auto sockets_vector =
      create_sockets(context, identities, driver_options.get_router_address());

  auto &dsp_to_driver = sockets_vector[0];
  auto &brian_to_driver = sockets_vector[1];
  auto &radctrl_to_driver = sockets_vector[2];

  auto pulse_samples = make_pulse(driver_options);
  for (int j = 0; j < driver_options.get_main_antenna_count(); j++) {
    dp.add_channels(j);
    auto samples = dp.add_channel_samples();

    int count = 0;
    for (auto &sm : pulse_samples) {
      samples->add_real(sm.real());
      samples->add_imag(sm.imag());
      count++;
    }
  }

  dp.set_txcenterfreq(12e6);
  dp.set_rxcenterfreq(14e6);

  bool SOB, EOB = false;

  std::vector<int> pulse_seq = PULSE16;

  auto first_time = true;
  auto seq_num = 0;

  while (1) {
    for (auto &pulse : pulse_seq) {
      std::chrono::steady_clock::time_point begin =
          std::chrono::steady_clock::now();

      if (pulse == pulse_seq.front()) {
        SOB = true;
      } else {
        SOB = false;
      }

      if (pulse == pulse_seq.back()) {
        EOB = true;
      } else {
        EOB = false;
      }
      std::cout << SOB << " " << EOB << std::endl;
      dp.set_sob(SOB);
      dp.set_eob(EOB);
      dp.set_txrate(driver_options.get_tx_rate());
      dp.set_timetosendsamples(pulse * 1500);
      dp.set_sequence_num(seq_num);

      auto mpinc = 1500 * 1e-6;
      auto num_recv_samps =
          (pulse_seq.back() * mpinc + 23.5e-3) * driver_options.get_rx_rate();
      dp.set_numberofreceivesamples(num_recv_samps);

      std::string msg_str;
      dp.SerializeToString(&msg_str);
      /*      zmq::message_t request (msg_str.size());
            memcpy ((void *) request.data (), msg_str.c_str(),
         msg_str.size());*/
      std::chrono::steady_clock::time_point end =
          std::chrono::steady_clock::now();
      std::cout << "Time difference to serialize(us) = "
                << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                         begin)
                       .count()
                << std::endl;
      std::cout << "Time difference to serialize(ns) = "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                        begin)
                       .count()
                << std::endl;

      begin = std::chrono::steady_clock::now();
      // rad_socket.send (request);
      SEND_REQUEST(radctrl_to_driver,
                   driver_options.get_driver_to_radctrl_identity(), msg_str);
      end = std::chrono::steady_clock::now();

      std::cout << "send time(us) = "
                << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                         begin)
                       .count()
                << std::endl;
      std::cout << "send time(ns) = "
                << std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                        begin)
                       .count()
                << std::endl;

      if (first_time == true) {
        dp.clear_channels();
        dp.clear_channel_samples();
        dp.clear_txcenterfreq();
        dp.clear_rxcenterfreq();
        first_time = false;
      }
      std::cout << dp.txcenterfreq() << std::endl;
    }

    auto message = std::string("Need metadata");
    SEND_REQUEST(dsp_to_driver, driver_options.get_driver_to_dsp_identity(),
                 message);
    auto reply =
        RECV_REPLY(dsp_to_driver, driver_options.get_driver_to_dsp_identity());

    rxsamplesmetadata::RxSamplesMetadata rx_metadata;
    rx_metadata.ParseFromString(reply);

    /*    SharedMemoryHandler shr_mem(rx_metadata.shrmemname());
        shr_mem.remove_shr_mem();

    */
    SEND_REQUEST(brian_to_driver, driver_options.get_driver_to_brian_identity(),
                 message);
    reply = RECV_REPLY(brian_to_driver,
                       driver_options.get_driver_to_brian_identity());

    rx_metadata.ParseFromString(reply);

    std::cout << std::endl
              << std::endl
              << "Got ack #" << rx_metadata.sequence_num() << " for seq #"
              << dp.sequence_num() << std::endl
              << std::endl;

    seq_num++;

    // usleep(100.0e3);
  }
}
