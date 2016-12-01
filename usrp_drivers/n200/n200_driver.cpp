/*Copyright 2016 SuperDARN*/
#include <unistd.h>
#include <stdint.h>
#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <zmq.hpp>
#include <chrono>
#include <complex>
#include <limits>       // std::numeric_limits
#include <iostream>
#include <utility>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <thread>

#include "utils/driver_options/driveroptions.hpp"
#include "usrp_drivers/n200/usrp.hpp"
#include "utils/protobuf/driverpacket.pb.h"
#include "utils/protobuf/computationpacket.pb.h"


std::vector<size_t> make_channels(const driverpacket::DriverPacket &dp) {
  std::vector<size_t> channels(dp.channels_size());
  for (int i = 0; i < dp.channels_size(); i++) {
    channels[i] = i;
  }
  return channels;
}

std::vector<std::vector<std::complex<float>>> make_samples(const driverpacket::DriverPacket &dp) {
  std::vector<std::vector<std::complex<float>>> samples(dp.samples_size());

  for (int i = 0 ; i < dp.samples_size(); i++) {
    auto num_samps = dp.samples(i).real_size();
    std::vector<std::complex<float>> v(num_samps);
    samples[i] = v;

    for (int j = 0; j < dp.samples(i).real_size(); j++) {
      samples[i][j] = std::complex<float>(dp.samples(i).real(j), dp.samples(i).imag(j));
    }
  }

  return samples;
}

void transmit(zmq::context_t* driver_c, std::shared_ptr<USRP> usrp_d,
                std::shared_ptr<DriverOptions> driver_options) {
  std::cout << "Enter transmit thread\n";


  zmq::socket_t packet_socket(*driver_c, ZMQ_SUB);
  packet_socket.connect("inproc://threads");
  packet_socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);
  std::cout << "TRANSMIT: Creating and connected to thread socket in control\n";
  zmq::message_t request;

  zmq::socket_t timing_socket(*driver_c, ZMQ_PAIR);
  timing_socket.connect("inproc://timing");

  zmq::socket_t ack_socket(*driver_c, ZMQ_PAIR);
  ack_socket.connect("inproc://ack");


  auto channels_set = false;
  auto center_freq_set = false;
  auto samples_set = false;

  std::vector<size_t> channels;

  uhd::time_spec_t time_zero;
  zmq::message_t timing (sizeof(time_zero));

  uhd::tx_streamer::sptr tx_stream;
  uhd::stream_args_t stream_args("fc32", "sc16");

  size_t samples_per_buff;
  std::vector<std::vector<std::complex<float>>> samples;

  auto atten_window_time_start_us = driver_options->get_atten_window_time_start();
  auto atten_window_time_end_us = driver_options->get_atten_window_time_end();
  auto tr_window_time_us = driver_options->get_tr_window_time();

  //default initialize SS. Needs to be outside of loop
  auto scope_sync_low = uhd::time_spec_t(0.0);

  auto rx_rate = driver_options->get_rx_rate();

  uint32_t sqn_num = -1;
  uint32_t expected_sqn_num = 0;

  while (1) {
    packet_socket.recv(&request);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Received in TRANSMIT\n";
    driverpacket::DriverPacket dp;
    std::string msg_str(static_cast<char*>(request.data()), request.size());
    dp.ParseFromString(msg_str);

    sqn_num = dp.sqnnum();

    if (sqn_num != expected_sqn_num){
      std::cout << "SEQUENCE NUMBER MISMATCH: SQN " << sqn_num << " EXPECTED: "
        << expected_sqn_num << std::endl;
      //TODO(keith) handle error
    }

    std::cout <<"TRANSMIT burst flags: SOB "  << dp.sob() << " EOB " << dp.eob() <<std::endl;
    std::chrono::steady_clock::time_point stream_begin = std::chrono::steady_clock::now();
    if (dp.channels_size() > 0 && dp.sob() == true && channels_set == false) {
      std::cout << "TRANSMIT STARTING NEW PULSE SEQUENCE" <<std::endl;
      channels = make_channels(dp);
      stream_args.channels = channels;
      usrp_d->set_tx_rate(dp.txrate());  // ~450us
      tx_stream = usrp_d->get_usrp()->get_tx_stream(stream_args);  // ~44ms
      channels_set = true;
    }
    std::chrono::steady_clock::time_point stream_end = std::chrono::steady_clock::now();

    std::cout << "TRANSMIT stream set up time: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (stream_end - stream_begin).count()
      << "us" << std::endl;


    std::chrono::steady_clock::time_point ctr_begin = std::chrono::steady_clock::now();

    if (dp.txcenterfreq() > 0) {
      std::cout << "TRANSMIT center freq " << dp.txcenterfreq() << std::endl;
      usrp_d->set_tx_center_freq(dp.txcenterfreq(), channels);
      center_freq_set = true;
    }

    std::chrono::steady_clock::time_point ctr_end = std::chrono::steady_clock::now();
    std::cout << "TRANSMIT center frq tuning time: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (ctr_end - ctr_begin).count()
      << "us" << std::endl;


    std::chrono::steady_clock::time_point sample_begin = std::chrono::steady_clock::now();

    if (dp.samples_size() > 0) {  // ~700us to unpack 4x1600 samples
      samples = make_samples(dp);
      samples_per_buff = samples[0].size();
      samples_set = true;
    }

    std::chrono::steady_clock::time_point sample_end = std::chrono::steady_clock::now();
    std::cout << "TRANSMIT sample unpack time: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (sample_end - sample_begin).count()
      << "us" << std::endl;

    if ( (channels_set == false) ||
      (center_freq_set == false) ||
      (samples_set == false) ) {
      // TODO(keith): throw error
      continue;
    }


    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    std::cout << "TRANSMIT Total set up time: "
          << std::chrono::duration_cast<std::chrono::microseconds>
                                                      (end - begin).count()
          << "us" << std::endl;



    if ( (dp.sob() == true) ) {
      time_zero = usrp_d->get_usrp()->get_time_now() + uhd::time_spec_t(10e-3);
    }

    auto pulse_delay = uhd::time_spec_t(dp.timetosendsamples()/1e6);
    auto pulse_start_time = time_zero + pulse_delay;
    auto pulse_len_time = uhd::time_spec_t(samples_per_buff/usrp_d->get_tx_rate());

    auto tr_time_high = pulse_start_time - uhd::time_spec_t(tr_window_time_us);
    auto atten_time_high = tr_time_high - uhd::time_spec_t(atten_window_time_start_us);
    auto scope_sync_high = atten_time_high;
    auto tr_time_low = pulse_start_time + pulse_len_time + uhd::time_spec_t(tr_window_time_us);
    auto atten_time_low = tr_time_low + uhd::time_spec_t(atten_window_time_end_us);

    if( dp.sob() == true) {
      auto sync_time = dp.numberofreceivesamples()/12e6;
      std::cout << "SYNC TIME " << sync_time << std::endl;
      scope_sync_low = atten_time_high + uhd::time_spec_t(sync_time);

      memcpy(timing.data (), &scope_sync_high, sizeof(scope_sync_high));
      timing_socket.send(timing);
    }

    begin = std::chrono::steady_clock::now();

    auto time_now = usrp_d->get_usrp()->get_time_now();
    std::cout << "TRANSMIT time zero: " << time_zero.get_frac_secs() << std::endl;
    std::cout << "TRANSMIT time now " << time_now.get_frac_secs() << std::endl;
    std::cout << "TRANSMIT timetosendsamples(us) " << dp.timetosendsamples() <<std::endl;
    std::cout << "TRANSMIT pulse delay " << pulse_delay.get_frac_secs() <<std::endl;
    std::cout << "TRANSMIT atten_time_high " << atten_time_high.get_frac_secs() << std::endl;
    std::cout << "TRANSMIT tr_time_high " << tr_time_high.get_frac_secs() << std::endl;
    std::cout << "TRANSMIT pulse start time " << pulse_start_time.get_frac_secs() <<std::endl;
    std::cout << "TRANSMIT tr_time_low " << tr_time_low.get_frac_secs() << std::endl;
    std::cout << "TRANSMIT atten_time_low " << atten_time_low.get_frac_secs() << std::endl;
    std::cout << "TRANSMIT scope_sync_low " << scope_sync_low.get_frac_secs() << std::endl;

    auto md = TXMetadata();
    md.set_has_time_spec(true);
    md.set_time_spec(pulse_start_time);
    md.set_start_of_burst(true);

    uint64_t num_samps_sent = 0;

    while (num_samps_sent < samples_per_buff) {
      auto num_samps_to_send = samples_per_buff - num_samps_sent;
      std::cout << "TRANSMIT Samples to send " << num_samps_to_send << std::endl;

      num_samps_sent = tx_stream->send(
        samples, num_samps_to_send, md.get_md());

      std::cout << "TRANSMIT Samples sent " << num_samps_sent << std::endl;

      md.set_start_of_burst(false);
      md.set_has_time_spec(false);
    }

    md.set_end_of_burst(true);
    tx_stream->send("", 0, md.get_md());

    end = std::chrono::steady_clock::now();

    std::cout << "TRANSMIT time to send to USRP: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                          (end - begin).count()
      << "us" << std::endl;

    std::chrono::steady_clock::time_point timing_start = std::chrono::steady_clock::now();

    usrp_d->get_usrp()->clear_command_time();

    if (dp.sob() == true) {
      std::cout << "TRANSMIT Scope sync high" << std::endl;
      usrp_d->get_usrp()->set_command_time(scope_sync_high);
      usrp_d->set_scope_sync();

    }

    usrp_d->get_usrp()->set_command_time(atten_time_high);
    usrp_d->set_atten();

    usrp_d->get_usrp()->set_command_time(tr_time_high);
    usrp_d->set_tr();

    usrp_d->get_usrp()->set_command_time(tr_time_low);
    usrp_d->clear_tr();

    usrp_d->get_usrp()->set_command_time(atten_time_low);
    usrp_d->clear_atten();

    if (dp.eob() == true) {
      usrp_d->get_usrp()->set_command_time(scope_sync_low);
      usrp_d->clear_scope_sync();
    }


    std::chrono::steady_clock::time_point timing_end = std::chrono::steady_clock::now();

    std::cout << "TRANSMIT Time to set up timing signals: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (timing_end - timing_start).count()
      << "us" << std::endl;

    if (dp.eob() == true) {
      //usrp_d->clear_scope_sync();
      driverpacket::DriverPacket ack;
      ack.set_sqnnum(sqn_num);
      expected_sqn_num += 1;

      std::string ack_str;
      ack.SerializeToString(&ack_str);

      zmq::message_t ack_msg(ack_str.size());
      memcpy((void*)ack_msg.data(), ack_str.c_str(),ack_str.size());
      ack_socket.send(ack_msg);
    }


  }
}












boost::interprocess::mapped_region shr_mem_create(const char* name, size_t size) {
  boost::interprocess::shared_memory_object::remove(name);

  // Create a shared memory object.
  auto create_mode = boost::interprocess::open_or_create;
  auto access_mode = boost::interprocess::read_write;
  boost::interprocess::shared_memory_object shm(create_mode, name, access_mode);

  // Set size
  shm.truncate(size);

  // Map the whole shared memory in this process
  boost::interprocess::mapped_region region(shm, access_mode);

  return region;
}

boost::interprocess::mapped_region mmap_create(const char* file_name, size_t size) {
  boost::interprocess::mapped_region region;
  try {
    // Create a file
    std::filebuf fbuf;
    fbuf.open(file_name, std::ios_base::in | std::ios_base::out
                         | std::ios_base::trunc | std::ios_base::binary);

    // Set the size
    fbuf.pubseekoff(size, std::ios_base::beg);
    fbuf.sputc(0);
    fbuf.close();

    // Create a file mapping.
    auto access_mode = boost::interprocess::read_write;
    boost::interprocess::file_mapping mapped_file(file_name, access_mode);

    // Map the whole file in this process
    region = boost::interprocess::mapped_region(mapped_file  // What to map
       , access_mode);  // Map it as read-write

    if (region.get_size() != size) {
      // TODO(keith): throw error
    }
  }
  catch(boost::interprocess::interprocess_exception &ex) {
    // TODO(keith): throw error
    std::remove(file_name);
    std::cout << ex.what() << std::endl;
  }

  return region;
}

void receive(zmq::context_t* driver_c, std::shared_ptr<USRP> usrp_d,
              std::shared_ptr<DriverOptions> driver_options) {
  std::cout << "Enter receive thread\n";

  zmq::socket_t packet_socket(*driver_c, ZMQ_SUB);
  packet_socket.connect("inproc://threads");
  packet_socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);
  zmq::message_t request;
  std::cout << "RECEIVE: Creating and connected to thread socket in control\n";

  zmq::socket_t timing_socket(*driver_c, ZMQ_PAIR);
  timing_socket.bind("inproc://timing");
  zmq::message_t timing;

  //zmq::context_t context(4);
  zmq::socket_t data_socket(*driver_c, ZMQ_PAIR);
  data_socket.connect("inproc://data");

  auto channels_set = false;
  auto center_freq_set = false;

  std::vector<size_t> channels;
  // std::vector<float> sample_buffer;
  // enum shr_mem_switcher {FIRST, SECOND};
  // shr_mem_switcher region_switcher = FIRST;
  // boost::interprocess::mapped_region first_region, second_region;
  // void *current_region_addr;

  uhd::time_spec_t time_zero;
  uhd::rx_streamer::sptr rx_stream;
  uhd::stream_args_t stream_args("fc32", "sc16");
  uhd::stream_cmd_t stream_cmd(
      uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);

  auto rx_rate = driver_options->get_rx_rate();

  computationpacket::ComputationPacket cp;

  while (1) {
    packet_socket.recv(&request);
    std::cout << "Received in RECEIVE\n";
    driverpacket::DriverPacket dp;
    std::string msg_str(static_cast<char*>(request.data()), request.size());
    dp.ParseFromString(msg_str);

    std::cout << "RECEIVE burst flags SOB " << dp.sob() << " EOB " << dp.eob() << std::endl;

    if ( dp.sob() == false ) continue;

    std::chrono::steady_clock::time_point stream_begin = std::chrono::steady_clock::now();
    if (dp.channels_size() > 0 && dp.sob() == true && channels_set == false) {
      channels = make_channels(dp);
      stream_args.channels = channels;
      usrp_d->set_rx_rate(rx_rate);  // ~450us
      rx_stream = usrp_d->get_usrp()->get_rx_stream(stream_args);  // ~44ms
      channels_set = true;
    }
    std::chrono::steady_clock::time_point stream_end = std::chrono::steady_clock::now();
    std::cout << "RECEIVE stream set up time: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (stream_end - stream_begin).count()
      << "us" << std::endl;


    std::chrono::steady_clock::time_point ctr_begin = std::chrono::steady_clock::now();

    if (dp.rxcenterfreq() > 0) {
      std::cout << "RECEIVE center freq " << dp.rxcenterfreq() << std::endl;
      usrp_d->set_rx_center_freq(dp.rxcenterfreq(), channels);
      center_freq_set = true;
    }

    std::chrono::steady_clock::time_point ctr_end = std::chrono::steady_clock::now();
    std::cout << "RECEIVE center frq tuning time: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (ctr_end - ctr_begin).count()
      << "us" << std::endl;


    if ((channels_set == false) || (center_freq_set == false)) {
      // TODO(keith): throw error
    }


/*    std::chrono::steady_clock::time_point shr_mem_begin= std::chrono::steady_clock::now();
    size_t mem_size = channels.size() * dp.numberofreceivesamples() * sizeof(float);
    if ((dp.numberofreceivesamples() != 0) &&
        (dp.numberofreceivesamples() != current_num_samples)) {

      first_region = mmap_create("/tmp/first_region",mem_size);
      second_region = mmap_create("/tmp/second_region",mem_size);
      sample_num_set = true;
      current_num_samples = dp.numberofreceivesamples();
      cp.set_size(channels.size() * dp.numberofreceivesamples());
    }

    if (region_switcher == FIRST) {
      current_region_addr = first_region.get_address();
      region_switcher = SECOND;
      cp.set_region_name("/tmp/first_region");
    }
    else {
      current_region_addr = second_region.get_address();
      region_switcher = FIRST;
      cp.set_region_name("/tmp/second_region");
    }

    std::chrono::steady_clock::time_point shr_mem_end= std::chrono::steady_clock::now();
    std::cout << "RECEIVE shared_memory timing: "
          << std::chrono::duration_cast<std::chrono::microseconds>(shr_mem_end - shr_mem_begin).count()
          << "us" << std::endl;*/

    std::cout << "Got to RECEIVE timing" << std::endl;
    timing_socket.recv(&timing);
    std::cout << "RECEIVED timing data" << std::endl;
    time_zero = *(reinterpret_cast<uhd::time_spec_t*>(timing.data()));

    std::cout << "timing data " << timing.data() << " size "<< timing.size();
    std::cout << "RECEIVE time_zero " << time_zero.get_frac_secs() << std::endl;

    std::chrono::steady_clock::time_point recv_begin = std::chrono::steady_clock::now();
    //TODO(keith) change to complex
    size_t mem_size = channels.size() * dp.numberofreceivesamples() * sizeof(float) * 2;
    zmq::message_t cp_message(mem_size);
    stream_cmd.num_samps = size_t(dp.numberofreceivesamples() * channels.size());
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = time_zero;
    rx_stream->issue_stream_cmd(stream_cmd);

    auto md = RXMetadata();
    size_t total_received_samples = 0;
    std::cout << "samples to receive: " << dp.numberofreceivesamples() << " mem size "
    << mem_size << std::endl;
    while (total_received_samples < dp.numberofreceivesamples()) {
      size_t num_rx_samps = rx_stream->recv(cp_message.data(),
        (size_t)dp.numberofreceivesamples(), md.get_md(), 25, false);

      if (md.get_error_code() == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
          std::cout << boost::format("Timeout while streaming") << std::endl;
          break;
      }
      if (md.get_error_code() == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
          // TODO(keith): throw error
      }
      if (md.get_error_code() != uhd::rx_metadata_t::ERROR_CODE_NONE) {
          // TODO(keith): throw error
      }

      total_received_samples += num_rx_samps;
      std::cout << "Total received samples " << total_received_samples <<std::endl;
    }
    std::chrono::steady_clock::time_point recv_end = std::chrono::steady_clock::now();
    std::cout << "RECEIVE receive timing: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (recv_end - recv_begin).count()
      << "us" << std::endl;

/*    for (int i =0; i<500;i++) {
      buffer[i] = 1.0;
    }*/
    std::cout << "RECEIVE total_received_samples " << total_received_samples << std::endl;

/*    std::string cp_str;
    cp.SerializeToString(&cp_str);

    memcpy ((void *) cp_message.data (), cp_str.c_str(), cp_str.size());*/
    std::chrono::steady_clock::time_point send_begin = std::chrono::steady_clock::now();
    data_socket.send(cp_message);
    std::chrono::steady_clock::time_point send_end = std::chrono::steady_clock::now();
    std::cout << "RECEIVE package and send timing: "
          << std::chrono::duration_cast<std::chrono::microseconds>(send_end - send_begin).count()
          << "us" << std::endl;
  }
}

void control(zmq::context_t* driver_c) {
  std::cout << "Enter control thread\n";

  std::cout << "Creating and connecting to thread socket in control\n";
/*    zmq::socket_t packet_socket(*thread_c, ZMQ_PAIR); // 1
  packet_socket.connect("inproc://threads"); // 2  */


  zmq::socket_t packet_socket(*driver_c, ZMQ_PUB);
  packet_socket.bind("inproc://threads");

  std::cout << "Creating and binding control socket\n";
  //zmq::context_t context(2);
  zmq::socket_t radarctrl_socket(*driver_c, ZMQ_PAIR);
  radarctrl_socket.bind("ipc:///tmp/feeds/0");

  zmq::message_t request;

  zmq::socket_t data_socket(*driver_c, ZMQ_PAIR);
  data_socket.bind("inproc://data");

  zmq::socket_t ack_socket(*driver_c, ZMQ_PAIR);
  ack_socket.bind("inproc://ack");

  zmq::socket_t computation_socket(*driver_c, ZMQ_PAIR);
  computation_socket.bind("ipc:///tmp/feeds/1");



  //Allows for set up of SUB/PUB socket
  sleep(1);

  while (1) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    radarctrl_socket.recv(&request);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    begin = std::chrono::steady_clock::now();
    driverpacket::DriverPacket dp;
    std::string msg_str(static_cast<char*>(request.data()), request.size());
    dp.ParseFromString(msg_str);

    std::cout << "Control " << dp.sob() << " " << dp.eob()
      << " " << dp.channels_size() << std::endl;

    if ((dp.sob() == true) && (dp.eob() == true)) {
      // TODO(keith): throw error
    }

    if (dp.samples_size() != dp.channels_size()) {
      // TODO(keith): throw error
    }

    for (int i = 0; i < dp.samples_size(); i++) {
      if (dp.samples(i).real_size() != dp.samples(i).imag_size()) {
        // TODO(keith): throw error
      }
    }

    packet_socket.send(request);

    end = std::chrono::steady_clock::now();
    std::cout << "Time difference to deserialize and send in control = "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (end - begin).count()
      <<std::endl;

    if (dp.eob() == true) {
      zmq::message_t ack;
      zmq::message_t data;

      data_socket.recv(&data);
      ack_socket.recv(&ack);

/*      computation_socket.send(data);
      radarctrl_socket.send(ack);*/

    }


  }
}




int UHD_SAFE_MAIN(int argc, char *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  auto driver_options = std::make_shared<DriverOptions>();

  std::cout << driver_options->get_device_args() << std::endl;
  std::cout << driver_options->get_tx_rate() << std::endl;
  std::cout << driver_options->get_pps() << std::endl;
  std::cout << driver_options->get_ref() << std::endl;
  std::cout << driver_options->get_tx_subdev() << std::endl;

  auto usrp_d = std::make_shared<USRP>(driver_options);


  //  Prepare our context
  zmq::context_t driver_context(1);
/*  zmq::context_t timing_context(3);*/

  std::vector<std::thread> threads;

  std::thread control_t(control, &driver_context);
  std::thread transmit_t(transmit, &driver_context, usrp_d, driver_options);
  std::thread receive_t(receive, &driver_context, usrp_d, driver_options);


  threads.push_back(std::move(transmit_t));
  threads.push_back(std::move(receive_t));
  threads.push_back(std::move(control_t));

  for (auto& th : threads) {
    th.join();
  }


  return EXIT_SUCCESS;
}
