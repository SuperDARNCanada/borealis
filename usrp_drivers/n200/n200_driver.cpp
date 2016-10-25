/*Copyright 2016 SuperDARN*/
#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
/*#include <boost/program_options.hpp>*/
/*#include <boost/math/special_functions/round.hpp>*/
/*#include <boost/foreach.hpp>*/
/*#include <boost/format.hpp>*/
/*#include <boost/thread.hpp>*/
/*#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/thread.hpp>*/
/*#include <boost/chrono.hpp>*/
#include <stdint.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <zmq.hpp>
#include <unistd.h>
#include "utils/driver_options/driveroptions.hpp"
#include "usrp_drivers/n200/usrp.hpp"
#include "utils/protobuf/driverpacket.pb.h"
#include <chrono>
#include <complex>
/*enum TRType {TRANSMIT,RECEIVE};
enum IntegrationPeriod {START,STOP,CONTINUE};

std::mutex TR_mutex;
std::mutex control_parameters;


TRType TR = transmit;

int current_integrations = 0;*/

std::vector<size_t> make_channels(const driverpacket::DriverPacket &dp){
  std::vector<size_t> channels(dp.channels_size());
  for (int i=0; i<dp.channels_size(); i++){
    channels[i] = i;
  }
  return channels;
}

std::vector<std::vector<std::complex<float>>> make_samples(const driverpacket::DriverPacket &dp){
  std::vector<std::vector<std::complex<float>>> samples(dp.samples_size());

  for(int i=0 ; i<dp.samples_size(); i++) {
    auto num_samps = dp.samples(i).real_size();
    std::vector<std::complex<float>> v(num_samps);
    samples[i] = v;

    for (int j=0; j<dp.samples(i).real_size(); j++){
      samples[i][j] = std::complex<float>(dp.samples(i).real(j),dp.samples(i).imag(j));
    }

  }

  return samples;
}

void transmit(zmq::context_t* thread_c,  zmq::context_t* timing_c,
        std::shared_ptr<USRP> usrp_d, std::shared_ptr<DriverOptions> driver_options) {

  std::cout << "Enter transmit thread\n";


  zmq::socket_t thread_socket(*thread_c, ZMQ_SUB);
  thread_socket.connect("inproc://threads");
  thread_socket.setsockopt(ZMQ_SUBSCRIBE,"",0);
  std::cout << "TRANSMIT: Creating and connected to thread socket in control\n";
  zmq::message_t request;  

  zmq::socket_t timing_socket(*timing_c, ZMQ_PAIR);
  thread_socket.connect("inproc://timing");

  auto channels_set = false;
  auto center_freq_set = false;
  auto samples_set = false;
  
  std::vector<size_t> channels;

  uhd::time_spec_t time_zero;
  uhd::tx_streamer::sptr tx_stream;
  uhd::stream_args_t stream_args("fc32", "sc16");
  
  size_t samples_per_buff;
  std::vector<std::vector<std::complex<float>>> samples;

  auto atten_window_time_start_us = driver_options->get_atten_window_time_start();
  auto atten_window_time_end_us = driver_options->get_atten_window_time_end();
  auto tr_window_time_us = driver_options->get_tr_window_time();

  while (1) {
    thread_socket.recv(&request);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Received in TRANSMIT\n";
    driverpacket::DriverPacket dp;
    std::string msg_str(static_cast<char*>(request.data()), request.size());
    dp.ParseFromString(msg_str);
     
    std::cout <<"BURST FLAGS: SOB "  << dp.sob() << " EOB " << dp.eob() <<std::endl;
    std::chrono::steady_clock::time_point stream_begin = std::chrono::steady_clock::now();
    if (dp.channels_size() > 0 && dp.sob() == true && channels_set == false) {
      std::cout << "STARTING NEW PULSE SEQUENCE" <<std::endl;
      channels = make_channels(dp);
      stream_args.channels = channels;
      usrp_d->set_tx_rate(dp.txrate()); //~450us 
      tx_stream = usrp_d->get_usrp()->get_tx_stream(stream_args); //~44ms
      channels_set = true;
    }
    std::chrono::steady_clock::time_point stream_end = std::chrono::steady_clock::now();
    std::cout << "stream set up time: "
          << std::chrono::duration_cast<std::chrono::microseconds>(stream_end - stream_begin).count()
          << "us" << std::endl;

    
    std::chrono::steady_clock::time_point ctr_begin = std::chrono::steady_clock::now();
    
    if (dp.txcenterfreq() > 0) {
      std::cout << "DP center freq " << dp.txcenterfreq() << std::endl;
      usrp_d->set_tx_center_freq(dp.txcenterfreq(),channels);
      center_freq_set = true;
    }
    
    std::chrono::steady_clock::time_point ctr_end= std::chrono::steady_clock::now();
    std::cout << "center frq tuning time: "
          << std::chrono::duration_cast<std::chrono::microseconds>(ctr_end - ctr_begin).count()
          << "us" << std::endl;


    std::chrono::steady_clock::time_point sample_begin = std::chrono::steady_clock::now();
    
    if (dp.samples_size() > 0) { // ~700us to unpack 4x1600 samples
/*            std::vector<std::vector<std::complex<float>>> samples(dp.samples_size(),
                                std::vector<std::complex<float>>(dp.samples(0).real_size()));*/
      samples = make_samples(dp);
      samples_per_buff = samples[0].size();
      samples_set = true; 
    }

    std::chrono::steady_clock::time_point sample_end= std::chrono::steady_clock::now();
    std::cout << "sample unpack time: "
          << std::chrono::duration_cast<std::chrono::microseconds>(sample_end - sample_begin).count()
          << "us" << std::endl;
    
    if( (channels_set == false) || 
      (center_freq_set == false) || 
      (samples_set == false) ) {

      //todo: throw error
    }


    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

    std::cout << "Total set up time: " 
          << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
          << "us" << std::endl;


    
    if ( (dp.sob() == true) && (dp.eob() == false) ) {
      auto time_now_tmp = usrp_d->get_usrp()->get_time_now();
      std::cout << "time now tmp " << time_now_tmp.get_frac_secs() << std::endl;
      time_zero = usrp_d->get_usrp()->get_time_now() + uhd::time_spec_t(10e-3);

      zmq::message_t timing (sizeof(time_zero));
      memcpy(timing.data (), &time_zero, sizeof(&time_zero));
      timing_socket.send(timing);
    }

    
/*    std::chrono::steady_clock::time_point timing_start= std::chrono::steady_clock::now();*/

    auto pulse_delay = uhd::time_spec_t(dp.timetosendsamples()/1e6);
    auto pulse_start_time = time_zero + pulse_delay;
    auto pulse_len_time = uhd::time_spec_t(samples_per_buff/dp.txrate());
    
    auto tr_time_high = pulse_start_time - uhd::time_spec_t(tr_window_time_us);
    auto atten_time_high = tr_time_high - uhd::time_spec_t(atten_window_time_start_us);    
    auto tr_time_low = pulse_start_time + pulse_len_time + uhd::time_spec_t(tr_window_time_us);
    auto atten_time_low = tr_time_low + uhd::time_spec_t(atten_window_time_end_us);


    begin = std::chrono::steady_clock::now();

    auto time_now = usrp_d->get_usrp()->get_time_now();
    std::cout << "timetosendsamples " << dp.timetosendsamples() <<std::endl;
    std::cout << "time zero: " << time_zero.get_frac_secs() << std::endl;
    std::cout << "pulse delay " << pulse_delay.get_frac_secs() <<std::endl; 
    std::cout << "atten_time_high " << atten_time_high.get_frac_secs() << std::endl;
    std::cout << "tr_time_high " << tr_time_high.get_frac_secs() << std::endl;
    std::cout << "pulse start time " << pulse_start_time.get_frac_secs() <<std::endl;
    std::cout << "tr_time_low " << tr_time_low.get_frac_secs() << std::endl;
    std::cout << "atten_time_low " << atten_time_low.get_frac_secs() << std::endl;
    std::cout << "time now " << time_now.get_frac_secs() << std::endl;
   

    auto md = TXMetadata();
    md.set_has_time_spec(true);
    md.set_time_spec(pulse_start_time);
    md.set_start_of_burst(true);
    
    uint64_t num_samps_sent = 0;

    while (num_samps_sent < samples_per_buff){
      auto num_samps_to_send = samples_per_buff - num_samps_sent;
      std::cout << "Samples to send " << num_samps_to_send << std::endl;

      num_samps_sent = tx_stream->send(
        samples, num_samps_to_send, md.get_md()
      );

      std::cout << "Samples sent " << num_samps_sent << std::endl;

      md.set_start_of_burst(false);
      md.set_has_time_spec(false);

    }

    md.set_end_of_burst(true);
    tx_stream->send("", 0, md.get_md());

    end= std::chrono::steady_clock::now();
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    std::cout << "time to send to USRP: " 
          << total_us
          << "us" << std::endl;

    std::chrono::steady_clock::time_point timing_start= std::chrono::steady_clock::now();
    usrp_d->get_usrp()->clear_command_time();
    if (dp.sob() == true ) {
      std::cout << "Scope sync high" << std::endl;   
      usrp_d->get_usrp()->set_command_time(atten_time_high);
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

    
    std::chrono::steady_clock::time_point timing_end= std::chrono::steady_clock::now();

    std::cout << "Time to set up timing signals: " 
      << std::chrono::duration_cast<std::chrono::microseconds>(timing_end - timing_start).count()
      << "us" << std::endl;
    if (dp.eob() == true) {
      usrp_d->clear_scope_sync();
    }


    //sleep(1);
  }

}

void receive(zmq::context_t* thread_c, zmq::context_t* timing_c,
            std::shared_ptr<USRP> usrp_d, std::shared_ptr<DriverOptions> driver_options) {

  std::cout << "Enter receive thread\n";

  zmq::socket_t thread_socket(*thread_c, ZMQ_SUB);
  thread_socket.connect("inproc://threads");
  thread_socket.setsockopt(ZMQ_SUBSCRIBE,"",0);
  zmq::message_t request;
  std::cout << "RECEIVE: Creating and connected to thread socket in control\n";

  zmq::socket_t timing_socket(*timing_c, ZMQ_PAIR);
  thread_socket.connect("inproc://timing");
  zmq::message_t timing;

  auto channels_set = false;
  auto center_freq_set = false;

  
  std::vector<size_t> channels;

  uhd::time_spec_t time_zero;
  uhd::rx_streamer::sptr rx_stream;
  uhd::stream_args_t stream_args("fc32", "sc16");
  uhd::stream_cmd_t stream_cmd( 
      uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE
    );

  auto rx_rate = driver_options->get_rx_rate();

  while (1) {
    thread_socket.recv(&request);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    std::cout << "Received in RECEIVE\n";
    driverpacket::DriverPacket dp;
    std::string msg_str(static_cast<char*>(request.data()), request.size());
    dp.ParseFromString(msg_str);
    
    std::chrono::steady_clock::time_point stream_begin = std::chrono::steady_clock::now();
    if (dp.channels_size() > 0 && dp.sob() == true && channels_set == false) {
      channels = make_channels(dp);
      stream_args.channels = channels;
      usrp_d->set_rx_rate(rx_rate); //~450us 
      rx_stream = usrp_d->get_usrp()->get_rx_stream(stream_args); //~44ms
      channels_set = true;
    }
    std::chrono::steady_clock::time_point stream_end = std::chrono::steady_clock::now();
    std::cout << "stream set up time: "
          << std::chrono::duration_cast<std::chrono::microseconds>(stream_end - stream_begin).count()
          << "us" << std::endl;

    
    std::chrono::steady_clock::time_point ctr_begin = std::chrono::steady_clock::now();
    
    if (dp.rxcenterfreq() > 0) {
      std::cout << "DP center freq " << dp.rxcenterfreq() << std::endl;
      usrp_d->set_rx_center_freq(dp.rxcenterfreq(),channels);
      center_freq_set = true;
    }
    
    std::chrono::steady_clock::time_point ctr_end= std::chrono::steady_clock::now();
    std::cout << "center frq tuning time: "
          << std::chrono::duration_cast<std::chrono::microseconds>(ctr_end - ctr_begin).count()
          << "us" << std::endl;


    if( (channels_set == false) || (center_freq_set == false)){

      //todo: throw error
    }

    timing_socket.recv(&timing);
    time_zero = *(reinterpret_cast<uhd::time_spec_t*>(timing.data()));

    std::vector<float> sample_buffer(dp.numberofreceivesamples());
    stream_cmd.num_samps = size_t(dp.numberofreceivesamples() * channels.size());
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = time_zero;
    rx_stream->issue_stream_cmd(stream_cmd);

    auto md = RXMetadata();
    size_t total_received_samples = 0;
    while(total_received_samples < dp.numberofreceivesamples()) {
      auto md_star = md.get_md();
      size_t num_rx_samps = rx_stream->recv(&sample_buffer.front(), 
        (size_t)dp.numberofreceivesamples(), md_star);

      if (md.get_error_code() == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
          std::cout << boost::format("Timeout while streaming") << std::endl;
          break;
      }
      if (md.get_error_code() == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) {
          //TODO: throw error
      }
      if (md.get_error_code() != uhd::rx_metadata_t::ERROR_CODE_NONE) {
          //TODO: throw error
      }

      total_received_samples += num_rx_samps;
    }

    std::cout << "total_received_samples " << total_received_samples << std::endl;

  }

}

void control(zmq::context_t* thread_c) {
  std::cout << "Enter control thread\n";

  std::cout << "Creating and connecting to thread socket in control\n";
/*    zmq::socket_t thread_socket(*thread_c, ZMQ_PAIR); // 1
  thread_socket.connect("inproc://threads"); // 2  */  


  zmq::socket_t thread_socket (*thread_c, ZMQ_PUB);
  thread_socket.bind("inproc://threads");

  std::cout << "Creating and binding control socket\n";
  zmq::context_t context(2);
  zmq::socket_t control_sock(context, ZMQ_PAIR);
  control_sock.bind("ipc:///tmp/feeds/0");

  zmq::message_t request;

  sleep(1);
  while (1) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    control_sock.recv(&request);
    std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
/*        std::cout << "recv time(ms) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
    std::cout << "recv time(ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() <<std::endl; */

    begin = std::chrono::steady_clock::now();
    driverpacket::DriverPacket dp;
    std::string msg_str(static_cast<char*>(request.data()), request.size());
    dp.ParseFromString(msg_str);

    if ((dp.sob() == true) && (dp.eob() == true)){
      //todo: throw error
    }

    if (dp.samples_size() != dp.channels_size()){
      //todo: throw error
    }

    for(int i=0; i<dp.samples_size(); i++){
      if (dp.samples(i).real_size() != dp.samples(i).imag_size()){
        //todo: throw error
      }
    }



    //zmq::message_t forward = request;
    thread_socket.send(request);

    end= std::chrono::steady_clock::now();
    std::cout << "Time difference to deserialize and send in control = " 
          << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() 
          <<std::endl;

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
  zmq::context_t control_context(1);
  zmq::context_t timing_context(3);

  std::vector<std::thread> threads;

  std::thread control_t(control,&control_context);
  std::thread transmit_t(transmit, &control_context, &timing_context, usrp_d, driver_options);
  std::thread receive_t(receive, &control_context, &timing_context ,usrp_d, driver_options);
  

  threads.push_back(std::move(transmit_t));
  threads.push_back(std::move(receive_t));
  threads.push_back(std::move(control_t));

  for (auto& th : threads) {
    th.join();
  }


  return EXIT_SUCCESS;
}
