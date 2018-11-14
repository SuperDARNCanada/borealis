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
#include <sstream>
#include <utility>
#include <fstream>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <cstdlib>
#include <cmath>
#include "utils/driver_options/driveroptions.hpp"
#include "usrp_drivers/n200/usrp.hpp"
#include "utils/protobuf/driverpacket.pb.h"
#include "utils/protobuf/rxsamplesmetadata.pb.h"
#include "utils/shared_memory/shared_memory.hpp"
#include "utils/shared_macros/shared_macros.hpp"
#include "utils/zmq_borealis_helpers/zmq_borealis_helpers.hpp"


//Delay needed for before any set_time_commands will work.
#define SET_TIME_COMMAND_DELAY 5e-3 // seconds




std::vector<std::complex<float>> buffer;

/**
 * @brief      Makes a vector of USRP TX channels from a driver packet.
 *
 * @param[in]  driver_packet    A received driver packet from radar_control.
 *
 * @return     A vector of TX channels to use.
 *
 * Values in a protobuffer have no contiguous underlying storage so values need to be
 * parsed into a vector.
 */
std::vector<size_t> make_tx_channels(const driverpacket::DriverPacket &driver_packet)
{
  std::vector<size_t> channels(driver_packet.channels_size());
  for (int i=0; i<driver_packet.channels_size(); i++) {
    channels[i] = driver_packet.channels(i);
  }
  return channels;
}

/**
 * @brief      Makes a set of vectors of the samples for each TX channel from the driver packet.
 *
 * @param[in]  driver_packet    A received driver packet from radar_control.
 *
 * @return     A set of vectors of TX samples for each USRP channel.
 *
 * Values in a protobuffer have no contiguous underlying storage so values need to be
 * parsed into a vector.
 */
std::vector<std::vector<std::complex<float>>> make_tx_samples(
                                                    const driverpacket::DriverPacket &driver_packet,
                                                    const DriverOptions &driver_options)
{
  // channel_samples_size() will get you # of channels (protobuf in c++)
  std::vector<std::vector<std::complex<float>>> samples(driver_packet.channel_samples_size());

  // With TXIO board, we pad 0s to correspond with the first signal to go high which is the atten
  // signal. A hardware delay will then activate TR.
/*  int tr_start_pad = std::ceil(driver_packet.txrate() *
                              (driver_options.get_atten_window_time_start() +
                               driver_options.get_tr_window_time()));

  // We pad 0s to the end to the first signal that drops low which is TR. The hardware delay will
  // then drop the atten signal low.
  int tr_end_pad = std::ceil(driver_packet.txrate() * driver_options.get_tr_window_time());
*/
  //std::vector<std::complex<float>> start_pad(tr_start_pad, std::complex<float>(0.0f,0.0f));
  //std::vector<std::complex<float>> end_pad(tr_end_pad, std::complex<float>(0.0f,0.0f));
  for (int channel=0; channel<driver_packet.channel_samples_size(); channel++) {
    // Get the number of real samples in this particular channel (_size() is from protobuf)
    auto num_samps = driver_packet.channel_samples(channel).real_size();
    std::vector<std::complex<float>> v(num_samps);
    // Type for smp? protobuf object, containing repeated double real and double imag
    auto smp = driver_packet.channel_samples(channel); 
    for (int smp_num = 0; smp_num < num_samps; smp_num++) {
      v[smp_num] = std::complex<float>(smp.real(smp_num), smp.imag(smp_num));
    }

    //v.insert(v.begin(), start_pad.begin(), start_pad.end());
    //v.insert(v.end(), end_pad.begin(), end_pad.end());
    samples[channel] = v;
  }

  for (auto &s : samples)
  {
    if (s.size() != samples[0].size())
    {
      //TODO(keith): Handle this error. Samples buffers are of different lengths.
    }
  }

  return samples;
}

/**
 * @brief      Generates a string of random characters
 *
 * @param[in]  length  The length of desired string.
 *
 * @return     A string of random characters.
 *
 * This string is used for creation of named shared memory.
 */
std::string random_string( size_t length )
{
    //Lambda expression to return a random character.
    auto randchar = []() -> char
    {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(length, 0);
    std::generate_n( str.begin(), length, randchar );
    return str;
}

void transmit(zmq::context_t &driver_c, USRP &usrp_d, const DriverOptions &driver_options)
{
  DEBUG_MSG("Enter transmit thread");
  //uhd::set_thread_priority_safe(1.0,true);

  auto identities = {driver_options.get_driver_to_radctrl_identity(),
                      driver_options.get_driver_to_dsp_identity(),
                      driver_options.get_driver_to_brian_identity()};

  auto sockets_vector = create_sockets(driver_c, identities, driver_options.get_router_address());

  auto receive_channels = driver_options.get_receive_channels();

  zmq::socket_t &driver_to_radar_control = sockets_vector[0];
  zmq::socket_t &driver_to_dsp = sockets_vector[1];
  zmq::socket_t &driver_to_brian = sockets_vector[2];

  zmq::socket_t start_trigger(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(start_trigger.connect("inproc://thread"))

  auto usrp_channels_set = false;
  auto tx_center_freq_set = false;
  auto rx_center_freq_set = false;
  auto samples_set = false;

  std::vector<size_t> channels;

  uhd::tx_streamer::sptr tx_stream;
  uhd::stream_args_t stream_args("fc32", "sc16");

  std::vector<std::vector<std::vector<std::complex<float>>>> samples;


  uint32_t sqn_num = 0;
  uint32_t expected_sqn_num = 0;

  /*This loop accepts pulse by pulse from the radar_control. It parses the samples, configures the
   *USRP, sets up the timing, and then sends samples/timing to the USRPs.
   */


  uhd::time_spec_t time_zero;
  uint32_t num_recv_samples;

  size_t ringbuffer_size;
  uhd::time_spec_t start_time;

  zmq::message_t request;
  start_trigger.recv(&request);
  memcpy(&start_time, static_cast<uhd::time_spec_t*>(request.data()), request.size());

  start_trigger.recv(&request);
  memcpy(&ringbuffer_size, static_cast<size_t*>(request.data()), request.size());

  std::vector<std::complex<float>*> ringbuffer_ptrs_start;

  for(uint32_t i=0; i<receive_channels.size(); i++){
    auto ptr = static_cast<std::complex<float>*>(buffer.data() + (i * ringbuffer_size));
    ringbuffer_ptrs_start.push_back(ptr);
  }

  double tx_center_freq = 0.0, rx_center_freq = 0.0;
  while (1)
  {
    auto more_pulses = true;
    std::vector<double> time_to_send_samples;
    while (more_pulses) {
      auto pulse_data = recv_data(driver_to_radar_control,
                                    driver_options.get_radctrl_to_driver_identity());
      driverpacket::DriverPacket driver_packet;

      //Here we accept our driver_packet from the radar_control. We use that info in order to
      //configure the USRP devices based on experiment requirements.
      TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " total setup time: ",
        [&]() {
          if (driver_packet.ParseFromString(pulse_data) == false)
          {
            //TODO(keith): handle error
          }

          sqn_num = driver_packet.sequence_num();
          if (sqn_num != expected_sqn_num){
            DEBUG_MSG("SEQUENCE NUMBER MISMATCH: SQN " << sqn_num << " EXPECTED: "
              << expected_sqn_num);
            //TODO(keith): handle error
          }

          DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " burst flags: SOB "  << driver_packet.sob() << " EOB "
            << driver_packet.eob());

          TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " stream set up time: ",
            [&]() {
              //On start of new sequence, check if there are new USRP channels and if so
              //set what USRP TX channels and rate(Hz) to use.
              if (driver_packet.channels_size() > 0 && driver_packet.sob() == true)
              {
                DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " starting something new");
                channels = make_tx_channels(driver_packet);
                stream_args.channels = channels;
                auto actual_tx_rate = usrp_d.set_tx_rate(driver_packet.txrate(), channels); // TODO(keith): Test that USRPs exist to match channels in config.
                tx_stream = usrp_d.get_usrp_tx_stream(stream_args);  // ~44ms TODO(keith): See what 0s look like on scope.
                usrp_channels_set = true;
              }
            }()
          );

          TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " center freq ",
            [&]() {
              //If there is new center frequency data, set TX center frequency for each USRP TX channel.
              if (tx_center_freq != driver_packet.txcenterfreq()){
                if (driver_packet.txcenterfreq() > 0.0 && driver_packet.sob() == true)
                {
                  DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " setting tx center freq to "
                            << driver_packet.txcenterfreq());
                  usrp_d.set_tx_center_freq(driver_packet.txcenterfreq(), channels);
                  tx_center_freq_set = true;
                  tx_center_freq = driver_packet.txcenterfreq();
                }
              }

              // rxcenterfreq() will return 0 if it hasn't changed, so check for changes here
              if (rx_center_freq != driver_packet.rxcenterfreq()){
                if (driver_packet.rxcenterfreq() > 0.0 && driver_packet.sob() == true)
                {
                  DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " setting rx center freq to "
                            << driver_packet.rxcenterfreq());
                  usrp_d.set_rx_center_freq(driver_packet.rxcenterfreq(), channels);
                  rx_center_freq_set = true;
                  rx_center_freq = driver_packet.rxcenterfreq();
                }
              }

            }()

          );

          TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " sample unpack time: ",
            [&]() {
              //Parse new samples from driver packet if they exist.
              if (driver_packet.channel_samples_size() > 0)
              {  // ~700us to unpack 4x1600 samples
                if (driver_packet.sob() == true) {
                  samples.clear();
                }
                auto s = make_tx_samples(driver_packet, driver_options);
                samples.push_back(s);
                //samples_per_buff.push_back(s.size());
                samples_set = true;
              }
            }()
          );
        }();

        time_to_send_samples.push_back(driver_packet.timetosendsamples());

        if(driver_packet.sob() == true) {
          num_recv_samples = driver_packet.numberofreceivesamples();
        }

        if (driver_packet.eob() == true) {
          more_pulses = false;
        }
      );
    }

    //In order to transmit, these parameters need to be set at least once.
    if ((usrp_channels_set == false) ||
      (tx_center_freq_set == false) ||
      (rx_center_freq_set == false) ||
      (samples_set == false))
    {
      // TODO(keith): throw error
      continue;
    }

/*    std::ofstream output_file(std::string("samples") + std::to_string(sqn_num), std::ios::binary);
    //output_file += ;

    // Loop over samples, which is a vector of vectors of vectors (channel, pulse, imag/real)
    for (auto &s : samples) {
      for (auto &data : s) {
        output_file.write(reinterpret_cast<char*>(data.data()), data.size() * sizeof(std::complex<float>));
      }
    }*/
    time_zero = usrp_d.get_current_usrp_time() + uhd::time_spec_t(SET_TIME_COMMAND_DELAY);
    // Here we are time-aligning our time_zero to the start of a sample. Do this by recalculating
    // time_zero using the calculated value of start_sample.
    // TODO: Account for offset btw TX/RX (seems to change with sampling rate at least)
    double future_start_sample = std::floor((time_zero.get_real_secs() - start_time.get_real_secs()) *
                                    driver_options.get_rx_rate());
    time_zero = uhd::time_spec_t(start_time + (future_start_sample/driver_options.get_rx_rate()));

    TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " full usrp time stuff ",
      [&]() {
        //We send the start time and samples to the USRP. This is done before IO signals are set
        //since those commands create a blocking buffer which causes the transfer of samples to be
        //late. This leads to no waveform output on the USRP.
        TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " time to send samples to USRP: ",
          [&]() {

            for (int i=0; i<samples.size(); i++){
              auto md = TXMetadata();
              md.set_has_time_spec(true);
              auto time = time_zero + uhd::time_spec_t(time_to_send_samples[i]/1.0e6);
              md.set_time_spec(time);
              //std::cout << "time diff :" << time.get_real_secs() - usrp_d.get_current_usrp_time().get_real_secs() <<std::endl;
              //std::cout << "start_sample: " << future_start_sample << std::endl;
              //The USRP tx_metadata start_of_burst and end_of_burst describe start and end of the pulse
              //samples.
              md.set_start_of_burst(true);

              uint64_t num_samps_sent = 0;

              //This will loop until all samples are sent to the usrp. Send will block until all samples sent
              //or timed out(too many samples to send within timeout period). Send has a default timing of
              //0.1 seconds.
              auto samples_per_buff = samples[i][0].size();
              // If grabbing start of vector using samples[i] it doesn't work (samples are firked)
              // You need to grab the ptr to the vector using samples[a][b].data(). See tx_waveforms
              // for how to do this properly. Also see uhd::tx_streamer::send(...) in the uhd docs
              // see 'const buffs_type &'' argument to the send function, the description should read
              // 'Typedef for a pointer to a single, or a collection of pointers to send buffers'.
              std::vector<std::complex<float> *> samples_ptrs(samples[i].size());
              for (int j=0; j< samples[j].size(); j++) {
                samples_ptrs[j] = samples[i][j].data(); 
              }
              while (num_samps_sent < samples_per_buff)
              {
                auto num_samps_to_send = samples_per_buff - num_samps_sent;
                DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " Samples to send " << num_samps_to_send);

                //Send behaviour can be found in UHD docs
                num_samps_sent = tx_stream->send(samples_ptrs, num_samps_to_send, md.get_md()); //TODO(keith): Determine timeout properties.
                DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " Samples sent " << num_samps_sent);

                md.set_start_of_burst(false);
                md.set_has_time_spec(false);
              }

              md.set_end_of_burst(true);
              tx_stream->send("", 0, md.get_md());
            }
          }()
        );
      }()
    );


    auto seqn_sampling_time = num_recv_samples/driver_options.get_rx_rate();
    auto usrp_time = usrp_d.get_current_usrp_time();
    auto time_diff = usrp_time - time_zero + uhd::time_spec_t(SET_TIME_COMMAND_DELAY);

    // sleep_time is how much longer we need to wait in tx thread before the end of the sampling time
    auto delay = uhd::time_spec_t(SET_TIME_COMMAND_DELAY);
    auto sleep_time = (seqn_sampling_time + 1.5 * delay.get_real_secs()) * 1e6;
/*      std::cout << "sleep_time " << sleep_time << std::endl;
    std::cout << "seqn_sampling_time " << seqn_sampling_time << std::endl;
    std::cout << "time_zero " << time_zero.get_real_secs() << std::endl;
    std::cout << "usrp_time " << usrp_time.get_real_secs() << std::endl;
    std::cout << "time_diff " << time_diff.get_real_secs() << std::endl;*/
    usleep(sleep_time);


    //auto post_sqn_work = [&](){
      auto shr_mem_name = random_string(25);
      SharedMemoryHandler shrmem(shr_mem_name);

      auto mem_size = receive_channels.size() * num_recv_samples
                          * sizeof(std::complex<float>);
      shrmem.create_shr_mem(mem_size);

      //create a vector of pointers to where each channel's data gets received.
      std::vector<std::complex<float>*> buffer_ptrs;
      for(uint32_t i=0; i<receive_channels.size(); i++){
        auto ptr = static_cast<std::complex<float>*>(shrmem.get_shrmem_addr()) +
                                    i*num_recv_samples;
        buffer_ptrs.push_back(ptr);
      }

      auto start_sample = uint32_t(std::fmod(((time_zero.get_real_secs() - start_time.get_real_secs()) *
                                    driver_options.get_rx_rate()), ringbuffer_size));


      if ((start_sample + num_recv_samples) > ringbuffer_size) {
        for (int i=0; i<receive_channels.size(); i++) {
          auto first_piece = ringbuffer_size - start_sample;
          auto second_piece = num_recv_samples - first_piece;

          auto first_dest = buffer_ptrs[i];
          auto second_dest = buffer_ptrs[i] + (first_piece);

          auto first_src = ringbuffer_ptrs_start[i] + start_sample;
          auto second_src = ringbuffer_ptrs_start[i];

          memcpy(first_dest, first_src, first_piece*sizeof(std::complex<float>));
          memcpy(second_dest, second_src, second_piece*sizeof(std::complex<float>));
        }

      }
      else {
        for (int i=0; i<receive_channels.size(); i++) {
          auto dest = buffer_ptrs[i];
          auto src = ringbuffer_ptrs_start[i] + start_sample;

          memcpy(dest, src, num_recv_samples * sizeof(std::complex<float>));
        }
      }

      rxsamplesmetadata::RxSamplesMetadata samples_metadata;
      samples_metadata.set_numberofreceivesamples(num_recv_samples);
      samples_metadata.set_shrmemname(shr_mem_name);
      samples_metadata.set_sequence_num(sqn_num);
      std::string samples_metadata_str;
      samples_metadata.SerializeToString(&samples_metadata_str);


      // Here we wait for a request from dsp for the samples metadata, then send it, bro!
// https://www.youtube.com/watch?v=WIrWyr3HgXI
      auto request = RECV_REQUEST(driver_to_dsp, driver_options.get_dsp_to_driver_identity());
      SEND_REPLY(driver_to_dsp, driver_options.get_dsp_to_driver_identity(), samples_metadata_str);

      // Here we wait for a request from brian for the samples metadata, then send it
      request = RECV_REQUEST(driver_to_brian, driver_options.get_brian_to_driver_identity());
      SEND_REPLY(driver_to_brian, driver_options.get_brian_to_driver_identity(), samples_metadata_str);
    //};

    //std::thread post_sqn_work_t(post_sqn_work);
    //post_sqn_work_t.detach();

    expected_sqn_num++;
    more_pulses = true;
    time_to_send_samples.clear();
    DEBUG_MSG(std::endl << std::endl);
  }

}


/**
 * @brief      Runs in a seperate thread to control receiving from the USRPs.
 *
 * @param[in]  driver_c        The driver ZMQ context.
 * @param[in]  usrp_d          The multi-USRP SuperDARN wrapper object.
 * @param[in]  driver_options  The driver options parsed from config.
 */
void receive(zmq::context_t &driver_c, USRP &usrp_d, const DriverOptions &driver_options) {
  DEBUG_MSG("Enter receive thread");

  zmq::socket_t start_trigger(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(start_trigger.bind("inproc://thread"));

  uhd::stream_args_t stream_args("fc32", "sc16");
  uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
  auto rx_rate_hz = driver_options.get_rx_rate();

  auto receive_channels = driver_options.get_receive_channels();
  stream_args.channels = receive_channels;

  usrp_d.set_rx_rate(rx_rate_hz, receive_channels);  // ~450us
  uhd::rx_streamer::sptr rx_stream = usrp_d.get_usrp_rx_stream(stream_args);  // ~44ms

  /* 100 is the arbitrary scaling for the usrp_buffer_size
     so there won't be fragmentation and the while(1) loop below
     with the recv runs less times
  */
  auto usrp_buffer_size = 100 * rx_stream->get_max_num_samps();
  // TODO: Put ringbuffer_size into the config file
  /* The ringbuffer_size is calculated this way because it's first truncated (size_t)
     then rescaled by usrp_buffer_size */
  size_t ringbuffer_size = (size_t(500.0e6)/sizeof(std::complex<float>)/usrp_buffer_size) *
                            usrp_buffer_size;

  buffer.resize(receive_channels.size() * ringbuffer_size);

  std::vector<std::complex<float>*> buffer_ptrs_start;

  for(uint32_t i=0; i<receive_channels.size(); i++){
    auto ptr = static_cast<std::complex<float>*>(buffer.data() + (i * ringbuffer_size));
    buffer_ptrs_start.push_back(ptr);
  }
  std::vector<std::complex<float>*> buffer_ptrs = buffer_ptrs_start;

  stream_cmd.stream_now = false;
  stream_cmd.num_samps = 0;
  stream_cmd.time_spec = usrp_d.get_current_usrp_time() + uhd::time_spec_t(SET_TIME_COMMAND_DELAY);

  rx_stream->issue_stream_cmd(stream_cmd);

  //auto kill_loop = false;
  uhd::rx_metadata_t meta;

  uint32_t buffer_inc = 0;
  uint32_t timeout_count = 0;
  uint32_t overflow_count = 0;
  uint32_t overflow_oos_count = 0;
  uint32_t late_count = 0;
  uint32_t bchain_count = 0;
  uint32_t align_count = 0;
  uint32_t badp_count = 0;

  zmq::message_t start_time(sizeof(stream_cmd.time_spec));
  memcpy(start_time.data(), &stream_cmd.time_spec, sizeof(stream_cmd.time_spec));
  start_trigger.send(start_time);

  zmq::message_t ring_size(sizeof(ringbuffer_size));
  memcpy(ring_size.data(), &ringbuffer_size, sizeof(ringbuffer_size));
  start_trigger.send(ring_size);


  //This loop receives 1 pulse sequence worth of samples.
  while (1) {
    // 3.0 is the timeout in seconds for the recv call, arbitrary number
    size_t num_rx_samples = rx_stream->recv(buffer_ptrs, usrp_buffer_size, meta, 3.0);
/*    std::cout << "Recv " << num_rx_samples << " samples" << std::endl;
    std::cout << "On ringbuffer idx: " << usrp_buffer_size * buffer_inc << std::endl;*/
    //timeout = 0.5;
    auto error_code = meta.error_code;
    //std::cout << "RX TIME: " << meta.time_spec.get_real_secs() << std::endl;
    switch(error_code) {
      case uhd::rx_metadata_t::ERROR_CODE_NONE :
        break;
      case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT : {
        std::cout << "Timed out!" << std::endl;
        timeout_count++;
        break;
      }
      case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW : {
        std::cout << "Overflow!" << std::endl;
        std::cout << "OOS: " << meta.out_of_sequence << std::endl;
        if (meta.out_of_sequence == 1) overflow_oos_count ++;
        overflow_count++;
        break;
      }
      case uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND : {
        std::cout << "LATE!" << std::endl;
        late_count++;
        break;
      }
      case uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN : {
        std::cout << "BROKEN CHAIN!" << std::endl;
        bchain_count++;
      }
      case uhd::rx_metadata_t::ERROR_CODE_ALIGNMENT : {
        std::cout << "ALIGNMENT!" << std::endl;
        align_count++;

      }
      case uhd::rx_metadata_t::ERROR_CODE_BAD_PACKET : {
        std::cout << "BAD PACKET!" << std::endl;
        badp_count++;
      }
      default :
        break;
    }

    if ((buffer_inc+1) * usrp_buffer_size < ringbuffer_size) {
      for (auto &buffer_ptr : buffer_ptrs) {
        buffer_ptr += usrp_buffer_size;
      }
      buffer_inc++;
    }
    else{
      buffer_ptrs = buffer_ptrs_start;
      buffer_inc = 0;
    }

/*    std::cout << "Timeout count: " << timeout_count << std::endl;
    std::cout << "Overflow count: " << overflow_count << std::endl;
    std::cout << "Overflow oos count: " << overflow_oos_count << std::endl;
    std::cout << "Late count: " << late_count << std::endl;
    std::cout << "Broken chain count: " << bchain_count << std::endl;
    std::cout << "Alignment count: " << align_count << std::endl;
    std::cout << "Bad packet count: " << badp_count << std::endl;
*/
  }

}



/**
 * @brief      UHD wrapped main function to start threads.
 *
 * @return     EXIT_SUCCESS
 *
 * Creates a new multi-USRP object using parameters from config file. Starts control, receive,
 * and transmit threads to operate on the multi-USRP object.
 */
int UHD_SAFE_MAIN(int argc, char *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  uhd::set_thread_priority_safe();

  DriverOptions driver_options;

  DEBUG_MSG(driver_options.get_device_args());
  DEBUG_MSG(driver_options.get_tx_rate());
  DEBUG_MSG(driver_options.get_pps());
  DEBUG_MSG(driver_options.get_ref());
  DEBUG_MSG(driver_options.get_tx_subdev());

  USRP usrp_d(driver_options);


  //  Prepare our context
  zmq::context_t driver_context(1);

  std::vector<std::thread> threads;

  // std::ref http://stackoverflow.com/a/15530639/1793295
  std::thread receive_t(receive, std::ref(driver_context), std::ref(usrp_d),
                          std::ref(driver_options));


  std::thread transmit_t(transmit, std::ref(driver_context), std::ref(usrp_d),
                          std::ref(driver_options));

  threads.push_back(std::move(transmit_t));
  threads.push_back(std::move(receive_t));

  for (auto& th : threads) {
    th.join();
  }


  return EXIT_SUCCESS;
}
