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

#include "utils/driver_options/driveroptions.hpp"
#include "usrp_drivers/n200/usrp.hpp"
#include "utils/protobuf/driverpacket.pb.h"
#include "utils/protobuf/rxsamplesmetadata.pb.h"
#include "utils/shared_memory/shared_memory.hpp"
#include "utils/shared_macros/shared_macros.hpp"
#include "utils/zmq_borealis_helpers/zmq_borealis_helpers.hpp"


//Delay needed for before any set_time_commands will work.
#define SET_TIME_COMMAND_DELAY 3e-3 // seconds




std::vector<std::complex<float>> buffer();
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
std::vector<std::vector<std::complex<float>>> make_tx_samples(const driverpacket::DriverPacket
                                                                &driver_packet)
{
  std::vector<std::vector<std::complex<float>>> samples(driver_packet.channel_samples_size());
  for (int channel=0; channel<driver_packet.channel_samples_size(); channel++) {
    auto num_samps = driver_packet.channel_samples(channel).real_size();
    std::vector<std::complex<float>> v(num_samps);
    auto smp = driver_packet.channel_samples(channel);
    for (int smp_num = 0; smp_num < num_samps; smp_num++) {
      v[smp_num] = std::complex<float>(smp.real(smp_num),smp.imag(smp_num));
    }
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


void transmit(zmq::context_t &driver_c, USRP &usrp_d, const DriverOptions &driver_options)
{
  DEBUG_MSG("Enter transmit thread");

  auto identities = {driver_options.get_driver_to_radctrl_identity(),
                      driver_options.get_driver_to_dsp_identity(),
                      driver_options.get_driver_to_brian_identity()};

  auto sockets_vector = create_sockets(driver_c, identities,driver_options.get_router_address());


  zmq::socket_t &driver_to_radar_control = sockets_vector[0];
  zmq::socket_t &driver_to_dsp = sockets_vector[1];
  zmq::socket_t &driver_to_brian = sockets_vector[2];


  auto usrp_channels_set = false;
  auto tx_center_freq_set = false;
  auto rx_center_freq_set = false;
  auto samples_set = false;

  std::vector<size_t> channels;

  uhd::tx_streamer::sptr tx_stream;
  uhd::stream_args_t stream_args("fc32", "sc16");

  size_t samples_per_buff;
  std::vector<std::vector<std::complex<float>>> samples;

  uint32_t sqn_num = 0;
  uint32_t expected_sqn_num = 0;

  /*This loop accepts pulse by pulse from the radar_control. It parses the samples, configures the
   *USRP, sets up the timing, and then sends samples/timing to the USRPs.
   */

  uhd::time_spec_t time_zero;
  uint32_t num_recv_samples;
  while (1)
  {
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
              auto actual_tx_rate = usrp_d.set_tx_rate(driver_packet.txrate(),channels); // TODO(keith): Test that USRPs exist to match channels in config.
              tx_stream = usrp_d.get_usrp_tx_stream(stream_args);  // ~44ms TODO(keith): See what 0s look like on scope.
              usrp_channels_set = true;
            }
          }()
        );

        TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " center freq ",
          [&]() {
            //If there is new center frequency data, set TX center frequency for each USRP TX channel.
            if (driver_packet.txcenterfreq() > 0.0)
            {
              DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " setting tx center freq to "
                        << driver_packet.txcenterfreq());
              usrp_d.set_tx_center_freq(driver_packet.txcenterfreq(),channels);
              tx_center_freq_set = true;
            }

            if (driver_packet.rxcenterfreq() > 0.0)
            {
              DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " setting rx center freq to "
                        << driver_packet.rxcenterfreq());
              usrp_d.set_rx_center_freq(driver_packet.rxcenterfreq(),channels);
              rx_center_freq_set = true;
            }

          }()

        );

        TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " sample unpack time: ",
          [&]() {
            //Parse new samples from driver packet if they exist.
            if (driver_packet.channel_samples_size() > 0)
            {  // ~700us to unpack 4x1600 samples
              samples = make_tx_samples(driver_packet);
              samples_per_buff = samples[0].size();
              samples_set = true;
            }
          }()
        );
      }()

    );

    //In order to transmit, these parameters need to be set at least once.
    if ((usrp_channels_set == false) ||
      (tx_center_freq_set == false) ||
      (rx_center_freq_set == false) ||
      (samples_set == false))
    {
      // TODO(keith): throw error
      continue;
    }

    if (driver_packet.sob() == true) {
      time_zero = usrp_d.get_current_usrp_time() + uhd::time_spec_t(DELAY);
      num_recv_samples = driver_packet.numberofreceivesamples();
    }

    TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " full usrp time stuff ",
      [&]() {
        //We send the start time and samples to the USRP. This is done before IO signals are set
        //since those commands create a blocking buffer which causes the transfer of samples to be
        //late. This leads to no waveform output on the USRP.
        TIMEIT_IF_DEBUG(COLOR_BLUE("TRANSMIT") << " time to send samples to USRP: ",
          [&]() {

            auto md = TXMetadata();
            md.set_has_time_spec(true);
            md.set_time_spec(time_zero + uhd::time_spec_t(driver_packet.timetosendsamples/1.0e6));
            //The USRP tx_metadata start_of_burst and end_of_burst describe start and end of the pulse
            //samples.
            md.set_start_of_burst(true);

            uint64_t num_samps_sent = 0;

            //This will loop until all samples are sent to the usrp. Send will block until all samples sent
            //or timed out(too many samples to send within timeout period). Send has a default timing of
            //0.1 seconds.
            while (num_samps_sent < samples_per_buff)
            {
              auto num_samps_to_send = samples_per_buff - num_samps_sent;
              DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " Samples to send " << num_samps_to_send);

              //Send behaviour can be found in UHD docs
              num_samps_sent = tx_stream->send(samples, num_samps_to_send, md.get_md()); //TODO(keith): Determine timeout properties.
              DEBUG_MSG(COLOR_BLUE("TRANSMIT") << " Samples sent " << num_samps_sent);

              md.set_start_of_burst(false);
              md.set_has_time_spec(false);
            }

            md.set_end_of_burst(true);
            tx_stream->send("", 0, md.get_md());
          }()
        );
      }()
    );

    if (driver_packet.eob() == true) {
      auto seqn_time = driver_options.get_rx_rate() * num_recv_samples;
      auto time_diff = usrp_d.get_current_usrp_time() - time_zero + uhd::time_spec_t(DELAY);

      auto sleep_time = (seqn_time - time_diff.get_real_secs()) * 1e6;
      usleep(sleep_time);

      auto start_sample = uint32_t((time_zero.get_real_secs() - start_time.get_real_secs()) *
                                    driver_options.get_rx_rate()) % ringbuffer_size;


      if ((start_sample + (seqn_time * driver_options.get_rx_rate())) > ringbuffer_size) {
        auto end_sample = uint32_t(start_sample + (seqn_time * driver_options.get_rx_rate())) -
                          ringbuffer_size;
      }
      else {
        auto end_sample = uint32_t(start_sample + (seqn_time * driver_options.get_rx_rate()));
      }
    }

   DEBUG_MSG(std::endl << std::endl);


  }
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
    std::string str(length,0);
    std::generate_n( str.begin(), length, randchar );
    return str;
}

    TIMEIT_IF_DEBUG("\033[33;40mRECEIVE\033[0m shared memory unpack timing: ",
      [&]() {
        mem_size = receive_channels.size() * driver_packet.numberofreceivesamples()
                            * sizeof(std::complex<float>);
        shrmem.create_shr_mem(mem_size);

        //create a vector of pointers to where each channel's data gets received.
        for(uint32_t i=0; i<receive_channels.size(); i++){
          auto ptr = static_cast<std::complex<float>*>(shrmem.get_shrmem_addr()) +
                                      i*driver_packet.numberofreceivesamples();
          buffer_ptrs.push_back(ptr);
        }
      }()
    );

/**
 * @brief      Runs in a seperate thread to control receiving from the USRPs.
 *
 * @param[in]  driver_c        The driver ZMQ context.
 * @param[in]  usrp_d          The multi-USRP SuperDARN wrapper object.
 * @param[in]  driver_options  The driver options parsed from config.
 */
void receive(zmq::context_t &driver_c, USRP &usrp_d, const DriverOptions &driver_options) {
  DEBUG_MSG("Enter receive thread");

  uhd::stream_args_t stream_args("fc32", "sc16");
  uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_START_CONTINUOUS);
  auto rx_rate_hz = driver_options.get_rx_rate();

  auto receive_channels = driver_options.get_receive_channels();
  stream_args.channels = receive_channels;

  usrp_d.set_rx_rate(rx_rate_hz,receive_channels);  // ~450us
  uhd::rx_streamer::sptr rx_stream = usrp_d.get_usrp_rx_stream(stream_args);  // ~44ms

  auto usrp_buffer_size = 100 * rx_stream->get_max_num_samps();
  ringbuffer_size = (size_t(500.0e6)/sizeof(std::complex<float>)/usrp_buffer_size) *
                            usrp_buffer_size;

  buffer.resize(receive_channels.size() * ringbuffer_size);

  std::vector<std::complex<float>*> buffer_ptrs_start;

  for(uint32_t i=0; i<rx_chans.size(); i++){
    auto ptr = static_cast<std::complex<float>*>(buffer.data() + (i * ringbuffer_size));
    buffer_ptrs_start.push_back(ptr);
  }
  std::vector<std::complex<float>*> buffer_ptrs = buffer_ptrs_start;

  stream_cmd.stream_now = false;
  stream_cmd.num_samps = 0;
  stream_cmd.time_spec = usrp_d->get_time_now() + uhd::time_spec_t(DELAY);

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

  auto first_time = true;


  //This loop receives 1 pulse sequence worth of samples.
  while (1) {
    size_t num_rx_samples = rx_stream->recv(buffer_ptrs, usrp_buffer_size, meta, 3.0);
    std::cout << "Recv " << num_rx_samples << " samples" << std::endl;
    std::cout << "On ringbuffer idx: " << usrp_buffer_size * buffer_inc << std::endl;
    //timeout = 0.5;
    auto error_code = meta.error_code;
    std::cout << "RX TIME: " << meta.time_spec.get_real_secs() << std::endl;
    if(first_time) {
      start_time = meta.time_spec;
      start_tx = true;
      first_time = false;
    }
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

    std::cout << "Timeout count: " << timeout_count << std::endl;
    std::cout << "Overflow count: " << overflow_count << std::endl;
    std::cout << "Overflow oos count: " << overflow_oos_count << std::endl;
    std::cout << "Late count: " << late_count << std::endl;
    std::cout << "Broken chain count: " << bchain_count << std::endl;
    std::cout << "Alignment count: " << align_count << std::endl;
    std::cout << "Bad packet count: " << badp_count << std::endl;

  }

}

/**
 * @brief      Runs in a seperate thread to act as an interface for the ingress and egress data.
 *
 * @param[in]  driver_c        The driver ZMQ context.
 */
void control(zmq::context_t &driver_c, const DriverOptions &driver_options) {
  DEBUG_MSG("Enter control thread");

  DEBUG_MSG("Creating and connecting to thread socket in control");

  zmq::socket_t driver_packet_pub_socket(driver_c, ZMQ_PUB);
  ERR_CHK_ZMQ(driver_packet_pub_socket.bind("inproc://threads"))

  DEBUG_MSG("Creating and binding control socket");
/*  zmq::socket_t radarctrl_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(radarctrl_socket.bind(driver_options.get_radar_control_to_driver_address()))
  zmq::message_t request;
*/
  zmq::socket_t data_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(data_socket.bind("inproc://data"))

  zmq::socket_t ack_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(ack_socket.bind("inproc://ack"))


/*  zmq::socket_t rx_dsp_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(rx_dsp_socket.connect(driver_options.get_driver_to_rx_dsp_address()))

*/

  //Sleep to handle "slow joiner" problem
  //http://zguide.zeromq.org/php:all#Getting-the-Message-Out
  sleep(1);

  while (1) {
    //radarctrl_socket.recv(&request);
    auto pulse_data = recv_data(driver_to_radar_control,
                                  driver_options.get_radctrl_to_driver_identity());

    driverpacket::DriverPacket driver_packet;
    TIMEIT_IF_DEBUG("CONTROL Time difference to deserialize and forward = ",
      [&]() {
        //std::string request_str(static_cast<char*>(request.data()), request.size());
        driver_packet.ParseFromString(pulse_data);

        DEBUG_MSG("Control " << driver_packet.sob() << " " << driver_packet.eob()
          << " " << driver_packet.channels_size());

        //TODO(keith): thinking about moving this err chking to transmit
        if (driver_packet.channel_samples_size() != driver_packet.channels_size()) {
          // TODO(keith): throw error
        }

        for (int channel = 0; channel < driver_packet.channel_samples_size(); channel++) {
          auto real_size = driver_packet.channel_samples(channel).real_size();
          auto imag_size = driver_packet.channel_samples(channel).imag_size();
          if (real_size != imag_size) {
            // TODO(keith): throw error
          }
        }
        if (driver_packet.sob() == false && driver_packet.timetosendsamples() == 0.0){
          //TODO(keith): throw error? this is really the best check i can think of for this field.
        }
        std::string whatisthis;
        driver_packet.SerializeToString(&whatisthis);
        zmq::message_t pulse(whatisthis.size());
        memcpy ((void *) pulse.data(), whatisthis.c_str(), whatisthis.size());

        driver_packet_pub_socket.send(pulse);
      }()
    );

    if (driver_packet.eob() == true) {
      //TODO(keith): handle potential errors.
      zmq::message_t ack, shr_mem_metadata;

      data_socket.recv(&shr_mem_metadata);
      //send data to dsp first so that processing can start before next sequence is aquired.
      //rx_dsp_socket.send(shr_mem_metadata);
      std::string meta_str(static_cast<char*>(shr_mem_metadata.data()),shr_mem_metadata.size());

      auto request = RECV_REQUEST(driver_to_dsp, driver_options.get_dsp_to_driver_identity());
      SEND_REPLY(driver_to_dsp, driver_options.get_dsp_to_driver_identity(), meta_str);

      request = RECV_REQUEST(driver_to_brian, driver_options.get_brian_to_driver_identity());
      SEND_REPLY(driver_to_brian, driver_options.get_brian_to_driver_identity(), meta_str);
      //ack_socket.recv(&ack);
      //radarctrl_socket.send(ack);

    }

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
  uhd::set_thread_priority_safe(1.0,true);

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
  std::thread control_t(control, std::ref(driver_context), std::ref(driver_options));
  std::thread transmit_t(transmit, std::ref(driver_context), std::ref(usrp_d),
                          std::ref(driver_options));
  std::thread receive_t(receive, std::ref(driver_context), std::ref(usrp_d),
                          std::ref(driver_options));


  threads.push_back(std::move(transmit_t));
  threads.push_back(std::move(receive_t));
  threads.push_back(std::move(control_t));

  for (auto& th : threads) {
    th.join();
  }


  return EXIT_SUCCESS;
}
