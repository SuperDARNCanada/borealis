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
#include "utils/protobuf/rxsamplesmetadata.pb.h"
#include "utils/shared_memory/shared_memory.hpp"

#ifdef DEBUG
#define DEBUG_MSG(x) do {std::cerr << x << std::endl;} while (0)
#else
#define DEBUG_MSG(x)
#endif

/**
 * @brief      Makes a vector of USRP TX channels from a driver packet.
 *
 * @param[in]  driver_packet    A received driver packet.
 *
 * @return     A vector of TX channels to use.
 *
 * Values in a protobuffer have no contiguous underlying storage so values need to be
 * parsed into a vector.
 */
std::vector<size_t> make_tx_channels(const driverpacket::DriverPacket &driver_packet)
{
  std::vector<size_t> channels(driver_packet.channels_size());
  for (int i = 0; i < driver_packet.channels_size(); i++) {
    channels[i] = i; // REVIEW # 0 why not = driver_packet.channels(i) ?
  }
  return channels;
}

/**
 * @brief      Makes a set of vectors of the samples for each TX channel from the driver packet.
 *
 * @param[in]  driver_packet    A received driver packet. // REVIEW #1 from the radar_control parsed via protobuf
 *
 * @return     A set of vectors of TX samples for each USRP channel.
 *
 * Values in a protobuffer have no contiguous underlying storage so values need to be
 * parsed into a vector.
 */
std::vector<std::vector<std::complex<float>>> make_tx_samples(const driverpacket::DriverPacket
                                                                &driver_packet)
{
  std::vector<std::vector<std::complex<float>>> samples(driver_packet.samples_size());
 // REVIEW # 15 are all channel buffers (vectors) the same length - should verify.
  for (int i = 0 ; i < driver_packet.samples_size(); i++) {
    auto num_samps = driver_packet.samples(i).real_size();
    std::vector<std::complex<float>> v(num_samps);
    samples[i] = v;

    for (int j = 0; j < driver_packet.samples(i).real_size(); j++) {
      samples[i][j] = std::complex<float>(driver_packet.samples(i).real(j),
                                           driver_packet.samples(i).imag(j));
    }
  }

  return samples;
}

/**
 * @brief      Runs in a seperate thread to control transmission from the USRPs.
 *
 * @param[in]  driver_c        The driver ZMQ context.
 * @param[in]  usrp_d          The multi-USRP SuperDARN wrapper object.
 * @param[in]  driver_options  The driver options parsed from config.
 */
void transmit(zmq::context_t &driver_c, USRP &usrp_d,
                const DriverOptions &driver_options)
{
  DEBUG_MSG("Enter transmit thread");


  zmq::socket_t packet_socket(driver_c, ZMQ_SUB);
  packet_socket.connect("inproc://threads");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)
  packet_socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);// REVIEW #37 check return value (0 if success, -1 if fail with errno set)
  DEBUG_MSG("TRANSMIT: Creating and connected to thread socket in control");
  zmq::message_t request;// REVIEW #32 Where should this be? The other message_t is initialized down below the socket connections, but this one is here, any reason why?

  zmq::socket_t timing_socket(driver_c, ZMQ_PAIR);
  timing_socket.connect("inproc://timing");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)

  zmq::socket_t ack_socket(driver_c, ZMQ_PAIR);
  ack_socket.connect("inproc://ack");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)


  auto usrp_channels_set = false;
  auto center_freq_set = false;
  auto samples_set = false;

  std::vector<size_t> channels;

  uhd::time_spec_t time_zero;
  zmq::message_t timing (sizeof(time_zero)); // REVIEW #26 Since the only thing this is used for is scope sync timing - maybe change the name to reflect that.

  uhd::tx_streamer::sptr tx_stream;
  uhd::stream_args_t stream_args("fc32", "sc16");

  size_t samples_per_buff;
  std::vector<std::vector<std::complex<float>>> samples;

  auto atten_window_time_start_s = driver_options.get_atten_window_time_start(); //seconds
  auto atten_window_time_end_s = driver_options.get_atten_window_time_end(); //seconds
  auto tr_window_time_s = driver_options.get_tr_window_time(); //seconds

  //default initialize SS. Needs to be outside of loop
  auto scope_sync_low = uhd::time_spec_t(0.0);

  uint32_t sqn_num = -1; // REVIEW #0 assigning -1 to uint does this work as intended? also unnecessary
  uint32_t expected_sqn_num = 0;
// REVIEW #0 If there are large periods of time between pulses, this while loop might be too speedy, resulting in overflows to the usrp - may have seen this in testing? - can we calculate an amount of time to sleep if that's the case?
  while (1) // REVIEW #35 Over 220 lines. are there any candidates for functions? (timing and cout code?)
  {// REVIEW #1 this while loop is for one pulse sequence, it does blah
    packet_socket.recv(&request);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now(); // REVIEW #26 begin_setup ?
    DEBUG_MSG("Received in TRANSMIT");
    driverpacket::DriverPacket driver_packet; // REVIEW #32 can move this and declaration of the packet_msg_str out of the while loop
    std::string packet_msg_str(static_cast<char*>(request.data()), request.size());
    driver_packet.ParseFromString(packet_msg_str); // REVIEW 37 check for return False

    sqn_num = driver_packet.sequence_num(); // REVIEW #0 default value for numeric types in protobuf is =0 this could create errors, should start at 1?

    if (sqn_num != expected_sqn_num){
      std::cout << "SEQUENCE NUMBER MISMATCH: SQN " << sqn_num << " EXPECTED: "
        << expected_sqn_num << std::endl;
      //TODO(keith): handle error
    }

    std::cout <<"TRANSMIT burst flags: SOB "  << driver_packet.sob() << " EOB " << driver_packet.eob() <<std::endl;
    std::chrono::steady_clock::time_point stream_begin = std::chrono::steady_clock::now();

    //On start of new sequence, check if there are new USRP channels and if so
    //set what USRP TX channels and rate(Hz) to use.
    if (driver_packet.channels_size() > 0 && driver_packet.sob() == true)
    {
      std::cout << "TRANSMIT starting something new" <<std::endl;
      channels = make_tx_channels(driver_packet);
      stream_args.channels = channels; // REVIEW #15 check that the channels from the driver packet are actually available (a mapping as mentioned in other comment would be handy here)
      usrp_d.set_tx_rate(driver_packet.txrate());  // ~450us REVIEW #15 check that it's not = 0 (default)
      tx_stream = usrp_d.get_usrp()->get_tx_stream(stream_args);  // ~44ms REVIEW : idea maybe could we set this up with all channels and then send 0s to any that we don't want to transmit on? then this tx_stream setup could be removed from while loop. 
      usrp_channels_set = true;
    }
    std::chrono::steady_clock::time_point stream_end = std::chrono::steady_clock::now(); // REVIEW #26 stream_setup_end ? (similarly stream_setup_begin) 

    std::cout << "TRANSMIT stream set up time: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (stream_end - stream_begin).count()
      << "us" << std::endl;


    std::chrono::steady_clock::time_point ctr_begin = std::chrono::steady_clock::now(); // REVIEW #26 ctr_setup_begin & ctr_setup_end:

    //If there is new center frequency data, set TX center frequency for each USRP TX channel.
    if (driver_packet.txcenterfreq() > 0.0)
    {
      std::cout << "TRANSMIT center freq " << driver_packet.txcenterfreq() << std::endl; // REVIEW #34 should print the actual freq after it's been set (return actual frequency from set_tx_center_freq)
      usrp_d.set_tx_center_freq(driver_packet.txcenterfreq(), channels);
      center_freq_set = true;
    }

    std::chrono::steady_clock::time_point ctr_end = std::chrono::steady_clock::now();
    std::cout << "TRANSMIT center frq tuning time: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (ctr_end - ctr_begin).count()
      << "us" << std::endl;


    std::chrono::steady_clock::time_point sample_begin = std::chrono::steady_clock::now(); // REVIEW #26 sample_unpack_begin

    //Parse new samples from driver packet if they exist.
    if (driver_packet.samples_size() > 0)
    {  // ~700us to unpack 4x1600 samples
      samples = make_tx_samples(driver_packet);
      samples_per_buff = samples[0].size();
      samples_set = true;
    }

    std::chrono::steady_clock::time_point sample_end = std::chrono::steady_clock::now();
    std::cout << "TRANSMIT sample unpack time: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (sample_end - sample_begin).count()
      << "us" << std::endl;

    //In order to transmit, these parameters need to be set at least once.
    if ((usrp_channels_set == false) || (center_freq_set == false) ||(samples_set == false))
    {
      // TODO(keith): throw error
      continue;
    }


    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now(); // REVIEW #26 end_setup

    std::cout << "TRANSMIT Total set up time: "
          << std::chrono::duration_cast<std::chrono::microseconds>
                                                      (end - begin).count()
          << "us" << std::endl;



    if (driver_packet.sob() == true)
    {
      //The USRP needs about a 10ms buffer into the future before time
      //commands will correctly work. This was found through testing and may be subject to change
      //with USRP firmware updates.
      time_zero = usrp_d.get_usrp()->get_time_now() + uhd::time_spec_t(10e-3); // REVIEW #29 could set this magic number in config or global const?
    }

// REVIEW #26 pulse_delay might not be the best name, why don't we just use timetosendsamples_s or get rid of it entirely, or something_better ? 
    auto pulse_delay = uhd::time_spec_t(driver_packet.timetosendsamples()/1e6); //convert us to s
    auto pulse_start_time = time_zero + pulse_delay;
    auto pulse_len_time = uhd::time_spec_t(samples_per_buff/usrp_d.get_tx_rate());

    auto tr_time_high = pulse_start_time - uhd::time_spec_t(tr_window_time_s);
    auto atten_time_high = tr_time_high - uhd::time_spec_t(atten_window_time_start_s);
    //This line lets us trigger TR with atten timing, but
    //we can still keep the existing logic if we want to use it. 
    tr_time_high = atten_time_high; // REVIEW #26 separate and create a new name for the new timing signal where x_high = atten_time_high and x_low = tr_time_low
    auto scope_sync_high = atten_time_high; // REVIEW #32 move this to within the next if(driver_packet.sob() == true) because it's only used in those instances ?
    auto tr_time_low = pulse_start_time + pulse_len_time + uhd::time_spec_t(tr_window_time_s);
    auto atten_time_low = tr_time_low + uhd::time_spec_t(atten_window_time_end_s);

    //To make sure tx and rx timing are synced, this thread sends when to receive to the receive
    //thread.
    if( driver_packet.sob() == true)
    {
      memcpy(timing.data(), &scope_sync_high, sizeof(scope_sync_high));
      timing_socket.send(timing);
    }

    begin = std::chrono::steady_clock::now(); // REVIEW #26 begin_send?  

    auto time_now = usrp_d.get_usrp()->get_time_now(); // REVIEW #34 making these print statements every single pulse is a lot. make them verbose debug output? This goes for all the cout statements in this while loop
    std::cout << "TRANSMIT time zero: " << time_zero.get_frac_secs() << std::endl; // REVIEW #34 make the print statments match the variable names - need a few underscores
    std::cout << "TRANSMIT time now " << time_now.get_frac_secs() << std::endl;
    std::cout << "TRANSMIT timetosendsamples(us) " << driver_packet.timetosendsamples() <<std::endl;
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
    md.set_start_of_burst(true); // REVIEW #3 different than the driver packet's sob() - maybe make a comment somewhere explaining the TXmetadata object and its purpose?

    uint64_t num_samps_sent = 0;
// REVIEW #3 comment how this loop is working. send has a default timeout of 0.1s, blocks... etc. The samples variable is a vector of buffers containing the complex samples, etc...
    while (num_samps_sent < samples_per_buff) {
      auto num_samps_to_send = samples_per_buff - num_samps_sent;
      std::cout << "TRANSMIT Samples to send " << num_samps_to_send << std::endl;

      num_samps_sent = tx_stream->send( // REVIEW #37 How do we know if it's timed out? Does it do the fragmentation of packets internal to the send(...) if above the max num samps value?
        samples, num_samps_to_send, md.get_md()); // REVIEW #0 Should we set the timeout to a reasonable value given our timing constraints? 100ms would be way too long to recover this pulse sequence. Maybe we can make a calculation based on how long this pulse is and set the timeout slightly beyond that - or beyond atten_time_low (at which point you should be in this loop for the next pulse) We can also get_max_num_samps to find out if this call will fragment into multiple packets, not sure of the behaviour there, does it return before the timeout??

      std::cout << "TRANSMIT Samples sent " << num_samps_sent << std::endl;

      md.set_start_of_burst(false);
      md.set_has_time_spec(false);
    }

    md.set_end_of_burst(true);
    tx_stream->send("", 0, md.get_md()); // REVIEW #43 If we know that we are under the max num samps value, then we could send start of burst and end of burst in the same send(...) call

    end = std::chrono::steady_clock::now(); // REVIEW #26 end_send?

    std::cout << "TRANSMIT time to send to USRP: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                          (end - begin).count()
      << "us" << std::endl;

    std::chrono::steady_clock::time_point timing_start = std::chrono::steady_clock::now();

    //Set high speed IO timing on the USRP now.
    usrp_d.get_usrp()->clear_command_time(); // REVIEW #0 The GPIO timing setup should be moved to above the samples sending, in case we timeout on sending samples (i.e. we sent half of them) then the RF waveform would be transmitting without timing signals, bad idea. Alternatively, what if there was a 4th thread for timing specifically?

    if (driver_packet.sob() == true) {
      auto sync_time = driver_packet.numberofreceivesamples()/driver_options.get_rx_rate();
      std::cout << "SYNC TIME " << sync_time << std::endl;
      scope_sync_low = atten_time_high + uhd::time_spec_t(sync_time); // REVIEW #26 use scope_sync_high instead of atten_time_high for clarity

      std::cout << "TRANSMIT Scope sync high" << std::endl; // REVIEW #34 this comment isn't really true, since the scope sync high happens in the 'future' since it's a timed command
      usrp_d.get_usrp()->set_command_time(scope_sync_high);
      usrp_d.set_scope_sync();

    }
// REVIEW #0 The other timing signal needs to happen here as well where x_high = atten_time_high and x_low = tr_time_low
    usrp_d.get_usrp()->set_command_time(atten_time_high);
    usrp_d.set_atten();

    usrp_d.get_usrp()->set_command_time(tr_time_high);
    usrp_d.set_tr();

    usrp_d.get_usrp()->set_command_time(tr_time_low);
    usrp_d.clear_tr();

    usrp_d.get_usrp()->set_command_time(atten_time_low);
    usrp_d.clear_atten();

    if (driver_packet.eob() == true) {
      usrp_d.get_usrp()->set_command_time(scope_sync_low);
      usrp_d.clear_scope_sync();
    }


    std::chrono::steady_clock::time_point timing_end = std::chrono::steady_clock::now();

    std::cout << "TRANSMIT Time to set up timing signals: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (timing_end - timing_start).count()
      << "us" << std::endl;

    //Final end of sequence work to acknowledge seq num.
    if (driver_packet.eob() == true) {
      //usrp_d.clear_scope_sync(); // REVIEW #33 use git
      driverpacket::DriverPacket ack; // REVIEW #33 Can this just be set up once at the start, otherwise duplicating this code. same with ack_str and ack_msg
      std::cout << "SEQUENCENUM " << sqn_num << std::endl;
      ack.set_sequence_num(sqn_num);
      expected_sqn_num += 1;

      std::string ack_str;
      ack.SerializeToString(&ack_str);

      zmq::message_t ack_msg(ack_str.size());
      memcpy((void*)ack_msg.data(), ack_str.c_str(),ack_str.size());
      ack_socket.send(ack_msg);
    }


  }
}

std::string random_string( size_t length ) // REVIEW #1 explain what this is for
{
    auto randchar = []() -> char // REVIEW #1 how does this work?
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

void receive(zmq::context_t &driver_c, USRP &usrp_d,
              const DriverOptions &driver_options) {
  std::cout << "Enter receive thread\n";

  zmq::socket_t packet_socket(driver_c, ZMQ_SUB);
  packet_socket.connect("inproc://threads");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)
  packet_socket.setsockopt(ZMQ_SUBSCRIBE, "", 0);// REVIEW #37 check return value (0 if success, -1 if fail with errno set)
  zmq::message_t request;
  std::cout << "RECEIVE: Creating and connected to thread socket in control\n";

  zmq::socket_t timing_socket(driver_c, ZMQ_PAIR); // REVIEW #26 rename to scope_sync_socket or something of the sort?
  timing_socket.bind("inproc://timing");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)
  zmq::message_t timing; // REVIEW #26 - as mentioned before change name

  //zmq::context_t context(4);  // REVIEW #33 remove
  zmq::socket_t data_socket(driver_c, ZMQ_PAIR);
  data_socket.connect("inproc://data");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)

  auto usrp_channels_changed = false; // REVIEW #26 rename to rx_streamer_set as they don't get changed, always receives on all and acts as a flag 
  auto center_freq_set = false;

  uhd::time_spec_t time_zero;
  uhd::rx_streamer::sptr rx_stream;
  uhd::stream_args_t stream_args("fc32", "sc16");
  uhd::stream_cmd_t stream_cmd(
      uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE); // REVIEW #3 why did we choose this stream_mode?

  auto rx_rate_hz = driver_options.get_rx_rate();

  auto receive_channels = usrp_d.get_receive_channels();

  while (1) { // REVIEW #1 comment this while loop is for one pulse sequence / scope sync time of receiving.
    packet_socket.recv(&request);
    std::cout << "Received in RECEIVE\n";
    driverpacket::DriverPacket driver_packet; // REVIEW #32 can move this and declaration of the packet_msg_str out of the while loop
    std::string msg_str(static_cast<char*>(request.data()), request.size()); // REVIEW #26 stay consistent with packet_msg_str
    driver_packet.ParseFromString(msg_str);

    std::cout << "RECEIVE burst flags SOB " << driver_packet.sob() << " EOB " << driver_packet.eob() << std::endl;

    if ( driver_packet.sob() == false ) continue;

    std::chrono::steady_clock::time_point stream_begin = std::chrono::steady_clock::now(); // REVIEW #26 rename to setup_rx_stream_begin or something more clear
    if (driver_packet.channels_size() > 0 && driver_packet.sob() == true && usrp_channels_changed == false) { // REVIEW #0 this would prevent us from doing receive-only operation. Should come up with another way to check this. Should do error handling if we receive first packet and sob = False ; also could be moved out of the while loop because doesn't depend on the driver packet information.
      stream_args.channels = receive_channels;
      usrp_d.set_rx_rate(rx_rate_hz);  // ~450us
      rx_stream = usrp_d.get_usrp()->get_rx_stream(stream_args);  // ~44ms
      usrp_channels_changed = true;
    }
    std::chrono::steady_clock::time_point stream_end = std::chrono::steady_clock::now(); // REVIEW #26 rename
    std::cout << "RECEIVE stream set up time: "
      << std::chrono::duration_cast<std::chrono::microseconds>(stream_end - stream_begin).count()
      << "us" << std::endl;


    std::chrono::steady_clock::time_point ctr_begin = std::chrono::steady_clock::now(); // REVIEW #26 rename

    if (driver_packet.rxcenterfreq() > 0) {
      std::cout << "RECEIVE center freq " << driver_packet.rxcenterfreq() << std::endl; // REVIEW #34 move this print statement below and use actual set rxcenterfreq
      usrp_d.set_rx_center_freq(driver_packet.rxcenterfreq(), receive_channels);
      center_freq_set = true;
    }

    std::chrono::steady_clock::time_point ctr_end = std::chrono::steady_clock::now(); // REVIEW #26
    std::cout << "RECEIVE center frq tuning time: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (ctr_end - ctr_begin).count()
      << "us" << std::endl;


    if ((usrp_channels_changed == false) || (center_freq_set == false)) {
      // TODO(keith): throw error // REVIEW #15 possible idea for recovering from these types of errors: to have a set of defaults that could be used to recover with 
    }

    // REVIEW #22 could do this only when mem_size needs to increase ?
    std::chrono::steady_clock::time_point shr_begin = std::chrono::steady_clock::now(); // REVIEW #26 
    size_t mem_size = receive_channels.size() * driver_packet.numberofreceivesamples() * sizeof(std::complex<float>);
    auto shr_mem_name = random_string(25); // REVIEW #1 why is this a different name every time ... explain
    SharedMemoryHandler shrmem(shr_mem_name);
    shrmem.create_shr_mem(mem_size); // REVIEW #41 would it be worthwhile to put this into the constructor?

    //create a vector of pointers to where each channel's data gets received.
    std::vector<std::complex<float> *> buffer_ptrs;
    for(uint32_t i=0; i<receive_channels.size(); i++){
      auto ptr = static_cast<std::complex<float>*>(shrmem.get_shrmem_addr()) +
                                  i*driver_packet.numberofreceivesamples(); // REVIEW #0 does this work auto = complex float * + int 
      buffer_ptrs.push_back(ptr);
    }

    std::chrono::steady_clock::time_point shr_end = std::chrono::steady_clock::now(); // REVIEW #26
    std::cout << "RECEIVE shr timing: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (shr_end - shr_begin).count()
      << "us" << std::endl;

    std::cout << "Got to RECEIVE timing" << std::endl;
    timing_socket.recv(&timing);
    std::cout << "RECEIVED timing data" << std::endl;
    time_zero = *(reinterpret_cast<uhd::time_spec_t*>(timing.data()));

    std::cout << "timing data " << timing.data() << " size "<< timing.size();
    std::cout << "RECEIVE time_zero " << time_zero.get_frac_secs() << std::endl;

    std::chrono::steady_clock::time_point recv_begin = std::chrono::steady_clock::now();

    //Documentation is unclear, but num samps is per channel
    stream_cmd.num_samps = size_t(driver_packet.numberofreceivesamples());
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = time_zero;
    rx_stream->issue_stream_cmd(stream_cmd);

    auto md = RXMetadata();
    size_t accumulated_received_samples = 0;
    std::cout << "RECEIVE total samples to receive: " << receive_channels.size() * driver_packet.numberofreceivesamples()
    << " mem size " << mem_size << std::endl;

    while (accumulated_received_samples < driver_packet.numberofreceivesamples()) {
      size_t num_rx_samps = rx_stream->recv(buffer_ptrs,
        (size_t)driver_packet.numberofreceivesamples(), md.get_md());

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

      accumulated_received_samples += num_rx_samps;
      std::cout << "Accumulated received samples " << accumulated_received_samples <<std::endl;
    }

    std::cout << "RECEIVE received samples per channel " << accumulated_received_samples << std::endl;

    std::chrono::steady_clock::time_point recv_end = std::chrono::steady_clock::now();
    std::cout << "RECEIVE receive timing: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (recv_end - recv_begin).count()
      << "us" << std::endl;


    std::chrono::steady_clock::time_point send_begin = std::chrono::steady_clock::now();

    rxsamplesmetadata::RxSamplesMetadata samples_metadata;
    samples_metadata.set_numberofreceivesamples(driver_packet.numberofreceivesamples());
    samples_metadata.set_shrmemname(shr_mem_name);
    samples_metadata.set_sequence_num(driver_packet.sequence_num());
    std::string samples_metadata_str;
    samples_metadata.SerializeToString(&samples_metadata_str);

    zmq::message_t samples_metadata_size_message(samples_metadata_str.size());
    memcpy ((void *) samples_metadata_size_message.data (), samples_metadata_str.c_str(),
            samples_metadata_str.size());

    data_socket.send(samples_metadata_size_message);
    //data_socket.send(cp_data_message);

    std::chrono::steady_clock::time_point send_end = std::chrono::steady_clock::now();
    std::cout << "RECEIVE package and send timing: "
          << std::chrono::duration_cast<std::chrono::microseconds>(send_end - send_begin).count()
          << "us" << std::endl;
  }
}

void control(zmq::context_t &driver_c) {
  std::cout << "Enter control thread\n";

  std::cout << "Creating and connecting to thread socket in control\n";
/*    zmq::socket_t packet_socket(*thread_c, ZMQ_PAIR); // 1
  packet_socket.connect("inproc://threads"); // 2  */


  zmq::socket_t packet_socket(driver_c, ZMQ_PUB);
  packet_socket.bind("inproc://threads");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)

  std::cout << "Creating and binding control socket\n";
  //zmq::context_t context(2);
  zmq::socket_t radarctrl_socket(driver_c, ZMQ_PAIR);// REVIEW #29 Should this be in a config file? Sort of a magic string
  radarctrl_socket.bind("ipc:///tmp/feeds/0");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)

  zmq::message_t request;

  zmq::socket_t data_socket(driver_c, ZMQ_PAIR);
  data_socket.bind("inproc://data");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)

  zmq::socket_t ack_socket(driver_c, ZMQ_PAIR);
  ack_socket.bind("inproc://ack");// REVIEW #37 check return value (0 if success, -1 if fail with errno set)

  zmq::socket_t computation_socket(driver_c, ZMQ_PAIR);// REVIEW #29 Should this be in a config file? Sort of a magic string
  computation_socket.connect("ipc:///tmp/feeds/1");



  //Allows for set up of SUB/PUB socket
  sleep(1);

  while (1) {
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    radarctrl_socket.recv(&request);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    begin = std::chrono::steady_clock::now();
    driverpacket::DriverPacket driver_packet;
    std::string msg_str(static_cast<char*>(request.data()), request.size());
    driver_packet.ParseFromString(msg_str);

    std::cout << "Control " << driver_packet.sob() << " " << driver_packet.eob()
      << " " << driver_packet.channels_size() << std::endl;

    if ((driver_packet.sob() == true) && (driver_packet.eob() == true)) {
      // TODO(keith): throw error
    }

    if (driver_packet.samples_size() != driver_packet.channels_size()) {
      // TODO(keith): throw error
    }

    for (int i = 0; i < driver_packet.samples_size(); i++) {
      if (driver_packet.samples(i).real_size() != driver_packet.samples(i).imag_size()) {
        // TODO(keith): throw error
      }
    }

    packet_socket.send(request);

    end = std::chrono::steady_clock::now();
    std::cout << "Time difference to deserialize and send in control = "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (end - begin).count()
      <<std::endl;

    if (driver_packet.eob() == true) {
      zmq::message_t ack, data, size;

      data_socket.recv(&size);
      computation_socket.send(size);

      ack_socket.recv(&ack);
      radarctrl_socket.send(ack);

    }


  }
}



// REVIEW #1 documentation
int UHD_SAFE_MAIN(int argc, char *argv[]) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  DriverOptions driver_options;

  std::cout << driver_options.get_device_args() << std::endl;
  std::cout << driver_options.get_tx_rate() << std::endl;
  std::cout << driver_options.get_pps() << std::endl;
  std::cout << driver_options.get_ref() << std::endl;
  std::cout << driver_options.get_tx_subdev() << std::endl;

  USRP usrp_d(driver_options);


  //  Prepare our context
  zmq::context_t driver_context(1);
/*  zmq::context_t timing_context(3);*/ // REVIEW #33 use git, don't need the commented out code

  std::vector<std::thread> threads;
// REVIEW #1 All threads work on same objects? Is that the reason for std::ref?
  std::thread control_t(control, std::ref(driver_context));
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
