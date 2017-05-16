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

#define ERR_CHK_ZMQ(x) try {x;} catch (zmq::error_t& e) {} //TODO(keith): handle error

//Delay needed for before any set_time_commands will work.
#define SET_TIME_COMMAND_DELAY 10.0e-3 // seconds
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
  std::vector<std::vector<std::complex<float>>> samples(driver_packet.samples_size());
  for (int i=0; i<driver_packet.samples_size(); i++) {
    auto num_samps = driver_packet.samples(i).real_size();
    std::vector<std::complex<float>> v(num_samps);
    samples[i] = v;

    for (int j = 0; j < driver_packet.samples(i).real_size(); j++) {
      samples[i][j] = std::complex<float>(driver_packet.samples(i).real(j),
                                           driver_packet.samples(i).imag(j));
    }
  }

  for (auto &s : samples)
  {
    if (s.size() != samples[0].size())
    {
      //TODO(keith): Handle this error.   REPLY Would this be a critical fail?
    }
  }

  return samples;
}


void transmit(zmq::context_t &driver_c, USRP &usrp_d,
                const DriverOptions &driver_options)
{
  DEBUG_MSG("Enter transmit thread");


  zmq::socket_t driver_packet_pub_socket(driver_c, ZMQ_SUB);
  ERR_CHK_ZMQ(driver_packet_pub_socket.connect("inproc://threads"))
  ERR_CHK_ZMQ(driver_packet_pub_socket.setsockopt(ZMQ_SUBSCRIBE, "", 0))
  DEBUG_MSG("TRANSMIT: Creating and connected to thread socket in control");

  zmq::socket_t receive_side_timing_socket(driver_c, ZMQ_PAIR);
  receive_side_timing_socket.connect("inproc://timing");

  zmq::socket_t ack_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(ack_socket.connect("inproc://ack"))


  auto usrp_channels_set = false;
  auto center_freq_set = false;
  auto samples_set = false;

  std::vector<size_t> channels;

  uhd::time_spec_t time_zero; //s
  zmq::message_t start_receive_timing (sizeof(time_zero));

  uhd::tx_streamer::sptr tx_stream;
  uhd::stream_args_t stream_args("fc32", "sc16");

  size_t samples_per_buff;
  std::vector<std::vector<std::complex<float>>> samples;

  auto atten_window_time_start_s = driver_options.get_atten_window_time_start(); //seconds
  auto atten_window_time_end_s = driver_options.get_atten_window_time_end(); //seconds
  auto tr_window_time_s = driver_options.get_tr_window_time(); //seconds

  //default initialize SS. Needs to be outside of loop
  auto scope_sync_low = uhd::time_spec_t(0.0);

  uint32_t sqn_num = 0;
  uint32_t expected_sqn_num = 0;
// REVIEW #0 If there are large periods of time between pulses, this while loop might be too speedy, resulting in overflows to the usrp - may have seen this in testing? - can we calculate an amount of time to sleep if that's the case?
// REPLY need to discuss this.

  /*This loop accepts pulse by pulse from the radar_control. It parses the samples, configures the
   *USRP, sets up the timing, and then sends samples/timing to the USRPs.
   */
  while (1) // REVIEW #35 Over 220 lines. are there any candidates for functions? (timing and cout code?)
  {         // REPLY can discuss
    zmq::message_t request;
    driver_packet_pub_socket.recv(&request);
    auto setup_begin = std::chrono::steady_clock::now();
    DEBUG_MSG("Received in TRANSMIT");
    driverpacket::DriverPacket driver_packet; // REVIEW #32 can move this and declaration of the packet_msg_str out of the while loop
                                              // REPLY http://stackoverflow.com/questions/7959573/declaring-variables-inside-loops-good-practice-or-bad-practice
    std::string packet_msg_str(static_cast<char*>(request.data()), request.size());
    if (driver_packet.ParseFromString(packet_msg_str) == false)
    {
      //TODO(keith): handle error
    }

    sqn_num = driver_packet.sequence_num(); // REVIEW #0 default value for numeric types in protobuf is =0 this could create errors, should start at 1?
                                            // I don't see this being an issue since the only time this is valid is the first pulse seq. If its not incremented on next then the error is caught
    if (sqn_num != expected_sqn_num){
      DEBUG_MSG("SEQUENCE NUMBER MISMATCH: SQN " << sqn_num << " EXPECTED: " << expected_sqn_num);
      //TODO(keith): handle error
    }

    DEBUG_MSG("TRANSMIT burst flags: SOB "  << driver_packet.sob() << " EOB "
      << driver_packet.eob());

    auto stream_setup_begin = std::chrono::steady_clock::now();

    //On start of new sequence, check if there are new USRP channels and if so
    //set what USRP TX channels and rate(Hz) to use.
    if (driver_packet.channels_size() > 0 && driver_packet.sob() == true)
    {
      DEBUG_MSG("TRANSMIT starting something new");
      channels = make_tx_channels(driver_packet);
      stream_args.channels = channels; // REVIEW #15 check that the channels from the driver packet are actually available (a mapping as mentioned in other comment would be handy here)
                                       // REPLY Need to explain this
      usrp_d.set_tx_rate(driver_packet.txrate());  // ~450us REVIEW #15 check that it's not = 0 (default)
                                                   // REPLY added this check to set_tx_rate
      tx_stream = usrp_d.get_usrp()->get_tx_stream(stream_args);  // ~44ms REVIEW : idea maybe could we set this up with all channels and then send 0s to any that we don't want to transmit on? then this tx_stream setup could be removed from while loop.
      usrp_channels_set = true;                                   // REPLY Yes. We will have to see what that looks like on the scope.
    }
    auto stream_setup_end = std::chrono::steady_clock::now();

    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                  (stream_setup_end - stream_setup_begin).count();
    DEBUG_MSG("TRANSMIT stream set up time: " << time_diff << "us");


    auto ctr_freq_setup_begin = std::chrono::steady_clock::now();

    //If there is new center frequency data, set TX center frequency for each USRP TX channel.
    if (driver_packet.txcenterfreq() > 0.0)
    {
      DEBUG_MSG("TRANSMIT center freq " << driver_packet.txcenterfreq()); // REVIEW #34 should print the actual freq after it's been set (return actual frequency from set_tx_center_freq)
                                                                                         // REPLY Need to discuss this
      usrp_d.set_tx_center_freq(driver_packet.txcenterfreq(), channels);
      center_freq_set = true;
    }

    auto ctr_freq_setup_end = std::chrono::steady_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                (ctr_freq_setup_end - ctr_freq_setup_begin).count();
    DEBUG_MSG("TRANSMIT center frq tuning time: " << time_diff << "us");


    auto sample_unpack_begin = std::chrono::steady_clock::now();

    //Parse new samples from driver packet if they exist.
    if (driver_packet.samples_size() > 0)
    {  // ~700us to unpack 4x1600 samples
      samples = make_tx_samples(driver_packet);
      samples_per_buff = samples[0].size();
      samples_set = true;
    }

    auto sample_unpack_end = std::chrono::steady_clock::now();

    time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                  (sample_unpack_end - sample_unpack_begin).count();
    DEBUG_MSG("TRANSMIT sample unpack time: " << time_diff << "us");

    //In order to transmit, these parameters need to be set at least once.
    if ((usrp_channels_set == false) || (center_freq_set == false) ||(samples_set == false))
    {
      // TODO(keith): throw error
      continue;
    }


    auto setup_end = std::chrono::steady_clock::now();

    time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                      (setup_end - setup_begin).count();
    DEBUG_MSG("TRANSMIT Total set up time: " << time_diff << "us");



    if (driver_packet.sob() == true)
    {
      //The USRP needs about a SET_TIME_COMMAND_DELAY buffer into the future before time
      //commands will correctly work. This was found through testing and may be subject to change
      //with USRP firmware updates.
      time_zero = usrp_d.get_usrp()->get_time_now() + uhd::time_spec_t(SET_TIME_COMMAND_DELAY);
    }

    //convert us to s
    auto time_to_send_pulse = uhd::time_spec_t(driver_packet.timetosendsamples()/1e6);

    auto pulse_start_time = time_zero + time_to_send_pulse; //s
    auto pulse_len_time = uhd::time_spec_t(samples_per_buff/usrp_d.get_tx_rate()); //s
    auto tr_time_high = pulse_start_time - uhd::time_spec_t(tr_window_time_s); //s
    auto atten_time_high = tr_time_high - uhd::time_spec_t(atten_window_time_start_s); //s
    auto scope_sync_high = atten_time_high; //s
    auto tr_time_low = pulse_start_time + pulse_len_time + uhd::time_spec_t(tr_window_time_s); //s
    auto atten_time_low = tr_time_low + uhd::time_spec_t(atten_window_time_end_s); //s

    //This line lets us trigger TR with atten timing, but
    //we can still keep the existing logic if we want to use it.
    auto x_high = atten_time_high; //s
    auto x_low = tr_time_low; //s

    //To make sure tx and rx timing are synced, this thread sends when to receive to the receive
    //thread.
    if( driver_packet.sob() == true)
    {
      memcpy(start_receive_timing.data(), &scope_sync_high, sizeof(scope_sync_high));
      receive_side_timing_socket.send(start_receive_timing);
    }

    auto begin_send = std::chrono::steady_clock::now();

    auto time_now = usrp_d.get_usrp()->get_time_now();
    DEBUG_MSG("TRANSMIT time_zero: " << time_zero.get_frac_secs());
    DEBUG_MSG("TRANSMIT time_now " << time_now.get_frac_secs());
    DEBUG_MSG("TRANSMIT timetosendsamples(us) " << driver_packet.timetosendsamples());
    DEBUG_MSG("TRANSMIT time_to_send_pulse " << time_to_send_pulse.get_frac_secs());
    DEBUG_MSG("TRANSMIT atten_time_high " << atten_time_high.get_frac_secs());
    DEBUG_MSG("TRANSMIT tr_time_high " << tr_time_high.get_frac_secs());
    DEBUG_MSG("TRANSMIT pulse_start_time " << pulse_start_time.get_frac_secs());
    DEBUG_MSG("TRANSMIT tr_time_low " << tr_time_low.get_frac_secs());
    DEBUG_MSG("TRANSMIT atten_time_low " << atten_time_low.get_frac_secs());
    DEBUG_MSG("TRANSMIT scope_sync_low " << scope_sync_low.get_frac_secs());

    auto md = TXMetadata();
    md.set_has_time_spec(true);
    md.set_time_spec(pulse_start_time);
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
      DEBUG_MSG("TRANSMIT Samples to send " << num_samps_to_send);

      //Send behaviour can be found in UHD docs
      num_samps_sent = tx_stream->send( // REVIEW #37 How do we know if it's timed out? Does it do the fragmentation of packets internal to the send(...) if above the max num samps value?
        samples, num_samps_to_send, md.get_md()); // REVIEW #0 Should we set the timeout to a reasonable value given our timing constraints? 100ms would be way too long to recover this pulse sequence. Maybe we can make a calculation based on how long this pulse is and set the timeout slightly beyond that - or beyond atten_time_low (at which point you should be in this loop for the next pulse) We can also get_max_num_samps to find out if this call will fragment into multiple packets, not sure of the behaviour there, does it return before the timeout??
                                                  // REPLY lets talk about this.
      DEBUG_MSG("TRANSMIT Samples sent " << num_samps_sent);

      md.set_start_of_burst(false);
      md.set_has_time_spec(false);
    }

    md.set_end_of_burst(true);
    tx_stream->send("", 0, md.get_md()); // REVIEW #43 If we know that we are under the max num samps value, then we could send start of burst and end of burst in the same send(...) call
                                         // REPLY would have to test that, but im not sure what that would gain. This is how its done in UHD examples.
    auto end_send = std::chrono::steady_clock::now();

    time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                          (end_send - begin_send).count();
    DEBUG_MSG("TRANSMIT time to send to USRP: " << time_diff << "us");

    auto timing_start = std::chrono::steady_clock::now();

    //Set high speed IO timing on the USRP now.
    usrp_d.get_usrp()->clear_command_time(); // REVIEW #0 The GPIO timing setup should be moved to above the samples sending, in case we timeout on sending samples (i.e. we sent half of them) then the RF waveform would be transmitting without timing signals, bad idea. Alternatively, what if there was a 4th thread for timing specifically?
                                             // REPLY I tried this. It doesnt work. If this is a problem then we need a circuit for this.
    if (driver_packet.sob() == true)
    {
      auto sync_time = driver_packet.numberofreceivesamples()/driver_options.get_rx_rate();
      DEBUG_MSG("SYNC TIME " << sync_time);

      scope_sync_low = scope_sync_high + uhd::time_spec_t(sync_time);

      DEBUG_MSG("TRANSMIT Scope sync high set");
      usrp_d.get_usrp()->set_command_time(scope_sync_high);
      usrp_d.set_scope_sync();

    }

    usrp_d.get_usrp()->set_command_time(atten_time_high);
    usrp_d.set_atten();

    usrp_d.get_usrp()->set_command_time(x_high);
    usrp_d.set_tr();

    usrp_d.get_usrp()->set_command_time(x_low);
    usrp_d.clear_tr();

    usrp_d.get_usrp()->set_command_time(atten_time_low);
    usrp_d.clear_atten();

    if (driver_packet.eob() == true)
    {
      usrp_d.get_usrp()->set_command_time(scope_sync_low);
      usrp_d.clear_scope_sync();
    }


    auto timing_end = std::chrono::steady_clock::now();

    time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                  (timing_end - timing_start).count();
    DEBUG_MSG("TRANSMIT Time to set up timing signals: " << time_diff << "us");

    //Final end of sequence work to acknowledge seq num.
    if (driver_packet.eob() == true) {
      driverpacket::DriverPacket ack; // REVIEW #33 Can this just be set up once at the start, otherwise duplicating this code. same with ack_str and ack_msg
                                      // REPLY same as above
      DEBUG_MSG("SEQUENCENUM " << sqn_num);
      ack.set_sequence_num(sqn_num);
      expected_sqn_num += 1;

      std::string ack_str;
      ack.SerializeToString(&ack_str);

      zmq::message_t ack_msg(ack_str.size());
      memcpy((void*)ack_msg.data(), ack_str.c_str(),ack_str.size());
      ack_socket.send(ack_msg); // REVIEW #43 let's make the ack have more information than just 'I received the packet' so we can make intelligent decisions elsewhere in the code if something unexpected/error happens
    }                           // REPLY need to discuss this.


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

/**
 * @brief      Runs in a seperate thread to control receiving from the USRPs.
 *
 * @param[in]  driver_c        The driver ZMQ context.
 * @param[in]  usrp_d          The multi-USRP SuperDARN wrapper object.
 * @param[in]  driver_options  The driver options parsed from config.
 */
void receive(zmq::context_t &driver_c, USRP &usrp_d,
              const DriverOptions &driver_options) {
  DEBUG_MSG("Enter receive thread");

  zmq::socket_t driver_packet_pub_socket(driver_c, ZMQ_SUB);
  ERR_CHK_ZMQ(driver_packet_pub_socket.connect("inproc://threads"))
  ERR_CHK_ZMQ(driver_packet_pub_socket.setsockopt(ZMQ_SUBSCRIBE, "", 0))
  zmq::message_t request;
  DEBUG_MSG("RECEIVE: Creating and connected to thread socket in control");

  zmq::socket_t receive_side_timing_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(receive_side_timing_socket.bind("inproc://timing"))
  zmq::message_t start_receive_timing;

  zmq::socket_t data_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(data_socket.connect("inproc://data"))

  auto center_freq_set = false;

  uhd::stream_args_t stream_args("fc32", "sc16");
  uhd::stream_cmd_t stream_cmd(
      uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
  auto rx_rate_hz = driver_options.get_rx_rate();

  auto receive_channels = usrp_d.get_receive_channels();
  stream_args.channels = receive_channels;
  usrp_d.set_rx_rate(rx_rate_hz);  // ~450us
  uhd::rx_streamer::sptr rx_stream = usrp_d.get_usrp()->get_rx_stream(stream_args);  // ~44ms

  //This loop receives 1 pulse sequence worth of samples.
  while (1) {
    driver_packet_pub_socket.recv(&request);
    DEBUG_MSG( "Received in RECEIVE");
    driverpacket::DriverPacket driver_packet;
    std::string packet_msg_str(static_cast<char*>(request.data()), request.size());
    driver_packet.ParseFromString(packet_msg_str);

    DEBUG_MSG("RECEIVE burst flags SOB " << driver_packet.sob() << " EOB " << driver_packet.eob());

    if ( driver_packet.sob() == false ) continue;

    auto set_ctr_freq_begin = std::chrono::steady_clock::now();

    if (driver_packet.rxcenterfreq() > 0) {
      DEBUG_MSG("RECEIVE center freq " << driver_packet.rxcenterfreq()); // REVIEW #34 move this print statement below and use actual set rxcenterfreq
      usrp_d.set_rx_center_freq(driver_packet.rxcenterfreq(), receive_channels);        // REPLY will discuss this.
      center_freq_set = true;
    }

    auto set_ctr_freq_end = std::chrono::steady_clock::now();

    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                  (set_ctr_freq_end - set_ctr_freq_begin).count();
    DEBUG_MSG("RECEIVE center frq tuning time: " << time_diff << "us");


    if (center_freq_set == false) {
      // TODO(keith): throw error // REVIEW #15 possible idea for recovering from these types of errors: to have a set of defaults that could be used to recover with
      // REPLY I think this is a risky/bad idea. This is part of reason that multi frequency epop sound ran incorrectly for months. I would rather hard fail in some places with a descriptive error
    }

    // REVIEW #22 could do this only when mem_size needs to increase ?
    // REPLY which part do you mean by 'this'
    auto create_shr_begin = std::chrono::steady_clock::now();
    size_t mem_size = receive_channels.size() * driver_packet.numberofreceivesamples()
                        * sizeof(std::complex<float>);
    auto shr_mem_name = random_string(25);
    //Use a random string to make a unique set of named shared memory
    SharedMemoryHandler shrmem(shr_mem_name);
    shrmem.create_shr_mem(mem_size); // REVIEW #41 would it be worthwhile to put this into the constructor?
                                     // REPLY No cause this handler can create or open. I would rather choose the method

    //create a vector of pointers to where each channel's data gets received.
    std::vector<std::complex<float> *> buffer_ptrs;
    for(uint32_t i=0; i<receive_channels.size(); i++){
      auto ptr = static_cast<std::complex<float>*>(shrmem.get_shrmem_addr()) +
                                  i*driver_packet.numberofreceivesamples(); // REVIEW #0 does this work auto = complex float * + int
      buffer_ptrs.push_back(ptr);                                           // REPLY This is the definition of how pointer arithmetic works :p
    }

    auto create_shr_end = std::chrono::steady_clock::now();

    time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                  (create_shr_end - create_shr_begin).count();
    DEBUG_MSG("RECEIVE shr timing: "<< time_diff << "us");

    DEBUG_MSG("Got to RECEIVE timing");
    receive_side_timing_socket.recv(&start_receive_timing);
    DEBUG_MSG("RECEIVED timing data");
    auto time_zero = *(reinterpret_cast<uhd::time_spec_t*>(start_receive_timing.data())); //s

    DEBUG_MSG("RECEIVE time_zero " << time_zero.get_frac_secs() << " s");

    auto recv_begin = std::chrono::steady_clock::now();

    //Documentation is unclear, but num samps is per channel
    stream_cmd.num_samps = size_t(driver_packet.numberofreceivesamples());
    stream_cmd.stream_now = false;
    stream_cmd.time_spec = time_zero; // REVIEW #15 If this is in the past does it just execute immediately? Should check if it's in the past and do something about it (recv samples would be offset from where we want them)? Docs say that in multi-device setup, should have a timespec in the future so that usrps are all aligned - so may need to change the time_zero to be in future and log the error. Maybe abandon the pulse sequence, in case you keep getting more behind. ERROR_CODE_LATE_COMMAND ? One thought is to send 'success' or 'fail' boolean or something along with the ACK to the radar_control so it can handle failed pulse sequences.
    rx_stream->issue_stream_cmd(stream_cmd); // REPLY We havent tested recv so we will have to see. Should talk about this.

    auto md = RXMetadata();

    DEBUG_MSG("RECEIVE total samples to receive: "
              << receive_channels.size() * driver_packet.numberofreceivesamples()
              << " mem size " << mem_size);

    size_t accumulated_received_samples = 0;
    while (accumulated_received_samples < driver_packet.numberofreceivesamples()) {
      size_t num_rx_samps = rx_stream->recv(buffer_ptrs,
        (size_t)driver_packet.numberofreceivesamples(), md.get_md());
// REVIEW #6 TODO: Calculate appropriate timeout for the recv command given number of receive samples and rx_rate
// REPLY what would an appropriate timeout?

      auto error_code = md.get_error_code();
      switch(error_code) {
        case uhd::rx_metadata_t::ERROR_CODE_NONE :
          break;
        case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT :
          DEBUG_MSG("Timeout while streaming");
          break;
          // TODO(keith): throw error
        case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW :
          break;
          //TODO(keith): throw error
        default :
          break;
          //TODO(keith): throw error
      } // REPLY I dont think there is a performance issue here, but i reworked this if you think it could look cleaner. We still need to compare against specific errors.

      //TODO(Keith): investigate status to see under what circumstances this could fail. Will need
      //additional info sent to radar control if true.

      // REVIEW #6 TODO: Handle fragmentation flag being set in metadata. If this happens that means the buffer was too small to hold the number of samples (shouldn't happen, so this would be an error)
      // REPLY We can talk about this. I don't think this is an error
      accumulated_received_samples += num_rx_samps;
      DEBUG_MSG("Accumulated received samples " << accumulated_received_samples);
    }

    DEBUG_MSG("RECEIVE received samples per channel " << accumulated_received_samples);

    auto recv_end = std::chrono::steady_clock::now();

    time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                  (recv_end - recv_begin).count();
    DEBUG_MSG("RECEIVE receive timing: " << time_diff << "us");


    auto send_begin = std::chrono::steady_clock::now();

    rxsamplesmetadata::RxSamplesMetadata samples_metadata;
    samples_metadata.set_numberofreceivesamples(driver_packet.numberofreceivesamples()); // REVIEW #0 Should this be accumulated_received_samples ?
    samples_metadata.set_shrmemname(shr_mem_name);                                       // REPLY The recv loop will not break until recving until that # of samples has been received. We want this exact number of samples to process.
    samples_metadata.set_sequence_num(driver_packet.sequence_num());
    std::string samples_metadata_str;
    samples_metadata.SerializeToString(&samples_metadata_str);

    zmq::message_t samples_metadata_size_message(samples_metadata_str.size());
    memcpy ((void *) samples_metadata_size_message.data (), samples_metadata_str.c_str(),
            samples_metadata_str.size());


    data_socket.send(samples_metadata_size_message);

    auto send_end = std::chrono::steady_clock::now();
    time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                  (send_end - send_begin).count();
    DEBUG_MSG("RECEIVE package and send timing: " << time_diff << "us");
  }
}

/**
 * @brief      Runs in a seperate thread to act as an interface for the ingress and egress data.
 *
 * @param[in]  driver_c        The driver ZMQ context.
 */
void control(zmq::context_t &driver_c) {
  DEBUG_MSG("Enter control thread");

  DEBUG_MSG("Creating and connecting to thread socket in control");

  zmq::socket_t driver_packet_pub_socket(driver_c, ZMQ_PUB);
  ERR_CHK_ZMQ(driver_packet_pub_socket.bind("inproc://threads"))

  DEBUG_MSG("Creating and binding control socket");
  zmq::socket_t radarctrl_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(radarctrl_socket.bind("ipc:///tmp/feeds/0"))
  zmq::message_t request;

  zmq::socket_t data_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(data_socket.bind("inproc://data"))

  zmq::socket_t ack_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(ack_socket.bind("inproc://ack"))

  zmq::socket_t rx_dsp_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(rx_dsp_socket.connect("ipc:///tmp/feeds/1"))



  //Sleep to handle "slow joiner" problem
  //http://zguide.zeromq.org/php:all#Getting-the-Message-Out
  sleep(1);

  while (1) {
    radarctrl_socket.recv(&request);

    auto begin_deserialize = std::chrono::steady_clock::now();
    driverpacket::DriverPacket driver_packet;
    std::string msg_str(static_cast<char*>(request.data()), request.size());
    driver_packet.ParseFromString(msg_str);

    DEBUG_MSG("Control " << driver_packet.sob() << " " << driver_packet.eob()
      << " " << driver_packet.channels_size());

    //TODO(keith): thinking about moving this err chking to transmit
    if (driver_packet.samples_size() != driver_packet.channels_size()) {
      // TODO(keith): throw error
    }

    for (int i = 0; i < driver_packet.samples_size(); i++) {
      if (driver_packet.samples(i).real_size() != driver_packet.samples(i).imag_size()) {
        // TODO(keith): throw error
      }
    }// REVIEW #6 TODO: Check that seq num is not 0 (assuming we change it to start at 1 -either way check that there is a seq num) TODO: Check that timetosendsamples is set - maybe check if in future? TODO: Check that channels, samples, ctrfreq, rxrate, txrate are set if sob==true
     // REPLY Pretty much all of this is getting checked elsewhere. How to check timetosendsamples?
    driver_packet_pub_socket.send(request);

    auto end_deserialize = std::chrono::steady_clock::now();

    auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>
                                                  (end_deserialize - begin_deserialize).count();
    DEBUG_MSG("Time difference to deserialize and send in control = " << time_diff << "us");

    if (driver_packet.eob() == true) {
      zmq::message_t ack, shr_mem_metadata;

      data_socket.recv(&shr_mem_metadata);
      //send data to dsp first so that processing can start before next sequence is aquired.
      rx_dsp_socket.send(shr_mem_metadata);

      ack_socket.recv(&ack);
      radarctrl_socket.send(ack);

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
  uhd::set_thread_priority_safe();

  DriverOptions driver_options;

  std::cout << driver_options.get_device_args() << std::endl;
  std::cout << driver_options.get_tx_rate() << std::endl;
  std::cout << driver_options.get_pps() << std::endl;
  std::cout << driver_options.get_ref() << std::endl;
  std::cout << driver_options.get_tx_subdev() << std::endl;

  USRP usrp_d(driver_options);


  //  Prepare our context
  zmq::context_t driver_context(1);

  std::vector<std::thread> threads;
  // REVIEW #1 All threads work on same objects? Is that the reason for std::ref?
  // http://stackoverflow.com/a/15530639/1793295
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
