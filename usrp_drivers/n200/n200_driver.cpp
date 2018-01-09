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

#ifdef DEBUG
#define TIMEIT_IF_DEBUG(msg,x) do { auto time_start = std::chrono::steady_clock::now();            \
                                    x;                                                             \
                                    auto time_end = std::chrono::steady_clock::now();              \
                                    auto time_diff = std::chrono::duration_cast                    \
                                                                        <std::chrono::microseconds>\
                                                    (time_end - time_start).count();               \
                                    DEBUG_MSG(msg << time_diff << "us"); } while (0)
#else
#define TIMEIT_IF_DEBUG(msg,x) do {x;} while(0)
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
  std::vector<std::vector<std::complex<float>>> samples(driver_packet.channel_samples_size());
  for (int channel=0; channel<driver_packet.channel_samples_size(); channel++) {
    auto num_samps = driver_packet.channel_samples(channel).real_size();
    std::vector<std::complex<float>> v(num_samps);
    samples[channel] = v;

    for (int smp_num = 0; smp_num < num_samps; smp_num++) {
      auto smp = driver_packet.channel_samples(channel);
      samples[channel][smp_num] = std::complex<float>(smp.real(smp_num),smp.imag(smp_num));
    }
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
//TODO(keith): If there are large periods of time between pulses, this while loop might be too speedy,
//resulting in overflows to the usrp - may have seen this in testing? - can we calculate an amount of
// time to sleep if that's the case? Discuss


  /*This loop accepts pulse by pulse from the radar_control. It parses the samples, configures the
   *USRP, sets up the timing, and then sends samples/timing to the USRPs.
   */
  while (1)
  {
    zmq::message_t request;
    ERR_CHK_ZMQ(driver_packet_pub_socket.recv(&request)) //TODO(keith): change to poll
    DEBUG_MSG("Received in TRANSMIT");
    driverpacket::DriverPacket driver_packet;


    //Here we accept our driver_packet from the radar_control. We use that info in order to
    //configure the USRP devices based on experiment requirements.
    TIMEIT_IF_DEBUG("TRANSMIT total setup time: ",
      [&]() {

        std::string packet_msg_str(static_cast<char*>(request.data()), request.size());
        if (driver_packet.ParseFromString(packet_msg_str) == false)
        {
          //TODO(keith): handle error
        }

        sqn_num = driver_packet.sequence_num();
        if (sqn_num != expected_sqn_num){
          DEBUG_MSG("SEQUENCE NUMBER MISMATCH: SQN " << sqn_num << " EXPECTED: "
            << expected_sqn_num);
          //TODO(keith): handle error
        }

        DEBUG_MSG("TRANSMIT burst flags: SOB "  << driver_packet.sob() << " EOB "
          << driver_packet.eob());

        TIMEIT_IF_DEBUG("TRANSMIT stream set up time: ",
          [&]() {
            //On start of new sequence, check if there are new USRP channels and if so
            //set what USRP TX channels and rate(Hz) to use.
            if (driver_packet.channels_size() > 0 && driver_packet.sob() == true)
            {
              DEBUG_MSG("TRANSMIT starting something new");
              channels = make_tx_channels(driver_packet);
              stream_args.channels = channels;
              auto actual_tx_rate = usrp_d.set_tx_rate(driver_packet.txrate(),channels); // TODO(keith): Test that USRPs exist to match channels in config.
              tx_stream = usrp_d.get_usrp_tx_stream(stream_args);  // ~44ms TODO(keith): See what 0s look like on scope.
              usrp_channels_set = true;
            }
          }()
        );

        TIMEIT_IF_DEBUG("TRANSMIT center freq ",
          [&]() {
            //If there is new center frequency data, set TX center frequency for each USRP TX channel.
            if (driver_packet.txcenterfreq() > 0.0)
            {
              DEBUG_MSG("TRANSMIT center freq " << driver_packet.txcenterfreq());

              center_freq_set = true;
            }
          }()
        );

        TIMEIT_IF_DEBUG("TRANSMIT sample unpack time: ",
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
    if ((usrp_channels_set == false) || (center_freq_set == false) ||(samples_set == false))
    {
      // TODO(keith): throw error
      continue;
    }






    //High speed IO and pulse times are calculated. These are the timings generated for the GPIO
    //pins that connect to the protection circuits. When these pins go high is relative to the times
    //the pulses go out.
    if (driver_packet.sob() == true)
    {
      //The USRP needs about a SET_TIME_COMMAND_DELAY buffer into the future before time
      //commands will correctly work. This was found through testing and may be subject to change
      //with USRP firmware updates.
      time_zero = usrp_d.get_current_usrp_time() + uhd::time_spec_t(SET_TIME_COMMAND_DELAY);
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

    DEBUG_MSG("TRANSMIT time_zero: " << time_zero.get_frac_secs());
    DEBUG_MSG("TRANSMIT time_now " << usrp_d.get_current_usrp_time().get_frac_secs());
    DEBUG_MSG("TRANSMIT timetosendsamples(us) " << driver_packet.timetosendsamples());
    DEBUG_MSG("TRANSMIT time_to_send_pulse " << time_to_send_pulse.get_frac_secs());
    DEBUG_MSG("TRANSMIT atten_time_high " << atten_time_high.get_frac_secs());
    DEBUG_MSG("TRANSMIT tr_time_high " << tr_time_high.get_frac_secs());
    DEBUG_MSG("TRANSMIT pulse_start_time " << pulse_start_time.get_frac_secs());
    DEBUG_MSG("TRANSMIT tr_time_low " << tr_time_low.get_frac_secs());
    DEBUG_MSG("TRANSMIT atten_time_low " << atten_time_low.get_frac_secs());
    DEBUG_MSG("TRANSMIT scope_sync_low " << scope_sync_low.get_frac_secs());






    //We send the start time and samples to the USRP. This is done before IO signals are set
    //since those commands create a blocking buffer which causes the transfer of samples to be
    //late. This leads to no waveform output on the USRP.
    TIMEIT_IF_DEBUG("TRANSMIT time to send samples to USRP: ",
      [&]() {

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
          num_samps_sent = tx_stream->send(samples, num_samps_to_send, md.get_md()); //TODO(keith): Determine timeout properties.
          DEBUG_MSG("TRANSMIT Samples sent " << num_samps_sent);

          md.set_start_of_burst(false);
          md.set_has_time_spec(false);
        }

        md.set_end_of_burst(true);
        tx_stream->send("", 0, md.get_md()); //TODO(keith): test sob/eob in same call. REVIEW #43 If we know that we are under the max num samps value, then we could send start of burst and end of burst in the same send(...) call
      }()
    );






    //Configure high speed IO on USRPs for pulses.
    TIMEIT_IF_DEBUG("TRANSMIT Time to set up timing signals: ",
      [&]() {
        //Set high speed IO timing on the USRP now.
        usrp_d.clear_command_times();
        if (driver_packet.sob() == true)
        {
          auto sync_time = driver_packet.numberofreceivesamples()/driver_options.get_rx_rate();
          DEBUG_MSG("SYNC TIME " << sync_time);

          scope_sync_low = scope_sync_high + uhd::time_spec_t(sync_time);

          DEBUG_MSG("TRANSMIT Scope sync high set");
          usrp_d.set_scope_sync(scope_sync_high);

        }

        usrp_d.set_atten(atten_time_high);
        usrp_d.set_tr(x_high);
        usrp_d.clear_tr(x_low);
        usrp_d.clear_atten(atten_time_low);

        if (driver_packet.eob() == true)
        {
          usrp_d.clear_scope_sync(scope_sync_low);
        }
      }()
    );






    //Final end of sequence work to acknowledge seq num.
    if (driver_packet.eob() == true) {
      driverpacket::DriverPacket ack;
      DEBUG_MSG("SEQUENCENUM " << sqn_num);
      ack.set_sequence_num(sqn_num);
      expected_sqn_num += 1;

      std::string ack_str;
      ack.SerializeToString(&ack_str);

      zmq::message_t ack_msg(ack_str.size());
      memcpy((void*)ack_msg.data(), ack_str.c_str(),ack_str.size());
      ack_socket.send(ack_msg); // TODO(keith): Potentially add other return statuses to ack.
    }


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
void receive(zmq::context_t &driver_c, USRP &usrp_d, const DriverOptions &driver_options) {
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

  auto receive_channels = driver_options.get_receive_channels();
  stream_args.channels = receive_channels;
  usrp_d.set_rx_rate(rx_rate_hz,receive_channels);  // ~450us
  uhd::rx_streamer::sptr rx_stream = usrp_d.get_usrp_rx_stream(stream_args);  // ~44ms

  //This loop receives 1 pulse sequence worth of samples.
  while (1) {
    driver_packet_pub_socket.recv(&request);
    DEBUG_MSG( "RECEIVE recv new request");
    driverpacket::DriverPacket driver_packet;
    std::string packet_msg_str(static_cast<char*>(request.data()), request.size());
    driver_packet.ParseFromString(packet_msg_str);

    DEBUG_MSG("RECEIVE burst flags SOB " << driver_packet.sob() << " EOB " << driver_packet.eob());

    //We only begin receiving if its the start of a pulse sequence. The rest of the pulses can be
    //ignored.
    if ( driver_packet.sob() == false ) continue;


    TIMEIT_IF_DEBUG("RECEIVE center frq tuning time: ",
      [&]() {
        if (driver_packet.rxcenterfreq() > 0) {
          auto set_freq = usrp_d.set_rx_center_freq(driver_packet.rxcenterfreq(), receive_channels);
          DEBUG_MSG("RECEIVE center freq " << set_freq);
          center_freq_set  = true;
        }
      }()
    );

    if (center_freq_set == false) {
      // TODO(keith): throw error
    }

    std::vector<std::complex<float> *> buffer_ptrs;
    size_t mem_size;
    std::string shr_mem_name;
    TIMEIT_IF_DEBUG("RECEIVE shared memory unpack timing: ",
      [&]() {
        mem_size = receive_channels.size() * driver_packet.numberofreceivesamples()
                            * sizeof(std::complex<float>);
        shr_mem_name = random_string(25);
        //Use a random string to make a unique set of named shared memory
        SharedMemoryHandler shrmem(shr_mem_name);
        shrmem.create_shr_mem(mem_size);

        //create a vector of pointers to where each channel's data gets received.
        for(uint32_t i=0; i<receive_channels.size(); i++){
          auto ptr = static_cast<std::complex<float>*>(shrmem.get_shrmem_addr()) +
                                      i*driver_packet.numberofreceivesamples();
          buffer_ptrs.push_back(ptr);
        }
      }()
    );

    DEBUG_MSG("Got to RECEIVE timing");
    receive_side_timing_socket.recv(&start_receive_timing);
    DEBUG_MSG("RECEIVED timing data");
    auto time_zero = *(reinterpret_cast<uhd::time_spec_t*>(start_receive_timing.data())); //s

    DEBUG_MSG("RECEIVE time_zero " << time_zero.get_frac_secs() << " s");

    TIMEIT_IF_DEBUG("RECEIVE time to recv from USRP: ",
      [&]() {
        //Documentation is unclear, but num samps is per channel
        stream_cmd.num_samps = size_t(driver_packet.numberofreceivesamples());
        stream_cmd.stream_now = false;
        stream_cmd.time_spec = time_zero; //TODO(keith): test late time zero, perhaps check against current time(either here or at creation)
        rx_stream->issue_stream_cmd(stream_cmd);

        auto md = RXMetadata();

        DEBUG_MSG("RECEIVE total samples to receive: "
                  << receive_channels.size() * driver_packet.numberofreceivesamples()
                  << " mem size " << mem_size);

        size_t accumulated_received_samples = 0;
        while (accumulated_received_samples < driver_packet.numberofreceivesamples()) {
          size_t num_rx_samps = rx_stream->recv(buffer_ptrs,
            (size_t)driver_packet.numberofreceivesamples(), md.get_md());

          auto error_code = md.get_error_code();
          switch(error_code) {
            case uhd::rx_metadata_t::ERROR_CODE_NONE :
              break;
            case uhd::rx_metadata_t::ERROR_CODE_TIMEOUT :
              DEBUG_MSG("Timeout while streaming");
              //TODO(keith): handle timeout situation.
              break;
              // TODO(keith): throw error
            case uhd::rx_metadata_t::ERROR_CODE_OVERFLOW :
              break;
              //TODO(keith): throw error
            default :
              break;
              //TODO(keith): throw error
          }
          accumulated_received_samples += num_rx_samps;
          DEBUG_MSG("Accumulated received samples " << accumulated_received_samples);
        }
        DEBUG_MSG("RECEIVE received samples per channel " << accumulated_received_samples);
      }()
    );



    TIMEIT_IF_DEBUG("RECEIVE package samples and send timing: ",
      [&]() {
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
      }();
    );

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
  zmq::socket_t radarctrl_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(radarctrl_socket.bind(driver_options.get_radar_control_to_driver_address()))
  zmq::message_t request;

  zmq::socket_t data_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(data_socket.bind("inproc://data"))

  zmq::socket_t ack_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(ack_socket.bind("inproc://ack"))

  zmq::socket_t rx_dsp_socket(driver_c, ZMQ_PAIR);
  ERR_CHK_ZMQ(rx_dsp_socket.connect(driver_options.get_driver_to_rx_dsp_address()))



  //Sleep to handle "slow joiner" problem
  //http://zguide.zeromq.org/php:all#Getting-the-Message-Out
  sleep(1);

  while (1) {
    radarctrl_socket.recv(&request);

    driverpacket::DriverPacket driver_packet;
    TIMEIT_IF_DEBUG("CONTROL Time difference to deserialize and forward = ",
      [&]() {
        std::string request_str(static_cast<char*>(request.data()), request.size());
        driver_packet.ParseFromString(request_str);

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
        driver_packet_pub_socket.send(request);
      }()
    );

    if (driver_packet.eob() == true) {
      //TODO(keith): handle potential errors.
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
