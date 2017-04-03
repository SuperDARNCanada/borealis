#include <vector>
#include <string>
#include <zmq.hpp> // REVIEW #4 Need to explain what we use from this lib in our general documentation
#include <thread>
#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdint.h>
#include <signal.h>
#include <cstdlib>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cuda_profiler_api.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "utils/protobuf/rxsamplesmetadata.pb.h"
#include "utils/protobuf/sigprocpacket.pb.h"
#include "utils/driver_options/driveroptions.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"
#include "utils/shared_memory/shared_memory.hpp"

#include "dsp.hpp"
#include "filtering.hpp"



int main(int argc, char **argv){
  GOOGLE_PROTOBUF_VERIFY_VERSION; // Verifies that header and lib are same version.

  //TODO(keith): verify config options.
  auto driver_options = DriverOptions();
  auto sig_options = SignalProcessingOptions(); // #26 REVIEW Should the naming be updated along with DSP ?
  auto rx_rate = driver_options.get_rx_rate(); // #5 REVIEW What units is rx_rate in?

  zmq::context_t sig_proc_context(1); // 1 is context num. Only need one per program as per examples

  zmq::socket_t driver_socket(sig_proc_context, ZMQ_PAIR);
  try {
    driver_socket.bind("ipc:///tmp/feeds/1"); // REVIEW #29 Should this be in a config file? Sort of a magic string right now
  }
  catch {
    //TODO(keith): handle error.
  }

  // REVIEW #1 Need a comment here to explain which blocks these 3 sockets talk to, the driver socket is obvious, but these three may not be
  zmq::socket_t radarctrl_socket(sig_proc_context, ZMQ_PAIR); // REVIEW #26 Name of radarctrl may need to be updated to be consistent with our discussion on Friday March 10th
  try {
    radarctrl_socket.bind("ipc:///tmp/feeds/2");
  }
  catch {
    //TODO(keith): handle error.
  }

  zmq::socket_t ack_socket(sig_proc_context, ZMQ_PAIR);
  try {
    ack_socket.bind("ipc:///tmp/feeds/3");
  }
  catch {
    //TODO(keith): handle error.
  }

  zmq::socket_t timing_socket(sig_proc_context, ZMQ_PAIR);
  try {
    timing_socket.bind("ipc:///tmp/feeds/4");
  }
  catch {
    //TODO(keith): handle error.
  }

  auto gpu_properties = get_gpu_properties();
  print_gpu_properties(gpu_properties);

  uint32_t first_stage_dm_rate = 0, second_stage_dm_rate = 0, third_stage_dm_rate = 0;
  //Check for non integer dm rates
  if (fmod(rx_rate,sig_options.get_first_stage_sample_rate()) > 0.0) {
    //TODO(keith): handle error
  }
  else if (fmod(sig_options.get_first_stage_sample_rate(),
          sig_options.get_second_stage_sample_rate()) > 0.0) {
    //TODO(keith): handle error
  }
  else if(fmod(sig_options.get_second_stage_sample_rate(),
        sig_options.get_third_stage_sample_rate()) > 0.0) {
    //TODO(keith): handle error
  }
  else{
    auto float_dm_rate = rx_rate/sig_options.get_first_stage_sample_rate();
    first_stage_dm_rate = static_cast<uint32_t>(float_dm_rate);

    float_dm_rate = sig_options.get_first_stage_sample_rate()/
          sig_options.get_second_stage_sample_rate();
    second_stage_dm_rate = static_cast<uint32_t>(float_dm_rate);

    float_dm_rate = sig_options.get_second_stage_sample_rate()/
          sig_options.get_third_stage_sample_rate();
    third_stage_dm_rate = static_cast<uint32_t>(float_dm_rate);
  }

  std::cout << "1st stage dm rate: " << first_stage_dm_rate << std::endl
    << "2nd stage dm rate: " << second_stage_dm_rate << std::endl
    << "3rd stage dm rate: " << third_stage_dm_rate << std::endl;


  std::chrono::steady_clock::time_point timing_start = std::chrono::steady_clock::now();

  Filtering filters(rx_rate,sig_options);

  std::cout << "Number of 1st stage taps: " << filters.get_num_first_stage_taps() << std::endl
    << "Number of 2nd stage taps: " << filters.get_num_second_stage_taps() << std::endl // REVIEW #34 mention that it's the number of taps?
    << "Number of 3rd stage taps: " << filters.get_num_third_stage_taps() <<std::endl
    << "Number of 1st stage taps after padding: "
    << filters.get_first_stage_lowpass_taps().size() << std::endl
    << "Number of 2nd stage taps after padding: "
    << filters.get_second_stage_lowpass_taps().size() << std::endl
    << "Number of 3rd stage taps after padding: "
    << filters.get_third_stage_lowpass_taps().size() << std::endl;

  std::chrono::steady_clock::time_point timing_end = std::chrono::steady_clock::now();
  std::cout << "Time to create 3 filters: "
    << std::chrono::duration_cast<std::chrono::microseconds>(timing_end - timing_start).count()
    << "us" << std::endl;

    //TODO(Keith): fix saving filter to file
/*  save_filter_to_file(filtertaps_1,"filter1coefficients.dat");
  save_filter_to_file(filtertaps_2,"filter2coefficients.dat"); // REVIEW #36 where should the coefficient files be placed? Currently this will put them in the calling dir I think
                                 // REPLY It does, and I do not know. Up to us I guess.
  save_filter_to_file(filtertaps_3,"filter3coefficients.dat");*/

  while(1){
    //Receive packet from radar control
    zmq::message_t radctl_request;
    radarctrl_socket.recv(&radctl_request);
    sigprocpacket::SigProcPacket sp_packet;
    std::string radctrl_str(static_cast<char*>(radctl_request.data()), radctl_request.size());
    if (sp_packet.ParseFromString(radctrl_str) == false){
      //TODO(keith): handle error
    }
    // REVIEW #37 need to check the boolean return value from each ParseFromString call
    // http://stackoverflow.com/questions/22121922/how-can-i-get-more-details-about-errors-generated-during-protobuf-parsing-c
    // REPLY Should this just halt the program? The only feasible way this goes wrong is if someone sends
    //non protobufs

    // REVIEW #15 all protobuf fields are optional, check all fields you require are filled (similarly when using computationpacket)
    // REPLY I think this is only possible if default fields are not also valid values
    std::cout << "Got radar_control request" << std::endl; // REVIEW #34 Change to radar_control to indicate where the request actually came from? (same with 'drive' below, maybe name 'usrp_driver' to be consistent with naming of dir structure)
                               // REPLY okay. I was thinking of removing this though.
    //Then receive packet from driver
    zmq::message_t driver_request;
    driver_socket.recv(&driver_request);
    rxsamplesmetadata::RxSamplesMetadata rx_metadata;
    std::string driver_str(static_cast<char*>(driver_request.data()), driver_request.size());
    if (rx_metadata.ParseFromString(driver_str) == false) {
      //TODO(keith): handle error
    }

    std::cout << "Got driver request" << std::endl;

    //Verify driver and radar control packets align
    if (sp_packet.sequence_num() != rx_metadata.sequence_num()) {
      //TODO(keith): handle error
      std::cout << "SEQUENCE NUMBER mismatch radar_control: " << sp_packet.sequence_num() // REVIEW #34 - debug output 'rctl' should be renamed to radar_control and driver to 'usrp_driver'
        << " usrp_driver: " << rx_metadata.sequence_num();                                // REPLY this might be removed depending on how we handle errors
    }

    //Parse needed packet values now
    if (sp_packet.rxchannel_size() == 0) {
      //TODO(keith): handle error
    }
    std::vector<double> rx_freqs; // REVIEW #28 Shouldn't this just be a vector of integers? I don't think sub-hz resolution available or necessary? Careful of division elsewhere if it's changed
                    //REPLY probably. I just used what was in the protobuffer.
    for(int i=0; i<sp_packet.rxchannel_size(); i++) {
      rx_freqs.push_back(sp_packet.rxchannel(i).rxfreq());
    }

    timing_start = std::chrono::steady_clock::now();

    filters.mix_first_stage_to_bandpass(rx_freqs,rx_rate);

    timing_end = std::chrono::steady_clock::now();

    std::cout << "NCO mix timing: "
      << std::chrono::duration_cast<std::chrono::microseconds>(timing_end - timing_start).count()
      << "us" << std::endl;

/*    std::vector<std::complex<float>> filtertaps_2_h(filtertaps_2.size()); // REVIEW #26 what does _h mean?
                                        // REPLY cuda examples will often denote a distinction between host and device with d and h. I followed the same scheme. could rename
    std::vector<std::complex<float>> filtertaps_3_h(filtertaps_3.size()); // REVIEW #0 need filtertaps_3 and filtertaps_2.size() * rx_freqs.size()
                                        // REPLY I actually think this is wrong to supply a size here at all.
    for (uint32_t i=0; i< rx_freqs.size(); i++){ // REVIEW #28 iterator type not consistent with previous for loop, should it be of type size_t?
                           // REPLY I need more clarifying here
      filtertaps_2_h.insert(filtertaps_2_h.end(),filtertaps_2.begin(),filtertaps_2.end()); // REVIEW #22 is this duplication necessary?
                                                 // REPLY Maybe not. I now realize from reviewing that we go from one set of samples and multiple filters
                                                 // to multiple sets of samples and one filter
      filtertaps_3_h.insert(filtertaps_3_h.end(),filtertaps_3.begin(),filtertaps_3.end());
    }*/

    if (rx_metadata.shrmemname().empty()){
      //TODO(keith): handle missing name error
    }
    DSPCore *dp = new DSPCore(&ack_socket, &timing_socket,
                             sp_packet.sequence_num(), rx_metadata.shrmemname().c_str());


    auto total_antennas = sig_options.get_main_antenna_count() +
                sig_options.get_interferometer_antenna_count();

    if (rx_metadata.numberofreceivesamples() == 0){
      //TODO(keith): handle error for missing number of samples.
    }
    auto total_samples = rx_metadata.numberofreceivesamples() * total_antennas;

    std::cout << "Total samples in data message: " << total_samples // REVIEW #34 change 'elements' to 'samples' to be clear
      << std::endl;                                               // REPLY okay

    dp->allocate_and_copy_rf_samples(total_samples);
    dp->allocate_and_copy_first_stage_filters(filters.get_first_stage_bandpass_taps_h().data(),
                                                filters.get_first_stage_bandpass_taps_h().size());

    auto num_output_samples_1 = rx_freqs.size() * rx_metadata.numberofreceivesamples()/first_stage_dm_rate // REVIEW #28 - What if the division has a remainder?
                    * total_antennas;

    dp->allocate_first_stage_output(num_output_samples_1);

    gpuErrchk(cudaStreamAddCallback(dp->get_cuda_stream(),
                  DSPCore::initial_memcpy_callback, dp, 0));

    dp->call_decimate(dp->get_rf_samples_p(),
      dp->get_first_stage_output_p(),
      dp->get_first_stage_bp_filters_p(), first_stage_dm_rate,
      rx_metadata.numberofreceivesamples(), filters.get_first_stage_lowpass_taps().size(), rx_freqs.size(),
      total_antennas, "First stage of decimation");


    // When decimating, we go from one set of samples for each antenna in the first stage
    // to multiple sets of reduced samples for each frequency in further stages. Output samples are
    // grouped by frequency with all samples for each antenna following each other
    // before samples of another frequency start. In the first stage need a filter for each frequency,
  // but in the next stages we only need one filter for all data sets.
    dp->allocate_and_copy_second_stage_filter(filters.get_second_stage_lowpass_taps().data(),
                                                filters.get_second_stage_lowpass_taps().size());

    auto num_output_samples_2 = rx_freqs.size() * num_output_samples_1 / second_stage_dm_rate;
    dp->allocate_second_stage_output(num_output_samples_2);
    auto samples_per_antenna_2 = rx_metadata.numberofreceivesamples()/first_stage_dm_rate;
    dp->call_decimate(dp->get_first_stage_output_p(),
      dp->get_second_stage_output_p(),
      dp->get_second_stage_filters_p(), second_stage_dm_rate,
      samples_per_antenna_2, filters.get_second_stage_lowpass_taps().size(), rx_freqs.size(),
      total_antennas, "Second stage of decimation");


    dp->allocate_and_copy_third_stage_filter(filters.get_third_stage_lowpass_taps().data(),
                                               filters.get_third_stage_lowpass_taps().size());
    auto num_output_samples_3 = num_output_samples_2 / third_stage_dm_rate;
    dp->allocate_third_stage_output(num_output_samples_3);
    auto samples_per_antenna_3 = samples_per_antenna_2/second_stage_dm_rate;
    dp->call_decimate(dp->get_second_stage_output_p(),
      dp->get_third_stage_output_p(),
      dp->get_third_stage_filters_p(), third_stage_dm_rate,
      samples_per_antenna_3, filters.get_third_stage_lowpass_taps().size(), rx_freqs.size(),
      total_antennas, "Third stage of decimation");

    dp->allocate_and_copy_host_output(num_output_samples_3);

    // New in CUDA 5.0: Add a CPU callback which is called once all currently pending operations in the CUDA stream have finished
    gpuErrchk(cudaStreamAddCallback(dp->get_cuda_stream(),
                      DSPCore::cuda_postprocessing_callback, dp, 0));


  }


}
