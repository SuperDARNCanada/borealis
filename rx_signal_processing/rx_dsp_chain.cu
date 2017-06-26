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
#include "decimate.hpp"

#ifdef DEBUG
#define DEBUG_MSG(x) do {std::cerr << x << std::endl;} while (0)
#else
#define DEBUG_MSG(x)
#endif

#define ERR_CHK_ZMQ(x) try {x;} catch (zmq::error_t& e) {} //TODO(keith): handle error


int main(int argc, char **argv){
  GOOGLE_PROTOBUF_VERIFY_VERSION; // Verifies that header and lib are same version.

  //TODO(keith): verify config options.
  auto driver_options = DriverOptions();
  auto sig_options = SignalProcessingOptions();
  auto rx_rate = driver_options.get_rx_rate(); //Hz

  zmq::context_t sig_proc_context(1); // 1 is context num. Only need one per program as per examples

  zmq::socket_t driver_socket(sig_proc_context, ZMQ_PAIR);
  ERR_CHK_ZMQ(driver_socket.bind("ipc:///tmp/feeds/1"))


  //This socket is used to receive metadata about the sequence to process
  zmq::socket_t radar_control_socket(sig_proc_context, ZMQ_PAIR);
  ERR_CHK_ZMQ(radar_control_socket.bind("ipc:///tmp/feeds/2"))

  //This socket is used to acknowledge a completed sequence to radar_control
  zmq::socket_t ack_socket(sig_proc_context, ZMQ_PAIR);
  ERR_CHK_ZMQ(ack_socket.bind("ipc:///tmp/feeds/3"))

  //This socket is used to send the GPU kernel timing to radar_control to know if the processing
  //can be done in real-time.
  zmq::socket_t timing_socket(sig_proc_context, ZMQ_PAIR);
  ERR_CHK_ZMQ(timing_socket.bind("ipc:///tmp/feeds/4"))


  auto gpu_properties = get_gpu_properties();
  print_gpu_properties(gpu_properties);

  uint32_t first_stage_dm_rate = 0, second_stage_dm_rate = 0, third_stage_dm_rate = 0;
  //Check for non integer dm rates
  if (fmod(rx_rate,sig_options.get_first_stage_sample_rate()) > 0.0) {
    //TODO(keith): handle error
  } //TODO(keith): not sure these checks will work.
/*  else if (fmod(sig_options.get_first_stage_sample_rate(),
          sig_options.get_second_stage_sample_rate()) > 0.0) {
    //TODO(keith): handle error
  }
  else if(fmod(sig_options.get_second_stage_sample_rate(),
        sig_options.get_third_stage_sample_rate()) > 0.0) {
    //TODO(keith): handle error
  }*/
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


  auto filter_timing_start = std::chrono::steady_clock::now();

  Filtering filters(rx_rate,sig_options);

  DEBUG_MSG("Number of 1st stage taps: " << filters.get_num_first_stage_taps() << std::endl
    << "Number of 2nd stage taps: " << filters.get_num_second_stage_taps() << std::endl
    << "Number of 3rd stage taps: " << filters.get_num_third_stage_taps() <<std::endl
    << "Number of 1st stage taps after padding: "
    << filters.get_first_stage_lowpass_taps().size() << std::endl
    << "Number of 2nd stage taps after padding: "
    << filters.get_second_stage_lowpass_taps().size() << std::endl
    << "Number of 3rd stage taps after padding: "
    << filters.get_third_stage_lowpass_taps().size());

  auto filter_timing_end = std::chrono::steady_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(filter_timing_end -
                                                                       filter_timing_start).count();
  DEBUG_MSG("Time to create 3 filters: " << time_diff << "us");

  //FIXME(Keith): fix saving filter to file
  filters.save_filter_to_file(filters.get_first_stage_lowpass_taps(),"filter1coefficients.dat");
  filters.save_filter_to_file(filters.get_second_stage_lowpass_taps(),"filter2coefficients.dat");
  filters.save_filter_to_file(filters.get_third_stage_lowpass_taps(),"filter3coefficients.dat");

  while(1){
    //Receive packet from radar control
    zmq::message_t radctl_request;
    radar_control_socket.recv(&radctl_request);
    sigprocpacket::SigProcPacket sp_packet;
    std::string radctrl_str(static_cast<char*>(radctl_request.data()), radctl_request.size());
    if (sp_packet.ParseFromString(radctrl_str) == false){
      //TODO(keith): handle error
    }

    //Then receive packet from driver
    zmq::message_t driver_request;
    driver_socket.recv(&driver_request);
    rxsamplesmetadata::RxSamplesMetadata rx_metadata;
    std::string driver_str(static_cast<char*>(driver_request.data()), driver_request.size());
    if (rx_metadata.ParseFromString(driver_str) == false) {
      //TODO(keith): handle error
    }

    DEBUG_MSG("Got driver request");

    //Verify driver and radar control packets align
    if (sp_packet.sequence_num() != rx_metadata.sequence_num()) {
      //TODO(keith): handle error
      DEBUG_MSG("SEQUENCE NUMBER mismatch radar_control: " << sp_packet.sequence_num()
        << " usrp_driver: " << rx_metadata.sequence_num());
    }

    //Parse needed packet values now
    if (sp_packet.rxchannel_size() == 0) {
      //TODO(keith): handle error
    }
    std::vector<double> rx_freqs;
    for(int i=0; i<sp_packet.rxchannel_size(); i++) {
      rx_freqs.push_back(sp_packet.rxchannel(i).rxfreq());
    }

    auto mix_timing_start = std::chrono::steady_clock::now();

    filters.mix_first_stage_to_bandpass(rx_freqs,rx_rate);

    auto mix_timing_end = std::chrono::steady_clock::now();

    time_diff = std::chrono::duration_cast<std::chrono::microseconds>(mix_timing_end -
                                                                        mix_timing_start).count();

    DEBUG_MSG("NCO mix timing: " << time_diff<< "us");

    if (rx_metadata.shrmemname().empty()){
      //TODO(keith): handle missing name error
    }
    DSPCore *dp = new DSPCore(&ack_socket, &timing_socket,
                             sp_packet.sequence_num(), rx_metadata.shrmemname());


    auto total_antennas = sig_options.get_main_antenna_count() +
                sig_options.get_interferometer_antenna_count();

    if (rx_metadata.numberofreceivesamples() == 0){
      //TODO(keith): handle error for missing number of samples.
    }
    auto total_samples = rx_metadata.numberofreceivesamples() * total_antennas;

    DEBUG_MSG("Total samples in data message: " << total_samples);

    dp->allocate_and_copy_rf_samples(total_samples);
    dp->allocate_and_copy_first_stage_filters(filters.get_first_stage_bandpass_taps_h().data(),
                                                filters.get_first_stage_bandpass_taps_h().size());

    auto num_output_samples_1 = rx_freqs.size() *
                                  (rx_metadata.numberofreceivesamples()/first_stage_dm_rate) *
                                  total_antennas;

    dp->allocate_first_stage_output(num_output_samples_1);

    gpuErrchk(cudaStreamAddCallback(dp->get_cuda_stream(),
                  DSPCore::initial_memcpy_callback, dp, 0));

    call_decimate<DecimationType::bandpass>(dp->get_rf_samples_p(),
      dp->get_first_stage_output_p(),
      dp->get_first_stage_bp_filters_p(), first_stage_dm_rate,
      rx_metadata.numberofreceivesamples(), filters.get_first_stage_lowpass_taps().size(), rx_freqs.size(),
      total_antennas, "First stage of decimation", dp->get_cuda_stream());


    // When decimating, we go from one set of samples for each antenna in the first stage
    // to multiple sets of reduced samples for each frequency in further stages. Output samples are
    // grouped by frequency with all samples for each antenna following each other
    // before samples of another frequency start. In the first stage need a filter for each 
    // frequency, but in the next stages we only need one filter for all data sets.
    dp->allocate_and_copy_second_stage_filter(filters.get_second_stage_lowpass_taps().data(),
                                                filters.get_second_stage_lowpass_taps().size());

    auto num_output_samples_2 = num_output_samples_1 / second_stage_dm_rate;

    dp->allocate_second_stage_output(num_output_samples_2);

    // each antenna has a data set for each frequency after filtering.
    auto samples_per_antenna_2 = num_output_samples_1/total_antennas;
    call_decimate<DecimationType::lowpass>(dp->get_first_stage_output_p(),
      dp->get_second_stage_output_p(),
      dp->get_second_stage_filter_p(), second_stage_dm_rate,
      samples_per_antenna_2, filters.get_second_stage_lowpass_taps().size(), rx_freqs.size(),
      total_antennas, "Second stage of decimation", dp->get_cuda_stream());


    dp->allocate_and_copy_third_stage_filter(filters.get_third_stage_lowpass_taps().data(),
                                               filters.get_third_stage_lowpass_taps().size());
    auto num_output_samples_3 = num_output_samples_2 / third_stage_dm_rate;
    dp->allocate_third_stage_output(num_output_samples_3);
    auto samples_per_antenna_3 = samples_per_antenna_2/second_stage_dm_rate;
    call_decimate<DecimationType::lowpass>(dp->get_second_stage_output_p(),
      dp->get_third_stage_output_p(),
      dp->get_third_stage_filter_p(), third_stage_dm_rate,
      samples_per_antenna_3, filters.get_third_stage_lowpass_taps().size(), rx_freqs.size(),
      total_antennas, "Third stage of decimation", dp->get_cuda_stream());

    dp->allocate_and_copy_host_output(num_output_samples_3);

    gpuErrchk(cudaStreamAddCallback(dp->get_cuda_stream(),
                      DSPCore::cuda_postprocessing_callback, dp, 0));

    cudaDeviceSynchronize();

  }



}
