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
#include "utils/shared_macros/shared_macros.hpp"
#include "utils/zmq_borealis_helpers/zmq_borealis_helpers.hpp"
#include "dsp.hpp"
#include "filtering.hpp"
#include "decimate.hpp"


int main(int argc, char **argv){
  GOOGLE_PROTOBUF_VERIFY_VERSION; // Verifies that header and lib are same version.

  //TODO(keith): verify config options.
  auto sig_options = SignalProcessingOptions();
  auto rx_rate = sig_options.get_rx_rate(); //Hz


  zmq::context_t context(1); // 1 is context num. Only need one per program as per examples
  auto identities = {sig_options.get_dsp_radctrl_identity(),
                   sig_options.get_dsp_driver_identity(),
                   sig_options.get_dsp_exphan_identity(),
                   sig_options.get_dsp_dw_identity(),
                   sig_options.get_dspbegin_brian_identity(),
                   sig_options.get_dspend_brian_identity()};

  auto sockets_vector = create_sockets(context, identities, sig_options.get_router_address());

  zmq::socket_t &dsp_to_radar_control = sockets_vector[0];
  zmq::socket_t &dsp_to_driver = sockets_vector[1];
  zmq::socket_t &dsp_to_experiment_handler = sockets_vector[2];
  zmq::socket_t &dsp_to_data_write = sockets_vector[3];
  zmq::socket_t &dsp_to_brian_begin = sockets_vector[4];
  zmq::socket_t &dsp_to_brian_end = sockets_vector[5];

  auto gpu_properties = get_gpu_properties();
  print_gpu_properties(gpu_properties);

  uint32_t first_stage_dm_rate = 0, second_stage_dm_rate = 0, third_stage_dm_rate = 0;
  //Check for non integer dm rates
  if (fmod(rx_rate,sig_options.get_first_stage_sample_rate()) > 0.0) {
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

  std::cout << sig_options.get_first_stage_sample_rate() << std::endl;
  std::cout << sig_options.get_second_stage_sample_rate() << std::endl;
  std::cout << sig_options.get_third_stage_sample_rate() << std::endl;

  RUNTIME_MSG("1st stage dm rate: " << COLOR_YELLOW(first_stage_dm_rate));
  RUNTIME_MSG("2nd stage dm rate: " << COLOR_YELLOW(second_stage_dm_rate));
  RUNTIME_MSG("3rd stage dm rate: " << COLOR_YELLOW(third_stage_dm_rate));


  auto filter_timing_start = std::chrono::steady_clock::now();

  Filtering filters(rx_rate,sig_options);

  RUNTIME_MSG("Number of 1st stage taps: " << COLOR_YELLOW(filters.get_num_first_stage_taps()));
  RUNTIME_MSG("Number of 2nd stage taps: " << COLOR_YELLOW(filters.get_num_second_stage_taps()));
  RUNTIME_MSG("Number of 3rd stage taps: " << COLOR_YELLOW(filters.get_num_third_stage_taps()));

  RUNTIME_MSG("Number of 1st stage taps after padding: "
              << COLOR_YELLOW(filters.get_first_stage_lowpass_taps().size()));
  RUNTIME_MSG("Number of 2nd stage taps after padding: "
              << COLOR_YELLOW(filters.get_second_stage_lowpass_taps().size()));
  RUNTIME_MSG("Number of 3rd stage taps after padding: "
              << COLOR_YELLOW(filters.get_third_stage_lowpass_taps().size()));

  auto filter_timing_end = std::chrono::steady_clock::now();
  auto time_diff = std::chrono::duration_cast<std::chrono::microseconds>(filter_timing_end -
                                                                       filter_timing_start).count();
  RUNTIME_MSG("Time to create 3 filters: " << COLOR_MAGENTA(time_diff) << "us");

  //FIXME(Keith): fix saving filter to file
  filters.save_filter_to_file(filters.get_first_stage_lowpass_taps(),"filter1coefficients.dat");
  filters.save_filter_to_file(filters.get_second_stage_lowpass_taps(),"filter2coefficients.dat");
  filters.save_filter_to_file(filters.get_third_stage_lowpass_taps(),"filter3coefficients.dat");

  SharedMemoryHandler shrmem(sig_options.get_ringbuffer_name());
  std::vector<cuComplex*> ringbuffer_ptrs_start;

  auto first_time = true;
  for(;;){
    //Receive packet from radar control
    auto message =  std::string("Need metadata");
    SEND_REQUEST(dsp_to_radar_control, sig_options.get_radctrl_dsp_identity(), message);
    auto reply = RECV_REPLY(dsp_to_radar_control, sig_options.get_radctrl_dsp_identity());

    sigprocpacket::SigProcPacket sp_packet;
    if (sp_packet.ParseFromString(reply) == false){
      //TODO(keith): handle error
    }

    //Then receive packet from driver
    message = std::string("Need data to process");
    SEND_REQUEST(dsp_to_driver, sig_options.get_driver_dsp_identity(), message);
    reply = RECV_REPLY(dsp_to_driver, sig_options.get_driver_dsp_identity());

    rxsamplesmetadata::RxSamplesMetadata rx_metadata;
    if (rx_metadata.ParseFromString(reply) == false) {
      //TODO(keith): handle error
    }

    RUNTIME_MSG("Got driver request for sequence #" << COLOR_RED(rx_metadata.sequence_num()));

    auto total_antennas = sig_options.get_main_antenna_count() +
                sig_options.get_interferometer_antenna_count();

    if (first_time) {
      shrmem.open_shr_mem();
      if (rx_metadata.ringbuffer_size() == 0) {
        //TODO(keith): handle error
      }
      for(uint32_t i=0; i<total_antennas; i++){
        auto ptr = static_cast<cuComplex*>(shrmem.get_shrmem_addr()) +
                                              (i * rx_metadata.ringbuffer_size());
        ringbuffer_ptrs_start.push_back(ptr);
      }
      first_time = false;
    }
    //Verify driver and radar control packets align
    if (sp_packet.sequence_num() != rx_metadata.sequence_num()) {
      //TODO(keith): handle error
      RUNTIME_MSG("SEQUENCE NUMBER mismatch radar_control: " << COLOR_RED(sp_packet.sequence_num())
        << " usrp_driver: " << COLOR_RED(rx_metadata.sequence_num()));
    }

    //Parse needed packet values now
    if (sp_packet.rxchannel_size() == 0) {
      //TODO(keith): handle error
    }
    std::vector<double> rx_freqs;
    for(int i=0; i<sp_packet.rxchannel_size(); i++) {
      rx_freqs.push_back(sp_packet.rxchannel(i).rxfreq());
    }

    TIMEIT_IF_TRUE_OR_DEBUG(false, "   NCO mix timing: ",
      [&]() {
        filters.mix_first_stage_to_bandpass(rx_freqs,rx_rate);
      }()
    );


    DSPCore *dp = new DSPCore(&dsp_to_brian_begin, &dsp_to_brian_end, &dsp_to_data_write,
                             sig_options, sp_packet.sequence_num(),
                             rx_freqs, &filters);

    if (rx_metadata.numberofreceivesamples() == 0){
      //TODO(keith): handle error for missing number of samples.
    }

      //We need to sample early to account for propagating samples through filters.
    int64_t extra_samples = (first_stage_dm_rate * second_stage_dm_rate *
                        filters.get_num_third_stage_taps());

    auto samples_needed = rx_metadata.numberofreceivesamples() + 2 * extra_samples;
    auto total_samples = samples_needed * total_antennas;

    DEBUG_MSG("   Total samples in data message: " << total_samples);

    dp->allocate_and_copy_rf_samples(total_antennas, samples_needed, extra_samples,
                                rx_metadata.time_zero(), rx_metadata.start_time(),
                                rx_metadata.ringbuffer_size(), first_stage_dm_rate,
                                second_stage_dm_rate,ringbuffer_ptrs_start);
    dp->allocate_and_copy_frequencies(rx_freqs.data(), rx_freqs.size());
    dp->allocate_and_copy_first_stage_filters(filters.get_first_stage_bandpass_taps_h().data(),
                                                filters.get_first_stage_bandpass_taps_h().size());

    auto num_output_samples_per_antenna_1 = samples_needed / first_stage_dm_rate;
    auto total_output_samples_1 = rx_freqs.size() * num_output_samples_per_antenna_1 *
                                   total_antennas;

    dp->allocate_first_stage_output(total_output_samples_1);

    dp->initial_memcpy_callback();

    call_decimate<DecimationType::bandpass>(dp->get_rf_samples_p(),
      dp->get_first_stage_output_p(),dp->get_first_stage_bp_filters_p(), first_stage_dm_rate,
      samples_needed, filters.get_first_stage_lowpass_taps().size(),
      rx_freqs.size(), total_antennas, rx_rate, dp->get_frequencies_p(),
      "First stage of decimation", dp->get_cuda_stream());



    // When decimating, we go from one set of samples for each antenna in the first stage
    // to multiple sets of reduced samples for each frequency in further stages. Output samples are
    // grouped by frequency with all samples for each antenna following each other
    // before samples of another frequency start. In the first stage need a filter for each
    // frequency, but in the next stages we only need one filter for all data sets.
    dp->allocate_and_copy_second_stage_filter(filters.get_second_stage_lowpass_taps().data(),
                                                filters.get_second_stage_lowpass_taps().size());

    auto num_output_samples_per_antenna_2 = num_output_samples_per_antenna_1 / second_stage_dm_rate;
    auto total_output_samples_2 = rx_freqs.size() * num_output_samples_per_antenna_2 *
                                    total_antennas;

    dp->allocate_second_stage_output(total_output_samples_2);

    // each antenna has a data set for each frequency after filtering.
    auto samples_per_antenna_2 = total_output_samples_1/total_antennas/rx_freqs.size();
    call_decimate<DecimationType::lowpass>(dp->get_first_stage_output_p(),
      dp->get_second_stage_output_p(), dp->get_second_stage_filter_p(), second_stage_dm_rate,
      samples_per_antenna_2, filters.get_second_stage_lowpass_taps().size(), rx_freqs.size(),
      total_antennas, rx_rate, dp->get_frequencies_p(), "Second stage of decimation",
      dp->get_cuda_stream());


    dp->allocate_and_copy_third_stage_filter(filters.get_third_stage_lowpass_taps().data(),
                                               filters.get_third_stage_lowpass_taps().size());

    auto num_output_samples_per_antenna_3 = num_output_samples_per_antenna_2 / third_stage_dm_rate;
    auto total_output_samples_3 = rx_freqs.size() * num_output_samples_per_antenna_3 *
                                    total_antennas;

    dp->allocate_third_stage_output(total_output_samples_3);

    auto samples_per_antenna_3 = samples_per_antenna_2/second_stage_dm_rate;
    call_decimate<DecimationType::lowpass>(dp->get_second_stage_output_p(),
      dp->get_third_stage_output_p(), dp->get_third_stage_filter_p(), third_stage_dm_rate,
      samples_per_antenna_3, filters.get_third_stage_lowpass_taps().size(), rx_freqs.size(),
      total_antennas, rx_rate, dp->get_frequencies_p(), "Third stage of decimation",
      dp->get_cuda_stream());

    dp->allocate_and_copy_host_output(total_output_samples_3);

    dp->cuda_postprocessing_callback(rx_freqs, total_antennas,
                                      samples_needed,
                                      num_output_samples_per_antenna_1,
                                      num_output_samples_per_antenna_2,
                                      num_output_samples_per_antenna_3);



  }

}
