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
#include <numeric>
#include <functional>
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

  SharedMemoryHandler shrmem(sig_options.get_ringbuffer_name());
  std::vector<cuComplex*> ringbuffer_ptrs_start;

  Filtering filters;

  std::vector<uint32_t> dm_rates;
  double rx_rate;
  uint32_t total_antennas;
  double output_sample_rate;

  auto first_time = true;
  for(;;) {

    //Receive first packet from radar control
    //auto message =  std::string("Need metadata");
    //SEND_REQUEST(dsp_to_radar_control, sig_options.get_radctrl_dsp_identity(), message);
    auto reply = RECV_REPLY(dsp_to_radar_control, sig_options.get_radctrl_dsp_identity());

    sigprocpacket::SigProcPacket sp_packet;
    if (sp_packet.ParseFromString(reply) == false){
      //TODO(keith): handle error
    }

    if (first_time) {
      total_antennas = sig_options.get_main_antenna_count() +
                  sig_options.get_interferometer_antenna_count();

      // First time - set up rx rate and filters.
      rx_rate = sp_packet.rxrate(); //Hz
      output_sample_rate = sp_packet.output_sample_rate(); //Hz

      std::vector<std::vector<float>> filter_taps;
      for (uint32_t i=0; i<sp_packet.decimation_stages_size(); i++) {
        dm_rates.push_back(sp_packet.decimation_stages(i).dm_rate());

        std::vector<float> taps(sp_packet.decimation_stages(i).filter_taps().begin(),
        sp_packet.decimation_stages(i).filter_taps().end());
        filter_taps.push_back(taps);
      }

      RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "Decimation rates: ");
      for (auto &rate : dm_rates) {
        RUNTIME_MSG("   " << rate);
      }

      filters = Filtering(filter_taps);

      RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "Number of taps per stage: ");
      for (auto &taps : filter_taps) {
        RUNTIME_MSG("   " << COLOR_MAGENTA(taps.size()));
      }

      RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") <<
                  "Number of taps per stage after padding: ");
      for (auto &taps : filters.get_unmixed_filter_taps()) {
        RUNTIME_MSG("   " << COLOR_MAGENTA(taps.size()));
      }
    } // if (first_time)

    //Then receive first packet from driver
    auto message = std::string("Need data to process");
    SEND_REQUEST(dsp_to_driver, sig_options.get_driver_dsp_identity(), message);
    reply = RECV_REPLY(dsp_to_driver, sig_options.get_driver_dsp_identity());

    rxsamplesmetadata::RxSamplesMetadata rx_metadata;
    if (rx_metadata.ParseFromString(reply) == false) {
      //TODO(keith): handle error
    }

    if (first_time) {
      // First time - set up memory
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

    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "Got driver request for sequence #"
      << COLOR_RED(rx_metadata.sequence_num()));

    //Verify driver and radar control packets align
    if (sp_packet.sequence_num() != rx_metadata.sequence_num()) {
      //TODO(keith): handle error
      RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") <<"SEQUENCE NUMBER mismatch radar_control: "
        << COLOR_RED(sp_packet.sequence_num()) << " usrp_driver: "
        << COLOR_RED(rx_metadata.sequence_num()));
    }

    if (rx_metadata.rx_rate() != rx_rate) {
      //TODO handle error
    }

    //Parse needed packet values now
    if (sp_packet.rxchannel_size() == 0) {
      //TODO(keith): handle error
    }
    std::vector<double> rx_freqs;
    std::vector<uint32_t> slice_ids;
    for(uint32_t channel=0; channel<sp_packet.rxchannel_size(); channel++) {
      rx_freqs.push_back(sp_packet.rxchannel(channel).rxfreq());
      slice_ids.push_back(sp_packet.rxchannel(channel).slice_id());
    }


    // Parse out the beam phases from the radar control signal proc packet.
    std::vector<cuComplex> beam_phases;
    std::vector<uint32_t> beam_direction_counts;

    for (uint32_t channel=0; channel<sp_packet.rxchannel_size(); channel++) {
      // In this case each channel is the info for a new RX frequency
      auto rx_channel = sp_packet.rxchannel(channel);

      // Keep track of the number of beams each RX freq has. We will need this for beamforming.
      beam_direction_counts.push_back(rx_channel.beam_directions_size());

      // We are going to use two intermediate vectors here to rearrange the phase data so that
      // all M data comes first, followed by all I data. This way can we directly treat each
      // block of memory as a matrix for beamforming the individual arrays.
      std::vector<cuComplex> main_phases;
      std::vector<cuComplex> intf_phases;

      for (uint32_t beam_num=0; beam_num<rx_channel.beam_directions_size(); beam_num++) {
        // Go through each beam now and add the phases for each antenna to a vector.
        auto beam = rx_channel.beam_directions(beam_num);

        for(uint32_t phase_num=0; phase_num<beam.phase_size(); phase_num++) {
          auto phase = beam.phase(phase_num);
          cuComplex new_angle;
          new_angle.x = phase.real_phase();
          new_angle.y = phase.imag_phase();

          if (phase_num < sig_options.get_main_antenna_count()) {
            main_phases.push_back(new_angle);
          }
          else {
            intf_phases.push_back(new_angle);
          }
        }
      }

      // Combine the separated antenna phases back into a flat vector.
      for (auto &phase : main_phases) {
        beam_phases.push_back(phase);
      }

      for (auto &phase : intf_phases) {
        beam_phases.push_back(phase);
      }
    }

    TIMEIT_IF_TRUE_OR_DEBUG(false, "   NCO mix timing: ",
      [&]() {
        filters.mix_first_stage_to_bandpass(rx_freqs,rx_rate);
      }()
    );



    auto complex_taps = filters.get_mixed_filter_taps();

    DSPCore *dp = new DSPCore(&dsp_to_brian_begin, &dsp_to_brian_end, &dsp_to_data_write,
                             sig_options, sp_packet.sequence_num(), rx_rate, output_sample_rate,
                             rx_freqs, complex_taps, beam_phases,
                             beam_direction_counts, rx_metadata.initialization_time(),
                             rx_metadata.sequence_start_time(), slice_ids, dm_rates);

    if (rx_metadata.numberofreceivesamples() == 0){
      //TODO(keith): handle error for missing number of samples.
    }


    //We need to sample early to account for propagating samples through filters. The number of
    //required early samples is equal to the largest filter length in time (based on the rate at
    //that stage.) The following lines find the max filter length in time and convert that to
    //number of samples at the input rate.

    int64_t extra_samples = std::accumulate(dm_rates.begin(), dm_rates.end()-1, 1,
                                            std::multiplies<int64_t>()) *
                            complex_taps.back().size();

    // problem with this if you have too few stages. Need to figure this out.

    for (int32_t i=complex_taps.size()-2; i>=0; i--) {
      auto dm_index = dm_rates.size() - 2 - i;
      if (i == 0) {
        if (extra_samples < complex_taps[0].size()) {
          extra_samples = complex_taps[0].size();
        }
      }
      else if (i == 1) {
        if (extra_samples < (dm_rates[0] * complex_taps[1].size())) {
          extra_samples = dm_rates[0] * complex_taps[1].size();
        }
      }
      else {
        auto total_dm_rate = std::accumulate(dm_rates.begin(),
                                          dm_rates.end()-2-i,
                                          1,
                                          std::multiplies<uint32_t>());
        if (extra_samples < total_dm_rate * complex_taps[i].size()) {
          extra_samples = total_dm_rate * complex_taps[i].size();
        }
      }
    }

    auto total_dm_rate = std::accumulate(dm_rates.begin(), dm_rates.end(), 1,
                                            std::multiplies<int64_t>());

    auto samples_needed = rx_metadata.numberofreceivesamples() + 2 * extra_samples;
    samples_needed = uint32_t(std::ceil(float(samples_needed)/float(total_dm_rate)) *
                              total_dm_rate);
    auto total_samples = samples_needed * total_antennas;

    DEBUG_MSG("   Total samples in data message: " << total_samples);

    dp->allocate_and_copy_frequencies(rx_freqs.data(), rx_freqs.size());

    auto offset_to_first_rx_sample = uint32_t(sp_packet.offset_to_first_rx_sample() * rx_rate);
    dp->allocate_and_copy_rf_samples(total_antennas, samples_needed, extra_samples,
                                offset_to_first_rx_sample,
                                rx_metadata.initialization_time(),
                                rx_metadata.sequence_start_time(),
                                rx_metadata.ringbuffer_size(), ringbuffer_ptrs_start);

    dp->allocate_and_copy_bandpass_filters(complex_taps[0].data(), complex_taps[0].size());

    auto num_output_samples_per_antenna = samples_needed / dm_rates[0];
    auto total_output_samples_1 = rx_freqs.size() * num_output_samples_per_antenna *
                                   total_antennas;

    dp->allocate_output(total_output_samples_1);

    dp->initial_memcpy_callback();

    auto last_filter_output = dp->get_last_filter_output_d();
    call_decimate<DecimationType::bandpass>(dp->get_rf_samples_p(),
      last_filter_output, dp->get_bp_filters_p(), dm_rates[0],
      samples_needed, complex_taps[0].size(),
      rx_freqs.size(), total_antennas, rx_rate, dp->get_frequencies_p(),
      "Bandpass stage of decimation", dp->get_cuda_stream());


    std::vector<uint32_t> samples_per_antenna(complex_taps.size());
    std::vector<uint32_t> total_output_samples(complex_taps.size());

    samples_per_antenna[0] = num_output_samples_per_antenna;
    total_output_samples[0] = total_output_samples_1;

    // When decimating, we go from one set of samples for each antenna in the first stage
    // to multiple sets of reduced samples for each frequency in further stages. Output samples are
    // grouped by frequency with all samples for each antenna following each other
    // before samples of another frequency start. In the first stage need a filter for each
    // frequency, but in the next stages we only need one filter for all data sets.
    cuComplex* prev_output = last_filter_output;
    for (uint32_t i=1; i<complex_taps.size(); i++) {
      samples_per_antenna[i] = samples_per_antenna[i-1]/dm_rates[i];
      total_output_samples[i] = rx_freqs.size() * samples_per_antenna[i] * total_antennas;

      dp->allocate_and_copy_lowpass_filter(complex_taps[i].data(), complex_taps[i].size());
      dp->allocate_output(total_output_samples[i]);

      auto allocated_lp_filter = dp->get_last_lowpass_filter_d();
      last_filter_output = dp->get_last_filter_output_d();

      call_decimate<DecimationType::lowpass>(prev_output, last_filter_output, allocated_lp_filter,
        dm_rates[i], samples_per_antenna[i-1], complex_taps[i].size(), rx_freqs.size(),
        total_antennas, rx_rate, dp->get_frequencies_p(), " stage of decimation",
        dp->get_cuda_stream());

      prev_output = last_filter_output;
    }

    dp->cuda_postprocessing_callback(rx_freqs, total_antennas,
                                      samples_needed, samples_per_antenna, total_output_samples);

  } //for(;;)
}
