/*

Copyright 2017 SuperDARN Canada

See LICENSE for details

  \file dsp.cu
  This file contains the implementation for the all the needed GPU DSP work.
*/

#include "dsp.hpp"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <cuComplex.h>
#include <chrono>
#include <thread>
#include <complex>
#include <eigen3/Eigen/Dense>
#include "utils/zmq_borealis_helpers/zmq_borealis_helpers.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"
#include "utils/protobuf/sigprocpacket.pb.h"
#include "utils/protobuf/processeddata.pb.h"
#include "utils/shared_macros/shared_macros.hpp"
#include "filtering.hpp"
//TODO(keith): decide on handing gpu errors
//TODO(keith): potentially add multigpu support

//This keep postprocess local to this file.
namespace {
  /**
   * @brief      Sends an acknowledgment to the radar control and starts the timing after the
   *             RF samples have been copied.
   *
   * @param[in]  stream           CUDA stream this callback is associated with.
   * @param[in]  status           Error status of CUDA work in the stream.
   * @param[in]  processing_data  A pointer to the DSPCore associated with this CUDA stream.
   */
  void CUDART_CB initial_memcpy_callback_handler(cudaStream_t stream, cudaError_t status,
                          void *processing_data)
  {
    gpuErrchk(status);

    auto imc = [processing_data]()
    {
      auto dp = static_cast<DSPCore*>(processing_data);
      dp->send_ack();
      dp->start_decimate_timing();
      DEBUG_MSG(COLOR_RED("Finished initial memcpy handler for sequence #"
                 << dp->get_sequence_num() << ". Thread should exit here"));
    };

    std::thread start_imc(imc);
    start_imc.join();
  }


  /**
   * @brief      Drops samples contaminated by edge effects and filter roll off.
   *
   * @param      input_samples    The input samples.
   * @param      output_samples   The output samples.
   * @param      samps_per_stage  The number of output samples per stage.
   * @param      taps_per_stage   The number of filter taps per stage.
   * @param[in]  num_antennas     The number of antennas.
   * @param[in]  num_freqs        The number of freqs.
   *
   */
  void drop_bad_samples(cuComplex *input_samples, std::vector<cuComplex> &output_samples,
                        std::vector<uint32_t> &samps_per_stage,
                        std::vector<uint32_t> &taps_per_stage,
                        uint32_t num_antennas, uint32_t num_freqs)
  {
    std::vector<uint32_t> decimation_rates(samps_per_stage.size() - 1);

    for (uint32_t i=0; i<samps_per_stage.size()-1; i++) {
      decimation_rates[i] = samps_per_stage[i]/samps_per_stage[i+1];
    }

    auto original_undropped_sample_count = samps_per_stage.back();
    auto original_samples_per_frequency = num_antennas * original_undropped_sample_count;
    auto num_bad_samples = 0;
    for (int i=0; i<decimation_rates.size(); i++) {
      if (num_bad_samples >= decimation_rates[i]) {
        num_bad_samples = floor(num_bad_samples/decimation_rates[i]);
      }
      else {
        num_bad_samples = 0;
      }

      num_bad_samples += floor(taps_per_stage[i]/decimation_rates[i]);
      if (taps_per_stage[i] % decimation_rates[i] > samps_per_stage[i] % decimation_rates[i]){
          num_bad_samples++;
      }
      samps_per_stage[i+1] -= num_bad_samples;
    }

    auto samples_per_frequency = samps_per_stage.back() * num_antennas;
    output_samples.resize(num_freqs * samples_per_frequency);

    for (uint32_t freq_index=0; freq_index < num_freqs; freq_index++) {
      for (int i=0; i<num_antennas; i++){
        auto dest = output_samples.data() + freq_index*samples_per_frequency + i*samps_per_stage.back();
        auto src = input_samples + freq_index*original_samples_per_frequency + i*original_undropped_sample_count;
        auto num_bytes =  sizeof(cuComplex) * samps_per_stage.back();
        memcpy(dest, src, num_bytes);
      }
    }
  }

  /**
   * @brief      Beamforms the final samples
   *
   * @param      filtered_samples         A flat vector containing all the filtered samples for all
   *                                      RX frequencies.
   * @param      beamformed_samples_main  A vector where the beamformed and combined main array
   *                                      samples are placed.
   * @param      beamformed_samples_intf  A vector where the beamformed and combined intf array
   *                                      samples are placed.
   * @param      phases                   A flat vector of the phase delay offsets used to generate
   *                                      azimuthal directions. Phase offsets are complex
   *                                      exponential.
   * @param      num_main_ants            The number of main antennas.
   * @param      num_intf_ants            The number of intf antennas.
   * @param      beam_direction_counts    A vector containing the number of beam directions for each
   *                                      RX frequency.
   * @param      num_samples              The number of samples per antenna.
   *
   * This method extracts the offsets to the phases and samples needed for the beam directions of
   * each RX frequency. The Eigen library is then used to multiply the matrices to yield the final
   * beamformed samples. The main array and interferometer array are beamformed separately.
   */
  void beamform_samples(std::vector<cuComplex> &filtered_samples,
                        std::vector<cuComplex> &beamformed_samples_main,
                        std::vector<cuComplex> &beamformed_samples_intf,
                        std::vector<cuComplex> &phases, uint32_t num_main_ants,
                        uint32_t num_intf_ants, std::vector<uint32_t> beam_direction_counts,
                        uint32_t num_samples)
  {

    // Gonna make a lambda here to avoid repeated code. This is the main procedure that will
    // beamform the samples from offsets into the vectors.
    auto beamform_from_offsets = [&](cuComplex* samples_ptr,
                                      cuComplex* phases_ptr,
                                      cuComplex* result_ptr,
                                      uint32_t num_antennas, uint32_t num_beams)
    {

      // We work with cuComplex type for most DSP, but Eigen only knows the equivalent std lib type
      // so we cast to it for this context.
      auto samples_cast = reinterpret_cast<std::complex<float>*>(samples_ptr);
      auto phases_cast = reinterpret_cast<std::complex<float>*>(phases_ptr);

      // All we do here is map an existing set of memory to a structure that Eigen uses.
      Eigen::MatrixXcf samps = Eigen::Map<Eigen::Matrix<std::complex<float>,
                                                        Eigen::Dynamic,
                                                        Eigen::Dynamic,
                                                        Eigen::RowMajor>>(samples_cast,
                                                                          num_antennas,
                                                                          num_samples);
      Eigen::MatrixXcf phases = Eigen::Map<Eigen::Matrix<std::complex<float>,
                                                          Eigen::Dynamic,
                                                          Eigen::Dynamic,
                                                          Eigen::RowMajor>>(phases_cast,
                                                                            num_beams,
                                                                            num_antennas);

      // Result matrix has dimensions beams x num_samples. This means one set of samples for
      // each beam dir. Eigen overloads the * operator so we dont need to implement any matrix
      // work ourselves.
      auto result = phases * samps;

      // This piece of code just transforms the Eigen result back into our flat vector.
      auto beamformed_cast = reinterpret_cast<std::complex<float>*>(result_ptr);
      Eigen::Map<Eigen::Matrix<std::complex<float>, Eigen::Dynamic,
                                Eigen::Dynamic, Eigen::RowMajor>>(beamformed_cast, result.rows(),
                                                                  result.cols()) = result;
    };

    auto main_phase_offset = 0;
    auto main_results_offset = 0;

    // Now we calculate the offsets into the samples, phases, and results vector for each
    // RX frequency. Each RX frequency could have a different number of beams, so we increment
    // the phase and results offsets based off the accumulated number of beams. Once we have the
    // offsets, we can call the beamforming lambda.
    for (uint32_t rx_freq_num=0; rx_freq_num<beam_direction_counts.size(); rx_freq_num++) {

      auto num_beams = beam_direction_counts[rx_freq_num];

      // Increment to start of new frequency dataset.
      auto main_sample_offset = num_samples * (num_main_ants + num_intf_ants) * rx_freq_num;
      auto main_sample_ptr = filtered_samples.data() + main_sample_offset;

      auto main_phase_ptr = phases.data() + main_phase_offset;

      auto main_results_ptr = beamformed_samples_main.data() + main_results_offset;

      beamform_from_offsets(main_sample_ptr, main_phase_ptr, main_results_ptr,
                            num_main_ants, num_beams);

      // Only need to worry about beamforming the interferometer if its being used.
      if (num_intf_ants > 0) {

        // Skip the main array samples.
        auto intf_sample_offset = main_sample_offset + (num_samples * num_main_ants);
        auto intf_sample_ptr = filtered_samples.data() + intf_sample_offset;

        auto intf_phase_offset = main_phase_offset + (num_beams * num_main_ants);
        auto intf_phase_ptr = phases.data() + intf_phase_offset;

        // Result offsets will be the same. Each main and intf will have one set of samples for
        // each beam.
        auto intf_results_offset = main_results_offset;
        auto intf_results_ptr = beamformed_samples_intf.data() + intf_results_offset;

        beamform_from_offsets(intf_sample_ptr, intf_phase_ptr, intf_results_ptr,
                              num_intf_ants, num_beams);
      }

      //Possibly non uniform striding means we incremement the offset as we go.
      main_phase_offset += num_beams * (num_main_ants + num_intf_ants);
      main_results_offset += num_beams * num_samples;
    }

  }
  /**
   * @brief      Creates a data packet of processed data.
   *
   * @param      pd    A processeddata protobuf object.
   * @param      dp    A pointer to the DSPCore object with data to be extracted.
   *
   * This function extracts the processed data into a protobuf that data write can use.
   */
  void create_processed_data_packet(processeddata::ProcessedData &pd, DSPCore* dp)
  {

    std::vector<cuComplex> output_samples;

    std::vector<uint32_t> samps_per_stage = {dp->get_num_rf_samples(),
                                             dp->get_num_first_stage_samples_per_antenna(),
                                             dp->get_num_second_stage_samples_per_antenna(),
                                             dp->get_num_third_stage_samples_per_antenna(),
                                             dp->get_num_fourth_stage_samples_per_antenna()};
    std::vector<uint32_t> taps_per_stage = {dp->dsp_filters->get_num_first_stage_taps(),
                                            dp->dsp_filters->get_num_second_stage_taps(),
                                            dp->dsp_filters->get_num_third_stage_taps(),
                                            dp->dsp_filters->get_num_fourth_stage_taps()};

    drop_bad_samples(dp->get_host_output_h(), output_samples, samps_per_stage, taps_per_stage,
                     dp->get_num_antennas(), dp->get_rx_freqs().size());

    // For each antenna, for each frequency.
    auto num_samples_after_dropping = output_samples.size()/
                                      (dp->get_num_antennas()*dp->get_rx_freqs().size());




    auto total_beam_dirs = 0;
    auto beam_direction_counts = dp->get_beam_direction_counts();
    for(auto &beam_count : beam_direction_counts) {
      total_beam_dirs += beam_count;
    }

    std::vector<cuComplex> beamformed_samples_main(total_beam_dirs * num_samples_after_dropping);
    std::vector<cuComplex> beamformed_samples_intf(total_beam_dirs * num_samples_after_dropping);

    TIMEIT_IF_TRUE_OR_DEBUG(false,"Beamforming time: ",
      {
      auto beam_phases = dp->get_beam_phases();
      beamform_samples(output_samples, beamformed_samples_main, beamformed_samples_intf,
                        beam_phases,
                        dp->sig_options.get_main_antenna_count(),
                        dp->sig_options.get_interferometer_antenna_count(),
                        beam_direction_counts,
                        num_samples_after_dropping);
      }
    );

    // We have a lambda to extract the starting pointers of each set of output samples so that
    // we can use a consistent function to write either rf samples or stage data.
    auto make_ptrs_vec = [](cuComplex* output_p, uint32_t num_freqs, uint32_t num_antennas,
                              uint32_t num_samps_per_antenna)
    {
      auto stage_samps_per_set = num_antennas * num_samps_per_antenna;

      std::vector<std::vector<cuComplex*>> ptrs;
      for (uint32_t freq=0; freq<num_freqs; freq++) {
        std::vector<cuComplex*> stage_ptrs;
        for(uint32_t antenna=0; antenna<num_antennas; antenna++) {
          auto idx = freq * stage_samps_per_set + antenna * num_samps_per_antenna;
          stage_ptrs.push_back(output_p + idx);
        }
        ptrs.push_back(stage_ptrs);
      }

      return ptrs;
    };

    #ifdef ENGINEERING_DEBUG
      auto stage_1_ptrs = make_ptrs_vec(dp->get_first_stage_output_h(), dp->get_rx_freqs().size(),
                            dp->get_num_antennas(),dp->get_num_first_stage_samples_per_antenna());

      auto stage_2_ptrs = make_ptrs_vec(dp->get_second_stage_output_h(), dp->get_rx_freqs().size(),
                            dp->get_num_antennas(),dp->get_num_second_stage_samples_per_antenna());

      auto stage_3_ptrs = make_ptrs_vec(dp->get_third_stage_output_h(), dp->get_rx_freqs().size(),
                              dp->get_num_antennas(),dp->get_num_third_stage_samples_per_antenna());

      auto stage_4_ptrs = make_ptrs_vec(dp->get_fourth_stage_output_h(), dp->get_rx_freqs().size(),
                              dp->get_num_antennas(),dp->get_num_fourth_stage_samples_per_antenna());
    #endif

    auto output_ptrs = make_ptrs_vec(output_samples.data(), dp->get_rx_freqs().size(),
                          dp->get_num_antennas(), num_samples_after_dropping);

    auto beamformed_offset = 0;
    for(uint32_t i=0; i<dp->get_rx_freqs().size(); i++) {
      auto dataset = pd.add_outputdataset();
      // This lambda adds the stage data to the processed data for debug purposes.
      auto add_debug_data = [dataset,i](std::string stage_name, std::vector<cuComplex*> &data_ptrs,
                                          uint32_t num_antennas, uint32_t num_samps_per_antenna)
      {
        auto debug_samples = dataset->add_debugsamples();

        debug_samples->set_stagename(stage_name);
        for (uint32_t j=0; j<num_antennas; j++){
          auto antenna_data = debug_samples->add_antennadata();
          for(uint32_t k=0; k<num_samps_per_antenna; k++) {
            auto antenna_samp = antenna_data->add_antennasamples();
            antenna_samp->set_real(data_ptrs[j][k].x);
            antenna_samp->set_imag(data_ptrs[j][k].y);
          } // close loop over samples
        } // close loop over antennas
      };

      // Add our beamformed IQ data to the processed data packet that gets sent to data_write.
      for (uint32_t beam_count=0; beam_count<beam_direction_counts[i]; beam_count++) {
        auto beam = dataset->add_beamformedsamples();
        beam->set_beamnum(beam_count);

        for (uint32_t sample=0; sample<num_samples_after_dropping; sample++){
          auto main_sample = beam->add_mainsamples();
          main_sample->set_real(beamformed_samples_main[beamformed_offset + sample].x);
          main_sample->set_imag(beamformed_samples_main[beamformed_offset + sample].y);

          if (dp->sig_options.get_interferometer_antenna_count() > 0) {
            auto intf_sample = beam->add_intfsamples();
            intf_sample->set_real(beamformed_samples_intf[beamformed_offset + sample].x);
            intf_sample->set_imag(beamformed_samples_intf[beamformed_offset + sample].y);
          }
        } // close loop over samples.
      } // close loop over beams.

      // Keep track of offsets as we move along frequencies. Different frequencies can have
      // different beams.
      beamformed_offset += beam_direction_counts[i];


      #ifdef ENGINEERING_DEBUG
        add_debug_data("stage_1",stage_1_ptrs[i],dp->get_num_antennas(),
                    dp->get_num_first_stage_samples_per_antenna());
        add_debug_data("stage_2",stage_2_ptrs[i],dp->get_num_antennas(),
                    dp->get_num_second_stage_samples_per_antenna());
        add_debug_data("stage_3",stage_3_ptrs[i],dp->get_num_antennas(),
                    dp->get_num_third_stage_samples_per_antenna());
        add_debug_data("stage_4",stage_4_ptrs[i],dp->get_num_antennas(),
            dp->get_num_fourth_stage_samples_per_antenna());
      #endif

      add_debug_data("output_samples", output_ptrs[i], dp->get_num_antennas(),
        num_samples_after_dropping);

      dataset->set_slice_id(dp->get_slice_ids()[i]);
      DEBUG_MSG("Created dataset for sequence #" << COLOR_RED(dp->get_sequence_num()));
    } // close loop over frequencies.

    pd.set_rf_samples_location(dp->get_shared_memory_name());
    pd.set_sequence_num(dp->get_sequence_num());
    pd.set_rx_sample_rate(dp->get_rx_rate());
    pd.set_output_sample_rate(dp->get_output_sample_rate());
    pd.set_processing_time(dp->get_decimate_timing());
    pd.set_initialization_time(dp->get_driver_initialization_time());
    pd.set_sequence_start_time(dp->get_sequence_start_time());
  }

  /**
   * @brief      Spawns the postprocessing work after all work in the CUDA stream is completed.
   *
   * @param[in]  stream           CUDA stream this callback is associated with.
   * @param[in]  status           Error status of CUDA work in the stream.
   * @param[in]  processing_data  A pointer to the DSPCore associated with this CUDA stream.
   *
   * The callback itself cannot call anything CUDA related as it may deadlock. It can, however
   * spawn a new thread and then exit gracefully, allowing the thread to do the work.
   */
  void CUDART_CB postprocess(cudaStream_t stream, cudaError_t status, void *processing_data)
  {
    gpuErrchk(status);

    auto pp = [processing_data]()
    {
      auto dp = static_cast<DSPCore*>(processing_data);

      dp->stop_timing();
      dp->send_timing();


      processeddata::ProcessedData pd;

      TIMEIT_IF_TRUE_OR_DEBUG(false, "Fill + send processed data time ",
        [&]() {
          create_processed_data_packet(pd,dp);
          dp->send_processed_data(pd);
        }()
      );

      DEBUG_MSG("Cuda kernel timing: " << COLOR_GREEN(dp->get_decimate_timing()) << "ms");
      DEBUG_MSG("Complete process timing: " << COLOR_GREEN(dp->get_total_timing()) << "ms");
      auto sq_num = dp->get_sequence_num();
      delete dp;

      DEBUG_MSG(COLOR_RED("Deleted DP in postprocess for sequence #" << sq_num
                  << ". Thread should terminate here."));
    };

    std::thread start_pp(pp);
    start_pp.detach();
  }

}


/**
 * @brief      Gets the properties of each GPU in the system.
 *
 * @return     The gpu properties.
 */
std::vector<cudaDeviceProp> get_gpu_properties()
{
  std::vector<cudaDeviceProp> gpu_properties;
  int num_devices = 0;

  gpuErrchk(cudaGetDeviceCount(&num_devices));

  for(int i=0; i< num_devices; i++) {
      cudaDeviceProp properties;
      gpuErrchk(cudaGetDeviceProperties(&properties, i));
      gpu_properties.push_back(properties);
  }

  return gpu_properties;
}

/**
 * @brief      Prints the properties of each cudaDeviceProp in the vector.
 *
 * @param[in]  gpu_properties  A vector of cudaDeviceProp structs.
 *
 * More info on properties and calculations here:
 * https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
 */
void print_gpu_properties(std::vector<cudaDeviceProp> gpu_properties) {
  for(auto i : gpu_properties) {
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "Device name: " << i.name);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Max grid size x: " << i.maxGridSize[0]);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Max grid size y: " << i.maxGridSize[1]);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Max grid size z: " << i.maxGridSize[2]);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Max threads per block: "
                << i.maxThreadsPerBlock);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Max size of block dimension x: "
                << i.maxThreadsDim[0]);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Max size of block dimension y: "
                << i.maxThreadsDim[1]);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Max size of block dimension z: "
                << i.maxThreadsDim[2]);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Memory Clock Rate (GHz): "
                << i.memoryClockRate/1e6);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Memory Bus Width (bits): "
                << i.memoryBusWidth);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Peak Memory Bandwidth (GB/s): "
                << 2.0*i.memoryClockRate*(i.memoryBusWidth/8)/1.0e6);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Max shared memory per block: "
                << i.sharedMemPerBlock);
    RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "  Warpsize: " << i.warpSize);
  }
}


///TODO(keith): update docstring
/**
 * @brief      Initializes the parameters needed in order to do asynchronous DSP processing.
 *
 * @param      ack_s         A pointer to the socket used for acknowledging when the transfer of RF
 *                           samples has completed.
 * @param[in]  timing_s      A pointer to the socket used for reporting GPU kernel timing.
 * @param[in]  sq_num        The pulse sequence number for which will be acknowledged.
 * @param[in]  shr_mem_name  The char string used to open a section of shared memory with RF
 *                           samples.
 *
 * The constructor creates a new CUDA stream and initializes the timing events. It then opens
 * the shared memory with the received RF samples for a pulse sequence.
 */
DSPCore::DSPCore(zmq::socket_t *ack_socket, zmq::socket_t *timing_socket, zmq::socket_t *data_socket,
                  SignalProcessingOptions &sig_options, uint32_t sequence_num,
                  double rx_rate, double output_sample_rate, std::vector<double> rx_freqs, Filtering *dsp_filters,
                  std::vector<cuComplex> beam_phases, std::vector<uint32_t> beam_direction_counts,
                  double driver_initialization_time, double sequence_start_time,
                  std::vector<uint32_t> slice_ids) :
  sequence_num(sequence_num),
  rx_rate(rx_rate),
  output_sample_rate(output_sample_rate),
  ack_socket(ack_socket),
  timing_socket(timing_socket),
  data_socket(data_socket),
  rx_freqs(rx_freqs),
  sig_options(sig_options),
  dsp_filters(dsp_filters),
  beam_phases(beam_phases),
  beam_direction_counts(beam_direction_counts),
  driver_initialization_time(driver_initialization_time),
  sequence_start_time(sequence_start_time),
  slice_ids(slice_ids)
{

  //https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
  gpuErrchk(cudaStreamCreate(&stream));
  gpuErrchk(cudaEventCreate(&initial_start));
  gpuErrchk(cudaEventCreate(&kernel_start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk(cudaEventCreate(&mem_transfer_end));
  gpuErrchk(cudaEventRecord(initial_start, stream));

  shm = SharedMemoryHandler(random_string(20));


}

/**
 * @brief      Frees all associated pointers, events, and streams. Removes and deletes shared
 *             memory.
 */
DSPCore::~DSPCore()
{
  gpuErrchk(cudaEventDestroy(initial_start));
  gpuErrchk(cudaEventDestroy(kernel_start));
  gpuErrchk(cudaEventDestroy(stop));
  gpuErrchk(cudaFree(freqs_d));
  gpuErrchk(cudaFree(rf_samples_d));
  gpuErrchk(cudaFree(first_stage_bp_filters_d));
  gpuErrchk(cudaFree(second_stage_filter_d));
  gpuErrchk(cudaFree(third_stage_filter_d));
  gpuErrchk(cudaFree(fourth_stage_filter_d));
  gpuErrchk(cudaFree(first_stage_output_d));
  gpuErrchk(cudaFree(second_stage_output_d));
  gpuErrchk(cudaFree(third_stage_output_d));
  gpuErrchk(cudaFree(fourth_stage_output_d));
  #ifdef ENGINEERING_DEBUG
    gpuErrchk(cudaFreeHost(first_stage_output_h));
    gpuErrchk(cudaFreeHost(second_stage_output_h));
    gpuErrchk(cudaFreeHost(third_stage_output_h));
    gpuErrchk(cudaFreeHost(fourth_stage_output_h));
  #endif
  gpuErrchk(cudaFreeHost(host_output_h));
  gpuErrchk(cudaStreamDestroy(stream));

}

/**
 * @brief      Allocates device memory for the RF samples and then copies them to device.
 *
 * @param[in]  total_antennas         The total number of antennas.
 * @param[in]  num_samples_needed     The number of samples needed from each antenna ringbuffer.
 * @param[in]  extra_samples          The number of extra samples needed for filter propagation.
 * @param[in]  offset_to_first_pulse  Offset from sequence start to center of first pulse.
 * @param[in]  time_zero              The time the driver began collecting samples. seconds since
 *                                    epoch.
 * @param[in]  start_time             The start time of the pulse sequence. seconds since epoch.
 * @param[in]  ringbuffer_size        The ringbuffer size in number of samples.
 * @param      ringbuffer_ptrs_start  A vector of pointers to the start of each antenna ringbuffer.
 *
 * Samples are being stored in a shared memory ringbuffer. This function calculates where to index
 * into the ringbuffer for samples and copies them to the gpu. This function will also copy the
 * samples to a shared memory section that data write, or another process can access in order to
 * work with the raw RF samples.
 */
void DSPCore::allocate_and_copy_rf_samples(uint32_t total_antennas, uint32_t num_samples_needed,
                                int64_t extra_samples, uint32_t offset_to_first_pulse,
                                double time_zero, double start_time,
                                uint64_t ringbuffer_size,
                                std::vector<cuComplex*> &ringbuffer_ptrs_start)
{


  size_t rf_samples_size = total_antennas * num_samples_needed * sizeof(cuComplex);
  shm.create_shr_mem(rf_samples_size);
  gpuErrchk(cudaMalloc(&rf_samples_d, rf_samples_size));

  auto sample_time_diff = start_time - time_zero;
  auto sample_in_time = (sample_time_diff * rx_rate) +
                      offset_to_first_pulse -
                      extra_samples;
  auto start_sample = int64_t(std::fmod(sample_in_time, ringbuffer_size));

  if ((start_sample) < 0) {
   start_sample += ringbuffer_size;
  }

  if ((start_sample + num_samples_needed) > ringbuffer_size) {
    for (uint32_t i=0; i<total_antennas; i++) {
      auto first_piece = ringbuffer_size - start_sample;
      auto second_piece = num_samples_needed - first_piece;

      auto first_dest = rf_samples_d + (i*num_samples_needed);
      auto second_dest = rf_samples_d + (i*num_samples_needed) + (first_piece);

      auto first_src = ringbuffer_ptrs_start[i] + start_sample;
      auto second_src = ringbuffer_ptrs_start[i];

      gpuErrchk(cudaMemcpyAsync(first_dest, first_src, first_piece * sizeof(cuComplex),
                                 cudaMemcpyHostToDevice, stream));
      gpuErrchk(cudaMemcpyAsync(second_dest, second_src, second_piece * sizeof(cuComplex),
                                 cudaMemcpyHostToDevice, stream));

      auto mem_cast = static_cast<cuComplex*>(shm.get_shrmem_addr());
      auto first_dest_h = mem_cast + (i*num_samples_needed);
      auto second_dest_h = mem_cast + (i*num_samples_needed) + (first_piece);

      memcpy(first_dest_h, first_src, first_piece * sizeof(cuComplex));
      memcpy(second_dest_h, second_src, second_piece * sizeof(cuComplex));
    }

  }
  else {
    for (uint32_t i=0; i<total_antennas; i++) {
      auto dest = rf_samples_d + (i*num_samples_needed);
      auto src = ringbuffer_ptrs_start[i] + start_sample;

      gpuErrchk(cudaMemcpyAsync(dest, src, num_samples_needed * sizeof(cuComplex),
        cudaMemcpyHostToDevice, stream));

      auto mem_cast = static_cast<cuComplex*>(shm.get_shrmem_addr());
      auto dest_h = mem_cast + (i*num_samples_needed);
      memcpy(dest_h, src, num_samples_needed * sizeof(cuComplex));
    }
  }


}

/**
 * @brief      Allocates device memory for the filtering frequencies and then copies them to device.
 *
 * @param      freqs      A pointer to the filtering freqs.
 * @param[in]  num_freqs  The number of freqs.
 */
void DSPCore::allocate_and_copy_frequencies(void *freqs, uint32_t num_freqs) {
  size_t freqs_size = num_freqs * sizeof(double);
  gpuErrchk(cudaMalloc(&freqs_d, freqs_size));
  gpuErrchk(cudaMemcpyAsync(freqs_d, freqs, freqs_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocates device memory for the first stage filters and then copies them to the
 *             device.
 *
 * @param[in]  taps        A pointer to the first stage filter taps.
 * @param[in]  total_taps  The total number of taps for all filters.
 */
void DSPCore::allocate_and_copy_first_stage_filters(void *taps, uint32_t total_taps)
{
  size_t first_stage_bp_filters_size = total_taps * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&first_stage_bp_filters_d, first_stage_bp_filters_size));
  gpuErrchk(cudaMemcpyAsync(first_stage_bp_filters_d, taps,
        first_stage_bp_filters_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocates device memory for the second stage filter and then copies it to the
 *             device.
 *
 * @param[in]  taps        A pointer to the second stage filter taps.
 * @param[in]  total_taps  The total number of taps for all filters.
 */
void DSPCore::allocate_and_copy_second_stage_filter(void *taps, uint32_t total_taps)
{
  size_t second_stage_filter_size = total_taps * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&second_stage_filter_d, second_stage_filter_size));
  gpuErrchk(cudaMemcpyAsync(second_stage_filter_d, taps,
         second_stage_filter_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocates device memory for the third stage filter and then copies it to the
 *             device.
 *
 * @param[in]  taps        A pointer to the third stage filters.
 * @param[in]  total_taps  The total number of taps for all filters.
 */
void DSPCore::allocate_and_copy_third_stage_filter(void *taps, uint32_t total_taps)
{
  size_t third_stage_filter_size = total_taps * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&third_stage_filter_d, third_stage_filter_size));
  gpuErrchk(cudaMemcpyAsync(third_stage_filter_d, taps,
        third_stage_filter_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocates device memory for the fourth stage filter and then copies it to the
 *             device.
 *
 * @param[in]  taps        A pointer to the fourth stage filters.
 * @param[in]  total_taps  The total number of taps for all filters.
 */
void DSPCore::allocate_and_copy_fourth_stage_filter(void *taps, uint32_t total_taps)
{
  size_t fourth_stage_filter_size = total_taps * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&fourth_stage_filter_d, fourth_stage_filter_size));
  gpuErrchk(cudaMemcpyAsync(fourth_stage_filter_d, taps,
        fourth_stage_filter_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocates device memory for the output of the first stage filters.
 *
 * @param[in]  num_first_stage_output_samples  The total number of output samples from first
 *                                             stage.
 */
void DSPCore::allocate_first_stage_output(uint32_t num_first_stage_output_samples)
{
  size_t first_stage_output_size = num_first_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&first_stage_output_d, first_stage_output_size));
}

/**
 * @brief      Allocates device memory for the output of the second stage filters.
 *
 * @param[in]  num_second_stage_output_samples  The total number of output samples from second
 *             stage.
 */
void DSPCore::allocate_second_stage_output(uint32_t num_second_stage_output_samples)
{
  size_t second_stage_output_size = num_second_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&second_stage_output_d, second_stage_output_size));
}

/**
 * @brief      Allocates device memory for the output of the third stage filters.
 *
 * @param[in]  num_third_stage_output_samples  The total number of output samples from third
 *                                             stage.
 */
void DSPCore::allocate_third_stage_output(uint32_t num_third_stage_output_samples)
{
  size_t third_stage_output_size = num_third_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&third_stage_output_d, third_stage_output_size));
}

/**
 * @brief      Allocates device memory for the output of the fourth stage filters.
 *
 * @param[in]  num_fourth_stage_output_samples The total number of output samples from fourth
 *                                             stage.
 */
void DSPCore::allocate_fourth_stage_output(uint32_t num_fourth_stage_output_samples)
{
  size_t fourth_stage_output_size = num_fourth_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&fourth_stage_output_d, fourth_stage_output_size));
}

/**
 * @brief      Allocates host memory for final decimated samples and copies from device to host.
 *
 * @param[in]  num_host_samples  Number of host samples to copy back from device.
 */
void DSPCore::allocate_and_copy_host_output(uint32_t num_host_samples)
{
  size_t host_output_size = num_host_samples * sizeof(cuComplex);
  gpuErrchk(cudaHostAlloc(&host_output_h, host_output_size, cudaHostAllocDefault));
  gpuErrchk(cudaMemcpyAsync(host_output_h, fourth_stage_output_d,
        host_output_size, cudaMemcpyDeviceToHost,stream));
}


/**
 * @brief      Allocates host memory for the first stage samples and copies from device to host.
 *
 * @param[in]  num_first_stage_output_samples  The number of first stage output samples.
 */
void DSPCore::allocate_and_copy_first_stage_host(uint32_t num_first_stage_output_samples)
{
  size_t host_output_size = num_first_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaHostAlloc(&first_stage_output_h, host_output_size, cudaHostAllocDefault));
  gpuErrchk(cudaMemcpyAsync(first_stage_output_h, first_stage_output_d,
        host_output_size, cudaMemcpyDeviceToHost,stream));
}

/**
 * @brief      Allocates host memory for the second stage samples and copies from device to host.
 *
 * @param[in]  num_second_stage_output_samples  The number of second stage output samples.
 */
void DSPCore::allocate_and_copy_second_stage_host(uint32_t num_second_stage_output_samples)
{
  size_t host_output_size = num_second_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaHostAlloc(&second_stage_output_h, host_output_size, cudaHostAllocDefault));
  gpuErrchk(cudaMemcpyAsync(second_stage_output_h, second_stage_output_d,
        host_output_size, cudaMemcpyDeviceToHost,stream));
}

/**
 * @brief      Allocates host memory for the third stage samples and copies from device to host.
 *
 * @param[in]  num_third_stage_output_samples  The number of third stage output samples.
 */
void DSPCore::allocate_and_copy_third_stage_host(uint32_t num_third_stage_output_samples)
{
  size_t host_output_size = num_third_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaHostAlloc(&third_stage_output_h, host_output_size, cudaHostAllocDefault));
  gpuErrchk(cudaMemcpyAsync(third_stage_output_h, third_stage_output_d,
        host_output_size, cudaMemcpyDeviceToHost,stream));
}

/**
 * @brief      Allocates host memory for the fourth stage samples and copies from device to host.
 *
 * @param[in]  num_fourth_stage_output_samples  The number of fourth stage output samples.
 */
void DSPCore::allocate_and_copy_fourth_stage_host(uint32_t num_fourth_stage_output_samples)
{
  size_t host_output_size = num_fourth_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaHostAlloc(&fourth_stage_output_h, host_output_size, cudaHostAllocDefault));
  gpuErrchk(cudaMemcpyAsync(fourth_stage_output_h, fourth_stage_output_d,
        host_output_size, cudaMemcpyDeviceToHost,stream));
}

/**
 * @brief      Stops the timers that the constructor starts.
 */
void DSPCore::stop_timing()
{
  gpuErrchk(cudaEventRecord(stop, stream));
  gpuErrchk(cudaEventSynchronize(stop));

  gpuErrchk(cudaEventElapsedTime(&total_process_timing_ms, initial_start, stop));
  gpuErrchk(cudaEventElapsedTime(&decimate_kernel_timing_ms, kernel_start, stop));
  gpuErrchk(cudaEventElapsedTime(&mem_time_ms, initial_start, mem_transfer_end));
  RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "Cuda memcpy time: "
    << COLOR_GREEN(mem_time_ms) << "ms");
  RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "Decimate time: "
    << COLOR_GREEN(decimate_kernel_timing_ms) << "ms");

}

/**
 * @brief      Sends the GPU kernel timing to the radar control.
 *
 * The timing here is used as a rate limiter, so that the GPU doesn't become backlogged with data.
 * If the GPU is overburdened, this will result in less averages, but the system wont crash.
 */
void DSPCore::send_timing()
{
  sigprocpacket::SigProcPacket sp;
  sp.set_kerneltime(decimate_kernel_timing_ms);
  sp.set_sequence_num(sequence_num);

  std::string s_msg_str;
  sp.SerializeToString(&s_msg_str);

  auto request = RECV_REQUEST(*timing_socket, sig_options.get_brian_dspend_identity());
  SEND_REPLY(*timing_socket, sig_options.get_brian_dspend_identity(), s_msg_str);

  DEBUG_MSG(COLOR_RED("Sent timing after processing with sequence #" << sequence_num));

}


/**
 * @brief      Add the postprocessing callback to the stream.
 *
 */
void DSPCore::cuda_postprocessing_callback(std::vector<double> freqs, uint32_t total_antennas,
                                            uint32_t num_samples_rf,
                                            uint32_t num_output_samples_per_antenna_1,
                                            uint32_t num_output_samples_per_antenna_2,
                                            uint32_t num_output_samples_per_antenna_3,
                                            uint32_t num_output_samples_per_antenna_4)
{
    #ifdef ENGINEERING_DEBUG
      auto total_output_samples_1 = num_output_samples_per_antenna_1 * rx_freqs.size() *
                                      total_antennas;
      auto total_output_samples_2 = num_output_samples_per_antenna_2 * rx_freqs.size() *
                                      total_antennas;
      auto total_output_samples_3 = num_output_samples_per_antenna_3 * rx_freqs.size() *
                                      total_antennas;
      auto total_output_samples_4 = num_output_samples_per_antenna_4 * rx_freqs.size() *
                                      total_antennas;

      allocate_and_copy_first_stage_host(total_output_samples_1);
      allocate_and_copy_second_stage_host(total_output_samples_2);
      allocate_and_copy_third_stage_host(total_output_samples_3);
      allocate_and_copy_fourth_stage_host(total_output_samples_4);

    #endif

    rx_freqs = freqs;
    num_rf_samples = num_samples_rf;
    num_antennas = total_antennas;
    num_first_stage_samples_per_antenna = num_output_samples_per_antenna_1;
    num_second_stage_samples_per_antenna = num_output_samples_per_antenna_2;
    num_third_stage_samples_per_antenna = num_output_samples_per_antenna_3;
    num_fourth_stage_samples_per_antenna = num_output_samples_per_antenna_4;

    gpuErrchk(cudaStreamAddCallback(stream, postprocess, this, 0));

    DEBUG_MSG(COLOR_RED("Added stream callback for sequence #" << sequence_num));
}

/**
 * @brief      Sends the acknowledgment to the radar control that the RF samples have been
 *             transfered.
 *
 * RF samples of one pulse sequence can be transfered asynchronously while samples of another are
 * being processed. This means that it is possible to start running a new pulse sequence in the
 * driver as soon as the samples are copied. The asynchronous nature means only timing constraint
 * is the time needed to run the GPU kernels for decimation.
 */
void DSPCore::send_ack()
{
  sigprocpacket::SigProcPacket sp;
  sp.set_sequence_num(sequence_num);

  std::string s_msg_str;
  sp.SerializeToString(&s_msg_str);

  auto request = RECV_REQUEST(*ack_socket, sig_options.get_brian_dspbegin_identity());
  SEND_REPLY(*ack_socket, sig_options.get_brian_dspbegin_identity(), s_msg_str);

  DEBUG_MSG(COLOR_RED("Sent ack after copy for sequence_num #" << sequence_num));
}

/**
 * @brief      Sends a processed data packet to data write.
 *
 * @param      pd    A processeddata protobuf object.
 */
void DSPCore::send_processed_data(processeddata::ProcessedData &pd)
{
  std::string p_msg_str;
  pd.SerializeToString(&p_msg_str);

  SEND_REPLY(*data_socket, sig_options.get_dw_dsp_identity(), p_msg_str);

  DEBUG_MSG(COLOR_RED("Send processed data to data_write for sequence #" << sequence_num));
}


/**
 * @brief      Starts the timing before the GPU kernels execute.
 *
 */
void DSPCore::start_decimate_timing()
{
  gpuErrchk(cudaEventRecord(kernel_start, stream));
  gpuErrchk(cudaEventRecord(mem_transfer_end,stream));
}

/**
 * @brief      Adds the callback to the CUDA stream to acknowledge the RF samples have been copied.
 *
 */
void DSPCore::initial_memcpy_callback()
{
  gpuErrchk(cudaStreamAddCallback(stream, initial_memcpy_callback_handler, this, 0));
}


/**
 * @brief      Gets the device pointer to the RF samples.
 *
 * @return     The RF samples device pointer.
 */
cuComplex* DSPCore::get_rf_samples_p(){
  return rf_samples_d;
}

/**
 * @brief      Gets the host pointer to the RF samples.
 *
 * @return     The rf samples host pointer.
 */
std::vector<cuComplex> DSPCore::get_rf_samples_h() {
  return rf_samples_h;
}

/**
 * @brief      Gets the device pointer to the receive frequencies.
 *
 * @return     The frequencies device pointer.
 */
double* DSPCore::get_frequencies_p() {
  return freqs_d;
}

/**
 * @brief      Gets the device pointer to the first stage bandpass filters.
 *
 * @return     The first stage bandpass filters device pointer.
 */
cuComplex* DSPCore::get_first_stage_bp_filters_p(){
  return first_stage_bp_filters_d;
}

/**
 * @brief      Gets the device pointer to the second stage filters.
 *
 * @return     The second stage filter device pointer.
 */
cuComplex* DSPCore::get_second_stage_filter_p(){
  return second_stage_filter_d;
}

/**
 * @brief      Gets the device pointer to the third stage filters.
 *
 * @return     The third stage filter device pointer.
 */
cuComplex* DSPCore::get_third_stage_filter_p(){
  return third_stage_filter_d;
}

/**
 * @brief      Gets the device pointer to the fourth stage filters.
 *
 * @return     The fourth stage filter device pointer.
 */
cuComplex* DSPCore::get_fourth_stage_filter_p(){
  return fourth_stage_filter_d;
}

/**
 * @brief      Gets the device pointer to output of the first stage decimation.
 *
 * @return     The first stage output device pointer.
 */
cuComplex* DSPCore::get_first_stage_output_p(){
  return first_stage_output_d;
}

/**
 * @brief      Gets the device pointer to output of the second stage decimation.
 *
 * @return     The second stage output device pointer.
 */
cuComplex* DSPCore::get_second_stage_output_p(){
  return second_stage_output_d;
}

/**
 * @brief      Gets the device pointer to output of the third stage decimation.
 *
 * @return     The third stage output device pointer.
 */
cuComplex* DSPCore::get_third_stage_output_p(){
  return third_stage_output_d;
}

/**
 * @brief      Gets the device pointer to output of the fourth stage decimation.
 *
 * @return     The fourth stage output device pointer.
 */
cuComplex* DSPCore::get_fourth_stage_output_p(){
  return fourth_stage_output_d;
}

/**
 * @brief      Gets the host pointer to the output samples.
 *
 * @return     The host output pointer.
 */
cuComplex* DSPCore::get_host_output_h() {
  return host_output_h;
}

/**
 * @brief      Get the vector of host side frequencies.
 *
 * @return     The receive freqs vector.
 */
std::vector<double> DSPCore::get_rx_freqs()
{
  return rx_freqs;
}
/**
 * @brief      Gets the CUDA stream this DSPCore's work is associated to.
 *
 * @return     The CUDA stream.
 */
cudaStream_t DSPCore::get_cuda_stream(){
  return stream;
}

/**
 * @brief      Gets the total GPU process timing in milliseconds.
 *
 * @return     The total process timing.
 */
float DSPCore::get_total_timing()
{
  return total_process_timing_ms;
}

/**
 * @brief      Gets the total decimation timing in milliseconds.
 *
 * @return     The decimation timing.
 */
float DSPCore::get_decimate_timing()
{
  return decimate_kernel_timing_ms;
}

/**
 * @brief      Gets the host pointer for first stage output.
 *
 * @return     The first stage output host pointer.
 */
cuComplex* DSPCore::get_first_stage_output_h()
{
  return first_stage_output_h;
}

/**
 * @brief      Gets the host pointer for the second stage output.
 *
 * @return     The second stage output host pointer.
 */
cuComplex* DSPCore::get_second_stage_output_h()
{
  return second_stage_output_h;
}

/**
 * @brief      Gets the host pointer for the third stage output.
 *
 * @return     The third stage output host pointer.
 */
cuComplex* DSPCore::get_third_stage_output_h()
{
  return third_stage_output_h;
}

/**
 * @brief      Gets the host pointer for the fourth stage output.
 *
 * @return     The fourth stage output host pointer.
 */
cuComplex* DSPCore::get_fourth_stage_output_h()
{
  return fourth_stage_output_h;
}

/**
 * @brief      Gets the number of antennas.
 *
 * @return     The number of antennas.
 */
uint32_t DSPCore::get_num_antennas()
{
  return num_antennas;
}

/**
 * @brief      Gets the number of rf samples.
 *
 * @return     The number of rf samples.
 */
uint32_t DSPCore::get_num_rf_samples()
{
  return num_rf_samples;
}

/**
 * @brief      Gets the number first stage samples per antenna.
 *
 * @return     The number first stage samples per antenna.
 */
uint32_t DSPCore::get_num_first_stage_samples_per_antenna()
{
  return num_first_stage_samples_per_antenna;
}

/**
 * @brief      Gets the number second stage samples per antenna.
 *
 * @return     The number second stage samples per antenna.
 */
uint32_t DSPCore::get_num_second_stage_samples_per_antenna()
{
  return num_second_stage_samples_per_antenna;
}

/**
 * @brief      Gets the number third stage samples per antenna.
 *
 * @return     The number third stage samples per antenna.
 */
uint32_t DSPCore::get_num_third_stage_samples_per_antenna()
{
  return num_third_stage_samples_per_antenna;
}

/**
 * @brief      Gets the number fourth stage samples per antenna.
 *
 * @return     The number fourth stage samples per antenna.
 */
uint32_t DSPCore::get_num_fourth_stage_samples_per_antenna()
{
  return num_fourth_stage_samples_per_antenna;
}

/**
 * @brief      Gets the sequence number.
 *
 * @return     The sequence number.
 */
uint32_t DSPCore::get_sequence_num()
{
  return sequence_num;
}

/**
 * @brief      Gets the rx sample rate.
 *
 * @return     The rx sampling rate (samples per second).
 */
double DSPCore::get_rx_rate()
{
  return rx_rate;
}

/**
 * @brief      Gets the output sample rate.
 *
 * @return     The output decimated and filtered rate (samples per second).
 */
double DSPCore::get_output_sample_rate()
{
  return output_sample_rate;
}

/**
 * @brief     Gets the vector of beam phases.
 *
 * @return    The beam phases.
 */
std::vector<cuComplex> DSPCore::get_beam_phases()
{
  return beam_phases;
}

/**
 * @brief     Gets the vector of beam direction counts for each RX frequency.
 *
 * @return    The beam direction counts.
 */
std::vector<uint32_t> DSPCore::get_beam_direction_counts()
{
  return beam_direction_counts;
}

/**
 * @brief     Gets the name of the shared memory section.
 *
 * @return    The shared memory name string.
 */
std::string DSPCore::get_shared_memory_name()
{
  return shm.get_region_name();
}

/**
 * @brief      Gets the driver initialization timestamp.
 *
 * @return     The driver initialization timestamp.
 */
double DSPCore::get_driver_initialization_time()
{
  return driver_initialization_time;
}

/**
 * @brief      Gets the sequence start timestamp.
 *
 * @return     The sequence start timestamp.
 */
double DSPCore::get_sequence_start_time()
{
  return sequence_start_time;
}

/**
 * @brief      Gets the vector of slice identifiers.
 *
 * @return     The vector of slice identifiers.
 */
std::vector<uint32_t> DSPCore::get_slice_ids()
{
  return slice_ids;
}
