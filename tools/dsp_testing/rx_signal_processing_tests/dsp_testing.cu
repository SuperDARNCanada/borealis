/*

Copyright 2022 SuperDARN Canada

See LICENSE for details

  \file dsp_testing.cu
  This file contains the implementation for the all the needed GPU DSP work.
*/

#include "dsp_testing.hpp"

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <cuComplex.h>
#include <chrono>
#include <thread>
#include <numeric>
#include <complex>
#include <armadillo>
#include "utils/shared_macros/shared_macros.hpp"
//#include "rx_signal_processing/filtering.hpp"
//TODO(keith): decide on handing gpu errors
//TODO(keith): potentially add multigpu support

//This keep postprocess local to this file.
namespace {
  /**
   * @brief      Starts the timing after the RF samples have been copied.
   *
   * @param[in]  stream           CUDA stream this callback is associated with.
   * @param[in]  status           Error status of CUDA work in the stream.
   * @param[in]  processing_data  A pointer to the DSPCoreTesting associated with this CUDA stream.
   */
  void CUDART_CB initial_memcpy_callback_handler(cudaStream_t stream, cudaError_t status, void *processing_data)
  {
    gpuErrchk(status);

    auto imc = [processing_data]()
    {
      auto dp = static_cast<DSPCoreTesting*>(processing_data);
      dp->start_decimate_timing();
      RUNTIME_MSG(COLOR_RED("Finished initial memcpy handler for sequence. Thread should exit here"));
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
   * @param      dm_rates         The decimation rates of each stage.
   * @param[in]  num_antennas     The number of antennas.
   * @param[in]  num_freqs        The number of freqs.
   */
  void drop_bad_samples(cuComplex *input_samples, std::vector<cuComplex> &output_samples,
                        std::vector<uint32_t> &samps_per_stage,
                        std::vector<uint32_t> &taps_per_stage,
                        std::vector<uint32_t> &dm_rates,
                        uint32_t num_antennas, uint32_t num_freqs)
  {

    auto original_undropped_sample_count = samps_per_stage.back();
    auto original_samples_per_frequency = num_antennas * original_undropped_sample_count;

    // This accounts for the length of the filter extending past the length of input samples while
    // decimating.
    std::vector<uint32_t> bad_samples_per_stage;
    for (uint32_t i=0; i<dm_rates.size(); i++) {
      bad_samples_per_stage.push_back(uint32_t(std::floor(float(taps_per_stage[i]) /
                                                 float(dm_rates[i]))));
    }

    // Propagate the number of bad samples from the first stage through to the last stage.
    for (uint32_t i=1; i<bad_samples_per_stage.size(); i++) {
      bad_samples_per_stage[i] += std::ceil(float(bad_samples_per_stage[i-1])/(dm_rates[i]));
    }

    samps_per_stage.back() -= bad_samples_per_stage.back();
    auto samples_per_frequency = samps_per_stage.back() * num_antennas;

    output_samples.resize(num_freqs * samples_per_frequency);

    for (uint32_t freq_index=0; freq_index < num_freqs; freq_index++) {
      for (int i=0; i<num_antennas; i++){
        auto dest = output_samples.data() + (freq_index * samples_per_frequency) +
                    (i * samps_per_stage.back());
        auto src = input_samples + freq_index * (original_samples_per_frequency) +
                    (i * original_undropped_sample_count);
        auto num_bytes =  sizeof(cuComplex) * samps_per_stage.back();
        memcpy(dest, src, num_bytes);
      }
    }
  }


  /**
   * @brief      This method name is kept for ease of comparison with borealis/rx_signal_processing/dsp.cu.
   *             However, no processed_data_packet is used here.
   *
   * @param      dp    A pointer to the DSPCoreTesting object with data to be extracted.
   *
   * This function drops bad samples and writes the data to file.
   */
  void create_processed_data_packet(DSPCoreTesting* dp)
  {

    std::vector<cuComplex> output_samples;
    auto rx_slice_info = dp->get_slice_info();

    auto samples_per_antenna = dp->get_samples_per_antenna();

    // create a new vector with the number of input rf samples included. Basically the equivalent
    // of a list concat in Python.
    std::vector<uint32_t> samps_per_stage;
    samps_per_stage.push_back(dp->get_num_rf_samples());
    samps_per_stage.insert(samps_per_stage.end(),
                           samples_per_antenna.begin(),
                           samples_per_antenna.end());

    auto filter_taps = dp->get_filter_taps();
    std::vector<uint32_t> taps_per_stage(filter_taps.size());
    for (uint32_t i=0; i<filter_taps.size(); i++) {
      taps_per_stage[i] = filter_taps[i].size();
    }

    auto filter_outputs_h = dp->get_filter_outputs_h();
    auto dm_rates = dp->get_dm_rates();
    drop_bad_samples(filter_outputs_h.back(), output_samples, samps_per_stage, taps_per_stage,
                     dm_rates, dp->get_num_antennas(), rx_slice_info.size());

    // For each antenna, for each frequency.
    auto num_samples_after_dropping = output_samples.size()/
                                      (dp->get_num_antennas()*rx_slice_info.size());

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

    std::vector<std::vector<std::vector<cuComplex*>>> all_stage_ptrs;
    //#ifdef ENGINEERING_DEBUG - Removed this for testing purposes
      for (uint32_t i=0; i<filter_outputs_h.size(); i++) {
        auto ptrs = make_ptrs_vec(filter_outputs_h[i], rx_slice_info.size(),
                            dp->get_num_antennas(), samples_per_antenna[i]);
        all_stage_ptrs.push_back(ptrs);
      }
    //#endif

    auto output_ptrs = make_ptrs_vec(output_samples.data(), rx_slice_info.size(),
                          dp->get_num_antennas(), num_samples_after_dropping);

    // TODO(Remington): Figure out how to get this data to a useful form/file
    /*
    for(uint32_t slice_num=0; slice_num<rx_slice_info.size(); slice_num++) {
      auto dataset = pd.add_outputdataset();
      // This lambda adds the stage data to the processed data for debug purposes.
      auto add_debug_data = [dataset,slice_num](std::string stage_name,
                                                std::vector<cuComplex*> &data_ptrs,
                                                uint32_t num_antennas,
                                                uint32_t num_samps_per_antenna)
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
      for (uint32_t beam_count=0; beam_count<rx_slice_info[slice_num].beam_count; beam_count++) {
        auto beam = dataset->add_beamformedsamples();
        beam->set_beamnum(beam_count);

        for (uint32_t sample=0; sample<num_samples_after_dropping; sample++){
          auto main_sample = beam->add_mainsamples();
          auto beam_start = beam_count * num_samples_after_dropping;
          main_sample->set_real(beamformed_samples_main[slice_num][beam_start + sample].x);
          main_sample->set_imag(beamformed_samples_main[slice_num][beam_start + sample].y);

          if (dp->sig_options.get_interferometer_antenna_count() > 0) {
            auto intf_sample = beam->add_intfsamples();
            intf_sample->set_real(beamformed_samples_intf[slice_num][beam_start + sample].x);
            intf_sample->set_imag(beamformed_samples_intf[slice_num][beam_start + sample].y);
          }
        } // close loop over samples.
      } // close loop over beams.


      auto num_lags = rx_slice_info[slice_num].lags.size();
      auto num_ranges = rx_slice_info[slice_num].num_ranges;
      for (uint32_t beam_count=0; beam_count<rx_slice_info[slice_num].beam_count; beam_count++) {
        auto beam_offset = beam_count * (num_ranges * num_lags);

        for (uint32_t range=0; range<num_ranges; range++) {
          auto range_offset = range * num_lags;

          for (uint32_t lag=0; lag<num_lags; lag++) {
            auto mainacf = dataset->add_mainacf();
            auto val = main_acfs[slice_num][beam_offset + range_offset + lag];
            mainacf->set_real(val.x);
            mainacf->set_imag(val.y);

            if (dp->sig_options.get_interferometer_antenna_count() > 0) {
              auto xcf = dataset->add_xcf();
              auto intfacf = dataset->add_intacf();

              val = xcfs[slice_num][beam_offset + range_offset + lag];
              xcf->set_real(val.x);
              xcf->set_imag(val.y);

              val = intf_acfs[slice_num][beam_offset + range_offset + lag];
              intfacf->set_real(val.x);
              intfacf->set_imag(val.y);
            } // close intf scope
          } // close lag scope
        } // close range scope
      } // close beam scope

      // #ifdef ENGINEERING_DEBUG - Removed for testing purposes
        for (uint32_t j=0; j<all_stage_ptrs.size(); j++){
          auto stage_str = "stage_" + std::to_string(j);
          add_debug_data(stage_str, all_stage_ptrs[j][slice_num], dp->get_num_antennas(),
            samples_per_antenna[j]);
        }
      // #endif

      add_debug_data("antennas", output_ptrs[slice_num], dp->get_num_antennas(),
        num_samples_after_dropping);

      dataset->set_slice_id(rx_slice_info[slice_num].slice_id);
      dataset->set_num_ranges(rx_slice_info[slice_num].num_ranges);
      dataset->set_num_lags(rx_slice_info[slice_num].lags.size());

      DEBUG_MSG("Created dataset for sequence #" << COLOR_RED(dp->get_sequence_num()));
    } // close loop over frequencies (number of slices).

    pd.set_rf_samples_location(dp->get_shared_memory_name());
    pd.set_sequence_num(dp->get_sequence_num());
    pd.set_rx_sample_rate(dp->get_rx_rate());
    pd.set_output_sample_rate(dp->get_output_sample_rate());
    pd.set_processing_time(dp->get_decimate_timing());
    pd.set_initialization_time(dp->get_driver_initialization_time());
    pd.set_sequence_start_time(dp->get_sequence_start_time());
    */
  }

  /**
   * @brief      Spawns the postprocessing work after all work in the CUDA stream is completed.
   *
   * @param[in]  stream           CUDA stream this callback is associated with.
   * @param[in]  status           Error status of CUDA work in the stream.
   * @param[in]  processing_data  A pointer to the DSPCoreTesting associated with this CUDA stream.
   *
   * The callback itself cannot call anything CUDA related as it may deadlock. It can, however
   * spawn a new thread and then exit gracefully, allowing the thread to do the work.
   */
  void CUDART_CB postprocess(cudaStream_t stream, cudaError_t status, void *processing_data)
  {

    gpuErrchk(status);

    auto pp = [processing_data]()
    {
      auto dp = static_cast<DSPCoreTesting*>(processing_data);

      dp->stop_timing();

      TIMEIT_IF_TRUE_OR_DEBUG(true, "Fill + send processed data time ",
        [&]() {
          create_processed_data_packet(dp);
        }()
      );

      RUNTIME_MSG("Cuda kernel timing: " << COLOR_GREEN(dp->get_decimate_timing()) << "ms");
      RUNTIME_MSG("Complete process timing: " << COLOR_GREEN(dp->get_total_timing()) << "ms");
      delete dp;

      RUNTIME_MSG(COLOR_RED("Deleted DP in postprocess for sequence. Thread should terminate here."));
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


/**
 * @brief      Initializes the parameters needed in order to do asynchronous DSP processing.
 *
 * @param      sig_options                 The signal processing options.
 * @param[in]  sequence_num                The pulse sequence number for which will be acknowledged.
 * @param[in]  rx_rate                     The USRP sampling rate.
 * @param[in]  output_sample_rate          The final decimated output sample rate.
 * @param[in]  filter_taps                 The filter taps for each stage.
 * @param[in]  beam_phases                 The beam phases.
 * @param[in]  driver_initialization_time  The driver initialization time.
 * @param[in]  sequence_start_time         The sequence start time.
 * @param[in]  dm_rates                    The decimation rates.
 * @param[in]  slice_info                  The slice info given as a vector of rx_slice_test structs.
 *
 * The constructor creates a new CUDA stream and initializes the timing events. It then opens the
 * shared memory with the received RF samples for a pulse sequence.
 */
DSPCoreTesting::DSPCoreTesting(double rx_rate, double output_sample_rate, std::vector<std::vector<float>> filter_taps,
                               std::vector<uint32_t> dm_rates, std::vector<rx_slice_test> slice_info) :
  rx_rate(rx_rate),
  output_sample_rate(output_sample_rate),
  filter_taps(filter_taps),
  dm_rates(dm_rates),
  slice_info(slice_info)
{
  //https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
  gpuErrchk(cudaStreamCreate(&stream));
  gpuErrchk(cudaEventCreate(&initial_start));
  gpuErrchk(cudaEventCreate(&kernel_start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk(cudaEventCreate(&mem_transfer_end));
  gpuErrchk(cudaEventRecord(initial_start, stream));
}

/**
 * @brief      Frees all associated pointers, events, and streams. Removes and deletes shared
 *             memory.
 */
DSPCoreTesting::~DSPCoreTesting()
{
  gpuErrchk(cudaEventDestroy(initial_start));
  gpuErrchk(cudaEventDestroy(kernel_start));
  gpuErrchk(cudaEventDestroy(stop));
  gpuErrchk(cudaFree(freqs_d));
  gpuErrchk(cudaFree(rf_samples_d));
  gpuErrchk(cudaFree(bp_filters_d));
  for (auto &filter : lp_filters_d) {
    gpuErrchk(cudaFree(filter));
  }

  for (auto &filter_output : filter_outputs_d) {
    gpuErrchk(cudaFree(filter_output));
  }

  for (auto &filter_output : filter_outputs_h) {
    gpuErrchk(cudaFreeHost(filter_output));
  }

  gpuErrchk(cudaStreamDestroy(stream));

}

/**
 * @brief      Allocates device memory for the RF samples and then copies them to device.
 *
 * @param[in]  total_antennas         The total number of antennas.
 * @param[in]  num_samples_needed     The number of samples needed from each antenna ringbuffer.
 *
 * @param      input_samples          A pointer to the input samples.
 *
 * Samples are stored in a flat array, with all samples for the first channel coming before all
 * samples for the second channel, and so on.
 */
void DSPCoreTesting::allocate_and_copy_rf_samples(uint32_t total_antennas, uint32_t num_samples_needed,
                                                  void *input_samples)
{
  size_t rf_samples_size = total_antennas * num_samples_needed * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&rf_samples_d, rf_samples_size));
  gpuErrchk(cudaMemcpyAsync(rf_samples_d, input_samples, rf_samples_size, cudaMemcpyHostToDevice, stream));

}

/**
 * @brief      Allocates device memory for the filtering frequencies and then copies them to device.
 *
 * @param      freqs      A pointer to the filtering freqs.
 * @param[in]  num_freqs  The number of freqs.
 */
void DSPCoreTesting::allocate_and_copy_frequencies(void *freqs, uint32_t num_freqs) {
  size_t freqs_size = num_freqs * sizeof(double);
  gpuErrchk(cudaMalloc(&freqs_d, freqs_size));
  gpuErrchk(cudaMemcpyAsync(freqs_d, freqs, freqs_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocate and copy bandpass filters for all rx freqs to gpu.
 *
 * @param      taps        A pointer to the filter taps.
 * @param[in]  total_taps  The total amount of filter taps.
 */
void DSPCoreTesting::allocate_and_copy_bandpass_filters(void *taps, uint32_t total_taps)
{
  size_t bp_filters_size = total_taps * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&bp_filters_d, bp_filters_size));
  gpuErrchk(cudaMemcpyAsync(bp_filters_d, taps, bp_filters_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocate and copy a lowpass filter to the gpu.
 *
 * @param      taps        A pointer to the filter taps.
 * @param[in]  total_taps  The total amount of filter taps.
 */
void DSPCoreTesting::allocate_and_copy_lowpass_filter(void *taps, uint32_t total_taps)
{
  cuComplex *ptr_d;
  lp_filters_d.push_back(ptr_d);

  size_t filter_size = total_taps * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&lp_filters_d.back(), filter_size));
  gpuErrchk(cudaMemcpyAsync(lp_filters_d.back(), taps, filter_size, cudaMemcpyHostToDevice, stream));

}

/**
 * @brief      Gets the last filter output d.
 *
 * @return     The last filter output d.
 */
cuComplex* DSPCoreTesting::get_last_filter_output_d()
{
  return filter_outputs_d.back();
}

/**
 * @brief      Gets the last pointer stored in the lowpass filters vector.
 *
 * @return     The last lowpass filter pointer inserted into the vector.
 */
cuComplex* DSPCoreTesting::get_last_lowpass_filter_d() {
  return lp_filters_d.back();
}

/**
 * @brief      Gets the samples per antenna vector. Vector contains an element for each stage.
 *
 * @return     The samples per antenna vector.
 */
std::vector<uint32_t> DSPCoreTesting::get_samples_per_antenna() {
  return samples_per_antenna;
}

/**
 * @brief      The vector containing vectors of filter taps for each stage.
 *
 * @return     The filter taps vectors for each stage.
 */
std::vector<std::vector<float>> DSPCoreTesting::get_filter_taps() {
  return filter_taps;
}

/**
 * @brief      Allocate a filter output on the GPU.
 *
 * @param[in]  num_output_samples  The number output samples
 */
void DSPCoreTesting::allocate_output(uint32_t num_output_samples)
{
  cuComplex *ptr_d;
  filter_outputs_d.push_back(ptr_d);
  size_t output_size = num_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&filter_outputs_d.back(), output_size));

}

/**
 * @brief      Allocate a host pointer for decimation stage output and then copy data.
 *
 * @param[in]  num_output_samples  The number output samples needed.
 * @param      output_d            The device pointer from which to copy from.
 */
void DSPCoreTesting::allocate_and_copy_host(uint32_t num_output_samples, cuComplex *output_d)
{
  cuComplex *ptr_h;
  filter_outputs_h.push_back(ptr_h);

  size_t output_size = num_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaMallocHost(&filter_outputs_h.back(), output_size));
  gpuErrchk(cudaMemcpyAsync(filter_outputs_h.back(), output_d, output_size, cudaMemcpyDeviceToHost,stream));

}

/**
 * @brief      Stops the timers that the constructor starts.
 */
void DSPCoreTesting::stop_timing()
{
  gpuErrchk(cudaEventRecord(stop, stream));
  gpuErrchk(cudaEventSynchronize(stop));

  gpuErrchk(cudaEventElapsedTime(&total_process_timing_ms, initial_start, stop));
  gpuErrchk(cudaEventElapsedTime(&decimate_kernel_timing_ms, kernel_start, stop));
  gpuErrchk(cudaEventElapsedTime(&mem_time_ms, initial_start, mem_transfer_end));
  RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "Cuda memcpy time: " << COLOR_GREEN(mem_time_ms) << "ms");
  RUNTIME_MSG(COLOR_MAGENTA("SIGNAL PROCESSING: ") << "Decimate time: " << COLOR_GREEN(decimate_kernel_timing_ms) << "ms");
}

/**
 * @brief      Add the postprocessing callback to the stream.
 *
 * This function allocates the host space needed for filter stage data and then copies the data
 * from GPU into the allocated space. Certain DSPCore members needed for post processing are
 * assigned such as the rx freqs, the number of rf samples, the total antennas and the vector
 * of samples per antenna(each stage).
 */
void DSPCoreTesting::cuda_postprocessing_callback(uint32_t total_antennas, uint32_t num_samples_rf,
                                                  std::vector<uint32_t> samples_per_antenna,
                                                  std::vector<uint32_t> total_output_samples)
{
  // #ifdef ENGINEERING_DEBUG - Removed for testing purposes
    for (uint32_t i=0; i<filter_outputs_d.size()-1; i++) {
      allocate_and_copy_host(total_output_samples[i], filter_outputs_d[i]);
    }
  // #endif

  allocate_and_copy_host(total_output_samples.back(), filter_outputs_d.back());

  num_rf_samples = num_samples_rf;
  num_antennas = total_antennas;
  this->samples_per_antenna = samples_per_antenna;

  gpuErrchk(cudaStreamAddCallback(stream, postprocess, this, 0));

  DEBUG_MSG(COLOR_RED("Added stream callback for sequence #" << sequence_num));
}

/**
 * @brief      Starts the timing before the GPU kernels execute.
 *
 */
void DSPCoreTesting::start_decimate_timing()
{
  gpuErrchk(cudaEventRecord(kernel_start, stream));
}

/**
 * @brief      Adds the callback to the CUDA stream to acknowledge the RF samples have been copied.
 *
 */
void DSPCoreTesting::initial_memcpy_callback()
{
  gpuErrchk(cudaEventRecord(mem_transfer_end,stream));
  gpuErrchk(cudaStreamAddCallback(stream, initial_memcpy_callback_handler, this, 0));
}


/**
 * @brief      Gets the device pointer to the RF samples.
 *
 * @return     The RF samples device pointer.
 */
cuComplex* DSPCoreTesting::get_rf_samples_p(){
  return rf_samples_d;
}

/**
 * @brief      Gets the host pointer to the RF samples.
 *
 * @return     The rf samples host pointer.
 */
std::vector<cuComplex> DSPCoreTesting::get_rf_samples_h() {
  return rf_samples_h;
}

/**
 * @brief      Gets the device pointer to the receive frequencies.
 *
 * @return     The frequencies device pointer.
 */
double* DSPCoreTesting::get_frequencies_p() {
  return freqs_d;
}


/**
 * @brief      Gets the bandpass filters device pointer.
 *
 * @return     The bandpass filter pointer.
 */
cuComplex* DSPCoreTesting::get_bp_filters_p(){
  return bp_filters_d;
}

/**
 * @brief      Gets the vector of decimation rates.
 *
 * @return     The dm rates.
 */
std::vector<uint32_t> DSPCoreTesting::get_dm_rates()
{
  return dm_rates;
}

/**
 * @brief      Gets the vector of host side filter outputs.
 *
 * @return     The filter outputs host vector.
 */
std::vector<cuComplex*> DSPCoreTesting::get_filter_outputs_h()
{
  return filter_outputs_h;
}

/**
 * @brief      Gets the CUDA stream this DSPCore's work is associated to.
 *
 * @return     The CUDA stream.
 */
cudaStream_t DSPCoreTesting::get_cuda_stream(){
  return stream;
}

/**
 * @brief      Gets the total GPU process timing in milliseconds.
 *
 * @return     The total process timing.
 */
float DSPCoreTesting::get_total_timing()
{
  return total_process_timing_ms;
}

/**
 * @brief      Gets the total decimation timing in milliseconds.
 *
 * @return     The decimation timing.
 */
float DSPCoreTesting::get_decimate_timing()
{
  return decimate_kernel_timing_ms;
}

/**
 * @brief      Gets the number of antennas.
 *
 * @return     The number of antennas.
 */
uint32_t DSPCoreTesting::get_num_antennas()
{
  return num_antennas;
}

/**
 * @brief      Gets the number of rf samples.
 *
 * @return     The number of rf samples.
 */
uint32_t DSPCoreTesting::get_num_rf_samples()
{
  return num_rf_samples;
}

/**
 * @brief      Gets the rx sample rate.
 *
 * @return     The rx sampling rate (samples per second).
 */
double DSPCoreTesting::get_rx_rate()
{
  return rx_rate;
}

/**
 * @brief      Gets the output sample rate.
 *
 * @return     The output decimated and filtered rate (samples per second).
 */
double DSPCoreTesting::get_output_sample_rate()
{
  return output_sample_rate;
}

/**
 * @brief      Gets the vector of slice information, rx_slice_test structs.
 *
 * @return     The vector of rx_slice_test structs with slice information.
 */
 std::vector<rx_slice_test> DSPCoreTesting::get_slice_info()
 {
  return slice_info;
 }
