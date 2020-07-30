/*
Copyright 2017 SuperDARN Canada

See LICENSE for details
  \file decimate.hpp
*/

#ifndef DECIMATE_H
#define DECIMATE_H

#include "dsp.hpp"
#include "utils/shared_macros/shared_macros.hpp"

enum class DecimationType {lowpass, bandpass};

void bandpass_decimate1024_wrapper(cuComplex* input_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, double F_s, double *freqs, cudaStream_t stream);

void bandpass_decimate2048_wrapper(cuComplex* input_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, double F_s, double *freqs, cudaStream_t stream);

void lowpass_decimate1024_wrapper(cuComplex* input_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, cudaStream_t stream);

void lowpass_decimate2048_wrapper(cuComplex* input_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, cudaStream_t stream);

/**
 * @brief      Selects which decimate kernel to run.
 *
 * @param[in]  input_samples        A pointer to original input samples from each antenna to
 *                                  decimate.
 * @param[in]  decimated_samples    A pointer to a buffer to place output samples for each frequency
 *                                  after decimation.
 * @param[in]  filter_taps          A pointer to one or more filters needed for each frequency. If
 *                                  using lowpass, one filter is used. If using bandpass, there is
 *                                  one filter for each RX frequency.
 * @param[in]  dm_rate              Decimation rate.
 * @param[in]  samples_per_antenna  The number of samples per antenna in the input set of samples
 *                                  for one frequency.
 * @param[in]  num_total_taps       Number of taps per stage.
 * @param[in]  num_freqs            Number of receive frequencies.
 * @param[in]  num_antennas         Number of antennas for which there are samples.
 * @param[in]  F_s                  The original sampling frequency.
 * @param      freqs                A pointer to the filtering freqs.
 * @param[in]  output_msg           A simple character string that can be used to debug or
 *                                  distinguish different stages.
 * @param[in]  stream               The CUDA stream for which to run a run a kernel.
 *
 *             Based off the total number of filter taps, this function will choose what decimate
 *             kernel to use.
 *
 * @tparam     type                 { description }
 */
template <DecimationType type>
void call_decimate(cuComplex* input_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_total_taps, uint32_t num_freqs,
  uint32_t num_antennas, double F_s, double* freqs, const char *output_msg, cudaStream_t stream) {

  DEBUG_MSG(COLOR_BLUE("Decimate: ") << output_msg);

  auto gpu_properties = get_gpu_properties();


  if (type == DecimationType::bandpass) {
    DEBUG_MSG(COLOR_BLUE("Decimate: ") << "    Running bandpass");
    //For now we have a kernel that will process 2 samples per thread if need be
    if (num_total_taps> 2 * gpu_properties[0].maxThreadsPerBlock) {
      std::cerr << "Total taps exceeds the amount we can process!" << std::endl;
      exit(-1);
      //TODO(Keith) : handle error
    }
    else if (num_total_taps > gpu_properties[0].maxThreadsPerBlock) {
      bandpass_decimate2048_wrapper(input_samples, decimated_samples, filter_taps,  dm_rate,
        samples_per_antenna, num_total_taps, num_freqs, num_antennas, F_s, freqs, stream);
    }
    else {
      bandpass_decimate1024_wrapper(input_samples, decimated_samples, filter_taps,  dm_rate,
        samples_per_antenna, num_total_taps, num_freqs, num_antennas, F_s, freqs, stream);
    }
  }
  else if (type == DecimationType::lowpass){
    DEBUG_MSG(COLOR_BLUE("Decimate: ") << "    Running lowpass");
    if (num_total_taps > 2 * gpu_properties[0].maxThreadsPerBlock) {
      //TODO(Keith) : handle error
    }
    else if (num_total_taps > gpu_properties[0].maxThreadsPerBlock) {
      lowpass_decimate2048_wrapper(input_samples, decimated_samples, filter_taps,  dm_rate,
        samples_per_antenna, num_total_taps, num_freqs, num_antennas, stream);
    }
    else {
      lowpass_decimate1024_wrapper(input_samples, decimated_samples, filter_taps,  dm_rate,
        samples_per_antenna, num_total_taps, num_freqs, num_antennas, stream);
    }
  }


  // This is to detect invalid launch parameters.
  gpuErrchk(cudaPeekAtLastError());

}


#endif