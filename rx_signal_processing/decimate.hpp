/*
Copyright 2017 SuperDARN Canada

See LICENSE for details
  \file decimate.hpp
*/

#ifndef DECIMATE_H
#define DECIMATE_H

#include "dsp.hpp"

enum class DecimationType {lowpass, bandpass};

void bandpass_decimate1024_wrapper(cuComplex* original_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, cudaStream_t stream);

void bandpass_decimate2048_wrapper(cuComplex* original_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, cudaStream_t stream);

void lowpass_decimate1024_wrapper(cuComplex* original_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, cudaStream_t stream);

void lowpass_decimate2048_wrapper(cuComplex* original_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, cudaStream_t stream);
/**
 * @brief      Selects which decimate kernel to run.
 *
 * @param[in]  original_samples     A pointer to original input samples from
 *                                  each antenna to decimate.
 * @param[in]  decimated_samples    A pointer to a buffer to place output
 *                                  samples for each frequency after decimation.
 * @param[in]  filter_taps          A pointer to one or more filters needed for
 *                                  each frequency.
 * @param[in]  dm_rate              Decimation rate.
 * @param[in]  samples_per_antenna  The number of samples per antenna in the
 *                                  original set of samples.
 * @param[in]  num_taps_per_filter  Number of taps per filter.
 * @param[in]  num_freqs            Number of receive frequencies.
 * @param[in]  num_antennas         Number of antennas for which there are
 *                                  samples.
 * @param[in]  output_msg           A simple character string that can be used
 *                                  to debug or distinguish different stages.
 * @param[in]  stream               The CUDA stream for which to run a run a kernel.
 *
 *             Based off the total number of filter taps, this function will
 *             choose what decimate kernel to use.
 *
 */
template <DecimationType type>
void call_decimate(cuComplex* original_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, const char *output_msg, cudaStream_t stream) {

  std::cout << output_msg << std::endl;

  auto gpu_properties = get_gpu_properties();


  if (type == DecimationType::bandpass) {
    std::cout << "    Running bandpass" << std::endl;
    //For now we have a kernel that will process 2 samples per thread if need be
    if (num_taps_per_filter * num_freqs > 2 * gpu_properties[0].maxThreadsPerBlock) {
      //TODO(Keith) : handle error
    }
    else if (num_taps_per_filter * num_freqs > gpu_properties[0].maxThreadsPerBlock) {
      bandpass_decimate2048_wrapper(original_samples, decimated_samples, filter_taps,  dm_rate,
        samples_per_antenna, num_taps_per_filter, num_freqs, num_antennas, stream);
    }
    else {
      bandpass_decimate1024_wrapper(original_samples, decimated_samples, filter_taps,  dm_rate,
        samples_per_antenna, num_taps_per_filter, num_freqs, num_antennas, stream);
    }
  }
  else if (type == DecimationType::lowpass){
    std::cout << "    Running lowpass" << std::endl;
    if (num_taps_per_filter > 2 * gpu_properties[0].maxThreadsPerBlock) {
      //TODO(Keith) : handle error
    }
    else if (num_taps_per_filter > gpu_properties[0].maxThreadsPerBlock) {
      lowpass_decimate2048_wrapper(original_samples, decimated_samples, filter_taps,  dm_rate,
        samples_per_antenna, num_taps_per_filter, num_freqs, num_antennas, stream);
    }
    else {
      lowpass_decimate1024_wrapper(original_samples, decimated_samples, filter_taps,  dm_rate,
        samples_per_antenna, num_taps_per_filter, num_freqs, num_antennas, stream);
    }
  }


  // This is to detect invalid launch parameters.
  gpuErrchk(cudaPeekAtLastError());

}


#endif