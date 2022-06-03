/*
Copyright 2017 SuperDARN Canada

See LICENSE for details

This file contains the CUDA code used to process the large amount of data
involved with radar receive side processing.
*/

#include <cuComplex.h> //cuComplex type and all cuCmulf/cuCaddf functions.
#include <iostream>
#include <stdint.h>
#include "decimate.hpp"
//This keeps the contained functions local to this file.
namespace {
  /**
   * @brief      Creates a new set of grid dimensions for a bandpass decimate CUDA kernel.
   *
   * @param[in]  num_samples   Number of input samples.
   * @param[in]  dm_rate       Decimation rate.
   * @param[in]  num_antennas  Number of antennas for which there are samples.
   *
   * @return     New grid dimensions for the kernel.
   */
  dim3 create_bandpass_grid(uint32_t num_samples, uint32_t dm_rate, uint32_t num_antennas)
  {
    auto num_blocks_x = num_samples/dm_rate;
    auto num_blocks_y = num_antennas;
    auto num_blocks_z = 1;
    DEBUG_MSG(COLOR_BLUE("Decimate: ") << "    Grid size: " << num_blocks_x << " x "
      << num_blocks_y << " x "<< num_blocks_z);
    dim3 dimGrid(num_blocks_x,num_blocks_y,num_blocks_z);

    return dimGrid;
  }

  /**
   * @brief      Creates a new set of block dimensions for a bandpass decimate CUDA kernel.
   *
   * @param[in]  num_taps_per_filter  Number of taps per filter.
   * @param[in]  num_freqs            Number of receive frequencies.
   *
   * @return     New block dimensions for the kernel.
   */
  dim3 create_bandpass_block(uint32_t num_taps_per_filter, uint32_t num_freqs)
  {
    auto num_threads_x = num_taps_per_filter;
    auto num_threads_y = num_freqs;
    auto num_threads_z = 1;
    DEBUG_MSG(COLOR_BLUE("Decimate: ") << "    Block size: " << num_threads_x << " x "
      << num_threads_y << " x " << num_threads_z);
    dim3 dimBlock(num_threads_x,num_threads_y,num_threads_z);

    return dimBlock;
  }

  /**
   * @brief      Creates a new set of grid dimensions for a lowpass decimate CUDA kernel.
   *
   * @param[in]  num_samples   Number of input samples in a frequency dataset.
   * @param[in]  dm_rate       Decimation rate.
   * @param[in]  num_antennas  Number of antennas for which there are samples.
   * @param[in]  num_freqs     Number of receive frequencies.
   *
   * @return     New grid dimensions for the kernel.
   */
  dim3 create_lowpass_grid(uint32_t num_samples, uint32_t dm_rate, uint32_t num_antennas,
                           uint32_t num_freqs)
  {
    auto num_blocks_x = num_samples/dm_rate;
    auto num_blocks_y = num_antennas;
    auto num_blocks_z = num_freqs;
    DEBUG_MSG(COLOR_BLUE("Decimate: ") << "    Grid size: " << num_blocks_x << " x "
      << num_blocks_y << " x "<< num_blocks_z);
    dim3 dimGrid(num_blocks_x,num_blocks_y,num_blocks_z);

    return dimGrid;
  }

  /**
   * @brief      Creates a new set of block dimensions for a lowpass decimate CUDA kernel.
   *
   * @param[in]  num_taps_per_filter  Number of taps per filter.
   *
   * @return     New block dimensions for the kernel.
   */
  dim3 create_lowpass_block(uint32_t num_taps_per_filter)
  {
    auto num_threads_x = num_taps_per_filter;
    auto num_threads_y = 1;
    auto num_threads_z = 1;
    DEBUG_MSG(COLOR_BLUE("Decimate: ") << "    Block size: " << num_threads_x << " x "
      << num_threads_y << " x " << num_threads_z);
    dim3 dimBlock(num_threads_x,num_threads_y,num_threads_z);

    return dimBlock;
  }
}

/**
 * @brief      Overloads __shfl_down to handle cuComplex.
 *
 * @param[in]  var      cuComplex value to shuffle.
 * @param[in]  srcLane  Relative lane from within the warp that should shuffle its variable down.
 * @param[in]  width    Section of the warp to shuffle. Defaults to full warp size.
 *
 * @return     Shuffled cuComplex variable.
 *
 * __shfl can only shuffle 4 bytes at time. This overload utilizes a trick similar to the below
 * link in order to shuffle 8 byte values.
 * https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
 * http://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions
 */
__device__ inline cuComplex __shfl_down_sync(cuComplex var, unsigned int srcLane, int width=32){
  float2 a = *reinterpret_cast<float2*>(&var);
  a.x = __shfl_down_sync(0xFFFFFFFF, a.x, srcLane, width);
  a.y = __shfl_down_sync(0xFFFFFFFF, a.y, srcLane, width);
  return *reinterpret_cast<cuComplex*>(&a);
}

/**
 * @brief      Performs a parallel reduction to sum a series of values together.
 *
 * @param      data        A pointer to a set of cuComplex data to reduce.
 * @param[in]  tap_offset  The offset into the data from which to pull values.
 *
 * @return     Final sum after reduction.
 *
 * NVIDIA supplies many versions of optimized parallel reduction. This is a slightly modified
 * version of reduction #5 from NVIDIA examples.
 * /usr/local/cuda/samples/6_Advanced/reduction
 */
__device__ cuComplex parallel_reduce(cuComplex* data, uint32_t tap_offset) {
  auto filter_tap_num = threadIdx.x;
  auto num_filter_taps = blockDim.x;
  cuComplex total_sum = data[tap_offset];

  if ((num_filter_taps >= 1024) && (filter_tap_num < 512))
  {
    total_sum = cuCaddf(total_sum,data[tap_offset  + 512]);
    data[tap_offset] = total_sum;
  }

  __syncthreads();

  if ((num_filter_taps >= 512) && (filter_tap_num < 256))
  {
    total_sum = cuCaddf(total_sum,data[tap_offset  + 256]);
    data[tap_offset] = total_sum;
  }

  __syncthreads();

  if ((num_filter_taps >= 256) && (filter_tap_num < 128))
  {
    total_sum = cuCaddf(total_sum, data[tap_offset + 128]);
    data[tap_offset] = total_sum;
  }

   __syncthreads();

  if ((num_filter_taps >= 128) && (filter_tap_num <  64))
  {
    total_sum = cuCaddf(total_sum, data[tap_offset  +  64]);
    data[tap_offset] = total_sum;
  }

  __syncthreads();

  if ( filter_tap_num < 32 )
  {
    // Fetch final intermediate sum from 2nd warp
    if (num_filter_taps >=  64) total_sum = cuCaddf(total_sum, data[tap_offset + 32]);
    // Reduce final warp using shuffle
    // http://docs.nvidia.com/cuda/cuda-c-programming-guide/#built-in-variables
    // __shfl_down is used an optimization in the final warp to simultaneously move
    // values from upper threads to lower threads without needing __syncthreads().
    for (int offset = warpSize/2; offset > 0; offset /= 2)
    {
      total_sum = cuCaddf(total_sum,__shfl_down_sync(total_sum, offset));
    }
  }

  return total_sum;
}

/**
 * @brief      cuComplex version of exponential function.
 *
 * @param[in]  z     Complex number.
 *
 * @return     Complex exponential of input.
 */
__device__ __forceinline__ cuComplex _exp (cuComplex z)
{
    cuComplex res;
    float t = expf(z.x);
    sincosf(z.y, &res.y, &res.x);
    res.x *= t;
    res.y *= t;
    return res;
}

/**
 * @brief      Performs decimation using bandpass filters on a set of input RF samples.
 *
 * @param[in]  original_samples     A pointer to original input samples from each antenna to
 *                                  decimate.
 * @param[in]  decimated_samples    A pointer to a buffer to place output samples for each
 *                                  frequency after decimation.
 * @param[in]  filter_taps          A pointer to one or more filters needed for each frequency.
 * @param[in]  dm_rate              Decimation rate.
 * @param[in]  samples_per_antenna  The number of samples per antenna in the original set of
 *                                  samples.
 * @param[in]  F_s                  The sampling frequency in hertz.
 * @param[in]  freqs                A pointer to the frequencies used in mixing.
 *
 * @param[in]  stride               Number of filter products to calculate per thread.
 *
 * This function performs a parallel version of filtering+downsampling on the GPU to be able
 * process data in realtime. This algorithm will use 1 GPU thread per filter tap if there are less
 * than or equal to 1024 taps for all filters combined. Only works with power of two length filters, or a
 * filter that is zero padded to a power of two in length. This algorithm takes
 * a single set of wide band samples from the USRP driver, and produces an output data set for each
 * RX frequency. The phase of each output sample is corrected after decimating via modified
 * Frerking method.
 *
 *   gridDim.x - Total number of output samples there will be after decimation.
 *   gridDim.y - Total number of antennas.
 *
 *   blockIdx.x - Decimated output sample index.
 *   blockIdx.y - Antenna index.
 *
 *   blockDim.x - Number of filter taps divided by stride length.
 *   blockDim.y - Total number of filters. Corresponds to total receive frequencies.
 *
 *   threadIdx.x - Filter tap index.
 *   threadIdx.y - Filter index.
 */
__global__ void bandpass_decimate(cuComplex* original_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, double F_s, double *freqs, uint32_t stride) {

  // Since number of filter taps is calculated at runtime and we do not want to hardcode
  // values, the shared memory can be dynamically initialized at invocation of the kernel.
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared

  extern __shared__ cuComplex filter_products[];

  auto antenna_num = blockIdx.y;
  auto antenna_offset = antenna_num * samples_per_antenna;

  auto dec_sample_num = blockIdx.x;
  auto dec_sample_offset = dec_sample_num * dm_rate;

  auto product_offset = threadIdx.y * blockDim.x + threadIdx.x;     // for indexing filter_products
  auto tap_offset = product_offset * stride;                        // first index into filter_taps for this thread

  cuComplex sample;                             // the current iq sample from an antenna
  cuComplex intermediate_product = make_cuComplex(0.0f, 0.0f);   // the product of sample with its respective filter tap
  cuComplex intermediate_sum = make_cuComplex(0.0f, 0.0f);       // the sum of all intermediate products for this thread

  // Calculate the sum of 'stride' samples in this thread.
  // This is done so that if the number of filter taps * num frequencies is greater than
  // the max number of threads per block (1024), each thread can dynamically calculate
  // multiple taps * samples and sum them together.
  for (int i=0; i<stride; i++) {
    // If an offset should extend past the length of samples per antenna
    // then zeroes are used as to not segfault or run into the next buffer.
    // output samples convolved with these zeroes will be discarded after
    // the complete process as to not introduce edge effects.
    if ((dec_sample_offset + threadIdx.x*stride + i) >= samples_per_antenna) {
      sample = make_cuComplex(0.0f,0.0f);
    }
    else {
      auto final_offset = antenna_offset + dec_sample_offset + threadIdx.x*stride + i;
      sample = original_samples[final_offset];
    }
    intermediate_product = cuCmulf(sample, filter_taps[tap_offset + i]);
    intermediate_sum = cuCaddf(intermediate_sum, intermediate_product);
  }

  filter_products[product_offset] = intermediate_sum;
  // Synchronizes all threads in a block, meaning 1 output sample per rx freq
  // is ready to be calculated with the parallel reduce
  __syncthreads();

  auto calculated_output_sample = parallel_reduce(filter_products, product_offset);

  // When decimating, we go from one set of samples for each antenna
  // to multiple sets of reduced samples for each frequency. Output samples are
  // grouped by frequency with all samples for each antenna following each other
  // before samples of another frequency start.
  if (threadIdx.x == 0) {

    //Correct phase after filtering using modified Frerking technique.
    auto freq_idx = threadIdx.y;
    auto unwrapped_phase = 2.0 * M_PI * (freqs[freq_idx]/F_s) * dec_sample_num * dm_rate;
    auto phase = fmod(unwrapped_phase, 2.0 * M_PI);
    auto filter_phase = _exp(make_cuComplex(0.0f, 1 * phase));
    calculated_output_sample = cuCmulf(calculated_output_sample,filter_phase);

    antenna_offset = antenna_num * gridDim.x;
    auto total_antennas = gridDim.y;
    auto freq_offset = threadIdx.y * gridDim.x * total_antennas;
    auto total_offset = freq_offset + antenna_offset + dec_sample_num;
    decimated_samples[total_offset] = calculated_output_sample;

  }
}


/**
 * @brief      This function wraps the bandpass_decimate_general kernel so that it can be called from
 *             another file.
 *
 * @param[in]  original_samples     A pointer to original input samples from each antenna to
 *                                  decimate.
 * @param[in]  decimated_samples    A pointer to a buffer to place output samples for each frequency
 *                                  after decimation.
 * @param[in]  filter_taps          A pointer to one or more filters needed for each frequency.
 * @param[in]  dm_rate              Decimation rate.
 * @param[in]  samples_per_antenna  The number of samples per antenna in the original set of
 *                                  samples.
 * @param[in]  num_taps_per_filter  Number of taps per filter.
 * @param[in]  num_freqs            Number of receive frequencies.
 * @param[in]  num_antennas         Number of antennas for which there are samples.
 * @param[in]  F_s                  The original sampling frequency.
 * @param      freqs                A pointer to the frequencies being filtered.
 * @param[in]  stream               CUDA stream with which to associate the invocation of the
 *                                  kernel.
 */
void bandpass_decimate_wrapper(cuComplex* original_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, double F_s, double *freqs, cudaStream_t stream) {

  // num_threads is capped at 1024 per block, so we must figure out how many threads per filter
  // can be allocated, and how many samples each thread must calculate.
  uint32_t max_threads_per_freq = 1024 / num_freqs;
  uint32_t stride = 1;
  uint32_t threads_per_freq = num_taps_per_filter;

  // Here we are assuming that num_taps_per_filter is a power of 2, which is a necessary assumption
  // for bandpass_decimate_general to correctly do its calculations.
  while (threads_per_freq > max_threads_per_freq) {
    stride = stride << 1;
    threads_per_freq = threads_per_freq >> 1;
  }

  //Allocate shared memory on device for all intermediate sums of filter products.
  auto shr_mem_size = num_freqs * threads_per_freq * sizeof(cuComplex);
  DEBUG_MSG(COLOR_BLUE("Decimate: ") << "    Number of shared memory bytes: "<< shr_mem_size);

  auto dimGrid = create_bandpass_grid(samples_per_antenna, dm_rate, num_antennas);
  auto dimBlock = create_bandpass_block(threads_per_freq, num_freqs);
  bandpass_decimate<<<dimGrid,dimBlock,shr_mem_size,stream>>>(original_samples, decimated_samples,
        filter_taps, dm_rate, samples_per_antenna, F_s, freqs, stride);

}


/**
 * @brief      Performs decimation using a lowpass filter on one or more sets of baseband samples
 * corresponding to each RX frequency.
 *
 * @param[in]  original_samples     A pointer to input samples for one or more baseband datasets.
 * @param[in]  decimated_samples    A pointer to a buffer to place output samples for each frequency
 *                                  dataset after decimation.
 * @param[in]  filter_taps          A pointer to a lowpass filter used for further decimation.
 * @param[in]  dm_rate              Decimation rate.
 * @param[in]  samples_per_antenna  The number of samples per antenna in the original set of
 *                                  samples.
 * @param[in]  stride               The number of samples to calculate per thread.
 *
 * This function performs a parallel version of filtering+downsampling on the GPU to be able
 * process data in realtime. This algorithm will use 1 GPU thread per filter tap if there are less
 * than 1024 taps for all filters combined. Only works with power of two length filters, or a
 * filter that is zero padded to a power of two in length. This algorithm takes one or more
 * baseband datasets corresponding to each RX frequency and filters each one using a single lowpass
 * filter before downsampling.
 *
 *   gridDim.x - The number of decimated output samples for one antenna in one frequency data set.
 *   gridDim.y - Total number of antennas.
 *   gridDim.z - Total number of frequency data sets.
 *
 *   blockIdx.x - Decimated output sample index.
 *   blockIdx.y - Antenna index.
 *   blockIdx.z - Frequency dataset index.
 *
 *   blockDim.x - Number of filter taps in the lowpass filter.

 *   threadIdx.x - Filter tap indices.
 */
__global__ void lowpass_decimate(cuComplex* original_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t stride) {

  // Since number of filter taps is calculated at runtime and we do not want to hardcode
  // values, the shared memory can be dynamically initialized at invocation of the kernel.
  // http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared

  extern __shared__ cuComplex filter_products[];

  auto total_antennas = gridDim.y;

  auto data_set_idx = blockIdx.z;

  auto frequency_dataset_offset = data_set_idx * samples_per_antenna * total_antennas;

  auto antenna_num = blockIdx.y;
  auto antenna_offset = antenna_num * samples_per_antenna;

  auto dec_sample_num = blockIdx.x;
  auto dec_sample_offset = dec_sample_num * dm_rate;

  auto product_offset = threadIdx.x;
  auto tap_offset = product_offset * stride;


  cuComplex sample;
  cuComplex intermediate_product = make_cuComplex(0.0f, 0.0f);
  cuComplex intermediate_sum = make_cuComplex(0.0f, 0.0f);

  // Calculate the sum of 'stride' samples in this thread.
  // This is done so that if the number of filter taps is greater than
  // the max number of threads per block (1024), each thread can dynamically calculate
  // multiple taps * samples and sum them together.
  for (int i=0; i<stride; i++) {
    // If an offset should extend past the length of samples per antenna
    // then zeroes are used as to not segfault or run into the next buffer.
    // Output samples convolved with these zeroes will be discarded after
    // the complete process as to not introduce edge effects.
    if ((dec_sample_offset + tap_offset + i) >= samples_per_antenna) {
      sample = make_cuComplex(0.0f,0.0f);
    }
    else {
      auto final_offset = frequency_dataset_offset + antenna_offset + dec_sample_offset + tap_offset + i;
      sample = original_samples[final_offset];
    }
    intermediate_product = cuCmulf(sample, filter_taps[tap_offset]);
    intermediate_sum = cuCaddf(intermediate_sum, intermediate_product);
  }

  filter_products[product_offset] = intermediate_sum;

  // Synchronizes all threads in a block, meaning 1 output sample per rx freq
  // is ready to be calculated with the parallel reduce
  __syncthreads();

  auto calculated_output_sample = parallel_reduce(filter_products, product_offset);

  // When decimating, we go from one set of samples for each antenna
  // to multiple sets of reduced samples for each frequency. Output samples are
  // grouped by frequency with all samples for each antenna following each other
  // before samples of another frequency start.
  if (threadIdx.x == 0) {
    auto num_output_samples_per_antenna = gridDim.x;
    frequency_dataset_offset = data_set_idx * num_output_samples_per_antenna * total_antennas;
    antenna_offset = antenna_num * num_output_samples_per_antenna;
    auto total_offset = frequency_dataset_offset + antenna_offset + dec_sample_num;
    decimated_samples[total_offset] = calculated_output_sample;
  }
}


/**
 * @brief      This function wraps the lowpass_decimate_general kernel so that it can be called from
 *             another file.
 *
 * @param[in]  original_samples     A pointer to one or more baseband frequency datasets.
 * @param[in]  decimated_samples    A pointer to a buffer to place output samples for each frequency
 *                                  after decimation.
 * @param[in]  filter_taps          A pointer to one lowpass filter.
 * @param[in]  dm_rate              Decimation rate.
 * @param[in]  samples_per_antenna  The number of samples per antenna in each data set.
 * @param[in]  num_taps_per_filter  Number of taps per filter.
 * @param[in]  num_freqs            Number of receive frequency datasets.
 * @param[in]  num_antennas         Number of antennas for which there are samples.
 * @param[in]  stream               CUDA stream with which to associate the invocation of the
 *                                  kernel.
 */
void lowpass_decimate_wrapper(cuComplex* original_samples,
  cuComplex* decimated_samples,
  cuComplex* filter_taps, uint32_t dm_rate,
  uint32_t samples_per_antenna, uint32_t num_taps_per_filter, uint32_t num_freqs,
  uint32_t num_antennas, cudaStream_t stream) {

  // num_threads is capped at 1024 per block, so we must figure out how many threads per filter
  // can be allocated, and how many samples each thread must calculate.
  uint32_t num_threads = num_taps_per_filter;
  uint32_t stride = 1;

  // Here we are assuming that num_taps_per_filter is a power of 2, which is a necessary assumption
  // for lowpass_decimate_general to correctly do its calculations.
  while (num_threads > 1024) {
    stride = stride << 1;
    num_threads = num_threads >> 1;
  }

  //Allocate shared memory on device for all intermediate sums of filter products.
  auto shr_mem_size = num_freqs * num_threads * sizeof(cuComplex);
  DEBUG_MSG(COLOR_BLUE("Decimate: ") << "    Number of shared memory bytes: "<< shr_mem_size);

  auto dimGrid = create_lowpass_grid(samples_per_antenna, dm_rate, num_antennas, num_freqs);
  auto dimBlock = create_lowpass_block(num_threads);
  lowpass_decimate<<<dimGrid,dimBlock,shr_mem_size,stream>>>(original_samples,
    decimated_samples, filter_taps, dm_rate, samples_per_antenna, stride);
}
