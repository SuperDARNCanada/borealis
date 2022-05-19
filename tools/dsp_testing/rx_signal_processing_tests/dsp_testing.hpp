/*
Copyright 2022 SuperDARN Canada

See LICENSE for details

  \file dsp_testing.hpp
  This file contains the declarations for the DSPCoreTesting.
*/

#ifndef DIGITAL_PROCESSING_TESTING_H
#define DIGITAL_PROCESSING_TESTING_H

#include <cuComplex.h>
#include <complex>
#include <zmq.hpp>
#include <vector>
#include <stdint.h>
#include <cstdlib>
#include <thrust/device_vector.h>
#include "utils/shared_memory/shared_memory.hpp"
#include "rx_signal_processing/filtering.hpp"

//This is inlined and used to detect and throw on CUDA errors.
#define gpuErrchk(ans) { throw_on_cuda_error((ans), __FILE__, __LINE__); }
inline void throw_on_cuda_error(cudaError_t code, const char *file, int line)
{
  if(code != cudaSuccess)
  {
  std::stringstream ss;
  ss << file << "(" << line << ")";
  std::string file_and_line;
  ss >> file_and_line;
  throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }
}

std::vector<cudaDeviceProp> get_gpu_properties();
void print_gpu_properties(std::vector<cudaDeviceProp> gpu_properties);


typedef struct rx_slice
{
  double rx_freq; // kHz
  uint32_t slice_id;

  rx_slice(double rx_freq, uint32_t slice_id) :
    rx_freq(rx_freq),
    slice_id(slice_id){}
}rx_slice;

/**
 * @brief      Contains the core DSP work done on the GPU.
 */
class DSPCoreTesting {
 public:
  void cuda_postprocessing_callback(uint32_t total_antennas, uint32_t num_samples_rf,
                                    std::vector<uint32_t> samples_per_antenna,
                                    std::vector<uint32_t> total_output_samples);
  void initial_memcpy_callback();
  //http://en.cppreference.com/w/cpp/language/explicit
  explicit DSPCoreTesting(double rx_rate, double output_sample_rate,
                          std::vector<std::vector<float>> filter_taps,
                          std::vector<uint32_t> dm_rates, std::vector<rx_slice> slice_info);

  ~DSPCoreTesting(); //destructor
  void allocate_and_copy_frequencies(void *freqs, uint32_t num_freqs);
  void allocate_and_copy_rf_samples(uint32_t total_antennas, uint32_t num_samples_needed,
                                    std::vector<cuComplex*> &ringbuffer_ptrs_start);
  void allocate_and_copy_bandpass_filters(void *taps, uint32_t total_taps);
  std::vector<cuComplex*> get_filter_outputs_h();
  cuComplex* get_last_filter_output_d();
  std::vector<cuComplex*> get_lowpass_filters_d();
  cuComplex* get_last_lowpass_filter_d();
  std::vector<uint32_t> get_samples_per_antenna();
  std::vector<uint32_t> get_dm_rates();
  cuComplex* get_bp_filters_p();
  void allocate_and_copy_lowpass_filter(void *taps, uint32_t total_taps);
  void allocate_output(uint32_t num_output_samples);
  std::vector<std::vector<float>> get_filter_taps();
  uint32_t get_num_antennas();
  float get_total_timing();
  float get_decimate_timing();
  void allocate_and_copy_host(uint32_t num_output_samples, cuComplex *output_d);
  void clear_device_and_destroy();
  cuComplex* get_rf_samples_p();
  std::vector<cuComplex> get_rf_samples_h();
  double* get_frequencies_p();
  uint32_t get_num_rf_samples();
  double get_rx_rate();
  double get_output_sample_rate();
  std::vector<rx_slice> get_slice_info();
  cudaStream_t get_cuda_stream();
  void start_decimate_timing();
  void stop_timing();

  SignalProcessingOptions sig_options;
  Filtering *dsp_filters;


//TODO(keith): May remove sizes as member variables.
 private:

  //! CUDA stream the work will be associated with.
  cudaStream_t stream;

  //! Rx sampling rate for the data being processed.
  double rx_rate;

  //! Output sampling rate of the filtered, decimated, processed data.
  double output_sample_rate;

  //! Stores the total GPU process timing once all the work is done.
  float total_process_timing_ms;

  //! Stores the decimation timing.
  float decimate_kernel_timing_ms;

  //! Pointer to the device rx frequencies.
  double *freqs_d;

  //! Pointer to the RF samples on device.
  cuComplex *rf_samples_d;

  //! Pointer to the first stage bandpass filters on device.
  cuComplex *bp_filters_d;

  //! Vector of device side lowpass filter pointers.
  std::vector<cuComplex*> lp_filters_d;

  //! Vector of device side filter output pointers.
  std::vector<cuComplex*> filter_outputs_d;

  //! Vector of host side filter output pointers.
  std::vector<cuComplex*> filter_outputs_h;

  //! Vector of the samples per antenna at each stage of decimation.
  std::vector<uint32_t> samples_per_antenna;

  //! Vector of decimation rates at each stage.
  std::vector<uint32_t> dm_rates;

  //! Vector that holds the vectors of filter taps at each stage.
  std::vector<std::vector<float>> filter_taps;

  //! CUDA event to timestamp when the GPU processing begins.
  cudaEvent_t initial_start;

  //! CUDA event to timestamp when the kernels begin executing.
  cudaEvent_t kernel_start;

  //! CUDA event to timestamp when the GPU processing stops.
  cudaEvent_t stop;

  //! Cuda event to timestamp the transfer of RF samples to the GPU.
  cudaEvent_t mem_transfer_end;

  //! Stores the memory transfer timing.
  float mem_time_ms;

  //! A vector of pointers to the start of ringbuffers.
  std::vector<cuComplex*> ringbuffers;

  //! A host side vector for the rf samples.
  std::vector<cuComplex> rf_samples_h;

  //! The number of total antennas.
  uint32_t num_antennas;

  //! The number of rf samples per antenna.
  uint32_t num_rf_samples;

  //! Slice information given from rx_slice structs
  std::vector<rx_slice> slice_info;

  void allocate_and_copy_rf_from_device(uint32_t num_rf_samples);

};

void postprocess(DSPCoreTesting *dp);
#endif
