/*
Copyright 2017 SuperDARN Canada

See LICENSE for details

  \file dsp.hpp
  This file contains the declarations for the DSPCore.
*/

#ifndef DIGITAL_PROCESSING_H
#define DIGITAL_PROCESSING_H

#include <cuComplex.h>
#include <complex>
#include <zmq.hpp>
#include <vector>
#include <stdint.h>
#include <cstdlib>
#include <thrust/device_vector.h>
#include "utils/shared_memory/shared_memory.hpp"

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

/**
 * @brief      Contains the core DSP work done on the GPU.
 */
class DSPCore {
 public:
    static void CUDART_CB cuda_postprocessing_callback(cudaStream_t stream, cudaError_t status,
                                                void *processing_data);
    static void CUDART_CB initial_memcpy_callback(cudaStream_t stream, cudaError_t status,
                                                void *processing_data);
    //http://en.cppreference.com/w/cpp/language/explicit
    explicit DSPCore(zmq::socket_t *ack_s, zmq::socket_t *timing_s,
                                 uint32_t sq_num, const char* shr_mem_name);
    void allocate_and_copy_rf_samples(uint32_t total_samples);
    void allocate_and_copy_first_stage_filters(void *taps, uint32_t total_taps);
    void allocate_and_copy_second_stage_filter(void *taps, uint32_t total_taps);
    void allocate_and_copy_third_stage_filter(void *taps, uint32_t total_taps);
    void allocate_first_stage_output(uint32_t num_first_stage_output_samples);
    void allocate_second_stage_output(uint32_t num_second_stage_output_samples);
    void allocate_third_stage_output(uint32_t num_third_stage_output_samples);
    void allocate_and_copy_host_output(uint32_t num_host_samples);
    void copy_output_to_host();
    void clear_device_and_destroy();
    void call_decimate(cuComplex* original_samples,cuComplex* decimated_samples,
        cuComplex* filter_taps, uint32_t dm_rate,uint32_t samples_per_antenna,
        uint32_t num_taps_per_filter, uint32_t num_freqs, uint32_t num_antennas,
        const char *output_msg);
    cuComplex* get_rf_samples_p();
    cuComplex* get_first_stage_bp_filters_p();
    cuComplex* get_second_stage_filter_p();
    cuComplex* get_third_stage_filter_p();
    cuComplex* get_first_stage_output_p();
    cuComplex* get_second_stage_output_p();
    cuComplex* get_third_stage_output_p();
    float get_total_timing();
    float get_decimate_timing();
    cudaStream_t get_cuda_stream();
    void start_decimate_timing();
    void stop_timing();
    void send_ack();
    void send_timing();

//TODO(keith): May remove sizes as member variables.
 private:

    //! CUDA stream the work will be associated with.
    cudaStream_t stream;

    //! Sequence number used to identify and acknowledge a pulse sequence.
    uint32_t sequence_num;

    //! Pointer to the socket used to acknowledge the RF samples have been copied to device.
    zmq::socket_t *ack_socket;

    //! Pointer to the socket used to report the timing of GPU kernels.
    zmq::socket_t *timing_socket;

    //! Stores the total GPU process timing once all the work is done.
    float total_process_timing_ms;

    //! Stores the decimation timing.
    float decimate_kernel_timing_ms;

    //! Pointer to the RF samples on device.
    cuComplex *rf_samples_d;
    size_t rf_samples_size;

    //! Pointer to the first stage bandpass filters on device.
    cuComplex *first_stage_bp_filters_d;
    size_t first_stage_bp_filters_size;

    //! Pointer to the second stage filters on device.
    cuComplex *second_stage_filter_d;
    size_t second_stage_filter_size;

    //! Pointer to the third stage filters on device.
    cuComplex *third_stage_filter_d;
    size_t third_stage_filter_size;

    //! Pointer to the output of the first stage decimation on device.
    cuComplex *first_stage_output_d;
    size_t first_stage_output_size;

    //! Pointer to the output of the second stage decimation on device.
    cuComplex *second_stage_output_d;
    size_t second_stage_output_size;

    //! Pointer to the output of the third stage decimation on device.
    cuComplex *third_stage_output_d;
    size_t third_stage_output_size;

    //! Pointer to the host output samples.
    cuComplex *host_output_h;
    size_t host_output_size;

    //! CUDA event to timestamp when the GPU processing begins.
    cudaEvent_t initial_start;

    //! CUDA event to timestamp when the kernels begin executing.
    cudaEvent_t kernel_start;

    //! CUDA event to timestamp when the GPU processing stops.
    cudaEvent_t stop;

    //! A pointer to a shared memory handler that contains RF samples from the USRP driver.
    SharedMemoryHandler *shr_mem;


};

#endif
