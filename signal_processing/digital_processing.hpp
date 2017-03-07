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

class DigitalProcessing {
 public:
    static void CUDART_CB cuda_postprocessing_callback(cudaStream_t stream, cudaError_t status,
                                                void *processing_data);
    static void CUDART_CB initial_memcpy_callback(cudaStream_t stream, cudaError_t status,
                                                void *processing_data);
    explicit DigitalProcessing(zmq::socket_t *ack_s, zmq::socket_t *timing_s,
                                 uint32_t sq_num, const char* shr_mem_name);
    void allocate_and_copy_rf_samples(uint32_t total_samples);
    void allocate_and_copy_first_stage_filters(void *taps, uint32_t total_taps);
    void allocate_and_copy_second_stage_filters(void *taps, uint32_t total_taps);
    void allocate_and_copy_third_stage_filters(void *taps, uint32_t total_taps);
    void allocate_first_stage_output(uint32_t first_stage_samples);
    void allocate_second_stage_output(uint32_t second_stage_samples);
    void allocate_third_stage_output(uint32_t third_stage_samples);
    void allocate_and_copy_host_output(uint32_t host_samples);
    void copy_output_to_host();
    void clear_device_and_destroy();
    void call_decimate(cuComplex* original_samples,cuComplex* decimated_samples,
        cuComplex* filter_taps, uint32_t dm_rate,uint32_t samples_per_channel,
        uint32_t num_taps, uint32_t num_freqs, uint32_t num_channels, const char *output_msg);
    cuComplex* get_rf_samples_p();
    cuComplex* get_first_stage_bp_filters_p();
    cuComplex* get_second_stage_filters_p();
    cuComplex* get_third_stage_filters_p();
    cuComplex* get_first_stage_output_p();
    cuComplex* get_second_stage_output_p();
    cuComplex* get_third_stage_output_p();
    float get_total_timing();
    float get_decimate_timing();
    cudaStream_t get_cuda_stream();
/*    uint32_t get_sequence_num();
    zmq::socket_t *get_ack_socket();
    zmq::socket_t *get_timing_socket();*/
    void start_decimate_timing();
    void stop_timing();
    void send_ack();
    void send_timing();


 private:
    cudaStream_t stream;

    uint32_t sequence_num;
    zmq::socket_t *ack_socket;
    zmq::socket_t *timing_socket;

    float total_process_timing_ms;
    float decimate_kernel_timing_ms;

    cuComplex *rf_samples;
    size_t rf_samples_size;

    cuComplex *first_stage_bp_filters;
    size_t first_stage_bp_filters_size;

    cuComplex *second_stage_filters;
    size_t second_stage_filters_size;

    cuComplex *third_stage_filters;
    size_t third_stage_filters_size;

    cuComplex *first_stage_output;
    size_t first_stage_output_size;

    cuComplex *second_stage_output;
    size_t second_stage_output_size;

    cuComplex *third_stage_output;
    size_t third_stage_output_size;

    cuComplex *host_output;
    size_t host_output_size;

    cudaEvent_t initial_start, kernel_start, stop;

    SharedMemoryHandler *shr_mem;

    void callbackFunc();

};

#endif