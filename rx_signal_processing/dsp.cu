#include "dsp.hpp"
#include "utils/protobuf/sigprocpacket.pb.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cuComplex.h>
#include <chrono>
#include <thread>

// REVIEW #13 Multiple functions for 'allocate_and_copy', 'allocate_x_stage_output', and 'get_x_stage_filters', 'get_x_stage_output' could be combined into one for each?

extern void decimate1024_wrapper(cuComplex* original_samples,
    cuComplex* decimated_samples,
    cuComplex* filter_taps, uint32_t dm_rate,
    uint32_t samples_per_channel, uint32_t num_taps, uint32_t num_freqs,
    uint32_t num_channels, cudaStream_t stream);

extern void decimate2048_wrapper(cuComplex* original_samples,
    cuComplex* decimated_samples,
    cuComplex* filter_taps, uint32_t dm_rate,
    uint32_t samples_per_channel, uint32_t num_taps, uint32_t num_freqs,
    uint32_t num_channels, cudaStream_t stream);



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











DSPCore::DSPCore(zmq::socket_t *ack_s, zmq::socket_t *timing_s,
                                        uint32_t sq_num, const char* shr_mem_name) // REVIEW #28 why use c_str and not string?
{

    sequence_num = sq_num;
    ack_socket = ack_s;
    timing_socket = timing_s;

    gpuErrchk(cudaStreamCreate(&stream)); // REVIEW #1 explain what's going on here
    gpuErrchk(cudaEventCreate(&initial_start));
    gpuErrchk(cudaEventCreate(&kernel_start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(initial_start, stream));

    shr_mem = new SharedMemoryHandler(shr_mem_name);
    shr_mem->open_shr_mem();

}

void DSPCore::allocate_and_copy_rf_samples(uint32_t total_samples) 
{
// REVIEW #15 - What happens when total_samples is negative, 0 or really high?
    rf_samples_size = total_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&rf_samples, rf_samples_size));
    gpuErrchk(cudaMemcpyAsync(rf_samples,shr_mem->get_shrmem_addr(), rf_samples_size, cudaMemcpyHostToDevice, stream));

}

void DSPCore::allocate_and_copy_first_stage_filters(void *taps, uint32_t total_taps) // REVIEW #9 Consider passing in the size of taps so you don't assume it is a cuComplex (This goes for all allocate_and_copy functions)
{
    first_stage_bp_filters_size = total_taps * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&first_stage_bp_filters, first_stage_bp_filters_size));
    gpuErrchk(cudaMemcpyAsync(first_stage_bp_filters, taps,
                first_stage_bp_filters_size, cudaMemcpyHostToDevice, stream));
}

void DSPCore::allocate_and_copy_second_stage_filters(void *taps, uint32_t total_taps)
{
    second_stage_filters_size = total_taps * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&second_stage_filters, second_stage_filters_size));
    gpuErrchk(cudaMemcpyAsync(second_stage_filters, taps,
               second_stage_filters_size, cudaMemcpyHostToDevice, stream));
}

void DSPCore::allocate_and_copy_third_stage_filters(void *taps, uint32_t total_taps)
{
    third_stage_filters_size = total_taps * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&third_stage_filters, third_stage_filters_size));
    gpuErrchk(cudaMemcpyAsync(third_stage_filters, taps,
                third_stage_filters_size, cudaMemcpyHostToDevice, stream));
}

void DSPCore::allocate_first_stage_output(uint32_t first_stage_samples) // REVIEW # 26 'x_stage_samples' doesn't indicate that it is the number of samples. Same for 'host_samples'. Consider changing name to better reflect this
{
    first_stage_output_size = first_stage_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&first_stage_output, first_stage_output_size));
}

void DSPCore::allocate_second_stage_output(uint32_t second_stage_samples)
{
    second_stage_output_size = second_stage_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&second_stage_output, second_stage_output_size));
}

void DSPCore::allocate_third_stage_output(uint32_t third_stage_samples)
{
    third_stage_output_size = third_stage_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&third_stage_output, third_stage_output_size));
}

void DSPCore::allocate_and_copy_host_output(uint32_t host_samples)
{
    host_output_size = host_samples * sizeof(cuComplex);
    gpuErrchk(cudaHostAlloc(&host_output, host_output_size, cudaHostAllocDefault));
    gpuErrchk(cudaMemcpyAsync(host_output, third_stage_output,
                host_output_size, cudaMemcpyDeviceToHost,stream));
}

void DSPCore::copy_output_to_host()
{
    gpuErrchk(cudaMemcpy(host_output, third_stage_output,
               host_output_size, cudaMemcpyDeviceToHost));
}

void DSPCore::clear_device_and_destroy()
{
    gpuErrchk(cudaFree(rf_samples));
    gpuErrchk(cudaFree(first_stage_bp_filters));
    gpuErrchk(cudaFree(second_stage_filters));
    gpuErrchk(cudaFree(third_stage_filters));
    gpuErrchk(cudaFree(first_stage_output));
    gpuErrchk(cudaFree(second_stage_output));
    gpuErrchk(cudaFree(third_stage_output));
    gpuErrchk(cudaFreeHost(host_output));
    gpuErrchk(cudaEventDestroy(initial_start));
    gpuErrchk(cudaEventDestroy(kernel_start));
    gpuErrchk(cudaEventDestroy(stop));
    gpuErrchk(cudaStreamDestroy(stream));

    shr_mem->remove_shr_mem();
    delete shr_mem;

}

void DSPCore::stop_timing()
{
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));

    gpuErrchk(cudaEventElapsedTime(&total_process_timing_ms, initial_start, stop));
    gpuErrchk(cudaEventElapsedTime(&decimate_kernel_timing_ms, kernel_start, stop));

}

void DSPCore::send_timing()
{
    sigprocpacket::SigProcPacket sp;
    sp.set_kerneltime(decimate_kernel_timing_ms);
    sp.set_sequence_num(sequence_num);

    std::string s_msg_str;
    sp.SerializeToString(&s_msg_str);
    zmq::message_t s_msg(s_msg_str.size());
    memcpy ((void *) s_msg.data (), s_msg_str.c_str(), s_msg_str.size());

    timing_socket->send(s_msg);
    std::cout << "Sent timing after processing" << std::endl;

}

static void postprocess(DSPCore *dp)
{

    dp->stop_timing();
    dp->send_timing();
    std::cout << "Cuda kernel timing: " << dp->get_decimate_timing()
        << "ms" <<std::endl;
    std::cout << "Complete process timing: " << dp->get_total_timing()
        << "ms" <<std::endl;

    dp->clear_device_and_destroy(); // REVIEW #6 need a TODO here
    delete dp;
}

void CUDART_CB DSPCore::cuda_postprocessing_callback(cudaStream_t stream, cudaError_t status,
                                                        void *processing_data)
{
    gpuErrchk(status);
    std::thread start_pp(postprocess,static_cast<DSPCore*>(processing_data));
    start_pp.detach(); 
}

void DSPCore::send_ack()
{
    sigprocpacket::SigProcPacket sp;
    sp.set_sequence_num(sequence_num);

    std::string s_msg_str;
    sp.SerializeToString(&s_msg_str);
    zmq::message_t s_msg(s_msg_str.size());
    memcpy ((void *) s_msg.data (), s_msg_str.c_str(), s_msg_str.size());
    ack_socket->send(s_msg);
    std::cout << "Sent ack after copy" << std::endl;
}

void DSPCore::start_decimate_timing()
{
    gpuErrchk(cudaEventRecord(kernel_start, stream));
}

void initial_memcpy_callback_handler(DSPCore *dp)
{
    dp->send_ack();
    dp->start_decimate_timing();
}

void CUDART_CB DSPCore::initial_memcpy_callback(cudaStream_t stream, cudaError_t status, // REVIEW #0 Do you need to put CUDART_CB* ?
                                                void *processing_data)
{
    gpuErrchk(status);
    std::thread start_imc(initial_memcpy_callback_handler,
                            static_cast<DSPCore*>(processing_data));
    start_imc.join();

}
// REVIEW #1 should we comment this code to indicate filter_taps contains filters for each rx_freq?
void DSPCore::call_decimate(cuComplex* original_samples,
    cuComplex* decimated_samples,
    cuComplex* filter_taps, uint32_t dm_rate,
    uint32_t samples_per_channel, uint32_t num_taps, uint32_t num_freqs,
    uint32_t num_channels, const char *output_msg) { // REVIEW #26 -Again here channels/freqs/antennas is confused and needs to be consistent, maybe we avoid the word 'channel' altogether
// REVIEW #15 This function assumes filter_taps size has been set up properly, how do we assure this is done properly and we're not going out of bounds of the array? The function is called so it doesn't know it's working on the class' own private data, should it be set up this way? Shouldn't you just call_decimate from the while loop and have it act on its own private data? 
    std::cout << output_msg << std::endl; 


    auto gpu_properties = get_gpu_properties();


    //For now we have a kernel that will process 2 samples per thread if need be
    if (num_taps * num_freqs > 2 * gpu_properties[0].maxThreadsPerBlock) { // REVIEW #29 We should make gpu_device a config option, here it's just grabbing the first one but to be general we should make that a config option.
        //TODO(Keith) : handle error
    }
    else if (num_taps * num_freqs > gpu_properties[0].maxThreadsPerBlock) { // REVIEW #26 Where does the 1024 and 2048 in the function names come from? Is it from the fact that all our devices have 1024 max threads per block? 
        decimate2048_wrapper(original_samples, decimated_samples, filter_taps,  dm_rate,
            samples_per_channel, num_taps, num_freqs, num_channels, stream);
    }
    else { // REVIEW #30 Could the decimateXXXX_wrapper function be moved into call_decimate to reduce code size and duplication and readability?
        decimate1024_wrapper(original_samples, decimated_samples, filter_taps,  dm_rate,
            samples_per_channel, num_taps, num_freqs, num_channels, stream);
    }
    gpuErrchk(cudaPeekAtLastError()); // REVIEW #4 do we need to do a cudaDeviceSynchronize after calling cudaPeekAtLastError here?

}

cuComplex* DSPCore::get_rf_samples_p(){
    return rf_samples;
}

cuComplex* DSPCore::get_first_stage_bp_filters_p(){
    return first_stage_bp_filters;
}

cuComplex* DSPCore::get_second_stage_filters_p(){
    return second_stage_filters;
}

cuComplex* DSPCore::get_third_stage_filters_p(){
    return third_stage_filters;
}

cuComplex* DSPCore::get_first_stage_output_p(){
    return first_stage_output;
}

cuComplex* DSPCore::get_second_stage_output_p(){
    return second_stage_filters;
}

cuComplex* DSPCore::get_third_stage_output_p(){
    return third_stage_filters;
}

cudaStream_t DSPCore::get_cuda_stream(){
    return stream;
}

float DSPCore::get_total_timing()
{
    return total_process_timing_ms;
}

float DSPCore::get_decimate_timing()
{
    return decimate_kernel_timing_ms;
}



/*uint32_t DSPCore::get_sequence_num()
{
    return sequence_num;
}

zmq::socket_t* DSPCore::get_rctl_socket()
{
    return rctl_socket;
}

zmq::socket_t* DSPCore::get_timing_socket()
{
    return timing_socket;
}*/



