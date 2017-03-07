#include "digital_processing.hpp"
#include "utils/protobuf/sigprocpacket.pb.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cuComplex.h>
#include <chrono>
#include <thread>
#include "multithreading.h"

__global__ void decimate1024(cuComplex* original_samples,
    cuComplex* decimated_samples,
    cuComplex* filter_taps, uint32_t dm_rate,
    uint32_t samples_per_channel)
{

    extern __shared__ cuComplex filter_products[];

    auto channel_num = blockIdx.y;
    auto channel_offset = channel_num * samples_per_channel;

    auto dec_sample_num = blockIdx.x;
    auto dec_sample_offset = dec_sample_num * dm_rate;

    auto tap_offset = threadIdx.y * blockDim.y + threadIdx.x;

    cuComplex sample;
    if ((dec_sample_offset + threadIdx.x) >= samples_per_channel) {
        sample = make_cuComplex(0.0,0.0);
    }
    else {
        auto final_offset = channel_offset + dec_sample_offset + threadIdx.x;
        sample = original_samples[final_offset];
    }


    filter_products[threadIdx.x] = cuCmulf(sample,filter_taps[tap_offset]);

    __syncthreads();


    auto num_taps = blockDim.x;
    for (unsigned int s=num_taps/2; s>32; s>>=1) {
        if (tap_offset < s)
            filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                    filter_products[tap_offset + s]);
        __syncthreads();
    }
    if (tap_offset < 32){
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 32]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 16]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 8]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 4]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 2]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 1]);
    }

    if (threadIdx.x == 0) {
        channel_offset = channel_num * samples_per_channel/dm_rate;
        auto total_channels = blockDim.y;
        auto freq_offset = threadIdx.y * total_channels;
        auto total_offset = freq_offset + channel_offset + dec_sample_num;
        decimated_samples[total_offset] = filter_products[tap_offset];
    }
}

__global__ void decimate2048(cuComplex* original_samples,
    cuComplex* decimated_samples,
    cuComplex* filter_taps, uint32_t dm_rate,
    uint32_t samples_per_channel)
{

    extern __shared__ cuComplex filter_products[];

    auto channel_num = blockIdx.y;
    auto channel_offset = channel_num * samples_per_channel;

    auto dec_sample_num = blockIdx.x;
    auto dec_sample_offset = dec_sample_num * dm_rate;

    auto tap_offset = threadIdx.y * blockDim.y + 2 * threadIdx.x;

    cuComplex sample_1;
    cuComplex sample_2;
    if ((dec_sample_offset + 2 * threadIdx.x) >= samples_per_channel) {
        sample_1 = make_cuComplex(0.0,0.0);
        sample_2 = make_cuComplex(0.0,0.0);
    }
    else {
        auto final_offset = channel_offset + dec_sample_offset + 2*threadIdx.x;
        sample_1 = original_samples[final_offset];
        sample_2 = original_samples[final_offset+1];
    }


    filter_products[threadIdx.x] = cuCmulf(sample_1,filter_taps[tap_offset]);
    filter_products[threadIdx.x+1] = cuCmulf(sample_2, filter_taps[tap_offset+1]);

    __syncthreads();

    auto half_num_taps = blockDim.x;
    for (unsigned int s=half_num_taps; s>32; s>>=1) {
        if (tap_offset < s)
            filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                    filter_products[tap_offset + s]);
        __syncthreads();
    }
    if (tap_offset < 32){
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 32]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 16]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 8]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 4]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 2]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 1]);
    }

/*    auto half_num_taps = blockDim.x;
    for (unsigned int s=half_num_taps; s>32; s>>=1) {
        if (tap_offset < s)
            filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                    filter_products[tap_offset + s]);
            filter_products[tap_offset+1] = cuCaddf(filter_products[tap_offset+1],
                                                        filter_products[tap_offset+1 + s]);
        __syncthreads();
    }
    if (tap_offset < 32){
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 32]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 16]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 8]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 4]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 2]);
        filter_products[tap_offset] = cuCaddf(filter_products[tap_offset],
                                                filter_products[tap_offset + 1]);

        filter_products[tap_offset + 1] = cuCaddf(filter_products[tap_offset + 1],
                                                    filter_products[tap_offset + 1 + 32]);
        filter_products[tap_offset + 1] = cuCaddf(filter_products[tap_offset + 1],
                                                    filter_products[tap_offset + 1 + 16]);
        filter_products[tap_offset + 1] = cuCaddf(filter_products[tap_offset + 1],
                                                    filter_products[tap_offset + 1 + 8]);
        filter_products[tap_offset + 1] = cuCaddf(filter_products[tap_offset + 1],
                                                    filter_products[tap_offset + 1 + 4]);
        filter_products[tap_offset + 1] = cuCaddf(filter_products[tap_offset + 1],
                                                    filter_products[tap_offset + 1 + 2]);
        filter_products[tap_offset + 1] = cuCaddf(filter_products[tap_offset + 1],
                                                    filter_products[tap_offset + 1 + 1]);
    }*/
    if (threadIdx.x == 0) {
        channel_offset = channel_num * samples_per_channel/dm_rate;
        auto total_channels = blockDim.y;
        auto freq_offset = threadIdx.y * total_channels;
        auto total_offset = freq_offset + channel_offset + dec_sample_num;
        decimated_samples[total_offset] = filter_products[tap_offset];
    }
}

dim3 create_grid(uint32_t num_samples, uint32_t dm_rate, uint32_t num_channels)
{
    auto num_blocks_x = num_samples/dm_rate;
    auto num_blocks_y = num_channels;
    auto num_blocks_z = 1;
    std::cout << "    Grid size: " << num_blocks_x << " x " << num_blocks_y << " x "
        << num_blocks_z << std::endl;
    dim3 dimGrid(num_blocks_x,num_blocks_y,num_blocks_z);

    return dimGrid;
}

dim3 create_block(uint32_t num_taps, uint32_t num_freqs)
{
    auto num_threads_x = num_taps;
    auto num_threads_y = num_freqs;
    auto num_threads_z = 1;
    std::cout << "    Block size: " << num_threads_x << " x " << num_threads_y << " x "
        << num_threads_z << std::endl;
    dim3 dimBlock(num_threads_x,num_threads_y,num_threads_z);

    return dimBlock;
}

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


DigitalProcessing::DigitalProcessing(zmq::socket_t *ack_s, zmq::socket_t *timing_s,
                                        uint32_t sq_num, const char* shr_mem_name)
{

    sequence_num = sq_num;
    ack_socket = ack_s;
    timing_socket = timing_s;

    gpuErrchk(cudaStreamCreate(&stream));
    gpuErrchk(cudaEventCreate(&initial_start));
    gpuErrchk(cudaEventCreate(&kernel_start));
    gpuErrchk(cudaEventCreate(&stop));
    gpuErrchk(cudaEventRecord(initial_start, stream));

    shr_mem = new SharedMemoryHandler(shr_mem_name);
    shr_mem->open_shr_mem();

}

void DigitalProcessing::allocate_and_copy_rf_samples(uint32_t total_samples)
{

    rf_samples_size = total_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&rf_samples, rf_samples_size));
    gpuErrchk(cudaMemcpyAsync(rf_samples,shr_mem->get_shrmem_addr(), rf_samples_size, cudaMemcpyHostToDevice, stream));

}

void DigitalProcessing::allocate_and_copy_first_stage_filters(void *taps, uint32_t total_taps)
{
    first_stage_bp_filters_size = total_taps * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&first_stage_bp_filters, first_stage_bp_filters_size));
    gpuErrchk(cudaMemcpyAsync(first_stage_bp_filters, taps,
                first_stage_bp_filters_size, cudaMemcpyHostToDevice, stream));
}

void DigitalProcessing::allocate_and_copy_second_stage_filters(void *taps, uint32_t total_taps)
{
    second_stage_filters_size = total_taps * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&second_stage_filters, second_stage_filters_size));
    gpuErrchk(cudaMemcpyAsync(second_stage_filters, taps,
               second_stage_filters_size, cudaMemcpyHostToDevice, stream));
}

void DigitalProcessing::allocate_and_copy_third_stage_filters(void *taps, uint32_t total_taps)
{
    third_stage_filters_size = total_taps * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&third_stage_filters, third_stage_filters_size));
    gpuErrchk(cudaMemcpyAsync(third_stage_filters, taps,
                third_stage_filters_size, cudaMemcpyHostToDevice, stream));
}

void DigitalProcessing::allocate_first_stage_output(uint32_t first_stage_samples)
{
    first_stage_output_size = first_stage_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&first_stage_output, first_stage_output_size));
}

void DigitalProcessing::allocate_second_stage_output(uint32_t second_stage_samples)
{
    second_stage_output_size = second_stage_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&second_stage_output, second_stage_output_size));
}

void DigitalProcessing::allocate_third_stage_output(uint32_t third_stage_samples)
{
    third_stage_output_size = third_stage_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&third_stage_output, third_stage_output_size));
}

void DigitalProcessing::allocate_and_copy_host_output(uint32_t host_samples)
{
    host_output_size = host_samples * sizeof(cuComplex);
    gpuErrchk(cudaHostAlloc(&host_output, host_output_size, cudaHostAllocDefault));
    gpuErrchk(cudaMemcpyAsync(host_output, third_stage_output,
                host_output_size, cudaMemcpyDeviceToHost,stream));
}

void DigitalProcessing::copy_output_to_host()
{
    gpuErrchk(cudaMemcpy(host_output, third_stage_output,
               host_output_size, cudaMemcpyDeviceToHost));
}

void DigitalProcessing::clear_device_and_destroy()
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

void DigitalProcessing::stop_timing()
{
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));

    gpuErrchk(cudaEventElapsedTime(&total_process_timing_ms, initial_start, stop));
    gpuErrchk(cudaEventElapsedTime(&decimate_kernel_timing_ms, kernel_start, stop));

}

void DigitalProcessing::send_timing()
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

void postprocess(DigitalProcessing *dp)
{

    dp->stop_timing();
    dp->send_timing();
    std::cout << "Cuda kernel timing: " << dp->get_decimate_timing()
        << "ms" <<std::endl;
    std::cout << "Complete process timing: " << dp->get_total_timing()
        << "ms" <<std::endl;

    dp->clear_device_and_destroy();
    delete dp;
}


void CUDART_CB DigitalProcessing::cuda_postprocessing_callback(cudaStream_t stream, cudaError_t status,
                                                        void *processing_data)
{
    gpuErrchk(status);
    std::thread start_pp(postprocess,static_cast<DigitalProcessing*>(processing_data));
    start_pp.detach();
}


void DigitalProcessing::send_ack()
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

void DigitalProcessing::start_decimate_timing()
{
    gpuErrchk(cudaEventRecord(kernel_start, stream));
}

void initial_memcpy_callback_handler(DigitalProcessing *dp)
{
    dp->send_ack();
    dp->start_decimate_timing();
}

void CUDART_CB DigitalProcessing::initial_memcpy_callback(cudaStream_t stream, cudaError_t status,
                                                void *processing_data)
{
    gpuErrchk(status);
    std::thread start_imc(initial_memcpy_callback_handler,
                            static_cast<DigitalProcessing*>(processing_data));
    start_imc.join();

}

void DigitalProcessing::call_decimate(cuComplex* original_samples,
    cuComplex* decimated_samples,
    cuComplex* filter_taps, uint32_t dm_rate,
    uint32_t samples_per_channel, uint32_t num_taps, uint32_t num_freqs,
    uint32_t num_channels, const char *output_msg) {

    std::cout << output_msg << std::endl;


    auto gpu_properties = get_gpu_properties();
    auto shr_mem_taps = num_taps * sizeof(cuComplex);
    std::cout << "    Number of shared memory bytes: "<< shr_mem_taps << std::endl;

    auto dimGrid = create_grid(samples_per_channel, dm_rate, num_channels);



    //For now we have a kernel that will process 2 samples per thread if need be
    if (num_taps * num_freqs > 2 * gpu_properties[0].maxThreadsPerBlock) {
        //TODO(Keith) : handle error
    }
    else if (num_taps * num_freqs > gpu_properties[0].maxThreadsPerBlock) {
        auto dimBlock = create_block(num_taps/2, num_freqs);
        decimate2048<<<dimGrid,dimBlock,shr_mem_taps,stream>>>(original_samples, decimated_samples,
            filter_taps, dm_rate, samples_per_channel);
    }
    else {
        auto dimBlock = create_block(num_taps,num_freqs);
        decimate1024<<<dimGrid,dimBlock,shr_mem_taps,stream>>>(original_samples, decimated_samples,
            filter_taps, dm_rate, samples_per_channel);;
    }
    gpuErrchk(cudaPeekAtLastError());

}

cuComplex* DigitalProcessing::get_rf_samples_p(){
    return rf_samples;
}

cuComplex* DigitalProcessing::get_first_stage_bp_filters_p(){
    return first_stage_bp_filters;
}

cuComplex* DigitalProcessing::get_second_stage_filters_p(){
    return second_stage_filters;
}

cuComplex* DigitalProcessing::get_third_stage_filters_p(){
    return third_stage_filters;
}

cuComplex* DigitalProcessing::get_first_stage_output_p(){
    return first_stage_output;
}

cuComplex* DigitalProcessing::get_second_stage_output_p(){
    return second_stage_filters;
}

cuComplex* DigitalProcessing::get_third_stage_output_p(){
    return third_stage_filters;
}

cudaStream_t DigitalProcessing::get_cuda_stream(){
    return stream;
}

float DigitalProcessing::get_total_timing()
{
    return total_process_timing_ms;
}

float DigitalProcessing::get_decimate_timing()
{
    return decimate_kernel_timing_ms;
}



/*uint32_t DigitalProcessing::get_sequence_num()
{
    return sequence_num;
}

zmq::socket_t* DigitalProcessing::get_rctl_socket()
{
    return rctl_socket;
}

zmq::socket_t* DigitalProcessing::get_timing_socket()
{
    return timing_socket;
}*/



