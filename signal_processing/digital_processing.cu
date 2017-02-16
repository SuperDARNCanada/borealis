#include "digital_processing.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cuComplex.h>
#include <chrono>
#include "multithreading.h"

//#define gpuErrchk(ans) { throw_on_cuda_error((ans), __FILE__, __LINE__); }



__global__ void decimate1024(cuComplex* original_samples,
    cuComplex* decimated_samples,
    cuComplex* filter_taps, uint32_t dm_rate,
    uint32_t samples_per_channel) {

    extern __shared__ cuComplex filter_products[];

    auto channel_num = blockIdx.y;
    auto channel_offset = channel_num * samples_per_channel;

    auto dec_sample_num = blockIdx.x;
    auto dec_sample_offset = dec_sample_num * dm_rate;

    auto tap_offset = threadIdx.y * blockDim.y + threadIdx.x;

    cuComplex sample;
    if ((dec_sample_offset + threadIdx.x) >= samples_per_channel) {
/*        sample.real = 0.0;
        sample.imag = 0.0;*/
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
    uint32_t samples_per_channel) {

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

dim3 create_grid(uint32_t num_samples, uint32_t dm_rate, uint32_t num_channels){
    auto num_blocks_x = num_samples/dm_rate;
    auto num_blocks_y = num_channels;
    auto num_blocks_z = 1;
    std::cout << "    Grid size: " << num_blocks_x << " x " << num_blocks_y << " x "
        << num_blocks_z << std::endl;
    dim3 dimGrid(num_blocks_x,num_blocks_y,num_blocks_z);

    return dimGrid;
}

dim3 create_block(uint32_t num_taps, uint32_t num_freqs) {
    auto num_threads_x = num_taps;
    auto num_threads_y = num_freqs;
    auto num_threads_z = 1;
    std::cout << "    Block size: " << num_threads_x << " x " << num_threads_y << " x "
        << num_threads_z << std::endl;
    dim3 dimBlock(num_threads_x,num_threads_y,num_threads_z);

    return dimBlock;
}

std::vector<cudaDeviceProp> get_gpu_properties(){
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


DigitalProcessing::DigitalProcessing() {
    gpuErrchk(cudaStreamCreate(&stream));

    gpuErrchk(cudaEventCreate(&start));
    gpuErrchk(cudaEventCreate(&stop));

    gpuErrchk(cudaEventRecord(start));

    gpuErrchk(cudaEventRecord(start, stream));

}

void DigitalProcessing::allocate_and_copy_rf_samples(void *data, uint32_t total_samples) {

    rf_samples_size = total_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&rf_samples, rf_samples_size));
    gpuErrchk(cudaMemcpyAsync(rf_samples,data, rf_samples_size, cudaMemcpyHostToDevice, stream));

}

void DigitalProcessing::allocate_and_copy_first_stage_filters(void *taps, uint32_t total_taps) {
    first_stage_bp_filters_size = total_taps * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&first_stage_bp_filters, first_stage_bp_filters_size));

    gpuErrchk(cudaMemcpyAsync(first_stage_bp_filters, taps,
                first_stage_bp_filters_size, cudaMemcpyHostToDevice, stream));
}

void DigitalProcessing::allocate_and_copy_second_stage_filters(void *taps, uint32_t total_taps) {
    second_stage_filters_size = total_taps * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&second_stage_filters, second_stage_filters_size));

    gpuErrchk(cudaMemcpyAsync(second_stage_filters, taps,
                second_stage_filters_size, cudaMemcpyHostToDevice, stream));
}

void DigitalProcessing::allocate_and_copy_third_stage_filters(void *taps, uint32_t total_taps) {
    third_stage_filters_size = total_taps * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&third_stage_filters, third_stage_filters_size));

    gpuErrchk(cudaMemcpyAsync(third_stage_filters, taps,
                third_stage_filters_size, cudaMemcpyHostToDevice, stream));
}

void DigitalProcessing::allocate_first_stage_output(uint32_t first_stage_samples) {
    first_stage_output_size = first_stage_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&first_stage_output, first_stage_output_size));
}

void DigitalProcessing::allocate_second_stage_output(uint32_t second_stage_samples) {
    second_stage_output_size = second_stage_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&second_stage_output, second_stage_output_size));
}

void DigitalProcessing::allocate_third_stage_output(uint32_t third_stage_samples) {
    third_stage_output_size = third_stage_samples * sizeof(cuComplex);
    gpuErrchk(cudaMalloc(&third_stage_output, third_stage_output_size));
}

void DigitalProcessing::allocate_host_output(uint32_t host_samples) {
    host_output_size = host_samples * sizeof(std::complex<float>);
    //gpuErrchk(cudaHostAlloc(&host_output, host_output_size, cudaHostAllocDefault));
    host_output = std::vector<std::complex<float>>(host_samples);
}

void DigitalProcessing::copy_output_to_host() {
    gpuErrchk(cudaMemcpy(host_output.data(), third_stage_output,
                host_output_size, cudaMemcpyDeviceToHost));
}

void DigitalProcessing::clear_device_and_destroy(){
    gpuErrchk(cudaFree(rf_samples));
    gpuErrchk(cudaFree(first_stage_bp_filters));
    gpuErrchk(cudaFree(second_stage_filters));
    gpuErrchk(cudaFree(third_stage_filters));
    gpuErrchk(cudaFree(first_stage_output));
    gpuErrchk(cudaFree(second_stage_output));
    gpuErrchk(cudaFree(third_stage_output));
    host_output.clear();
    //gpuErrchk(cudaFreeHost(host_output));
    gpuErrchk(cudaStreamDestroy(stream));

    //delete this;

}

CUT_THREADPROC postprocess(void *void_arg)
{
    DigitalProcessing *dp = static_cast<DigitalProcessing*>(void_arg);
    // ... GPU is done with processing, continue on new CPU thread...
    std::chrono::steady_clock::time_point timing_start = std::chrono::steady_clock::now();
    dp->copy_output_to_host();
    std::chrono::steady_clock::time_point timing_end = std::chrono::steady_clock::now();
    std::cout << "Time to copy back to host: "
      << std::chrono::duration_cast<std::chrono::milliseconds>
                                                  (timing_end - timing_start).count()
      << "ms" << std::endl;


    dp->report_timing();
    dp->clear_device_and_destroy();
    delete dp;
    CUT_THREADEND;
}

void CUDART_CB DigitalProcessing::cuda_stream_callback(cudaStream_t stream, cudaError_t status,
                                                        void *processing_data)
{
    gpuErrchk(status);
    cutStartThread(postprocess, processing_data);
    //dp->callback_postprocess();
}

/*void DigitalProcessing::callback_postprocess()
{
    cutStartThread(postprocess_data, data);
}*/

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

void DigitalProcessing::report_timing(){
    gpuErrchk(cudaEventRecord(stop, stream));
    gpuErrchk(cudaEventSynchronize(stop));

    float milliseconds = 0;
    gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Complete memory transfer and decimation timing: " << milliseconds
        << "ms" <<std::endl;

}
