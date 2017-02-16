#include <complex>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>


#define T_DEVICE_V(x) thrust::device_vector<x>
#define T_HOST_V(x) thrust::host_vector<x>
#define T_COMPLEX_F thrust::complex<float>

#define NUM_SAMPLES 8333
#define NUM_CHANNELS 20
#define NUM_FILTER_TAPS 360
#define NUM_FREQUENCIES 3
#define DECIMATION_RATE 30


void throw_on_cuda_error(cudaError_t code, const char *file, int line)
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


//I have recently learned this technique is similar to strided
//convolution.
__global__ void decimatev1(T_COMPLEX_F* original_samples,
    T_COMPLEX_F* decimated_samples,
    T_COMPLEX_F* filter_taps, uint32_t dm_rate,
    uint32_t num_original_samples) {

    extern __shared__ T_COMPLEX_F filter_products[];

    //These offsets are just to index the correct spot in a flattened array.
    auto channel_num = blockIdx.y;
    auto channel_offset = channel_num * num_original_samples;

    auto dec_sample_num = blockIdx.x;
    auto dec_sample_offset = dec_sample_num * dm_rate;

    auto tap_offset = threadIdx.x;
    auto bp_filter_offset = blockIdx.z * blockDim.x;

    T_COMPLEX_F sample;
    //If thread runs out of bounds on the samples...
    //if ((dec_sample_offset + tap_offset) >= num_original_samples) {
       // sample = T_COMPLEX_F(0.0,0.0);
    //}
    //else {
        auto final_offset = channel_offset + dec_sample_offset + tap_offset;
        sample = original_samples[final_offset];
    //}


    filter_products[tap_offset] = sample * filter_taps[bp_filter_offset + tap_offset];

    __syncthreads();


    //Simple parallel sum/reduction algorithm
    //http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
    auto num_taps = blockDim.x;
    for(uint32_t stride=num_taps/2; stride>0; stride>>=1) {
        if (tap_offset < stride) {
            filter_products[tap_offset] = filter_products[tap_offset] +
                                            filter_products[tap_offset + stride];

        }
        __syncthreads();
    }

    //Calculating the output sample index.
    if (tap_offset == 0) {
        channel_offset = channel_num * num_original_samples/dm_rate;
        auto total_channels = blockDim.y;
        auto freq_offset = blockIdx.z * total_channels;
        auto total_offset = freq_offset + channel_offset + dec_sample_num;
        decimated_samples[total_offset] = filter_products[tap_offset];
    }

}



int roundUp(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}


__global__ void decimatev2(T_COMPLEX_F* original_samples,
    T_COMPLEX_F* decimated_samples,
    T_COMPLEX_F* filter_taps, uint32_t dm_rate,
    uint32_t num_original_samples) {

    extern __shared__ T_COMPLEX_F filter_products[];

    auto channel_num = blockIdx.y;
    auto channel_offset = channel_num * num_original_samples;

    auto dec_sample_num = blockIdx.x;
    auto dec_sample_offset = dec_sample_num * dm_rate;

    auto tap_offset = threadIdx.y * blockDim.y + 2 * threadIdx.x;

    T_COMPLEX_F sample_1;
    T_COMPLEX_F sample_2;
    if ((dec_sample_offset + 2 * threadIdx.x) >= num_original_samples) {
        sample_1 = T_COMPLEX_F(0.0,0.0);
        sample_2 = T_COMPLEX_F(0.0,0.0);
    }
    else {
        auto final_offset = channel_offset + dec_sample_offset + 2*threadIdx.x;
        sample_1 = original_samples[final_offset];
        sample_2 = original_samples[final_offset+1];
    }


    filter_products[threadIdx.x] = sample_1 * filter_taps[tap_offset];
    filter_products[threadIdx.x+1] = sample_2 * filter_taps[tap_offset+1];

    __syncthreads();


    //Simple parallel sum/reduction algorithm
    //http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
    auto half_num_taps = blockDim.x;
/*    for(uint32_t stride=half_num_taps; stride>0; stride>>=1) {
        if (tap_offset < stride) {
            filter_products[tap_offset] = filter_products[tap_offset] +
                                            filter_products[tap_offset + stride];
            filter_products[tap_offset + 1] = filter_products[tap_offset + 1] +
                                            filter_products[tap_offset + 1 + stride];

        }
        __syncthreads();
    }*/
    for (unsigned int s=half_num_taps; s>32; s>>=1) {
        if (tap_offset < s)
            filter_products[tap_offset] += filter_products[tap_offset + s];
            filter_products[tap_offset+1] += filter_products[tap_offset+1 + s];
        __syncthreads();
    }
    if (tap_offset < 32){
        filter_products[tap_offset] += filter_products[tap_offset + 32];
        filter_products[tap_offset] += filter_products[tap_offset + 16];
        filter_products[tap_offset] += filter_products[tap_offset + 8];
        filter_products[tap_offset] += filter_products[tap_offset + 4];
        filter_products[tap_offset] += filter_products[tap_offset + 2];
        filter_products[tap_offset] += filter_products[tap_offset + 1];

        filter_products[tap_offset + 1] += filter_products[tap_offset + 1 + 32];
        filter_products[tap_offset + 1] += filter_products[tap_offset + 1 + 16];
        filter_products[tap_offset + 1] += filter_products[tap_offset + 1 + 8];
        filter_products[tap_offset + 1] += filter_products[tap_offset + 1 + 4];
        filter_products[tap_offset + 1] += filter_products[tap_offset + 1 + 2];
        filter_products[tap_offset + 1] += filter_products[tap_offset + 1 + 1];
    }
    if (threadIdx.x == 0) {
        channel_offset = channel_num * num_original_samples/dm_rate;
        auto total_channels = blockDim.y;
        auto freq_offset = threadIdx.y * total_channels;
        auto total_offset = freq_offset + channel_offset + dec_sample_num;
        decimated_samples[total_offset] = filter_products[tap_offset];
    }
}

__global__ void decimatev3(T_COMPLEX_F* original_samples,
    T_COMPLEX_F* decimated_samples,
    T_COMPLEX_F* filter_taps, uint32_t dm_rate,
    uint32_t num_original_samples) {

    extern __shared__ T_COMPLEX_F filter_products[];

    auto channel_num = blockIdx.y;
    auto channel_offset = channel_num * num_original_samples;

    auto dec_sample_num = blockIdx.x;
    auto dec_sample_offset = dec_sample_num * dm_rate;

    auto tap_offset = threadIdx.y * blockDim.y + threadIdx.x;

    T_COMPLEX_F sample;
    if ((dec_sample_offset + threadIdx.x) >= num_original_samples) {
        sample = T_COMPLEX_F(0.0,0.0);
    }
    else {
        auto final_offset = channel_offset + dec_sample_offset + threadIdx.x;
        sample = original_samples[final_offset];
    }


    filter_products[threadIdx.x] = sample * filter_taps[tap_offset];

    __syncthreads();


    //Simple parallel sum/reduction algorithm
    //http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
    auto num_taps = blockDim.x;
/*    for(uint32_t stride=num_taps/2; stride>0; stride>>=1) {
        if (tap_offset < stride) {
            filter_products[tap_offset] = filter_products[tap_offset] +
                                            filter_products[tap_offset + stride];

        }
        __syncthreads();
    }*/
    for (unsigned int s=num_taps/2; s>32; s>>=1) {
        if (tap_offset < s)
            filter_products[tap_offset] += filter_products[tap_offset + s];
        __syncthreads();
    }
    if (tap_offset < 32){
        filter_products[tap_offset] += filter_products[tap_offset + 32];
        filter_products[tap_offset] += filter_products[tap_offset + 16];
        filter_products[tap_offset] += filter_products[tap_offset + 8];
        filter_products[tap_offset] += filter_products[tap_offset + 4];
        filter_products[tap_offset] += filter_products[tap_offset + 2];
        filter_products[tap_offset] += filter_products[tap_offset + 1];
    }

    if (threadIdx.x == 0) {
        channel_offset = channel_num * num_original_samples/dm_rate;
        auto total_channels = blockDim.y;
        auto freq_offset = threadIdx.y * total_channels;
        auto total_offset = freq_offset + channel_offset + dec_sample_num;
        decimated_samples[total_offset] = filter_products[tap_offset];
    }
}

int main() {
    /**
    Thrust vectors seemed to give me less trouble than cudaMalloc, but they
    don't support multidimensional access.
    */

    //samples are in a NUM_CHANNEL x NUM_SAMPLES vector, but flattened
    T_HOST_V(T_COMPLEX_F) samples_h(NUM_CHANNELS * NUM_SAMPLES,T_COMPLEX_F(1.0,1.0));

    //Filter taps are in a NUM_FREQUENCIES * NUM_FILTER_TAPS vector, but flattened
    T_HOST_V(T_COMPLEX_F) filter_h(NUM_FREQUENCIES * NUM_FILTER_TAPS,T_COMPLEX_F(1.0,1.0));

    T_DEVICE_V(T_COMPLEX_F) samples_d = samples_h;
    T_DEVICE_V(T_COMPLEX_F) filter_d = filter_h;
    T_DEVICE_V(T_COMPLEX_F) output_d(NUM_SAMPLES/DECIMATION_RATE * NUM_CHANNELS);
    //T_DEVICE_V(T_COMPLEX_F) filter_prods(NUM_SAMPLES/DECIMATION_RATE * NUM_CHANNELS*NUM_FREQUENCIES)

    // A block.x for each output sample
    auto num_blocks_x = NUM_SAMPLES/DECIMATION_RATE;

    // A block.y for each channel
    auto num_blocks_y = NUM_CHANNELS;

    // A block.z for each frequency
    auto num_blocks_z = 1;

    dim3 dimGrid(num_blocks_x,num_blocks_y,num_blocks_z);

    // Dedicate a thread for every filter tap
    auto num_threads_x = NUM_FILTER_TAPS/2;
    auto num_threads_y = NUM_FREQUENCIES;

    dim3 dimBlock(num_threads_x,num_threads_y);

    float time = 0.0;
    cudaEvent_t start_t, stop_t;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);
    cudaEventRecord(start_t, 0);

    auto samples_p = thrust::raw_pointer_cast(samples_d.data());
    auto output_p = thrust::raw_pointer_cast(output_d.data());
    auto filter_p = thrust::raw_pointer_cast(filter_d.data());
    auto shr_mem_taps = NUM_FILTER_TAPS * sizeof(T_COMPLEX_F) * NUM_FREQUENCIES;

    decimatev2<<<dimGrid,dimBlock,shr_mem_taps>>>(samples_p, output_p, filter_p,
                DECIMATION_RATE, NUM_SAMPLES);
    throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);
    cudaDeviceSynchronize();

    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&time, start_t, stop_t);
    printf("decimate %d samples timing:  %3.5f ms \n",NUM_SAMPLES, time);

}