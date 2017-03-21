#include <cuComplex.h>
#include <iostream>
#include <stdint.h>

/*Overloaded __shfl_down function. Default does not recognize cuComplex but
does for equivalent float2 type.
https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/
http://docs.nvidia.com/cuda/cuda-c-programming-guide/#warp-shuffle-functions
*/
__device__ inline cuComplex __shfl_down(cuComplex var, unsigned int srcLane, int width=32){
    float2 a = *reinterpret_cast<float2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<cuComplex*>(&a);
}

/*Slightly modified version of reduction #5 from NVIDIA examples
/usr/local/cuda/samples/6_Advanced/reduction
*/
__device__ cuComplex parallel_reduce(cuComplex* data, int tap_offset) {

    auto filter_tap_num = threadIdx.x;
    auto num_filter_taps = blockDim.x;
    cuComplex total_sum = data[tap_offset];


    if ((num_filter_taps >= 512) && (filter_tap_num < 256))
    {
        data[tap_offset] = total_sum = cuCaddf(total_sum,data[tap_offset  + 256]);
    }

    __syncthreads();

    if ((num_filter_taps >= 256) && (filter_tap_num < 128))
    {
            data[tap_offset] = total_sum = cuCaddf(total_sum, data[tap_offset + 128]);
    }

     __syncthreads();

    if ((num_filter_taps >= 128) && (filter_tap_num <  64))
    {
       data[tap_offset] = total_sum = cuCaddf(total_sum, data[tap_offset  +  64]);
    }

    __syncthreads();

    if ( filter_tap_num < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        if (num_filter_taps >=  64) total_sum = cuCaddf(total_sum, data[tap_offset + 32]);
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
            total_sum = cuCaddf(total_sum,__shfl_down(total_sum, offset));
        }
    }

    return total_sum;
}

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


    filter_products[tap_offset] = cuCmulf(sample,filter_taps[tap_offset]);

    __syncthreads();

    auto total_sum = parallel_reduce(filter_products, tap_offset);

    if (threadIdx.x == 0) {
        channel_offset = channel_num * samples_per_channel/dm_rate;
        auto total_channels = blockDim.y;
        auto freq_offset = threadIdx.y * total_channels;
        auto total_offset = freq_offset + channel_offset + dec_sample_num;
        decimated_samples[total_offset] = total_sum;
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

    filter_products[tap_offset] = cuCmulf(sample_1,filter_taps[tap_offset]);
    filter_products[tap_offset+1] = cuCmulf(sample_2, filter_taps[tap_offset+1]);

    __syncthreads();

    auto total_sum = parallel_reduce(filter_products, tap_offset);
    if (threadIdx.x == 0) {
        channel_offset = channel_num * samples_per_channel/dm_rate;
        auto total_channels = blockDim.y;
        auto freq_offset = threadIdx.y * total_channels;
        auto total_offset = freq_offset + channel_offset + dec_sample_num;
        decimated_samples[total_offset] = total_sum;
    }
}

static dim3 create_grid(uint32_t num_samples, uint32_t dm_rate, uint32_t num_channels)
{
    auto num_blocks_x = num_samples/dm_rate;
    auto num_blocks_y = num_channels;
    auto num_blocks_z = 1;
    std::cout << "    Grid size: " << num_blocks_x << " x " << num_blocks_y << " x "
        << num_blocks_z << std::endl;
    dim3 dimGrid(num_blocks_x,num_blocks_y,num_blocks_z);

    return dimGrid;
}

static dim3 create_block(uint32_t num_taps, uint32_t num_freqs)
{
    auto num_threads_x = num_taps;
    auto num_threads_y = num_freqs;
    auto num_threads_z = 1;
    std::cout << "    Block size: " << num_threads_x << " x " << num_threads_y << " x "
        << num_threads_z << std::endl;
    dim3 dimBlock(num_threads_x,num_threads_y,num_threads_z);

    return dimBlock;
}

void decimate1024_wrapper(cuComplex* original_samples,
    cuComplex* decimated_samples,
    cuComplex* filter_taps, uint32_t dm_rate,
    uint32_t samples_per_channel, uint32_t num_taps, uint32_t num_freqs,
    uint32_t num_channels, cudaStream_t stream) {

    auto shr_mem_taps = num_freqs * num_taps * sizeof(cuComplex);
    std::cout << "    Number of shared memory bytes: "<< shr_mem_taps << std::endl;

    auto dimGrid = create_grid(samples_per_channel, dm_rate, num_channels);
    auto dimBlock = create_block(num_taps,num_freqs);
    decimate1024<<<dimGrid,dimBlock,shr_mem_taps,stream>>>(original_samples, decimated_samples,
                filter_taps, dm_rate, samples_per_channel);

}

void decimate2048_wrapper(cuComplex* original_samples,
    cuComplex* decimated_samples,
    cuComplex* filter_taps, uint32_t dm_rate,
    uint32_t samples_per_channel, uint32_t num_taps, uint32_t num_freqs,
    uint32_t num_channels, cudaStream_t stream) {

    auto shr_mem_taps = num_freqs * num_taps * sizeof(cuComplex);
    std::cout << "    Number of shared memory bytes: "<< shr_mem_taps << std::endl;

    auto dimGrid = create_grid(samples_per_channel, dm_rate, num_channels);
    auto dimBlock = create_block(num_taps/2, num_freqs);
    decimate2048<<<dimGrid,dimBlock,shr_mem_taps,stream>>>(original_samples, decimated_samples,
        filter_taps, dm_rate, samples_per_channel);
}
