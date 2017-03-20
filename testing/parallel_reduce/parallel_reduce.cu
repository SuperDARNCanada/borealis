#include <cuComplex.h>
#include <iostream>
#include <vector>

#define NUMELEMENTS 64
__device__ inline cuComplex __shfl_down(cuComplex var, unsigned int srcLane, int width=32){
    float2 a = *reinterpret_cast<float2*>(&var);
    a.x = __shfl_down(a.x, srcLane, width);
    a.y = __shfl_down(a.y, srcLane, width);
    return *reinterpret_cast<cuComplex*>(&a);
}

__device__ cuComplex parallel_reduce(cuComplex* data, int tap_offset) {

    auto filter_tap_num = threadIdx.x;
    auto num_filter_taps = blockDim.x;
    cuComplex total_sum = data[tap_offset];


    if ((num_filter_taps >= 512) && (filter_tap_num < 256))
    {
        data[tap_offset] = total_sum = cuCaddf(total_sum,data[tap_offset + 256]);
    }

    __syncthreads();

    if ((num_filter_taps >= 256) && (filter_tap_num < 128))
    {
            data[tap_offset] = total_sum = cuCaddf(total_sum, data[tap_offset + 128]);
    }

     __syncthreads();

    if ((num_filter_taps >= 128) && (filter_tap_num <  64))
    {
       data[tap_offset] = total_sum = cuCaddf(total_sum, data[tap_offset +  64]);
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

__device__ cuComplex parallel_reduce2(cuComplex* data, int tap_offset){


    for (unsigned int s=blockDim.x/2; s>32; s>>=1) {
        if (threadIdx.x < s)
            data[tap_offset] = cuCaddf(data[tap_offset],
                                                    data[tap_offset + s]);
        __syncthreads();
    }
    if (threadIdx.x < 32){
        data[tap_offset] = cuCaddf(data[tap_offset],
                                                data[tap_offset + 32]);
        __syncthreads();
        data[tap_offset] = cuCaddf(data[tap_offset],
                                                data[tap_offset + 16]);
        __syncthreads();
        data[tap_offset] = cuCaddf(data[tap_offset],
                                                data[tap_offset + 8]);
        __syncthreads();
        data[tap_offset] = cuCaddf(data[tap_offset],
                                                data[tap_offset + 4]);
        __syncthreads();
        data[tap_offset] = cuCaddf(data[tap_offset],
                                                data[tap_offset + 2]);
        __syncthreads();
        data[tap_offset] = cuCaddf(data[tap_offset],
                                                data[tap_offset + 1]);
        __syncthreads();
    }

    return data[0];
}

__global__ void add_numbers(cuComplex* data, cuComplex* reduced_sum){
    extern __shared__ cuComplex shr_data[];

    shr_data[threadIdx.x] = data[threadIdx.x];
    __syncthreads();

   *reduced_sum = parallel_reduce2(shr_data,threadIdx.x);

}


int main(){
    std::vector<cuComplex> data(NUMELEMENTS,make_cuComplex(1.0,1.0));
    cuComplex sum;

    cuComplex *data_d, *sum_d;

    size_t total_bytes = data.size() * sizeof(cuComplex);

    cudaMalloc(&data_d, total_bytes);
    cudaMalloc(&sum_d, sizeof(cuComplex));

    cudaMemcpy(data_d,data.data(),total_bytes,cudaMemcpyHostToDevice);

    dim3 dimGrid(1,1);
    dim3 dimBlock(NUMELEMENTS,1);
    add_numbers<<<dimGrid,dimBlock, total_bytes>>>(data_d,sum_d);

    cudaMemcpy(&sum,sum_d,sizeof(cuComplex),cudaMemcpyDeviceToHost);

    std::cout << "Resulting sum: (" << sum.x << "," << sum.y << ")" << std::endl;


}