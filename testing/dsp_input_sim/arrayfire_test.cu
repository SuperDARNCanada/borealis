#include "arrayfire.h"
#include <af/cuda.h>
#include <vector>
#include <string>
#include <thread>
#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdint.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#define DATA_SIZE 10000
#define TEST_FREQ 1
#define RX_RATE 1
#define ANTS 1
#define TAPS 72
#define T_DEVICE_V(x) thrust::device_vector<x>
#define T_HOST_V(x) thrust::device_vector<x>
#define T_COMPLEX_F thrust::complex<float>

af::array my_mult (const af::array &lhs, const af::array &rhs){
    return lhs * rhs;
}

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

/*af::cfloat NCO_func(af::seq i,float sampling_freq) {
        float radians = fmod(sampling_freq * i,2 * M_PI);
        float I = cos(radians);
        float Q = sin(radians);

        af::cfloat sig = {I,Q};
        return sig;
}*/

__global__ void make_NCO (af::cfloat* NCO, float sampling_freq) {

    auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < DATA_SIZE) {
        auto radians = fmod(sampling_freq * tid, 2 * (float)M_PI);
        NCO[tid].real = cos(radians);
        NCO[tid].imag = sin(radians);
        //NCO[tid] = (af::cfloat)thrust::complex<float>(I,Q);
    }

}

int main(){
    af::array load_lib = af::randu(1);

    std::vector<std::complex<float>> data_1(ANTS*DATA_SIZE,std::complex<float>(1.0,1.0));
    std::vector<std::complex<float>> data_2(ANTS*DATA_SIZE,std::complex<float>(1.0,1.0));
    std::vector<std::complex<float>> data_3(ANTS*DATA_SIZE,std::complex<float>(1.0,1.0));
    std::vector<std::complex<float>> filter(TAPS,std::complex<float>(3.14,3.14));

    std::vector<std::complex<float>> v(DATA_SIZE);


    auto NCO_timing_start = std::chrono::steady_clock::now();
    auto sampling_freq = 2 * M_PI * TEST_FREQ/RX_RATE;
    for (uint i=0; i<v.size(); i++) {
        auto radians = fmod(sampling_freq * i,2 * M_PI);
        auto I = cos(radians);
        auto Q = sin(radians);
        v[i] = std::complex<float>(I,Q);
    }
    auto NCO_timing_end = std::chrono::steady_clock::now();
    std::cout << "Host NCO fill timing: "
        << std::chrono::duration_cast<std::chrono::milliseconds>(NCO_timing_end - NCO_timing_start).count()
        << "ms" << std::endl;
    for (int i=0; i<10; i++){
        std::cout << " " << v[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;





/*    cudaEvent_t start_t, stop_t;
    float time = 0.0;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);
    cudaEventRecord(start_t, 0);
    thrust::device_vector<thrust::complex<float>> thrust_NCO(DATA_SIZE);
    auto thrust_NCO_p = thrust::raw_pointer_cast(thrust_NCO.data());
    dim3 grid_dim(DATA_SIZE/1024);
    dim3 blk_dim(1024);
    make_NCO<<<grid_dim,blk_dim>>>(thrust_NCO_p,sampling_freq);
    throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_t, 0);
    cudaEventSynchronize(stop_t);
    cudaEventElapsedTime(&time, start_t, stop_t);
    printf("GPU user NCO time:  %3.5f ms \n", time);
    for (int i=0; i<10; i++){
        std::cout << " " << thrust_NCO[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;

    std::cout << "sizeof thrust complex " << sizeof(thrust::complex<float>)
        << std::endl;
    std::cout << "sizeof cfloat " << sizeof(af::cfloat)
        << std::endl;*/
    auto af_mix_timer = af::timer::start();
    af::array NCO(af::dim4(1,DATA_SIZE),c32);

    af::cfloat *NCO_d = NCO.device<af::cfloat>();
    printf("NCO maker: %g\n", af::timer::stop(af_mix_timer));
    af::sync();

    int af_id = af::getDevice();
    int cuda_id = afcu::getNativeId(af_id);
    cudaStream_t af_cuda_stream = afcu::getStream(cuda_id);

    dim3 grid_dim(1);
    dim3 blk_dim(1024);

    make_NCO<<<grid_dim,blk_dim,0,af_cuda_stream>>>(NCO_d,sampling_freq);
    throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);
    cudaDeviceSynchronize();
    NCO.unlock();
/*    for (int i=0; i<10; i++){
        af_print(NCO(i));
        std::cout << " ";
    }*/
    af::print("NCO",NCO);
    std::cout << std::endl;
    std::cout << std::endl;


    af_mix_timer = af::timer::start();
    auto ptr1 = reinterpret_cast<af::cfloat*>(data_1.data());
    af::array signal_array1 (1,DATA_SIZE,ANTS,ptr1);

    af::dim4 dims = NCO.dims();
    printf("dims = [%lld %lld %lld %lld]\n", dims[0], dims[1], dims[2], dims[3]);

    dims = signal_array1.dims();
    printf("dims = [%lld %lld %lld %lld]\n", dims[0], dims[1], dims[2], dims[3]);


    af::array mixed_sig = af::batchFunc(NCO,signal_array1,my_mult);
    throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);
    dims = mixed_sig.dims();
    printf("dims = [%lld %lld %lld %lld]\n", dims[0], dims[1], dims[2], dims[3]);
    af::print("Mixed sig", mixed_sig);
    printf("Mixing time: %g\n", af::timer::stop(af_mix_timer));


    af::timer af_conv_timer = af::timer::start();
    auto taps_ptr = reinterpret_cast<af::cfloat*>(filter.data());
    af::array filter_array (1,TAPS,1,taps_ptr);
    auto result = af::convolve(signal_array1,filter_array,AF_CONV_DEFAULT,AF_CONV_SPATIAL);
    printf("Convolve time: %g\n", af::timer::stop(af_conv_timer));

/*    auto timer = af::timer::start();

    auto ptr2 = reinterpret_cast<af::cfloat*>(data_2.data());
    auto ptr3 = reinterpret_cast<af::cfloat*>(data_3.data());
    auto ptr4 = reinterpret_cast<af::cfloat*>(filter.data());

    af::array signal_array1 (1,DATA_SIZE,ANTS,ptr1);
    af::array signal_array2 (1,DATA_SIZE,ANTS,ptr2);
    af::array signal_array3 (1,DATA_SIZE,ANTS,ptr3);
    af::array filter_array (1,TAPS,1,ptr4);

    auto result = af::convolve(signal_array1,filter_array,AF_CONV_DEFAULT,AF_CONV_SPATIAL);
    result = af::convolve(signal_array2,filter_array,AF_CONV_DEFAULT,AF_CONV_SPATIAL);
    result = af::convolve(signal_array3,filter_array,AF_CONV_DEFAULT,AF_CONV_SPATIAL);

    printf("elapsed seconds: %g\n", af::timer::stop(timer));*/
    //af::print("result",result);

    return 0;
}