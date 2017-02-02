#include <stdio.h>
#include <thrust/complex.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
__global__ void EmptyKernel() {
    //extern __shared__ thrust::complex<float> filter_products[];
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

int main() {

    const int N = 100;

    float time, cumulative_time = 0.f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i=0; i<N; i++) {

        cudaEventRecord(start, 0);
        dim3 dimGrid(83333,20,1);
        dim3 dimBlock(1024);
        auto bytes = 1024 * sizeof(thrust::complex<float>);
        EmptyKernel<<<dimGrid,dimBlock>>>();
        throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        cumulative_time = cumulative_time + time;

    }

    printf("Kernel launch overhead time:  %3.5f ms \n", cumulative_time / N);
    return 0;
}