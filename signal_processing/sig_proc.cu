#include <vector>
#include <string>
#include <zmq.hpp>
#include <thread>
#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdint.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include "utils/protobuf/computationpacket.pb.h"
#include "utils/protobuf/receiverpacket.pb.h"
#include "utils/driver_options/driveroptions.hpp"

extern "C" {
    #include "remez.h"
}

#define T_DEVICE_V(x) thrust::device_vector<x>
#define T_HOST_V(x) thrust::device_vector<x>
#define T_COMPLEX_F thrust::complex<float>

#define FIRST_STAGE_SAMPLE_RATE 1.0e6 //1 MHz
#define SECOND_STAGE_SAMPLE_RATE 0.1e6 // 100 kHz
#define THIRD_STAGE_SAMPLE_RATE (10000.0/3.0) //3.33 kHz

#define FIRST_STAGE_FILTER_CUTOFF 1.0e6
#define FIRST_STAGE_FILTER_TRANSITION (FIRST_STAGE_FILTER_CUTOFF * 0.5)

#define SECOND_STAGE_FILTER_CUTOFF 0.1e6
#define SECOND_STAGE_FILTER_TRANSITION (SECOND_STAGE_FILTER_CUTOFF * 0.5)

#define THIRD_STAGE_FILTER_CUTOFF (10000.0/3.0)
#define THIRD_STAGE_FILTER_TRANSITION (THIRD_STAGE_FILTER_CUTOFF * 0.25)

#define k 3 //from formula 7-6

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

std::vector<double> create_normalized_lowpass_filter_bands(float cutoff, float transition_band,
                        float Fs) {
    std::vector<double> filterbands;
    filterbands.push_back(0.0);
    filterbands.push_back(cutoff/Fs);
    filterbands.push_back((cutoff + transition_band)/Fs);
    filterbands.push_back(0.5);

    return filterbands;
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

std::vector<cudaDeviceProp> get_gpu_properties(){
    std::vector<cudaDeviceProp> gpu_properties;
    int num_devices = 0;

    throw_on_cuda_error(cudaGetDeviceCount(&num_devices), __FILE__,__LINE__);

    for(int i=0; i<num_devices; i++) {
            cudaDeviceProp properties;
            cudaGetDeviceProperties(&properties, i);
            gpu_properties.push_back(properties);
    }

    return gpu_properties;
}

__global__ void decimate(T_COMPLEX_F* original_samples,
    T_COMPLEX_F* decimated_samples,
    T_COMPLEX_F* filter_taps, uint32_t num_taps,
    uint32_t num_original_samples) {

    extern __shared__ T_COMPLEX_F filter_products[];

    auto channel_num = blockIdx.y;
    auto channel_idx = channel_num * num_original_samples;


    auto dec_sample_num = blockIdx.x;
    auto dec_sample_idx = dec_sample_num * blockDim.x;

    auto tap_idx = threadIdx.x;

    T_COMPLEX_F sample;
    if ((dec_sample_idx + tap_idx) >= num_original_samples) {
        sample = T_COMPLEX_F(0.0,0.0);
    }
    else {
        auto final_idx = channel_idx + dec_sample_idx + tap_idx;
        sample = original_samples[final_idx];
    }


    filter_products[tap_idx] = sample * filter_taps[tap_idx];

    __syncthreads();


    //Simple parallel sum/reduction algorithm
    //http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
    for(uint32_t stride=num_taps/2; stride>0; stride>>=1) {
        if (tap_idx < stride) {
            filter_products[tap_idx] = filter_products[tap_idx] +
                                            filter_products[tap_idx + stride];

        }
        __syncthreads();
    }

    if (tap_idx == 0) {
        channel_idx = channel_num * blockDim.x;
        decimated_samples[channel_idx + dec_sample_num] = filter_products[tap_idx];
    }

}


void print_gpu_properties(std::vector<cudaDeviceProp> gpu_properties) {
    for(auto i : gpu_properties) {
        std::cout << "Device name: " << i.name << std::endl;
        std::cout << "  Max grid size x: " << i.maxGridSize[0] << std::endl;
        std::cout << "  Max grid size y: " << i.maxGridSize[1] << std::endl;
        std::cout << "  Max grid size z: " << i.maxGridSize[2] << std::endl;
        std::cout << "  Max threads per block: " << i.maxThreadsPerBlock
            << std::endl;
        std::cout << "  Max size of block dimension x: " << i.maxThreadsDim[0]
                << std::endl;
        std::cout << "  Max size of block dimension y: " << i.maxThreadsDim[1]
                << std::endl;
        std::cout << "  Max size of block dimension z: " << i.maxThreadsDim[2]
                << std::endl;
        std::cout << "  Memory Clock Rate (GHz): " << i.memoryClockRate/1e6
            << std::endl;
        std::cout << "  Memory Bus Width (bits): " << i.memoryBusWidth
            << std::endl;
        std::cout << "  Peak Memory Bandwidth (GB/s): " <<
           2.0*i.memoryClockRate*(i.memoryBusWidth/8)/1.0e6 << std::endl;
        std::cout << "  Max shared memory per block: " << i.sharedMemPerBlock
            << std::endl;
    }
}

int main(int argc, char **argv){
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto driver_options = DriverOptions();
    auto rx_rate = driver_options.get_rx_rate();

    zmq::context_t sig_proc_context(1);

    zmq::socket_t driver_socket(sig_proc_context, ZMQ_PAIR);
    driver_socket.bind("ipc:///tmp/feeds/1");
    zmq::message_t driver_request;

    zmq::socket_t radarctrl_socket(sig_proc_context, ZMQ_PAIR);
    radarctrl_socket.bind("ipc:///tmp/feeds/2");
    zmq::message_t radctl_request;

    auto gpu_properties = get_gpu_properties();
    print_gpu_properties(gpu_properties);

    uint32_t first_stage_dm_rate, second_stage_dm_rate, third_stage_dm_rate;
    if (fmod(rx_rate,FIRST_STAGE_SAMPLE_RATE) > 0.0){
        //TODO(keith): handle error
    }
    else{
        auto rate_f = rx_rate/FIRST_STAGE_SAMPLE_RATE;
        first_stage_dm_rate = static_cast<uint32_t>(rate_f);

        rate_f = FIRST_STAGE_SAMPLE_RATE/SECOND_STAGE_SAMPLE_RATE;
        second_stage_dm_rate = static_cast<uint32_t>(rate_f);

        rate_f = SECOND_STAGE_SAMPLE_RATE/THIRD_STAGE_SAMPLE_RATE;
        third_stage_dm_rate = static_cast<uint32_t>(rate_f);
    }

    std::cout << "1st stage dm rate: " << first_stage_dm_rate << std::endl
        << "2nd stage taps: " << second_stage_dm_rate << std::endl
        << "3rd stage taps: " << third_stage_dm_rate <<std::endl;


    auto S_lowpass1 = k * (rx_rate/FIRST_STAGE_FILTER_TRANSITION);
    auto S_lowpass2 = k * (FIRST_STAGE_SAMPLE_RATE/SECOND_STAGE_FILTER_TRANSITION);
    auto S_lowpass3 = k * (SECOND_STAGE_SAMPLE_RATE/THIRD_STAGE_FILTER_TRANSITION);

    std::cout << "1st stage taps: " << S_lowpass1 << std::endl << "2nd stage taps: "
        << S_lowpass2 << std::endl << "3rd stage taps: " << S_lowpass3 <<std::endl;


    std::chrono::steady_clock::time_point timing_start = std::chrono::steady_clock::now();
    std::vector<double> filterbands_1;
    filterbands_1 = create_normalized_lowpass_filter_bands(FIRST_STAGE_FILTER_CUTOFF,
                        FIRST_STAGE_FILTER_TRANSITION, rx_rate);

    std::vector<double> filterbands_2;
    filterbands_2 = create_normalized_lowpass_filter_bands(SECOND_STAGE_FILTER_CUTOFF,
                        SECOND_STAGE_FILTER_TRANSITION, FIRST_STAGE_SAMPLE_RATE);

    std::vector<double> filterbands_3;
    filterbands_3 = create_normalized_lowpass_filter_bands(THIRD_STAGE_FILTER_CUTOFF,
                        THIRD_STAGE_FILTER_TRANSITION, SECOND_STAGE_SAMPLE_RATE);

    std::vector<double> filtertaps_1(S_lowpass1+1);
    std::vector<double> filtertaps_2(S_lowpass2+1);
    std::vector<double> filtertaps_3(S_lowpass3+1);

    std::vector<double> desired_band_gain = {1.0,0.0};
    std::vector<double> weight = {1.0,1.0};

    auto converges = remez(filtertaps_1.data(),filtertaps_1.capacity(),(filterbands_1.size()/2),
        filterbands_1.data(),desired_band_gain.data(),weight.data(),BANDPASS,GRIDDENSITY);
    if (converges < 0){
        std::cerr << "Filter 1 failed to converge!" << std::endl;
        //TODO(keith): throw error
    }

    converges = remez(filtertaps_2.data(),filtertaps_2.capacity(),(filterbands_2.size()/2),
        filterbands_2.data(),desired_band_gain.data(),weight.data(),BANDPASS,GRIDDENSITY);
    if (converges < 0){
        std::cerr << "Filter 2 failed to converge!" << std::endl;
        //TODO(keith): throw error
    }

    converges = remez(&filtertaps_3[0],filtertaps_3.capacity(),(filterbands_3.size()/2),
        filterbands_3.data(),desired_band_gain.data(),weight.data(),BANDPASS,GRIDDENSITY);
    if (converges < 0){
        std::cerr << "Filter 3 failed to converge!" << std::endl;
        //TODO(keith): throw error
    }

    std::chrono::steady_clock::time_point timing_end = std::chrono::steady_clock::now();
    std::cout << "Time to create 3 filters: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (timing_end - timing_start).count()
      << "us" << std::endl;

    std::ofstream filter1;
    filter1.open("filter1coefficients.dat");
    for (auto i : filtertaps_1){
        filter1 << i << std::endl;
    }
    filter1.close();

    std::ofstream filter2;
    filter2.open("filter2coefficients.dat");
    for (auto i : filtertaps_2){
        filter2 << i << std::endl;
    }
    filter2.close();

    std::ofstream filter3;
    filter3.open("filter3coefficients.dat");
    for (auto i : filtertaps_3){
        filter3 << i << std::endl;
    }
    filter3.close();

    while(1){
        //Receive packet from radar control
        radarctrl_socket.recv(&radctl_request);
        receiverpacket::ReceiverPacket rp;
        std::string r_msg_str(static_cast<char*>(radctl_request.data()), radctl_request.size());
        rp.ParseFromString(r_msg_str);

        //Then receive packet from driver
        driver_socket.recv(&driver_request);
        computationpacket::ComputationPacket cp;
        std::string c_msg_str(static_cast<char*>(driver_request.data()), driver_request.size());
        cp.ParseFromString(c_msg_str);

        //Verify driver and radar control packets align
        if (rp.sequence_num() != cp.sequence_num()) {
            //TODO(keith): handle error
        }

        //Receive driver samples now
        driver_socket.recv(&driver_request);
        auto start = static_cast<T_COMPLEX_F *>(driver_request.data());
        auto data_size = static_cast<size_t>(driver_request.size());
        auto num_elements = data_size/sizeof(T_COMPLEX_F);
        auto total_samples = cp.numberofreceivesamples() * rp.num_channels();

        std::cout << "Number of elements in data message: " << num_elements
            << "\nNumber of elements calculated from packets: " << total_samples
            << std::endl;

        T_DEVICE_V(T_COMPLEX_F) rf_samples(start,start+num_elements);
        throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);
        for(int i = 0; i < 10; i++) {
            std::cout << "rf_samples[" << i << "] = " << rf_samples[i] << std::endl;
        }

        //Parse needed packet values now
        std::vector<double> rx_freqs;
        for(int i=0; i<rp.rxfreqs_size(); i++) {
            rx_freqs.push_back(rp.rxfreqs(i));
        }

        timing_start = std::chrono::steady_clock::now();
        //Creates a vector that holds a host vector of filter coefficients for
        //each frequency. Each host vector is defaulted to the size of the filter
        std::vector<T_HOST_V(T_COMPLEX_F)> filter1_bp_coefs_h(rx_freqs.size(),
                T_HOST_V(T_COMPLEX_F)(filtertaps_1.size()));

        for (int i=0; i<rx_freqs.size(); i++) {
            auto sampling_freq = 2 * M_PI * rx_freqs[i]/rx_rate;
            auto vec_d = filter1_bp_coefs_h[i];

            for(int j=0;j < filtertaps_1.size(); j++) {
                auto radians = fmod(sampling_freq * j,2 * M_PI);
                auto I = filtertaps_1[j] * cos(radians);
                auto Q = filtertaps_1[j] * sin(radians);
                vec_d[j] = std::complex<float>(I,Q);
            }
        }

        std::vector<T_DEVICE_V(T_COMPLEX_F)> filter1_bp_coefs_d(rx_freqs.size());
        for (int i=0; i<rx_freqs.size(); i++) {
            filter1_bp_coefs_d[i] = filter1_bp_coefs_h[i];
        }
        throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);

        timing_end = std::chrono::steady_clock::now();

        std::cout << "NCO mix timing: "
          << std::chrono::duration_cast<std::chrono::microseconds>(timing_end - timing_start).count()
          << "us" << std::endl;

        timing_start = std::chrono::steady_clock::now();

        auto num_blocks_x = cp.numberofreceivesamples()/first_stage_dm_rate;
        std::cout << num_blocks_x << std::endl;
        auto num_blocks_y = rp.num_channels();
        std::cout << num_blocks_y << std::endl;
        dim3 dimGrid(num_blocks_x,num_blocks_y);

        for (auto prop :gpu_properties) {
            if (filtertaps_1.size() > prop.maxThreadsPerBlock) {
                //TODO(Keith) : handle error
            }
        }

        auto num_threads_x = filtertaps_1.size();
        std::cout << num_threads_x << std::endl;
        dim3 dimBlock(num_threads_x);

        auto num_output_samples = rx_freqs.size() * cp.numberofreceivesamples()/first_stage_dm_rate;
        auto stage_1_output = T_DEVICE_V(T_COMPLEX_F)(num_output_samples,T_COMPLEX_F(1.0,0.0));
        throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);

        auto rf_samples_p = thrust::raw_pointer_cast(rf_samples.data());
        auto output_p = thrust::raw_pointer_cast(stage_1_output.data());
        auto num_bytes_taps = filtertaps_1.size() * sizeof(T_COMPLEX_F);
        std::cout << num_bytes_taps << std::endl;

        for (auto filter : filter1_bp_coefs_d) {
            auto filter_p = thrust::raw_pointer_cast(filter.data());
            decimate<<<dimGrid,dimBlock,num_bytes_taps>>>(rf_samples_p, output_p, filter_p,
                        filtertaps_1.size(), cp.numberofreceivesamples());
            throw_on_cuda_error(cudaPeekAtLastError(), __FILE__,__LINE__);
        }
        cudaDeviceSynchronize();
        for(int i = 0; i < 10; i++) {
            std::cout << "output_samples[" << i << "] = " << stage_1_output[i] << std::endl;
        }

        timing_end = std::chrono::steady_clock::now();

        std::cout << "First stage decimate timing: "
          << std::chrono::duration_cast<std::chrono::microseconds>(timing_end - timing_start).count()
          << "us" << std::endl;

        for (auto vec_d : filter1_bp_coefs_d) {
            vec_d.clear();
            vec_d.shrink_to_fit();
        }
    }

}