#include <vector>
#include <string>
#include <zmq.hpp>
#include <thread>
#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdint.h>
#include <signal.h>
#include <cstdlib>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cuda_profiler_api.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "utils/protobuf/computationpacket.pb.h"
#include "utils/protobuf/sigprocpacket.pb.h"
#include "utils/driver_options/driveroptions.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"
#include "utils/shared_memory/shared_memory.hpp"

#include "digital_processing.hpp"

extern "C" {
    #include "remez.h"
}

/*#define T_DEVICE_V(x) thrust::device_vector<x>
#define T_HOST_V(x) thrust::host_vector<x,thrust::cuda::experimental::pinned_allocator<x>>
#define T_COMPLEX_F thrust::complex<float>

#define sig_options.get_first_stage_sample_rate() 1.0e6 //1 MHz
#define sig_options.get_second_stage_sample_rate() 0.1e6 // 100 kHz
#define sig_options.get_third_stage_sample_rate() (10000.0/3.0) //3.33 kHz

#define sig_options.get_first_stage_filter_cutoff() 1.0e6
#define sig_options.get_first_stage_filter_transition() (sig_options.get_first_stage_filter_cutoff() * 0.5)

#define sig_options.get_second_stage_filter_cutoff() 0.1e6
#define sig_options.get_second_stage_filter_transition() (sig_options.get_second_stage_filter_cutoff() * 0.5)

#define sig_options.get_third_stage_filter_cutoff() (10000.0/3.0)
#define sig_options.get_third_stage_filter_transition() (sig_options.get_third_stage_filter_cutoff() * 0.25)
*/
std::vector<double> create_normalized_lowpass_filter_bands(float cutoff, float transition_band,
                        float Fs) {
    std::vector<double> filterbands;
    filterbands.push_back(0.0);
    filterbands.push_back(cutoff/Fs);
    filterbands.push_back((cutoff + transition_band)/Fs);
    filterbands.push_back(0.5);

    return filterbands;
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

uint32_t calculate_num_filter_taps(float rate, float transition_width) {
    auto k = 3; //from formula 7-6 of Lyons text
    return k * (rate/transition_width);
}

std::vector<std::complex<float>> create_filter(uint32_t num_taps, float filter_cutoff, float transition_width,
                                    float rate) {

    std::vector<double> desired_band_gain = {1.0,0.0};
    std::vector<double> weight = {1.0,1.0};

    auto filter_bands = create_normalized_lowpass_filter_bands(filter_cutoff,transition_width,
                            rate);

    std::vector<double> filter_taps(num_taps+1); //remez returns number of taps + 1

    auto converges = remez(filter_taps.data(),filter_taps.capacity(),(filter_bands.size()/2),
        filter_bands.data(),desired_band_gain.data(),weight.data(),BANDPASS,GRIDDENSITY);
    if (converges < 0){
        std::cerr << "Filter failed to converge with cutoff of " << filter_cutoff
            << ", transition width " << transition_width << ", and rate "
            << rate << std::endl;
        //TODO(keith): throw error
    }

    std::vector<std::complex<float>> complex_taps(num_taps+1);
    for (auto &i : filter_taps) {
        complex_taps[i] = std::complex<float>(i,0.0);
    }

    return complex_taps;
}

void save_filter_to_file(std::vector<std::complex<float>> filter_taps, const char* name) {
    std::ofstream filter;
    filter.open(name);
    for (auto &i : filter_taps){
        filter << i << std::endl;
    }
    filter.close();
}


int main(int argc, char **argv){
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto driver_options = DriverOptions();
    auto sig_options = SignalProcessingOptions();
    auto rx_rate = driver_options.get_rx_rate();
    zmq::context_t sig_proc_context(1);

    zmq::socket_t driver_socket(sig_proc_context, ZMQ_PAIR);
    driver_socket.bind("ipc:///tmp/feeds/1");

    zmq::socket_t radarctrl_socket(sig_proc_context, ZMQ_PAIR);
    radarctrl_socket.bind("ipc:///tmp/feeds/2");

    zmq::socket_t ack_socket(sig_proc_context, ZMQ_PAIR);
    ack_socket.bind("ipc:///tmp/feeds/3");

    zmq::socket_t timing_socket(sig_proc_context, ZMQ_PAIR);
    timing_socket.bind("ipc:///tmp/feeds/4");

    auto gpu_properties = get_gpu_properties();
    print_gpu_properties(gpu_properties);

    uint32_t first_stage_dm_rate, second_stage_dm_rate, third_stage_dm_rate = 0;
    if (fmod(rx_rate,sig_options.get_first_stage_sample_rate()) > 0.0){
        //TODO(keith): handle error
    }
    else{
        auto rate_f = rx_rate/sig_options.get_first_stage_sample_rate();
        first_stage_dm_rate = static_cast<uint32_t>(rate_f);

        rate_f = sig_options.get_first_stage_sample_rate()/
                    sig_options.get_second_stage_sample_rate();
        second_stage_dm_rate = static_cast<uint32_t>(rate_f);

        rate_f = sig_options.get_second_stage_sample_rate()/
                    sig_options.get_third_stage_sample_rate();
        third_stage_dm_rate = static_cast<uint32_t>(rate_f);
    }

    std::cout << "1st stage dm rate: " << first_stage_dm_rate << std::endl
        << "2nd stage dm rate: " << second_stage_dm_rate << std::endl
        << "3rd stage dm rate: " << third_stage_dm_rate <<std::endl;


    auto S_lowpass1 = calculate_num_filter_taps(rx_rate,
                                    sig_options.get_first_stage_filter_transition());
    auto S_lowpass2 = calculate_num_filter_taps(sig_options.get_first_stage_sample_rate(),
                                    sig_options.get_second_stage_filter_transition());
    auto S_lowpass3 = calculate_num_filter_taps(sig_options.get_second_stage_sample_rate(),
                                    sig_options.get_third_stage_filter_transition());

    std::cout << "1st stage taps: " << S_lowpass1 << std::endl << "2nd stage taps: "
        << S_lowpass2 << std::endl << "3rd stage taps: " << S_lowpass3 <<std::endl;


    std::chrono::steady_clock::time_point timing_start = std::chrono::steady_clock::now();


    auto filtertaps_1 = create_filter(S_lowpass1, sig_options.get_first_stage_filter_cutoff(),
                        sig_options.get_first_stage_filter_transition(), rx_rate);
    auto filtertaps_2 = create_filter(S_lowpass2,sig_options.get_second_stage_filter_cutoff(),
                        sig_options.get_second_stage_filter_transition(),
                        sig_options.get_first_stage_sample_rate());
    auto filtertaps_3 = create_filter(S_lowpass3,sig_options.get_third_stage_filter_cutoff(),
                        sig_options.get_third_stage_filter_transition(),
                        sig_options.get_second_stage_sample_rate());

    std::chrono::steady_clock::time_point timing_end = std::chrono::steady_clock::now();
    std::cout << "Time to create 3 filters: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (timing_end - timing_start).count()
      << "us" << std::endl;

    save_filter_to_file(filtertaps_1,"filter1coefficients.dat");
    save_filter_to_file(filtertaps_2,"filter2coefficients.dat");
    save_filter_to_file(filtertaps_3,"filter3coefficients.dat");

    while(1){
        //Receive packet from radar control
        zmq::message_t radctl_request;
        radarctrl_socket.recv(&radctl_request);
        sigprocpacket::SigProcPacket sp;
        std::string r_msg_str(static_cast<char*>(radctl_request.data()), radctl_request.size());
        sp.ParseFromString(r_msg_str);

        //Then receive packet from driver
        zmq::message_t driver_request;
        driver_socket.recv(&driver_request);
        computationpacket::ComputationPacket cp;
        std::string c_msg_str(static_cast<char*>(driver_request.data()), driver_request.size());
        cp.ParseFromString(c_msg_str);

        //Verify driver and radar control packets align
        if (sp.sequence_num() != cp.sequence_num()) {
            //TODO(keith): handle error
        }


        //Receive driver samples now
        //timing_start = std::chrono::steady_clock::now();
        //driver_socket.recv(&driver_request);
        //timing_end = std::chrono::steady_clock::now();
        //std::cout << "recv: "
        //  << std::chrono::duration_cast<std::chrono::microseconds>(timing_end - timing_start).count()
        //  << "us" << std::endl;

        //auto start = static_cast<T_COMPLEX_F *>(driver_request.data());
        //auto data_size = static_cast<size_t>(driver_request.size());
        //auto num_elements = data_size/sizeof(T_COMPLEX_F);



        //Parse needed packet values now
        std::vector<double> rx_freqs;
        for(int i=0; i<sp.rxchannel_size(); i++) {
            rx_freqs.push_back(sp.rxchannel(i).rxfreq());
        }

        timing_start = std::chrono::steady_clock::now();

        std::vector<std::complex<float>> filtertaps_1_bp_h(rx_freqs.size()*filtertaps_1.size());
        for (int i=0; i<rx_freqs.size(); i++) {
            auto sampling_freq = 2 * M_PI * rx_freqs[i]/rx_rate;

            for(int j=0;j < filtertaps_1.size(); j++) {
                auto radians = fmod(sampling_freq * j,2 * M_PI);
                auto I = filtertaps_1[j].real() * cos(radians);
                auto Q = filtertaps_1[j].real() * sin(radians);
                filtertaps_1_bp_h[i*filtertaps_1.size() + j] = std::complex<float>(I,Q);
            }
        }

        timing_end = std::chrono::steady_clock::now();

        std::cout << "NCO mix timing: "
          << std::chrono::duration_cast<std::chrono::microseconds>(timing_end - timing_start).count()
          << "us" << std::endl;

        std::vector<std::complex<float>> filtertaps_2_h(filtertaps_2.size());
        std::vector<std::complex<float>> filtertaps_3_h(filtertaps_3.size());
        for (uint32_t i=0; i< rx_freqs.size(); i++){
            filtertaps_2_h.insert(filtertaps_2_h.end(),filtertaps_2.begin(),filtertaps_2.end());
            filtertaps_3_h.insert(filtertaps_3_h.end(),filtertaps_3.begin(),filtertaps_3.end());
        }

        DigitalProcessing *dp = new DigitalProcessing(&ack_socket, &timing_socket,
                                                         sp.sequence_num(), cp.name().c_str());

        auto total_samples = cp.numberofreceivesamples() * sig_options.get_total_receive_antennas();

        std::cout << "Total elements in data message: " << total_samples
            << std::endl;

        dp->allocate_and_copy_rf_samples(total_samples);
        dp->allocate_and_copy_first_stage_filters(filtertaps_1_bp_h.data(), filtertaps_1_bp_h.size());


        auto num_output_samples_1 = rx_freqs.size() * cp.numberofreceivesamples()/first_stage_dm_rate
                                        * sig_options.get_total_receive_antennas();
        dp->allocate_first_stage_output(num_output_samples_1);

        gpuErrchk(cudaStreamAddCallback(dp->get_cuda_stream(),
                                    DigitalProcessing::initial_memcpy_callback, dp, 0));

        dp->call_decimate(dp->get_rf_samples_p(),
            dp->get_first_stage_output_p(),
            dp->get_first_stage_bp_filters_p(), first_stage_dm_rate,
            cp.numberofreceivesamples(), filtertaps_1.size(), rx_freqs.size(),
            sig_options.get_total_receive_antennas(), "First stage of decimation");



        dp->allocate_and_copy_second_stage_filters(filtertaps_2_h.data(), filtertaps_2_h.size());
        auto num_output_samples_2 = num_output_samples_1 / second_stage_dm_rate;
        dp->allocate_second_stage_output(num_output_samples_2);
        auto num_samps_2 = cp.numberofreceivesamples()/first_stage_dm_rate;
        dp->call_decimate(dp->get_first_stage_output_p(),
            dp->get_second_stage_output_p(),
            dp->get_second_stage_filters_p(), second_stage_dm_rate,
            num_samps_2, filtertaps_2.size(), rx_freqs.size(),
            sig_options.get_total_receive_antennas(), "Second stage of decimation");



        dp->allocate_and_copy_third_stage_filters(filtertaps_3_h.data(), filtertaps_3_h.size());
        auto num_output_samples_3 = num_output_samples_2 / third_stage_dm_rate;
        dp->allocate_third_stage_output(num_output_samples_3);
        auto num_samps_3 = num_samps_2/second_stage_dm_rate;
        dp->call_decimate(dp->get_second_stage_output_p(),
            dp->get_third_stage_output_p(),
            dp->get_third_stage_filters_p(), third_stage_dm_rate,
            num_samps_3, filtertaps_3.size(), rx_freqs.size(),
            sig_options.get_total_receive_antennas(), "Third stage of decimation");

        dp->allocate_and_copy_host_output(num_output_samples_3);

        // New in CUDA 5.0: Add a CPU callback which is called once all currently pending operations in the CUDA stream have finished
        gpuErrchk(cudaStreamAddCallback(dp->get_cuda_stream(),
                                            DigitalProcessing::cuda_postprocessing_callback, dp, 0));


    }


}