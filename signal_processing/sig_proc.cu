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

#include "utils/protobuf/computationpacket.pb.h"
#include "utils/driver_options/driveroptions.hpp"

extern "C" {
    #include "remez.h"
}

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



std::vector<double> create_normalized_lowpass_filter_bands(float cutoff, float transition_band,
                        float Fs) {
    std::vector<double> filterbands;
    filterbands.push_back(0.0);
    filterbands.push_back(cutoff/Fs);
    filterbands.push_back((cutoff + transition_band)/Fs);
    filterbands.push_back(0.5);

    return filterbands;
}

int main(int argc, char **argv){
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto driver_options = DriverOptions();
    auto rx_rate = driver_options.get_rx_rate();

    zmq::context_t sig_proc_context(1);

    zmq::socket_t sig_proc_socket(sig_proc_context, ZMQ_PAIR);
    sig_proc_socket.connect("ipc:///tmp/feeds/1");
    zmq::message_t request;

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

/*    while(1){
        sig_proc_socket.recv(&request);
        computationpacket::ComputationPacket cp;
        std::string msg_str(static_cast<char*>(request.data()), request.size());
        cp.ParseFromString(msg_str);

        sleep(1);

    }*/

}