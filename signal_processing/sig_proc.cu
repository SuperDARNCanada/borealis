#include <vector>
#include <string>
#include <zmq.hpp>
#include <thread>
#include <complex>
#include <iostream>
#include <stdint.h>
#include <math.h>

#include "utils/protobuf/computationpacket.pb.h"
#include "utils/driver_options/driveroptions.hpp"

extern "C" {
    #include "remez.h"
}

#define FIRST_STAGE_SAMPLE_RATE 1e6 //1 MHz
#define SECOND_STAGE_SAMPLE_RATE 0.1e6 // 100 kHz
#define THIRD_STAGE_SAMPLE_RATE 10000.0/3.0 //3.33 kHz

#define FIRST_STAGE_FILTER_CUTOFF 1e6
#define FIRST_STAGE_FILTER_TRANSITION FIRST_STAGE_FILTER_CUTOFF * 0.5

#define SECOND_STAGE_FILTER_CUTOFF 0.1e6
#define SECOND_STAGE_FILTER_TRANSITION SECOND_STAGE_FILTER_CUTOFF * 0.5

#define THIRD_STAGE_FILTER_CUTOFF 10000.0/3.0
#define THIRD_STAGE_FILTER_TRANSITION THIRD_STAGE_FILTER_CUTOFF * 0.25

#define k 3 //from formula 7-6
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

    auto S_lowpass1 = k * (rx_rate/FIRST_STAGE_FILTER_TRANSITION);
    auto S_lowpass2 = k * (FIRST_STAGE_SAMPLE_RATE/SECOND_STAGE_FILTER_TRANSITION);
    auto S_lowpass3 = k * (SECOND_STAGE_SAMPLE_RATE/THIRD_STAGE_FILTER_TRANSITION);

    std::cout << "1st stage taps: " << S_lowpass1 << std::endl << "2nd stage taps: "
        << S_lowpass2 << std::endl << "3rd stage taps: " << S_lowpass3 <<std::endl;

    while(1){
/*        sig_proc_socket.recv(&request);
        computationpacket::ComputationPacket cp;
        std::string msg_str(static_cast<char*>(request.data()), request.size());
        cp.ParseFromString(msg_str);*/

        sleep(1);

    }

}