#include <iostream>
#include <vector>
#include <complex>
#include <thread>
#include <unistd.h>
#include <zmq.hpp>
#include "utils/protobuf/computationpacket.pb.h"
#include "utils/protobuf/sigprocpacket.pb.h"
#include "utils/shared_memory/shared_memory.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"
#include "utils/driver_options/driveroptions.hpp"

std::string random_string( size_t length )
{
    auto randchar = []() -> char
    {
        const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
        const size_t max_index = (sizeof(charset) - 1);
        return charset[ rand() % max_index ];
    };
    std::string str(length,0);
    std::generate_n( str.begin(), length, randchar );
    return str;
}

int main(int argc, char** argv){

    zmq::context_t context(1);
    zmq::socket_t driver_socket(context, ZMQ_PAIR);
    driver_socket.connect("ipc:///tmp/feeds/1");

    zmq::socket_t radctrl_socket(context, ZMQ_PAIR);
    radctrl_socket.connect("ipc:///tmp/feeds/2");

    zmq::socket_t ack_socket(context, ZMQ_PAIR);
    ack_socket.connect("ipc:///tmp/feeds/3");

    zmq::socket_t timing_socket(context, ZMQ_PAIR);
    timing_socket.connect("ipc:///tmp/feeds/4");

    sigprocpacket::SigProcPacket sp;

    auto driver_options = DriverOptions();
    auto rx_rate = driver_options.get_rx_rate();


    std::vector<float> sample_buffer;
    enum shr_mem_switcher {FIRST, SECOND};
    shr_mem_switcher region_switcher = FIRST;
    boost::interprocess::mapped_region first_region, second_region;
    void *current_region_addr;

    std::vector<double> rxfreqs = {0.0,0.0,0.0};
    for (int i=0; i<rxfreqs.size(); i++) {
        auto rxchan = sp.add_rxchannel();
        rxchan->set_rxfreq(rxfreqs[i]);
        rxchan->set_nrang(75);
        rxchan->set_frang(180);

    }

    auto num_channels = driver_options.get_main_antenna_count() +
                            driver_options.get_interferometer_antenna_count();


    computationpacket::ComputationPacket cp;

    auto num_samples = int(rx_rate * 0.1);
    cp.set_numberofreceivesamples(num_samples);

    auto default_v = std::complex<float>(0.0,0.0);
    std::vector<std::complex<float>> samples(num_samples*num_channels,default_v);


    for (int i=0; i<samples.size(); i++) {
        auto nco_point = std::complex<float>(0.0,0.0);
        for (auto freq : rx_freqs) {
            auto sampling_freq = 2 * M_PI * freq/rx_rate;

            auto radians = fmod(sampling_freq * i, 2 * M_PI);
            auto I = cos(radians);
            auto Q = sin(radians);

            nco_point += std::complex<float>(I,Q);
        }
        samples[i] = nco_point;
    }

    zmq::message_t data(samples.data(),samples.size()*sizeof(std::complex<float>));



    auto sqn_num = 0;


    while(1) {

        sp.set_sequence_num(sqn_num);
        cp.set_sequence_num(sqn_num);

        std::string r_msg_str;
        sp.SerializeToString(&r_msg_str);
        zmq::message_t r_msg (r_msg_str.size());
        memcpy ((void *) r_msg.data (), r_msg_str.c_str(), r_msg_str.size());

        auto name_str = random_string(10);

        auto shr_start = std::chrono::steady_clock::now();
        SharedMemoryHandler shrmem(name_str);
        auto size = samples.size() * sizeof(std::complex<float>);
        shrmem.create_shr_mem(size);
        memcpy(shrmem.get_shrmem_addr(), samples.data(), size);
        auto shr_end = std::chrono::steady_clock::now();
        std::cout << "shrmem + memcpy for " << sp.sequence_num()
            << " after "
            << std::chrono::duration_cast<std::chrono::milliseconds>(shr_end - shr_start).count()
            << "ms" << std::endl;

        std::cout << "Sending data with sequence_num: " << sqn_num << std::endl;

        auto timing_ack_start = std::chrono::steady_clock::now();

        radctrl_socket.send(r_msg);

        cp.set_name(name_str.c_str());

        std::string c_msg_str;
        cp.SerializeToString(&c_msg_str);
        zmq::message_t c_msg (c_msg_str.size());
        memcpy ((void *) c_msg.data (), c_msg_str.c_str(), c_msg_str.size());

        driver_socket.send(c_msg);

/*        zmq::message_t data(samples.data(),samples.size()*sizeof(std::complex<float>));
        driver_socket.send(data);*/

        zmq::message_t ack;
        ack_socket.recv(&ack);
        sigprocpacket::SigProcPacket sp;
        std::string s_msg_str1(static_cast<char*>(ack.data()), ack.size());
        sp.ParseFromString(s_msg_str1);

        auto timing_ack_end = std::chrono::steady_clock::now();

        std::cout << "Received ack for sequence_num " << sp.sequence_num()
            << " after "
            << std::chrono::duration_cast<std::chrono::milliseconds>(timing_ack_end - timing_ack_start).count()
            << "ms" << std::endl;


        auto timing_timing_start = std::chrono::steady_clock::now();

        zmq::message_t timing;
        timing_socket.recv(&timing);
        std::string s_msg_str2(static_cast<char*>(timing.data()), timing.size());
        sp.ParseFromString(s_msg_str2);

        auto timing_timing_end = std::chrono::steady_clock::now();

        std::cout << "Received timing for sequence_num " << sp.sequence_num()
            << " after "
            << std::chrono::duration_cast<std::chrono::milliseconds>(timing_timing_end - timing_timing_start).count()
            << "ms with decimation timing of " << sp.kerneltime() << "ms" <<  std::endl;
        sqn_num += 1;

        //usleep(250000);

    }


}
