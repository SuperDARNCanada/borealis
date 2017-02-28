#include <iostream>
#include <vector>
#include <complex>
#include <thread>
#include <unistd.h>
#include <zmq.hpp>
#include "utils/protobuf/computationpacket.pb.h"
#include "utils/protobuf/sigprocpacket.pb.h"



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

    std::vector<double> rxfreqs = {0.,0.,0.};
    for (int i=0; i<rxfreqs.size(); i++) {
        auto rxchan = sp.add_rxchannel();
        rxchan->set_rxfreq(rxfreqs[i]);
        rxchan->set_nrang(75);
        rxchan->set_frang(180);

    }

    auto num_channels = 20;

/*    std::vector<double> beam_dir(num_channels,1.0);
    auto beam_dirs = sp.add_beam_directions();
    for(int i=0; i<beam_dir.size(); i++) {
        beam_dirs->add_phase(i);
        beam_dirs->set_phase(i,beam_dir[i]);
    }*/


    computationpacket::ComputationPacket cp;

    auto num_samples = 1000000;
    cp.set_numberofreceivesamples(num_samples);

    auto default_v = std::complex<float>(2.0,2.0);
    std::vector<std::complex<float>> samples(num_samples*num_channels, default_v);
    zmq::message_t data(samples.data(),samples.size()*sizeof(std::complex<float>));



    auto sqn_num = 0;
    while(1) {

        sp.set_sequence_num(sqn_num);
        cp.set_sequence_num(sqn_num);

        std::string r_msg_str;
        sp.SerializeToString(&r_msg_str);
        zmq::message_t r_msg (r_msg_str.size());
        memcpy ((void *) r_msg.data (), r_msg_str.c_str(), r_msg_str.size());

        std::string c_msg_str;
        cp.SerializeToString(&c_msg_str);
        zmq::message_t c_msg (c_msg_str.size());
        memcpy ((void *) c_msg.data (), c_msg_str.c_str(), c_msg_str.size());

        std::cout << "Sending data with sequence_num: " << sqn_num << std::endl;

        auto timing_ack_start = std::chrono::steady_clock::now();

        radctrl_socket.send(r_msg);
        driver_socket.send(c_msg);

        zmq::message_t data(samples.data(),samples.size()*sizeof(std::complex<float>));
        driver_socket.send(data);

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
