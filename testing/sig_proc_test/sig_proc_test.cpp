#include <iostream>
#include <vector>
#include <complex>
#include <thread>
#include <unistd.h>
#include <zmq.hpp>
#include "utils/protobuf/computationpacket.pb.h"
#include "utils/protobuf/receiverpacket.pb.h"



int main(int argc, char** argv){

    zmq::context_t context(1);
    zmq::socket_t driver_socket(context, ZMQ_PAIR);
    driver_socket.connect("ipc:///tmp/feeds/1");

    zmq::socket_t radctrl_socket(context, ZMQ_PAIR);
    radctrl_socket.connect("ipc:///tmp/feeds/2");

    receiverpacket::ReceiverPacket rp;

    std::vector<double> rxfreqs = {0.};
    for (int i=0; i<rxfreqs.size(); i++) {
        rp.add_rxfreqs(i);
        rp.set_rxfreqs(i,rxfreqs[i]);
    }

    auto num_channels = 20;

    std::vector<double> beam_dir(num_channels,1.0);
    auto beam_dirs = rp.add_beam_directions();
    for(int i=0; i<beam_dir.size(); i++) {
        beam_dirs->add_phase(i);
        beam_dirs->set_phase(i,beam_dir[i]);
    }

    rp.set_num_channels(num_channels);
    rp.set_nrang(75);
    rp.set_frang(180);


    computationpacket::ComputationPacket cp;

    auto num_samples = 1000000;
    cp.set_numberofreceivesamples(num_samples);

    auto default_v = std::complex<float>(2.0,2.0);
    std::vector<std::complex<float>> samples(num_samples*num_channels, default_v);




    auto sqn_num = 0;
    while(1) {

        rp.set_sequence_num(sqn_num);
        cp.set_sequence_num(sqn_num);

        std::string r_msg_str;
        rp.SerializeToString(&r_msg_str);
        zmq::message_t r_msg (r_msg_str.size());
        memcpy ((void *) r_msg.data (), r_msg_str.c_str(), r_msg_str.size());

        std::string c_msg_str;
        cp.SerializeToString(&c_msg_str);
        zmq::message_t c_msg (c_msg_str.size());
        memcpy ((void *) c_msg.data (), c_msg_str.c_str(), c_msg_str.size());

        radctrl_socket.send(r_msg);
        driver_socket.send(c_msg);

        zmq::message_t data(samples.data(),samples.size()*sizeof(std::complex<float>));
        driver_socket.send(data);

        sleep(1);

    }


}