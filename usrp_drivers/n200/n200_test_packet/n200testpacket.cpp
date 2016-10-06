#include <zmq.hpp>
#include "utils/protobuf/driverpacket.pb.h"
#include <thread>
#include <unistd.h>
#include <iostream>
int main(int argc, char *argv[]){

    driverpacket::DriverPacket dp;
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);
    socket.connect("tcp://10.65.0.25:33033");

    for (int j=0; j<16; j++){
        dp.add_channels(j);
        dp.add_centerfreq(12e6);
        dp.add_waveformfreq(1e6);
        dp.add_txrate(5e6);
        dp.add_rxrate(3.33e3);
        auto samples = dp.add_samples();

        for (int k=0; k<1600; k++){
            samples->add_real(k);
            samples->add_imag(k);
        }
    }

    while (1){
        for (int i=0; i<8; i++){
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
            bool SOB, EOB = false;
            double timetoio = 0.0;
            if (i == 0){
                SOB = true;
            }
            if (i == 7){
                EOB = true;
            }

            dp.set_sob(SOB);
            dp.set_eob(EOB);
            dp.set_timetoio(timetoio);
            dp.set_timetosendsamples(i);


            for (int j=0; j<16; j++){
                dp.set_channels(j,j);
                dp.set_centerfreq(j,12e6);
                dp.set_waveformfreq(j,1e6);
                dp.set_txrate(i,5e6);
                dp.set_rxrate(i,3.33e3);

                for (int k=0; k<1600; k++){
                    dp.mutable_samples(j)->set_real(k,k);
                    dp.mutable_samples(j)->set_imag(k,k);
                }
            }

            std::string msg_str;
            dp.SerializeToString(&msg_str);
            zmq::message_t request (msg_str.size());
            memcpy ((void *) request.data (), msg_str.c_str(), msg_str.size());
            std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
            std::cout << "Time difference to serialize(ms) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
            std::cout << "Time difference to serialize(ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() <<std::endl;

            begin = std::chrono::steady_clock::now();
            socket.send (request);
            end= std::chrono::steady_clock::now();

            zmq::message_t reply;
            socket.recv(&reply);

            std::cout << "send time(ms) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
            std::cout << "send time(ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() <<std::endl;            
            sleep(1);
        }
    
    }

}