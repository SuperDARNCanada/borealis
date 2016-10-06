/*Copyright 2016 SuperDARN*/
#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
/*#include <boost/program_options.hpp>*/
/*#include <boost/math/special_functions/round.hpp>*/
/*#include <boost/foreach.hpp>*/
/*#include <boost/format.hpp>*/
/*#include <boost/thread.hpp>*/
/*#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/thread.hpp>*/
/*#include <boost/chrono.hpp>*/
#include <stdint.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <mutex>
#include <zmq.hpp>
#include <unistd.h>
#include "utils/driver_options/driveroptions.hpp"
#include "usrp_drivers/n200/usrp.hpp"
#include "utils/protobuf/driverpacket.pb.h"
#include <chrono>
/*enum TRType {TRANSMIT,RECEIVE};
enum IntegrationPeriod {START,STOP,CONTINUE};

std::mutex TR_mutex;
std::mutex control_parameters;


TRType TR = transmit;

int current_integrations = 0;*/


void transmit() {
    std::cout << "Enter transmit thread" << std::endl;
    while (1) {
        sleep(1);
    }

}

void receive() {
    std::cout << "Enter receive thread" << std::endl;
    while (1) {
        sleep(1);
    }

}

void control() {
    std::cout << "Enter control thread" << std::endl;

    std::cout << "Creating and binding control socket" << std::endl;
    zmq::context_t context(1);
    zmq::socket_t control_sock(context, ZMQ_PAIR);
    control_sock.bind("tcp://10.65.0.17:5555");

    zmq::message_t request;

    while (1) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        control_sock.recv(&request);
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
        std::cout << "recv time(ms) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
        std::cout << "recv time(ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() <<std::endl; 

        begin = std::chrono::steady_clock::now();
        driverpacket::DriverPacket dp;
        std::string msg_str(static_cast<char*>(request.data()), request.size());

        dp.ParseFromString(msg_str);

        if (dp.samples_size() != dp.channels_size()){
            //todo: throw error
        }

        for(int i=0; i<dp.samples_size(); i++){
            if (dp.samples(i).real_size() != dp.samples(i).imag_size()){
                //todo: throw error
            }
        }

        std::vector<std::vector<std::complex<float>>> samples(dp.samples_size(),std::vector<std::complex<float>>(dp.samples(0).real_size()));

        for (int i=0; i<dp.samples_size(); i++){
            for (int j=0; j<dp.samples(i).real_size(); j++){
                samples[i][j] = std::complex<float>(dp.samples(i).real(j),dp.samples(i).imag(j));
            }
        }
        end= std::chrono::steady_clock::now();
        std::cout << "Time difference to deserialize = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
        std::cout << "Time difference to deserialize = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() <<std::endl;

    }

}




int UHD_SAFE_MAIN(int argc, char *argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    auto driver_options = std::make_shared<DriverOptions>();

    std::cout << driver_options->get_device_args() << std::endl;
    std::cout << driver_options->get_tx_rate() << std::endl;
    std::cout << driver_options->get_pps() << std::endl;
    std::cout << driver_options->get_ref() << std::endl;
    std::cout << driver_options->get_tx_subdev() << std::endl;

/*    auto usrp_d = std::make_shared<USRP>(driver_options);

    std::vector<size_t> channels {0,1,2,3};//,4,5,6,7,8,9,10,11,12,13,14,15};

    double tx_freq = 12e6;
    double rx_freq = 12e6;

    usrp_d->set_tx_center_freq(tx_freq,channels);
    usrp_d->set_rx_center_freq(rx_freq,channels);

    std::cout << usrp_d->to_string(channels);*/


    //  Prepare our context

   
    std::vector<std::thread> threads;

    std::thread transmit_t(transmit);
    std::thread receive_t(receive);
    std::thread control_t(control);

    threads.push_back(std::move(transmit_t));
    threads.push_back(std::move(receive_t));
    threads.push_back(std::move(control_t));

    for (auto& th : threads) {
        th.join();
    }


    return EXIT_SUCCESS;
}
