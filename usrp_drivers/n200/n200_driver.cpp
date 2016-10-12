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
#include <complex>
/*enum TRType {TRANSMIT,RECEIVE};
enum IntegrationPeriod {START,STOP,CONTINUE};

std::mutex TR_mutex;
std::mutex control_parameters;


TRType TR = transmit;

int current_integrations = 0;*/

std::vector<size_t> make_channels(const driverpacket::DriverPacket &dp){
    std::vector<size_t> channels(dp.channels_size());
    for (int i=0; i<dp.channels_size(); i++){
        channels[i] = i;
    }
    return channels;
}

std::vector<std::vector<std::complex<float>>> make_samples(const driverpacket::DriverPacket &dp){
    std::vector<std::vector<std::complex<float>>> samples(dp.samples_size());

    for(int i=0 ; i<dp.samples_size(); i++) {
        auto num_samps = dp.samples(i).real_size();
        std::vector<std::complex<float>> v(num_samps);
        samples[i] = v;

        for (int j=0; j<dp.samples(i).real_size(); j++){
            samples[i][j] = std::complex<float>(dp.samples(i).real(j),dp.samples(i).imag(j));
        }

    }

    return samples;
}

void transmit(zmq::context_t* thread_c,std::shared_ptr<USRP> usrp_d) {
    std::cout << "Enter transmit thread\n";

    std::cout << "Creating and connecting to thread socket in control\n";
    zmq::socket_t thread_socket(*thread_c, ZMQ_PAIR);
    thread_socket.connect("inproc://threads");

    zmq::message_t request;  

    uhd::stream_args_t stream_args("fc32", "sc16");

    auto channels_set = false;
    auto center_freq_set = false;
    auto samples_set = false;
    uhd::time_spec_t start_time;
    std::vector<size_t> channels;
    uhd::tx_streamer::sptr tx_stream;
    while (1) {
        thread_socket.recv(&request);
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::cout << "Received in transmit\n";
        driverpacket::DriverPacket dp;
        std::string msg_str(static_cast<char*>(request.data()), request.size());
        dp.ParseFromString(msg_str);
        std::cout << "Scope sync high" << std::endl;    
        if (dp.sob() == true) {
          usrp_d->set_scope_sync();
        }
        std::cout <<"BURSTS: " << dp.sob() << " " << dp.eob() <<std::endl;
        std::cout << "pulse number: " <<dp.timetoio() << std::endl;
        std::chrono::steady_clock::time_point stream_begin = std::chrono::steady_clock::now();
        if (dp.channels_size() > 0 && dp.sob() == true && channels_set == false) {
            std::cout << "STARTING NEW PULSE SEQUENCE" <<std::endl;
            channels = make_channels(dp);
            stream_args.channels = channels;
            usrp_d->set_tx_rate(dp.txrate()); //~450us 
            tx_stream = usrp_d->get_usrp()->get_tx_stream(stream_args); //~44ms
            channels_set = true;
        }
        std::chrono::steady_clock::time_point stream_end = std::chrono::steady_clock::now();
        std::cout << "stream set up time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(stream_end - stream_begin).count()
                  << "us" << std::endl;

        std::chrono::steady_clock::time_point ctr_begin = std::chrono::steady_clock::now();
        if (dp.centerfreq() > 0) {
            std::cout << "DP center freq " << dp.centerfreq() << std::endl;
            usrp_d->set_tx_center_freq(dp.centerfreq(),channels);
            center_freq_set = true;
        }
        std::chrono::steady_clock::time_point ctr_end= std::chrono::steady_clock::now();

        std::cout << "center frq tuning time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(ctr_end - ctr_begin).count()
                  << "us" << std::endl;

        std::chrono::steady_clock::time_point sample_begin = std::chrono::steady_clock::now();
        std::vector<std::vector<std::complex<float>>> samples;
        if (dp.samples_size() > 0) { // ~700us to unpack 4x1600 samples
/*            std::vector<std::vector<std::complex<float>>> samples(dp.samples_size(),
                                                              std::vector<std::complex<float>>(dp.samples(0).real_size()));*/
            samples = make_samples(dp);

            samples_set = true; 
        }
        std::chrono::steady_clock::time_point sample_end= std::chrono::steady_clock::now();
        std::cout << "sample unpack time: "
                  << std::chrono::duration_cast<std::chrono::microseconds>(sample_end - sample_begin).count()
                  << "us" << std::endl;
        
        if ( (dp.sob() == true) && (dp.eob() == false) ) {
            //todo: enable start signals
            start_time = usrp_d->get_usrp()->get_time_now() + uhd::time_spec_t(.05);
            std::cout << "start time: " << start_time.get_frac_secs() << std::endl;
        }

        if ( (dp.sob() == false) && (dp.eob() == true) ) {
            //todo: do end of sequence stuff
        }

        if( (channels_set == false) || 
            (center_freq_set == false) || 
            (samples_set == false) ) {

            //todo: throw error
        }

        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

        std::cout << "Total set up time: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << "us" << std::endl;

        auto md = TXMetadata();
        md.set_has_time_spec(true);
        auto time_now = usrp_d->get_usrp()->get_time_now();
        auto time_delay = uhd::time_spec_t(dp.timetosendsamples()/1e6);
        auto send_time = start_time + uhd::time_spec_t(dp.timetosendsamples()/1e6);
        
        usrp_d->get_usrp()->set_command_time(start_time+uhd::time_spec_t((dp.timetosendsamples()-10)/1e6);
        usrp_d->set_tr();
        usrp_d->get_usrp()->clear_command_time();

        std::cout << "timetosendsamples " << dp.timetosendsamples() <<std::endl;
        std::cout << "start time: " << start_time.get_frac_secs() << std::endl;
        std::cout << "time delay " << time_delay.get_frac_secs() <<std::endl; 
        std::cout << "send time " << send_time.get_frac_secs() <<std::endl;
        std::cout << "time now " << time_now.get_frac_secs() << std::endl;
        md.set_time_spec(send_time);
        

        auto samples_per_buff = samples[0].size();

        uint64_t num_samps_sent = 0;

        begin = std::chrono::steady_clock::now();
        while (num_samps_sent < samples_per_buff){
            auto num_samps_to_send = samples_per_buff - num_samps_sent;
            std::cout << "Samples to send " << num_samps_to_send << std::endl;

            num_samps_sent = tx_stream->send(
                samples, num_samps_to_send, md.get_md()
            );

            std::cout << "Samples sent " << num_samps_sent << std::endl;

            md.set_start_of_burst(false);
            md.set_has_time_spec(false);

        }

        md.set_end_of_burst(true);
        tx_stream->send("", 0, md.get_md());
        usrp_d->clear_tr();
        if (dp.eob() == true) {
          usrp_d->clear_scope_sync();
        }
        end= std::chrono::steady_clock::now();

        std::cout << "time to send to USRP: " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count()
                  << "us" << std::endl;

        //sleep(1);
    }

}

void receive() {
    std::cout << "Enter receive thread\n";
    while (1) {
        sleep(1);
    }

}

void control(zmq::context_t* thread_c) {
    std::cout << "Enter control thread\n";

    std::cout << "Creating and connecting to thread socket in control\n";
/*    zmq::socket_t thread_socket(*thread_c, ZMQ_PAIR); // 1
    thread_socket.connect("inproc://threads"); // 2  */  


    zmq::socket_t thread_socket (*thread_c, ZMQ_PAIR);
    thread_socket.bind("inproc://threads");

    std::cout << "Creating and binding control socket\n";
    zmq::context_t context(2);
    zmq::socket_t control_sock(context, ZMQ_PAIR);
    control_sock.bind("ipc:///tmp/feeds/0");

    zmq::message_t request;

    while (1) {
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        control_sock.recv(&request);
        std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();
/*        std::cout << "recv time(ms) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() <<std::endl;
        std::cout << "recv time(ns) = " << std::chrono::duration_cast<std::chrono::nanoseconds> (end - begin).count() <<std::endl; */

        begin = std::chrono::steady_clock::now();
        driverpacket::DriverPacket dp;
        std::string msg_str(static_cast<char*>(request.data()), request.size());
        dp.ParseFromString(msg_str);

        if ((dp.sob() == true) && (dp.eob() == true)){
            //todo: throw error
        }

        if (dp.samples_size() != dp.channels_size()){
            //todo: throw error
        }

        for(int i=0; i<dp.samples_size(); i++){
            if (dp.samples(i).real_size() != dp.samples(i).imag_size()){
                //todo: throw error
            }
        }



        //zmq::message_t forward = request;
        thread_socket.send(request);

        end= std::chrono::steady_clock::now();
        std::cout << "Time difference to deserialize and send in control = " 
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() 
                  <<std::endl;

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

    auto usrp_d = std::make_shared<USRP>(driver_options);


    //  Prepare our context
    zmq::context_t context(1);

    std::vector<std::thread> threads;

    std::thread transmit_t(transmit,&context,usrp_d);
    std::thread receive_t(receive);
    std::thread control_t(control,&context);

    threads.push_back(std::move(transmit_t));
    threads.push_back(std::move(receive_t));
    threads.push_back(std::move(control_t));

    for (auto& th : threads) {
        th.join();
    }


    return EXIT_SUCCESS;
}
