/*Copyright 2016 SuperDARN*/
#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <boost/program_options.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/foreach.hpp>
#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <stdint.h>
#include <iostream>
#include <memory>
#include <vector>
#include <string>

#include "driveroptions.hpp"
#include "usrp.hpp"



void tx_thread() {

}

void rx_thread() {
    
}




int UHD_SAFE_MAIN(int argc, char *argv[]) {


    auto driver_options = std::make_shared<DriverOptions>();

    std::cout << driver_options->get_device_args() << std::endl;
    std::cout << driver_options->get_tx_rate() << std::endl;
    std::cout << driver_options->get_pps() << std::endl;
    std::cout << driver_options->get_ref() << std::endl;
    std::cout << driver_options->get_tx_subdev() << std::endl;

    auto usrp_d = std::make_shared<USRP>(driver_options);

    std::vector<size_t> channels {0,1,2,3};//,4,5,6,7,8,9,10,11,12,13,14,15};

    double tx_freq = 12e6;
    double rx_freq = 12e6;

    usrp_d->set_tx_center_freq(tx_freq,channels);
    usrp_d->set_rx_center_freq(rx_freq,channels);

    std::cout << usrp_d->to_string(channels);



    return EXIT_SUCCESS;
}
