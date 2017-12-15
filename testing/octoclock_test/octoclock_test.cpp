// octoclock test code 
// 201712 - Kevin Krieger

#include <stdlib.h>

#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
//#include <thread>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>

#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/thread_priority.hpp>
#include <uhd/usrp_clock/multi_usrp_clock.hpp>

// Test program for Ettus Octoclocks
// .130 and .132 are regular octoclocks, externally referenced off of .131, which is an octoclock-g (GPS unit built-in)
namespace po = boost::program_options;            // Options on command line or config file

float speed_of_light = 299792458.0; // meters per second
static const std::string  DEVICE_DEFAULT_MULTI_USRP_CLOCK_ARGS          = "addr0=192.168.10.130,addr1=192.168.10.131,addr2=192.168.132";
static const uint32_t UPDATE_PERIOD_IN_S				= 10;


// Signal handling (CTRL-C)
static bool stop_signal_called = false;
void sig_int_handler(int) {stop_signal_called = true;}

int main(int argc, char *argv[]) {

  // register signal handler
  std::signal(SIGINT, &sig_int_handler);

  // Variables for program options
  std::string multi_usrp_clock_args;

  uint32_t update_period_in_s;

  // Set up program options
  po::options_description config("Configuration");
  config.add_options()
    ("help", "help message")
    ("multi_usrp_clock_args", po::value<std::string>(&multi_usrp_clock_args)->default_value(DEVICE_DEFAULT_MULTI_USRP_CLOCK_ARGS)->composing(), "multi usrp clock address args")
    ("print_diagnostic_info,d", "Print out various information about the devices and exit")
    ("print_time,g", "Print out the time from the octoclocks periodically (period defined by update_period)")
    ("update_period", po::value<uint32_t>(&update_period_in_s)->default_value(UPDATE_PERIOD_IN_S), "If printing values, how often to print them (s)")
  ;

  po::positional_options_description p;
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(config).positional(p).run(), vm);
  std::ifstream config_filename("octoclock_test.cfg");
  po::store(po::parse_config_file(config_filename, config, false), vm);
  po::notify(vm);

  std::cout << std::endl;

  // Print help messages
  if (vm.count("help")) {
    std::cout << config << std::endl;
    return 0;
  }

  // Create multi usrp clock device (get shared pointer) 
  uhd::usrp_clock::multi_usrp_clock::sptr clock;
  try {
    clock = uhd::usrp_clock::multi_usrp_clock::make(multi_usrp_clock_args);
  } catch(...) {
    std::cout << "Failed to get usrp clock device" << std::endl;
    return -1;
  }

  if(vm.count("print_diagnostic_info")) {
    
    return 0;
  } 

  while (not stop_signal_called) {
    // If user asked for updates, then print out every x seconds, otherwise just sleep
    if (vm.count("print_time")) {
      std::cout << "Octoclock 0 time: " << clock->get_time(0) << std::endl;
      std::cout << "Octoclock 1 time: " << clock->get_time(1) << std::endl;
      std::cout << "Octoclock 2 time: " << clock->get_time(2) << std::endl;
	
      sleep(update_period_in_s);
    } else {
      sleep(1);
    }
  }
  std::cout << "Stop signal called, exiting" << std::endl;
  return 0;
}


  

