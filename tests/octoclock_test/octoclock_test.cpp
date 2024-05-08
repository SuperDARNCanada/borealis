// octoclock test code
// Copyright 2017 SuperDARN Canada
// Author: Kevin Krieger

#include <stdlib.h>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <uhd/usrp_clock/multi_usrp_clock.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/thread.hpp>

#include "src/usrp_drivers/utils/shared_macros.hpp"

// Test program for Ettus Octoclocks
// .130 and .132 are regular octoclocks, externally referenced off of .131,
// which is an octoclock-g (GPS unit built-in)
namespace po =
    boost::program_options;  // Options on command line or config file

float speed_of_light = 299792458.0;  // meters per second
static const std::string DEVICE_DEFAULT_MULTI_USRP_CLOCK_ARGS =
    "addr0=192.168.10.130,addr1=192.168.10.131,addr2=192.168.10.132";
static const uint32_t UPDATE_PERIOD_IN_S = 10;

// Signal handling (CTRL-C)
static bool stop_signal_called = false;
void sig_int_handler(int) { stop_signal_called = true; }

int main(int argc, char *argv[]) {
  // register signal handler
  std::signal(SIGINT, &sig_int_handler);

  // Variables for program options
  std::string multi_usrp_clock_args;

  uint32_t update_period_in_s;

  // Set up program options
  po::options_description config("Configuration");
  config.add_options()("help", "help message")(
      "multi_usrp_clock_args",
      po::value<std::string>(&multi_usrp_clock_args)
          ->default_value(DEVICE_DEFAULT_MULTI_USRP_CLOCK_ARGS)
          ->composing(),
      "multi usrp clock address args")(
      "print_diagnostic_info,d",
      "Print out various information about the devices and exit")(
      "print_time,g",
      "Print out the time from the octoclocks periodically (period defined by "
      "update_period)")("update_period",
                        po::value<uint32_t>(&update_period_in_s)
                            ->default_value(UPDATE_PERIOD_IN_S),
                        "If printing values, how often to print them (s)");

  po::positional_options_description p;
  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(config).positional(p).run(),
      vm);
  std::ifstream config_filename("octoclock_test.cfg");
  po::store(po::parse_config_file(config_filename, config, false), vm);
  po::notify(vm);

  std::cout << std::endl;

  // Print help messages
  if (vm.count("help")) {
    std::cout << config << std::endl;
    return 0;
  }
  std::string gps_lock_str;
  std::string gps_time_str;
  std::string gps_gpgga_str;
  std::string gps_gprmc_str;
  std::string gps_servo_str;

  // Create multi usrp clock device (get shared pointer)
  uhd::usrp_clock::multi_usrp_clock::sptr clock;
  try {
    clock = uhd::usrp_clock::multi_usrp_clock::make(multi_usrp_clock_args);
  } catch (...) {
    std::cout << "Failed to get usrp clock device" << std::endl;
    return -1;
  }
  if (vm.count("print_diagnostic_info")) {
    std::cout << "Octoclock number of boards: " << clock->get_num_boards()
              << std::endl;
    std::cout << clock->get_pp_string() << std::endl;
    for (size_t boardnumber = 0; boardnumber < clock->get_num_boards();
         boardnumber++) {
      std::cout << "Octoclock board: " << boardnumber << std::endl;
      std::cout << "  gps detected: "
                << clock->get_sensor("gps_detected", boardnumber).value
                << std::endl;
      std::cout << "  reference: "
                << clock->get_sensor("using_ref", boardnumber).value
                << std::endl;
      std::cout << "  external reference detected: "
                << clock->get_sensor("ext_ref_detected", boardnumber).value
                << std::endl;
      std::cout << "  switch position: "
                << clock->get_sensor("switch_pos", boardnumber).value
                << std::endl;
      if (clock->get_sensor("gps_detected", boardnumber).value == "true") {
        std::cout << std::endl << "GPS sentences" << std::endl;
        std::cout << clock->get_sensor("gps_gpgga", boardnumber).value
                  << std::endl;
        std::cout << clock->get_sensor("gps_gprmc", boardnumber).value
                  << std::endl;
        std::cout << "GPS Time: "
                  << clock->get_sensor("gps_time", boardnumber).value
                  << std::endl;
        std::cout << "GPS locked: "
                  << clock->get_sensor("gps_locked", boardnumber).value
                  << std::endl;
        std::cout << "GPS servo status: "
                  << clock->get_sensor("gps_servo", boardnumber).value
                  << std::endl;
        std::cout << std::endl;
      }
    }
  }

  auto box_time = clock->get_time(1);
  auto system_time = std::chrono::system_clock::now();
  auto system_since_epoch =
      std::chrono::duration<double>(system_time.time_since_epoch());
  auto gps_host_time_diff = system_since_epoch.count() - box_time;
  std::cout << "Box time: " << box_time
            << " System time: " << system_since_epoch.count() << std::endl;

  while (not stop_signal_called) {
    // If user asked for updates, then print out every x seconds, otherwise just
    // sleep
    if (vm.count("print_time")) {
      for (size_t boardnumber = 0; boardnumber < clock->get_num_boards();
           boardnumber++) {
        if (clock->get_sensor("gps_detected", boardnumber).value == "true") {
          std::cout << std::endl << "GPS sentences" << std::endl;
          auto system_time = std::chrono::system_clock::now();
          TIMEIT_IF_TRUE_OR_DEBUG(
              true, "time to get system time: ",
              system_since_epoch = std::chrono::duration<double>(
                  system_time.time_since_epoch()));
          TIMEIT_IF_TRUE_OR_DEBUG(true, "time to get box time: ",
                                  box_time = clock->get_time(boardnumber));
          gps_host_time_diff = system_since_epoch.count() - box_time;
          std::cout << std::endl
                    << "System time: " << int(system_since_epoch.count())
                    << " Box time: " << box_time
                    << " Diff: " << gps_host_time_diff << std::endl;
          TIMEIT_IF_TRUE_OR_DEBUG(
              true, "Time to get gps_gpgga: ",
              gps_gpgga_str =
                  clock->get_sensor("gps_gpgga", boardnumber).value);
          std::cout << "GPS GPGGA: " << gps_gpgga_str << std::endl;
          TIMEIT_IF_TRUE_OR_DEBUG(
              true, "Time to get gps_gprmc: ",
              gps_gprmc_str =
                  clock->get_sensor("gps_gprmc", boardnumber).value);
          std::cout << "GPS GPRMC: " << gps_gprmc_str << std::endl;
          TIMEIT_IF_TRUE_OR_DEBUG(
              true, "Time to get gps_servo: ",
              gps_servo_str =
                  clock->get_sensor("gps_servo", boardnumber).value);
          std::cout << "GPS Servo: " << gps_servo_str << std::endl;
          TIMEIT_IF_TRUE_OR_DEBUG(
              true, "Time to get gps_time: ",
              gps_time_str = clock->get_sensor("gps_time", boardnumber).value);
          std::cout << "GPS Time: " << gps_time_str << std::endl;
          TIMEIT_IF_TRUE_OR_DEBUG(
              true, "Time to get gps_locked: ",
              gps_lock_str =
                  clock->get_sensor("gps_locked", boardnumber).value);
          std::cout << "GPS locked: " << gps_lock_str << std::endl;
          std::cout << std::endl;
        }
        sleep(update_period_in_s);
      }
    } else {
      sleep(1);
    }
  }
  std::cout << "Stop signal called, exiting" << std::endl;
  return 0;
}
