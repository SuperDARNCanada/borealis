// N200 gpio test code
// 201609 - Kevin Krieger

#include <stdlib.h>

#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
// #include <thread>

#include <boost/assign/list_of.hpp>
#include <boost/cstdint.hpp>
#include <boost/format.hpp>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/thread_priority.hpp>

// Test program for SuperDARN N200 GPIOs
// Takes in values for all multi-usrp GPIO either from a
// file or on the command line, or optionally produces gpio outputs for a
// specified pulse sequence

namespace po =
    boost::program_options;  // Options on command line or config file

float speed_of_light = 299792458.0;  // meters per second

static const std::string DEVICE_DEFAULT_MULTI_USRP_ARGS = "addr0=192.168.10.2";
static const std::string GPIO_DEFAULT_CPU_FORMAT = "fc32";
static const std::string GPIO_DEFAULT_OTW_FORMAT = "sc16";
static const std::string GPIO_DEFAULT_GPIO_BANK =
    "RXA";  // RXA is the bank on RX daughterboard
static const size_t GPIO_DEFAULT_NUM_BITS =
    16;  // There are 16 gpio on the LFRX and LFTX
static const std::string GPIO_DEFAULT_GPIO_DDR = "0x0";  // all as inputs
static const std::string GPIO_DEFAULT_GPIO_OUT =
    "0x0";  // GPIO out values, all low
static const std::string GPIO_DEFAULT_GPIO_CTRL =
    "0x0";  // GPIO ctrl values, all low
static const uint32_t GPIO_DEFAULT_UPDATE_PERIOD_IN_S =
    10;  // 10 second period to update users on GPIO registers to stdout
static const uint32_t SEQUENCE_DEFAULT_TAU_IN_US =
    2400;  // Tau value for pulse sequence
static const uint32_t SEQUENCE_DEFAULT_PULSE_LENGTH_IN_US =
    300;                                              // How long is each pulse?
static const float SEQUENCE_DEFAULT_RSEP_IN_KM = 45;  // Range separation
static const float SEQUENCE_DEFAULT_NUM_RANGES = 75;  // Number of ranges
static const float SEQUENCE_DEFAULT_FIRST_RANGE_IN_KM =
    180;  // Distance to the first range
static const uint32_t SEQUENCE_DEFAULT_SCOPE_SYNC_DELAY_IN_US =
    (SEQUENCE_DEFAULT_NUM_RANGES * SEQUENCE_DEFAULT_RSEP_IN_KM +
     SEQUENCE_DEFAULT_FIRST_RANGE_IN_KM) *
    1000.0 * 1000000 * 2.0 /
    speed_of_light;  // How long after last pulse do we keep scope sync high?
static const uint32_t SEQUENCE_DEFAULT_SCOPE_SYNC_PRE_DELAY_IN_US =
    50;  // How long before TR to set atten signal high
static const uint32_t SEQUENCE_DEFAULT_ATTEN_PRE_DELAY_IN_US =
    10;  // How long before TR to set atten signal high
static const uint32_t SEQUENCE_DEFAULT_TR_PRE_DELAY_IN_US =
    5;  // How long before pulse transmission to set TR signal high
static const uint32_t SEQUENCE_DEFAULT_ATTEN_POST_DELAY_IN_US =
    10;  // How long after TR signal goes low to keep atten signal high for
static const uint32_t SEQUENCE_DEFAULT_TR_POST_DELAY_IN_US =
    5;  // How long after pulse transmission to keep TR signal high
static const size_t SEQUENCE_DEFAULT_NUM_PULSES =
    8;  // How many pulses in a pulse sequence?

// Signal handling (CTRL-C)
static bool stop_signal_called = false;
void sig_int_handler(int) { stop_signal_called = true; }

std::string to_bit_string(boost::uint32_t val, const size_t num_bits) {
  std::string out;
  for (int i = num_bits - 1; i >= 0; i--) {
    std::string bit = ((val >> i) & 1) ? "1" : "0";
    out += "  ";
    out += bit;
  }
  return out;
}

void output_reg_values(const std::string bank,
                       const uhd::usrp::multi_usrp::sptr &usrp,
                       const size_t num_bits) {
  std::vector<std::string> attrs = boost::assign::list_of("CTRL")("DDR")(
      "ATR_0X")("ATR_RX")("ATR_TX")("ATR_XX")("OUT")("READBACK");
  std::cout << (boost::format("%10s ") % "Bit");
  for (int i = num_bits - 1; i >= 0; i--) {
    std::cout << (boost::format(" %2d") % i);
  }
  std::cout << std::endl;
  BOOST_FOREACH (std::string &attr, attrs) {
    std::cout << (boost::format("%10s:%s") % attr %
                  to_bit_string(
                      boost::uint32_t(usrp->get_gpio_attr(bank, attr)),
                      num_bits))
              << std::endl;
  }
}
uint64_t fib(uint64_t x) {
  if (x == 0) return 0;
  if (x == 1) return 1;
  return fib(x - 1) + fib(x - 2);
}

void do_something_that_takes_a_while_ms(uint32_t n) {
  //  double x = 0.0;
  //  for (uint32_t i = 0; i<1000; i++) {
  //    x = i*i/2.0*3.14*15125.0*i*4;
  //  }
  //  boost::this_thread::sleep( boost::posix_time::milliseconds(800) );
  //  std::this_thread::sleep_for(std::chrono::milliseconds(x));
  std::cout << fib(n);
}

void print_gpio_banks(const uhd::usrp::multi_usrp::sptr &usrp) {
  std::cout << "Device GPIO banks: ";
  std::vector<std::string> all_banks = usrp->get_gpio_banks(0);
  for (std::vector<std::string>::const_iterator i = all_banks.begin();
       i != all_banks.end(); i++) {
    std::cout << *i << " ";
  }
  std::cout << std::endl;
}

int UHD_SAFE_MAIN(int argc, char *argv[]) {
  // Set the thread priority. Takes float between -1 and 1
  // a higher value is higher priority. 0 is default normal priority.
  if (!uhd::set_thread_priority_safe(1.0)) {
    std::cout << "Setting thread priority failed" << std::endl;
  }

  // register signal handler
  std::signal(SIGINT, &sig_int_handler);

  // Variables for program options
  std::string multi_usrp_args;
  std::string cpu_format, otw_format;
  std::string gpio_bank;
  std::string gpio_ddr;
  std::string gpio_out;
  std::string gpio_ctrl;
  size_t num_bits;
  size_t num_pulses;
  uint32_t tau_in_us;
  uint32_t pulse_length_in_us;
  uint32_t scope_sync_delay_in_us;
  uint32_t scope_sync_pre_delay_in_us;
  uint32_t atten_pre_delay_in_us;
  uint32_t tr_pre_delay_in_us;
  uint32_t atten_post_delay_in_us;
  uint32_t tr_post_delay_in_us;
  uint32_t update_period_in_s;
  uint32_t fib_sequence_number;
  uint32_t start_delay_us;

  // Set up program options
  po::options_description config("Configuration");
  config.add_options()("help", "help message")(
      "multi_usrp_args",
      po::value<std::string>(&multi_usrp_args)
          ->default_value(DEVICE_DEFAULT_MULTI_USRP_ARGS)
          ->composing(),
      "multi uhd device address args")
      //    ("cpu_format",
      //    po::value<std::string>(&cpu_format)->default_value(GPIO_DEFAULT_CPU_FORMAT),
      //    "CPU data format")
      //    ("otw_format",
      //    po::value<std::string>(&otw_format)->default_value(GPIO_DEFAULT_OTW_FORMAT),
      //    "OTW data format")
      ("gpio_bank",
       po::value<std::string>(&gpio_bank)
           ->default_value(GPIO_DEFAULT_GPIO_BANK),
       "GPIO bank string")("gpio_ddr",
                           po::value<std::string>(&gpio_ddr)->default_value(
                               GPIO_DEFAULT_GPIO_DDR),
                           "GPIO DDR value string")(
          "gpio_out",
          po::value<std::string>(&gpio_out)->default_value(
              GPIO_DEFAULT_GPIO_OUT),
          "GPIO OUT value string")("gpio_ctrl",
                                   po::value<std::string>(&gpio_ctrl)
                                       ->default_value(GPIO_DEFAULT_GPIO_CTRL),
                                   "GPIO CTRL value string")(
          "num_bits",
          po::value<size_t>(&num_bits)->default_value(GPIO_DEFAULT_NUM_BITS),
          "Number of bits in GPIO bank")("scope_sync",
                                         "Generate scope sync signal")(
          "pulse_sequence", "Generate pulse sequence")(
          "num_pulses",
          po::value<size_t>(&num_pulses)
              ->default_value(SEQUENCE_DEFAULT_NUM_PULSES),
          "Number of pulses in the sequence to send")(
          "print_diagnostic_info,d",
          "Print out various information about the device and exit")(
          "print_gpio_values,g",
          "Print out the gpio registers periodically (period defined by "
          "update_period)")(
          "update_period",
          po::value<uint32_t>(&update_period_in_s)
              ->default_value(GPIO_DEFAULT_UPDATE_PERIOD_IN_S),
          "If printing gpio values, how often to print them (s)")(
          "tau",
          po::value<uint32_t>(&tau_in_us)
              ->default_value(SEQUENCE_DEFAULT_TAU_IN_US),
          "Tau value for pulse sequence (us)")(
          "pulse_length",
          po::value<uint32_t>(&pulse_length_in_us)
              ->default_value(SEQUENCE_DEFAULT_PULSE_LENGTH_IN_US),
          "Pulse length (us)")(
          "scope_sync_delay",
          po::value<uint32_t>(&scope_sync_delay_in_us)
              ->default_value(SEQUENCE_DEFAULT_SCOPE_SYNC_DELAY_IN_US),
          "Scope sync delay at end of pulse sequence (us)")(
          "scope_sync_pre_delay",
          po::value<uint32_t>(&scope_sync_pre_delay_in_us)
              ->default_value(SEQUENCE_DEFAULT_SCOPE_SYNC_PRE_DELAY_IN_US),
          "Scope sync signal pre-delay before first ATTEN signal (us)")(
          "atten_pre_delay",
          po::value<uint32_t>(&atten_pre_delay_in_us)
              ->default_value(SEQUENCE_DEFAULT_ATTEN_PRE_DELAY_IN_US),
          "Attenuator signal pre-delay before TR signal (us)")(
          "tr_pre_delay",
          po::value<uint32_t>(&tr_pre_delay_in_us)
              ->default_value(SEQUENCE_DEFAULT_TR_PRE_DELAY_IN_US),
          "TR signal pre delay before pulse output (us)")(
          "atten_post_delay",
          po::value<uint32_t>(&atten_post_delay_in_us)
              ->default_value(SEQUENCE_DEFAULT_ATTEN_POST_DELAY_IN_US),
          "Attenuator signal post-delay after TR signal (us)")(
          "tr_post_delay",
          po::value<uint32_t>(&tr_post_delay_in_us)
              ->default_value(SEQUENCE_DEFAULT_TR_POST_DELAY_IN_US),
          "TR signal post delay after pulse output (us)")(
          "fib", po::value<uint32_t>(&fib_sequence_number)->default_value(38),
          "TESTING: fibonacci seq number to calculate up to in pulse train "
          "loop")("delay",
                  po::value<uint32_t>(&start_delay_us)->default_value(200),
                  "TESTING: Start delay in us after scope sync to start first "
                  "pulse's ATTEN signal");

  po::positional_options_description p;
  p.add("blah-example", -1);
  po::variables_map vm;
  po::store(
      po::command_line_parser(argc, argv).options(config).positional(p).run(),
      vm);
  std::ifstream config_filename("gpio_test.cfg");
  po::store(po::parse_config_file(config_filename, config, false), vm);
  po::notify(vm);

  std::cout << std::endl;

  // Print help messages
  if (vm.count("help")) {
    std::cout << config << std::endl;
    return 0;
  }

  // Create multi usrp device (get shared pointer)
  uhd::usrp::multi_usrp::sptr usrp;
  try {
    usrp = uhd::usrp::multi_usrp::make(multi_usrp_args);
  } catch (...) {
    std::cout << "Failed to get usrp device" << std::endl;
    return -1;
  }

  if (vm.count("print_diagnostic_info")) {
    output_reg_values(gpio_bank, usrp, num_bits);
    std::cout << usrp->get_pp_string() << std::endl;
    print_gpio_banks(usrp);
    std::cout << "Number of motherboards: " << usrp->get_num_mboards()
              << std::endl;
    std::cout << "Low level registers: " << std::endl;
    std::vector<std::string> low_level_registers = usrp->enumerate_registers();
    for (std::vector<std::string>::const_iterator i =
             low_level_registers.begin();
         i != low_level_registers.end(); i++) {
      std::cout << *i << std::endl;
    }
    return 0;
  }

  // Get a mask for how many gpio the boards actually have
  uint32_t gpio_mask = (1 << num_bits) - 1;
  double accum_time = 0.0;
  size_t pulse_number = 0;
  uint32_t proper_interpulse_delay_in_us = 0;
  std::vector<size_t> pulses;
  pulses.push_back(0);
  pulses.push_back(14);
  pulses.push_back(22);
  pulses.push_back(24);
  pulses.push_back(27);
  pulses.push_back(31);
  pulses.push_back(42);
  pulses.push_back(43);
  //  pulses = {0, 14, 22, 24, 27, 31, 42, 43};
  std::cout << "Pulse sequence: ";
  for (std::vector<std::size_t>::const_iterator i = pulses.begin();
       i != pulses.end(); i++) {
    std::cout << *i << " ";
  }
  std::cout << std::endl;

  // Generate pulse sequence
  if (vm.count("pulse_sequence")) {
    // First 4 bits are outputs for Scope sync, ATTEN, TR and the mock pulse
    usrp->set_gpio_attr(gpio_bank, "DDR", 0x000F, gpio_mask);
    std::cout << "Generating pulse sequences and waiting for CTRL-C to exit..."
              << std::endl;
    while (not stop_signal_called) {
      std::chrono::steady_clock::time_point pulse_seq_begin =
          std::chrono::steady_clock::now();

      usrp->set_time_now(uhd::time_spec_t(0.0));
      std::chrono::steady_clock::time_point set_time_end =
          std::chrono::steady_clock::now();
      accum_time = 0.0;
      accum_time += (double)(start_delay_us) / 1000000.0;
      usrp->set_command_time(uhd::time_spec_t(accum_time));
      std::chrono::steady_clock::time_point set_command_time_end =
          std::chrono::steady_clock::now();
      usrp->set_gpio_attr(gpio_bank, "OUT", 0xFFFF, 0x0001);  // scope sync high
      std::chrono::steady_clock::time_point set_gpio_end =
          std::chrono::steady_clock::now();
      accum_time += scope_sync_pre_delay_in_us / 1000000.0;
      usrp->set_command_time(uhd::time_spec_t(
          accum_time));  // Wait for delay to start pulse sequence

      for (pulse_number = 0; pulse_number < num_pulses; pulse_number++) {
        usrp->set_gpio_attr(gpio_bank, "OUT", 0xFFFF, 0x0002);  // atten high

        accum_time += atten_pre_delay_in_us / 1000000.0;
        usrp->set_command_time(
            uhd::time_spec_t(accum_time));  // Wait for delay to TR signal
        usrp->set_gpio_attr(gpio_bank, "OUT", 0xFFFF,
                            0x0004);  // TR, scopy sync and atten high

        accum_time += tr_pre_delay_in_us / 1000000.0;
        usrp->set_command_time(
            uhd::time_spec_t(accum_time));  // Wait for delay to pulse
        usrp->set_gpio_attr(gpio_bank, "OUT", 0xFFFF,
                            0x0008);  // Mock pulse high

        accum_time += pulse_length_in_us / 1000000.0;
        usrp->set_command_time(
            uhd::time_spec_t(accum_time));  // Wait for pulse to end
        usrp->set_gpio_attr(gpio_bank, "OUT", 0x0000,
                            0x0008);  // Mock pulse low

        accum_time += tr_post_delay_in_us / 1000000.0;
        usrp->set_command_time(
            uhd::time_spec_t(accum_time));  // delay to TR low
        usrp->set_gpio_attr(gpio_bank, "OUT", 0x0000, 0x0004);  // TR low

        accum_time += atten_post_delay_in_us / 1000000.0;
        usrp->set_command_time(
            uhd::time_spec_t(accum_time));  // delay to atten low
        usrp->set_gpio_attr(gpio_bank, "OUT", 0x0000, 0x0002);  // atten low

        if (pulse_number == num_pulses - 1) {
          // Last pulse, no need to delay again
          std::cout << "Last pulse" << std::endl;
        } else {
          // Delay is from start of one pulse to start of next pulse, so perform
          // proper calculation here
          proper_interpulse_delay_in_us =
              tau_in_us * (pulses[pulse_number + 1] - pulses[pulse_number]);
          proper_interpulse_delay_in_us -= tr_post_delay_in_us;
          proper_interpulse_delay_in_us -= atten_post_delay_in_us;
          proper_interpulse_delay_in_us -= atten_pre_delay_in_us;
          proper_interpulse_delay_in_us -= tr_pre_delay_in_us;
          proper_interpulse_delay_in_us -= pulse_length_in_us;
          //   std::cout << "Pulse sequence #: " << pulses[pulse_number] << "
          //   next: " << pulses[pulse_number+1] << " Proper interpulse delay
          //   (us): " << proper_interpulse_delay_in_us << std::endl;
          accum_time += (double)proper_interpulse_delay_in_us / 1000000.0;
          usrp->set_command_time(
              uhd::time_spec_t(accum_time));  // delay to next pulse
        }
      }

      accum_time += scope_sync_delay_in_us / 1000000.0;
      usrp->set_command_time(
          uhd::time_spec_t(accum_time));  // delay to scope sync low
      usrp->set_gpio_attr(gpio_bank, "OUT", 0x0000, 0x0001);  // scope sync low
      usrp->clear_command_time();
      //      boost::this_thread::sleep(
      //      boost::posix_time::milliseconds(accum_time/1000.0 +100) );
      //      sleep(accum_time+0.01);
      // uhd::time_spec_t curtime = usrp->get_time_now();
      // std::cout << "Accum time: " << accum_time << std::endl;
      // std::cout << boost::format("Current usrp time: %f") %
      // curtime.get_real_secs() << std::endl;
      do_something_that_takes_a_while_ms(fib_sequence_number);
      std::cout << "Set time now: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       set_time_end - pulse_seq_begin)
                       .count()
                << "us"
                << " Set command time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       set_command_time_end - set_time_end)
                       .count()
                << "us"
                << " GPIO attribute time: "
                << std::chrono::duration_cast<std::chrono::microseconds>(
                       set_gpio_end - set_command_time_end)
                       .count()
                << "us" << std::endl;
    }
  } else {
    // Set the GPIO to the values given in command line options
    std::cout
        << "Setting GPIO to given values and waiting for CTRL-C to exit..."
        << std::endl;
    // Note that strtoul has undefined behaviour if the string passed to it is
    // invalid
    uint32_t ddr = strtoul(gpio_ddr.c_str(), NULL, 0);
    uint32_t out = strtoul(gpio_out.c_str(), NULL, 0);
    uint32_t ctrl = strtoul(gpio_ctrl.c_str(), NULL, 0);
    usrp->set_gpio_attr(gpio_bank, "CTRL", ctrl, gpio_mask);
    usrp->set_gpio_attr(gpio_bank, "DDR", ddr, gpio_mask);
    usrp->set_gpio_attr(gpio_bank, "OUT", out, gpio_mask);
    while (not stop_signal_called) {
      // If user asked for updates, then print out every x seconds, otherwise
      // just sleep
      if (vm.count("print_gpio_values")) {
        output_reg_values(gpio_bank, usrp, num_bits);
        sleep(update_period_in_s);
      } else {
        sleep(1);
      }
    }
    std::cout << "Stop signal called, exiting" << std::endl;
  }

  return 0;
}
