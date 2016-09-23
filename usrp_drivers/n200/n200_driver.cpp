/*Copyright 2016 SuperDARN*/
#include <uhd/utils/thread_priority.hpp>
#include <uhd/utils/safe_main.hpp>
#include <uhd/utils/static.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/exception.hpp>
#include <boost/program_options.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
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

namespace pt = boost::property_tree;



class Options {
 protected:
        pt::ptree config_pt;
        void parse_config_file() {
            std::ifstream json_file("../../config.ini");
            boost::property_tree::read_json(json_file, config_pt);
        }

};

class DriverOptions: public Options {
 public:
        std::string devices;
        std::string tx_subdev;
        std::string rx_subdev;
        std::string pps;
        std::string ref;
        double tx_sample_rate;
        double rx_sample_rate;
        std::string cpu;
        std::string otw;

        DriverOptions() {
            parse_config_file();

            devices = config_pt.get<std::string>("devices");
            /*Remove whitespace/new lines from device list*/
            boost::remove_erase_if (devices, boost::is_any_of(" \n"));

            tx_subdev = config_pt.get<std::string>("tx_subdev");
            rx_subdev = config_pt.get<std::string>("rx_subdev");
            pps = config_pt.get<std::string>("pps");
            ref = config_pt.get<std::string>("ref");
            tx_sample_rate = config_pt.get<double>("tx_sample_rate");
            rx_sample_rate = config_pt.get<double>("rx_sample_rate");
            cpu = config_pt.get<std::string>("cpu");
            otw = config_pt.get<std::string>("overthewire"); 
        }

        double get_tx_rate() {
            return tx_sample_rate;
        }

        double get_rx_rate() {
            return rx_sample_rate;
        }

        std::string get_device_args() {
            return devices;
        }

        std::string get_tx_subdev() {
            return tx_subdev;
        }

        std::string get_rx_subdev() {
            return rx_subdev;
        }

        std::string get_pps() {
            return pps;
        }

        std::string get_ref() {
            return ref;
        }

        std::string get_cpu() {
            return cpu;
        }

        std::string get_otw() {
            return otw;
        }

};

class USRP{
 public:
        uhd::usrp::multi_usrp::sptr usrp;

        explicit USRP(std::shared_ptr<DriverOptions> driver_options) {
            usrp = uhd::usrp::multi_usrp::make(driver_options->get_device_args());
            set_usrp_clock_source(driver_options->get_ref());
            set_tx_subdev(driver_options->get_tx_subdev());
            set_tx_rate(driver_options->get_tx_rate());
            set_rx_subdev(driver_options->get_rx_subdev());
            set_rx_rate(driver_options->get_rx_rate());

            set_time_source(driver_options->get_pps());
            check_ref_locked();

        }

        void set_usrp_clock_source(std::string source) {
            usrp->set_clock_source(source);
        }

        void set_tx_subdev(std::string tx_subdev) {
            usrp->set_tx_subdev_spec(tx_subdev);
        }

        void set_tx_rate(double tx_rate) {
            usrp->set_tx_rate(tx_rate);

            double actual_rate = usrp->get_tx_rate();

            if (actual_rate != tx_rate) {
                /*TODO: something*/
            }
        }

        void set_tx_center_freq(double freq, std::vector<size_t> chs) {
            uhd::tune_request_t tune_request(freq);

            for(auto &channel : chs) {
                usrp->set_tx_freq(tune_request, channel);

                double actual_freq = usrp->get_tx_freq(channel);
                if (actual_freq != freq) {
                    /*TODO: something*/
                }

            }

            /*boost::this_thread::sleep(boost::posix_time::seconds(1)); */
        }

        void set_rx_subdev(std::string rx_subdev) {
            usrp->set_rx_subdev_spec(rx_subdev);
        }

        void set_rx_rate(double rx_rate) {
            usrp->set_rx_rate(rx_rate);

            double actual_rate = usrp->get_rx_rate();

            if (actual_rate != rx_rate) {
                /*TODO: something*/
            }
        }

        void set_rx_center_freq(double freq, std::vector<size_t> chs) {
            uhd::tune_request_t tune_request(freq);

            for(auto &channel : chs) {
                usrp->set_rx_freq(tune_request, channel);

                double actual_freq = usrp->get_rx_freq(channel);
                if (actual_freq != freq) {
                    /*TODO: something*/
                }

            }

            /*boost::this_thread::sleep(boost::posix_time::seconds(1)); */
        }        

        void set_time_source(std::string source) {
            usrp->set_time_source(source);
            usrp->set_time_unknown_pps(uhd::time_spec_t(0.0));
        }

        void check_ref_locked() {
            size_t num_boards = usrp->get_num_mboards();

            for(size_t i = 0; i < num_boards; i++) {
                std::vector<std::string> sensor_names;
                sensor_names = usrp->get_mboard_sensor_names(i);
                if ((std::find(sensor_names.begin(), sensor_names.end(), "ref_locked") != sensor_names.end())) {
                    uhd::sensor_value_t ref_locked = usrp->get_mboard_sensor("ref_locked", i);
                    /*
                    TODO: something like this
                    UHD_ASSERT_THROW(ref_locked.to_bool());
                    */
                }

            }
        }

        std::string to_string(std::vector<size_t> chs) {
            std::stringstream device_str;

            device_str << "Using device " << usrp->get_pp_string() << std::endl
                       << "TX rate " << usrp->get_tx_rate()/1e6 << " Msps" << std::endl
                       << "RX rate " << usrp->get_rx_rate()/1e6 << " Msps" << std::endl;
                         
 /*           device_str = boost::format(device_str) % 
                         (usrp->to_pp_string(),usrp->get_tx_rate()/1e6,usrp->get_rx_rate()/1e6);*/

            for(auto &channel : chs) {
/*                device_str += boost::format("TX channel %d freq %f MHz\n") %
                              (channel,usrp->get_tx_freq(channel));*/
                device_str << "TX channel " << channel << " freq " 
                           << usrp->get_tx_freq(channel) << " MHz" << std::endl;
            }

            for(auto &channel : chs) {
                device_str << "RX channel " << channel << " freq " 
                           << usrp->get_tx_freq(channel) << " MHz" << std::endl;
            }

            return device_str.str();
                          
        }


};

class TXMetadata{
 public:
        uhd::tx_metadata_t md;

        TXMetadata(bool start_of_burst,bool end_of_burst,
                   bool has_time_spec,uhd::time_spec_t time_spec) {

            md.start_of_burst = start_of_burst;
            md.end_of_burst = end_of_burst;
            md.has_time_spec = has_time_spec;
            md.time_spec = time_spec;

        }

        uhd::tx_metadata_t get_md() {
            return md;
        }

        void set_start_of_burst(bool start_of_burst) {
            md.start_of_burst = start_of_burst;

        }

        void set_end_of_burst(bool end_of_burst) {
            md.end_of_burst = end_of_burst;
        }

        void set_has_time_spec(bool has_time_spec) {
            md.has_time_spec = has_time_spec;
        }

        void set_time_spec(uhd::time_spec_t time_spec) {
            md.time_spec = time_spec;
        }

};

class RXMetadata{
 public:
        uhd::rx_metadata_t md;

        RXMetadata();

        uhd::rx_metadata_t get_md() {
            return md;
        }

        bool get_end_of_burst() {
            return md.end_of_burst;
        }

        uhd::rx_metadata_t::error_code_t get_error_code() {
            return md.error_code;
        }

        size_t get_fragment_offset() {
            return md.fragment_offset;
        }

        bool get_has_time_spec() {
            return md.has_time_spec;
        }

        bool get_out_of_sequence() {
            return md.out_of_sequence;
        }

        bool get_start_of_burst() {
            return md.start_of_burst;
        }

        uhd::time_spec_t get_time_spec() {
            return md.time_spec;
        }

};

void streamer_thread() {

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
