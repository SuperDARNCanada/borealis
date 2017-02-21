/*Copyright 2016 SuperDARN*/
#include <boost/property_tree/ini_parser.hpp>
#include <boost/range/algorithm_ext/erase.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include <string>
#include "utils/options/options.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"


SignalProcessingOptions::SignalProcessingOptions() {
    Options::parse_config_file();

    total_receive_antennas = boost::lexical_cast<uint32_t>(
                                config_pt.get<std::string>("total_receive_antennas"));
    first_stage_sample_rate = boost::lexical_cast<double>(
                                config_pt.get<std::string>("first_stage_sample_rate"));
    second_stage_sample_rate = boost::lexical_cast<double>(
                                config_pt.get<std::string>("second_stage_sample_rate"));
    third_stage_sample_rate = boost::lexical_cast<double>(
                                config_pt.get<std::string>("third_stage_sample_rate"));
    first_stage_filter_cutoff = boost::lexical_cast<double>(
                                config_pt.get<std::string>("first_stage_filter_cutoff"));
    first_stage_filter_transition = boost::lexical_cast<double>(
                                config_pt.get<std::string>("first_stage_filter_transition"));
    second_stage_filter_cutoff = boost::lexical_cast<double>(
                                config_pt.get<std::string>("second_stage_filter_cutoff"));
    second_stage_filter_transition = boost::lexical_cast<double>(
                                config_pt.get<std::string>("second_stage_filter_transition"));
    third_stage_filter_cutoff = boost::lexical_cast<double>(
                                config_pt.get<std::string>("third_stage_filter_cutoff"));
    third_stage_filter_transition = boost::lexical_cast<double>(
                                config_pt.get<std::string>("third_stage_filter_transition"));
}

uint32_t SignalProcessingOptions::get_total_receive_antennas(){
    return total_receive_antennas;
}

double SignalProcessingOptions::get_first_stage_sample_rate(){
    return first_stage_sample_rate;
}

double SignalProcessingOptions::get_second_stage_sample_rate(){
    return second_stage_sample_rate;
}

double SignalProcessingOptions::get_third_stage_sample_rate(){
    return third_stage_sample_rate;
}

double SignalProcessingOptions::get_first_stage_filter_cutoff(){
    return first_stage_filter_cutoff;
}

double SignalProcessingOptions::get_first_stage_filter_transition(){
    return first_stage_filter_transition;
}

double SignalProcessingOptions::get_second_stage_filter_cutoff(){
    return second_stage_filter_cutoff;
}

double SignalProcessingOptions::get_second_stage_filter_transition(){
    return second_stage_filter_transition;
}

double SignalProcessingOptions::get_third_stage_filter_cutoff(){
    return third_stage_filter_cutoff;
}

double SignalProcessingOptions::get_third_stage_filter_transition(){
    return third_stage_filter_transition;
}
