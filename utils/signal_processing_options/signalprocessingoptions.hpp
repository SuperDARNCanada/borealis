/*Copyright 2016 SuperDARN*/
#ifndef SIGNALPROCESSINGOPTIONS_H
#define SIGNALPROCESSINGOPTIONS_H

#include <stdint.h>
#include <string>
#include "utils/options/options.hpp"

class SignalProcessingOptions: public Options {
 public:
    uint32_t total_receive_antennas;

    double first_stage_sample_rate;
    double second_stage_sample_rate;
    double third_stage_sample_rate;
    double first_stage_filter_cutoff;
    double first_stage_filter_transition;
    double second_stage_filter_cutoff;
    double second_stage_filter_transition;
    double third_stage_filter_cutoff;
    double third_stage_filter_transition;


    SignalProcessingOptions();
    uint32_t get_total_receive_antennas();
    double get_first_stage_sample_rate();
    double get_second_stage_sample_rate();
    double get_third_stage_sample_rate();
    double get_first_stage_filter_cutoff();
    double get_first_stage_filter_transition();
    double get_second_stage_filter_cutoff();
    double get_second_stage_filter_transition();
    double get_third_stage_filter_cutoff();
    double get_third_stage_filter_transition();
};

#endif
