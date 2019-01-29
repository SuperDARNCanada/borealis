/*
Copyright 2017 SuperDARN Canada

See LICENSE for details
  \file filtering.hpp
*/

#ifndef FILTERING_H
#define FILTERING_H

#include <vector>
#include <complex>
#include "utils/signal_processing_options/signalprocessingoptions.hpp"
#include <string>


/**
 * @brief      Class for filtering.
 */
class Filtering {
  public:
    //http://en.cppreference.com/w/cpp/language/explicit
    explicit Filtering(std::vector<float> &first_stage_taps, 
                       std::vector<float> &second_stage_taps, 
                       std::vector<float> &third_stage_taps, 
                       std::vector<float> &fourth_stage_taps);
    void save_filter_to_file(const std::vector<std::complex<float>> &filter_taps,
                              std::string name);
    void mix_first_stage_to_bandpass(const std::vector<double> &rx_freqs,
                                      double initial_rx_rate);
    uint32_t get_num_first_stage_taps();
    uint32_t get_num_second_stage_taps();
    uint32_t get_num_third_stage_taps();
    uint32_t get_num_fourth_stage_taps();
    std::vector<std::complex<float>> get_first_stage_lowpass_taps();
    std::vector<std::complex<float>> get_second_stage_lowpass_taps();
    std::vector<std::complex<float>> get_third_stage_lowpass_taps();
    std::vector<std::complex<float>> get_fourth_stage_lowpass_taps();    
    std::vector<std::complex<float>> get_first_stage_bandpass_taps_h();

  private:
    //! Number of taps in the first stage filter.
    uint32_t num_first_stage_taps;

    //! Number of taps in the second stage filter.
    uint32_t num_second_stage_taps;

    //! Number of taps in the third stage filter.
    uint32_t num_third_stage_taps;

    //! Number of taps in the fourth stage filter.
    uint32_t num_fourth_stage_taps;

    //! A vector of taps for the first stage lowpass filter.
    std::vector<std::complex<float>> first_stage_lowpass_taps;

    //! A vector of taps for the second stage lowpass filter.
    std::vector<std::complex<float>> second_stage_lowpass_taps;

    //! A vector of taps for the third stage lowpass filter.
    std::vector<std::complex<float>> third_stage_lowpass_taps;

    //! A vector of taps for the fourth stage lowpass filter.
    std::vector<std::complex<float>> fourth_stage_lowpass_taps;

    //! A host side vector that holds the taps for all first stage bandpass filters.
    std::vector<std::complex<float>> first_stage_bandpass_taps_h;

    std::vector<std::complex<float>> fill_filter(std::vector<float> &filter_taps);

};

#endif