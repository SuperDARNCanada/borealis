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
    Filtering() = default;
    explicit Filtering(std::vector<std::vector<float>> input_filter_taps);
    void save_filter_to_file(const std::vector<std::complex<float>> &filter_taps,
                              std::string name);
    void mix_first_stage_to_bandpass(const std::vector<double> &rx_freqs,
                                      double initial_rx_rate);

    std::vector<std::vector<std::complex<float>>> get_filter_taps();

  private:
    //! Vector that holds the vectors of filter taps at each stage.
    std::vector<std::vector<std::complex<float>>> filter_taps;

    //! A vector to hold the bandpass taps after first stage filter has been mixed.
    std::vector<std::complex<float>> bandpass_taps;

    std::vector<std::complex<float>> fill_filter(std::vector<float> &filter_taps);

};

#endif