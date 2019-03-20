/*
Copyright 2017 SuperDARN Canada

See LICENSE for details

  \file filtering.cpp
  This file contains the implemenation for the all the DSP filtering details.
*/

#include <iostream>
#include <fstream>
#include <system_error>

#include "filtering.hpp"

#include <cmath>
/**
 * @brief      The constructor finds the number of filter taps for each stage and then a lowpass
 *             filter for each stage.
 *
 * @param[in]  input_filter_taps  The filter taps sent from radar control.
 */
Filtering::Filtering(std::vector<std::vector<float>> input_filter_taps) {
  for (auto &taps : input_filter_taps) {
    filter_taps.push_back(fill_filter(taps));
  }
}


/**
 * @brief      Fills the lowpass filter taps with zero to a size that is a power of 2.
 *
 * @param[in]  filter_taps          The filter taps provided, will be real.
 *
 * @return     A vector of filter taps. Filter is real, but represented using complex<float> form
 *             R + i0 for each tap. The vector is filled with zeros at the end to reach a length
 *             that is a power of 2 for processing.
 *
 */
std::vector<std::complex<float>> Filtering::fill_filter(std::vector<float> &filter_taps) {

  //Adding a 0 for the imaginary part to be able to handle complex math in CUDA
  std::vector<std::complex<float>> complex_taps;
  for (auto &i : filter_taps) {
    auto complex_tap = std::complex<float>(i,0.0);
    complex_taps.push_back(complex_tap);
  }

  //Quick and small lambda to get the next power of 2. Returns the same
  //number if already power of two.
  auto next_pwr2 = [](uint32_t n) -> uint32_t {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;

  };

  uint32_t size_difference = 0;
  //Need minimum of 32
  if (complex_taps.size() < 32) {
    size_difference = 32 - complex_taps.size();
  }
  else {
    size_difference = next_pwr2(complex_taps.size()) - complex_taps.size();
  }
  //pad the filter with the result.
  for (uint32_t i=0 ; i<size_difference; i++) {
    complex_taps.push_back(std::complex<float>(0.0,0.0));
  }

  return complex_taps;
}


/**
 * @brief      Mixes the first stage lowpass filter to bandpass filters for each RX frequency.
 *
 * @param[in]  rx_freqs                rx_freqs A reference to a vector of RX frequencies in Hz.
 * @param[in]  initial_rx_sample_rate  initial_rx_sample_rate The USRP RX sampling rate in Hz.
 *
 * Creates a flatbuffer with a bandpass filter for each RX frequency to be used in decimation.
 */
void Filtering::mix_first_stage_to_bandpass(const std::vector<double> &rx_freqs,
                                              double initial_rx_sample_rate) {
    bandpass_taps.clear(); //clear any previously mixed filter taps.
    for (uint32_t i=0; i<rx_freqs.size(); i++) {
    // TODO(keith): comment the protobuf with what rx freqs are. Offset or center.
      auto sampling_freq = 2 * M_PI * rx_freqs[i]/initial_rx_sample_rate; //radians per sample

      for(int j=0;j < filter_taps[0].size(); j++) {
        auto radians = fmod(sampling_freq * j,2 * M_PI);
        auto amplitude = std::abs(filter_taps[0][j]);
        auto I = amplitude * cos(radians);
        auto Q = amplitude * sin(radians);
        bandpass_taps.push_back(std::complex<float>(I,Q));
      }
    }
}

/**
 * @brief      Writes out a set of filter taps to file in case they need to be tested.
 *
 * @param[in]  filter_taps  A reference to a vector of filter taps.
 * @param[in]  name         A output file name.
 */
void Filtering::save_filter_to_file(const std::vector<std::complex<float>> &filter_taps,
                                     std::string name)
{
  std::ofstream filter;
  filter.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  try {
    filter.open(name);
  } catch (std::system_error &e) {
    //TODO(keith): handle error
  }

  if (filter_taps.size() == 0) {
    //TODO(keith): handle error
  }
  for (auto &i : filter_taps){
    filter << i << std::endl;
  }
  filter.close();
}

/**
 * @brief      Gets the mixed filter taps at each stage.
 *
 * @return     The mixed filter taps.
 *
 * A temp vector is created. The first stage taps are replaced with the bandpass taps.
 */
std::vector<std::vector<std::complex<float>>> Filtering::get_mixed_filter_taps() {
  auto temp_taps = filter_taps;
  temp_taps[0] = bandpass_taps;
  return temp_taps;
}

/**
 * @brief      Gets the unmixed filter taps at each stage.
 *
 * @return     The unmixed filter taps.
 *
 * The unmixed filter taps are returned.
 */
std::vector<std::vector<std::complex<float>>> Filtering::get_unmixed_filter_taps() {
  return filter_taps;
}