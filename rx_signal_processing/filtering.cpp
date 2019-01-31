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
 * @param[in]  first_stage_taps            A reference to the first stage filter taps.
 * @param[in]  second_stage_taps           A reference to the second stage filter taps.
 * @param[in]  third_stage_taps            A reference to the third stage filter taps.
 * @param[in]  fourth_stage_taps           A reference to the fourth stage filter taps. 
 */
Filtering::Filtering(std::vector<float> &first_stage_taps, 
                     std::vector<float> &second_stage_taps, 
                     std::vector<float> &third_stage_taps, 
                     std::vector<float> &fourth_stage_taps) {
  num_first_stage_taps = first_stage_taps.size();
  num_second_stage_taps = second_stage_taps.size();
  num_third_stage_taps = third_stage_taps.size();
  num_fourth_stage_taps = fourth_stage_taps.size();

  first_stage_lowpass_taps = fill_filter(first_stage_taps);
  second_stage_lowpass_taps = fill_filter(second_stage_taps);
  third_stage_lowpass_taps = fill_filter(third_stage_taps);
  fourth_stage_lowpass_taps = fill_filter(fourth_stage_taps);
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

  //TODO(keith): either pad to pwr of 2 or make pwr of 2 filter len
  //Pads to at least len 32 or next pwr of 2 to stop seg fault for now.

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
    first_stage_bandpass_taps_h.clear(); //clear any previously mixed filter taps.
    for (uint32_t i=0; i<rx_freqs.size(); i++) {
    // TODO(keith): comment the protobuf with what rx freqs are. Offset or center.
      auto sampling_freq = 2 * M_PI * rx_freqs[i]/initial_rx_sample_rate; //radians per sample

      //TODO(keith): verify that this is okay.
      for(int j=0;j < first_stage_lowpass_taps.size(); j++) {
        auto radians = fmod(sampling_freq * j,2 * M_PI);
        auto amplitude = std::abs(first_stage_lowpass_taps[j]);
        auto I = amplitude * cos(radians);
        auto Q = amplitude * sin(radians);
        first_stage_bandpass_taps_h.push_back(std::complex<float>(I,Q));
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
 * @brief      Gets the number of first stage taps.
 *
 * @return     The number of first stage taps.
 */
uint32_t Filtering::get_num_first_stage_taps() {
  return num_first_stage_taps;
}

/**
 * @brief      Gets the number of second stage taps.
 *
 * @return     The number of second stage taps.
 */
uint32_t Filtering::get_num_second_stage_taps() {
  return num_second_stage_taps;
}

/**
 * @brief      Gets the number of third stage taps.
 *
 * @return     The number of third stage taps.
 */
uint32_t Filtering::get_num_third_stage_taps() {
  return num_third_stage_taps;
}

/**
 * @brief      Gets the number of fourth stage taps.
 *
 * @return     The number of fourth stage taps.
 */
uint32_t Filtering::get_num_fourth_stage_taps() {
  return num_fourth_stage_taps;
}

/**
 * @brief      Gets the vector of the first stage lowpass filter taps.
 *
 * @return     The first stage lowpass taps.
 */
std::vector<std::complex<float>> Filtering::get_first_stage_lowpass_taps() {
  return first_stage_lowpass_taps;
}

/**
 * @brief      Gets the vector of the second stage lowpass filter taps.
 *
 * @return     The second stage lowpass taps.
 */
std::vector<std::complex<float>> Filtering::get_second_stage_lowpass_taps() {
  return second_stage_lowpass_taps;
}

/**
 * @brief      Gets the vector of the third stage lowpass taps.
 *
 * @return     The third stage lowpass taps.
 */
std::vector<std::complex<float>> Filtering::get_third_stage_lowpass_taps() {
  return third_stage_lowpass_taps;
}

/**
 * @brief      Gets the vector of the fourth stage lowpass taps.
 *
 * @return     The fourth stage lowpass taps.
 */
std::vector<std::complex<float>> Filtering::get_fourth_stage_lowpass_taps() {
  return fourth_stage_lowpass_taps;
}

/**
 * @brief      Gets the vector containing bandpass filter taps for all RX frequencies.
 *
 * @return     The first stage bandpass taps.
 *
 * As an example, if there are 3 Rx frequency filters with 32 taps each, this vector will be 96 taps in size.
 */
std::vector<std::complex<float>> Filtering::get_first_stage_bandpass_taps_h() {
  //TODO(keith): maybe rename this?
  return first_stage_bandpass_taps_h;
}
