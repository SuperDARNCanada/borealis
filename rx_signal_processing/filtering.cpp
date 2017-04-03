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

extern "C" {
  #include "remez.h"
}


/**
 * @brief      The constructor finds the number of filter taps for each stage and then a lowpass
 *             filter for each stage.
 *
 * @param[in]  initial_rx_sample_rate  The USRP RX sampling rate in Hz.
 * @param[in]  sig_options             A reference to the signal processing config options.
 */
Filtering::Filtering(double initial_rx_sample_rate, const SignalProcessingOptions &sig_options) {
  num_first_stage_taps = calculate_num_filter_taps(initial_rx_sample_rate,
                  sig_options.get_first_stage_filter_transition());
  num_second_stage_taps = calculate_num_filter_taps(sig_options.get_first_stage_sample_rate(),
                  sig_options.get_second_stage_filter_transition());
  num_third_stage_taps = calculate_num_filter_taps(sig_options.get_second_stage_sample_rate(),
                  sig_options.get_third_stage_filter_transition());

  first_stage_lowpass_taps = create_filter(num_first_stage_taps,
                                              sig_options.get_first_stage_filter_cutoff(),
                                              sig_options.get_first_stage_filter_transition(),
                                              initial_rx_sample_rate);
  second_stage_lowpass_taps = create_filter(num_second_stage_taps,
                                              sig_options.get_second_stage_filter_cutoff(),
                                              sig_options.get_second_stage_filter_transition(),
                                              sig_options.get_first_stage_sample_rate());
  third_stage_lowpass_taps = create_filter(num_third_stage_taps,
                                              sig_options.get_third_stage_filter_cutoff(),
                                              sig_options.get_third_stage_filter_transition(),
                                              sig_options.get_second_stage_sample_rate());
}


/**
 * @brief      Creates the band edges needed to create lowpass filter with remez.
 *
 * @param[in]  cutoff           Cutoff frequency for the lowpass filter in Hz.
 * @param[in]  transition_band  Width of transition from passband to stopband in Hz.
 * @param[in]  Fs               Sampling frequency in Hz.
 *
 * @return     A vector of calculated lowpass filter bands
 */
std::vector<double> Filtering::create_normalized_lowpass_filter_bands(double cutoff,
                                                                        double transition_band,
                                                                        double Fs) {
  std::vector<double> filterbands; //TODO(keith): describe band edges in documentation
  filterbands.push_back(0.0);
  filterbands.push_back(cutoff/Fs);
  filterbands.push_back((cutoff + transition_band)/Fs);
  filterbands.push_back(0.5);

  return filterbands;
}

/**
 * @brief      Calculates the number filter taps.
 *
 * @param[in]  rate             The sampling rate of the input samples to the filter in Hz.
 * @param[in]  transition_band  Width of transition of band of the filter in Hz.
 *
 * @return     Number of calculated filter taps.
 *
 * Calculates number of filter taps according to Lyon's Understanding Digital
 * Signal Processing(1st edition). Uses Eqn 7-6 to calculate how many filter taps should be used
 * for a given stage. The choice in k=3 was used in the book seems to minimize the amount of
 * ripple in filter. The number of taps will always truncate down to an int.
 */
uint32_t Filtering::calculate_num_filter_taps(double rate, double transition_band) { // REVIEW #26 transition_width and transition_band are both used?
  auto k = 3;
  auto num_taps = k * (rate/transition_band);

  return num_taps;
}

/**
 * @brief      Creates and returns a set of lowpass filter taps using the remez exchange method.
 *
 * @param[in]  num_taps         Number of taps the filter should have.
 * @param[in]  filter_cutoff    Cutoff frequency in Hz for the lowpass filter.
 * @param[in]  transition_band  Width of transition of band of the filter in Hz.
 * @param[in]  rate             Sampling rate of input samples.
 *
 * @return     A vector of filter taps. Filter is real, but represented using complex<float> form
 *             R + i0 for each tap.
 *
 * The GNU Octive remez algorithm being used always returns number of taps + 1. The filter is real,
 * but converted to complex<float> with imaginary part being zero since the CUDA kernels will be
 * doing operations on complex numbers.
 */
std::vector<std::complex<float>> Filtering::create_filter(uint32_t num_taps, double filter_cutoff,
                                                            double transition_band, double rate) {
  // TODO(Keith): explain filter algorithm
  std::vector<double> desired_band_gain = {1.0,0.0};
  std::vector<double> weight = {1.0,1.0};

  auto filter_bands = create_normalized_lowpass_filter_bands(filter_cutoff,transition_band,
              rate);

  //remez returns number of taps + 1. Should we use num_taps + 1
  //TODO(Keith): Investigate number of taps behaviour - does this depend on num_taps being odd or even?
  std::vector<double> filter_taps(num_taps + 1,0.0);
  auto num_filter_band_edges = filter_bands.size()/2;
  auto converges = remez(filter_taps.data(), num_taps + 1, num_filter_band_edges,
    filter_bands.data(),desired_band_gain.data(),weight.data(),BANDPASS,GRIDDENSITY); // REVIEW #15 are we passing the right values, should error check before passing these params
                                              //REPLY We should maybe error check the function inputs. we should have a strong error case in the failure to converge, or verify the filters saved to file.
  if (converges == false){
    std::cerr << "Filter failed to converge with cutoff of " << filter_cutoff
      << "Hz, transition width " << transition_band << "Hz, and rate "
      << rate << "Hz" << std::endl;
    //TODO(keith): throw error
  }

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
    for (uint32_t i=0; i<rx_freqs.size(); i++) { // TODO(keith): comment the protobuf with what rx freqs are. Offset or center.
      auto sampling_freq = 2 * M_PI * rx_freqs[i]/initial_rx_sample_rate; //radians per sample

      //TODO(keith): verify that this is okay.
      for(int j=0;j < first_stage_lowpass_taps.size(); j++) {
        auto radians = fmod(sampling_freq * j,2 * M_PI);
        auto amplitude = first_stage_lowpass_taps[j].real();
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
void Filtering::save_filter_to_file(const std::vector<std::complex<float>> &filter_taps, std::string name) {
  std::ofstream filter;
  filter.exceptions(std::ofstream::failbit | std::ofstream::badbit);
  try {
    filter.open(name); // REVIEW #15 Should check that name isn't null, also - can we use c++ string here?
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
 * @brief      Gets the vector containing bandpass filter taps for each RX frequency.
 *
 * @return     The first stage bandpass taps.
 */
std::vector<std::complex<float>> Filtering::get_first_stage_bandpass_taps_h() {
  //TODO(keith): maybe rename this?
  return first_stage_bandpass_taps_h;
}
