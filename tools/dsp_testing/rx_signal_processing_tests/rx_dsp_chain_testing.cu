/*
Copyright 2022 SuperDARN Canada

See LICENSE for details

This file contains C++ code to test the CUDA kernels defined in rx_signal_processing/decimate.cu
*/

#include <cuComplex.h>  // cuComplex type and all cuCmulf/cuCaddf functions.
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <complex>
#include <cmath>
#include <vector>
#include "dsp_testing.hpp"
#include "rx_signal_processing/filtering.hpp"
#include "rx_signal_processing/decimate.hpp"


#define FREQS {-1.25e6, 1.25e6}
#define RX_RATE (5.0e6)
#define DM_RATES {10, 5, 6, 5}
#define NUM_SAMPS 451500
#define NUM_CHANNELS 20
#define TAU_SPACING_US 2400
#define PULSE_LENGTH_US 300
#define PULSE_LIST {0, 9, 12, 20, 22, 26, 27}

/**
 * @brief      Creates an array of ideal samples for a 7-pulse SuperDARN sequence.
 *
 * @param[in]  dm_rates           List of downsampling rates for each stage of filtering.
 * @param[in]  rx_rate            RX rate of the system.
 * @param[in]  num_channels       Number of channels to generate samples for.
 * @param[in]  rx_freqs           List of carrier frequencies to receive on.
 * @param[in]  filter_taps        List of filter taps for each stage.
 */
std::vector<std::complex<float>> make_samples(std::vector<uint32_t> dm_rates, double rx_rate, uint32_t num_channels,
                                              std::vector<double> rx_freqs, std::vector<std::vector<float>> filter_taps)
{
  // We need to sample early to account for propagating samples through filters. The number of
  // required early samples is equal to adding half the filter length of each stage, starting with
  // the last stage so that the center point of the filter (point of highest gain) aligns with the
  // center of the pulse. This is the exact number of extra samples needed so that the output
  // data after decimation correctly aligns to the center of the first pulse.
  int64_t extra_samples = 0;

  for (int32_t i=dm_rates.size()-1; i>=0; i--) {
    extra_samples = (extra_samples * dm_rates[i]) + (filter_taps[i].size()/2);
  }

  std::vector<uint32_t> pulse_starts_in_samps;
  std::vector<uint32_t> pulse_ends_in_samps;
  std::vector<uint32_t> pulse_list = PULSE_LIST;
  uint32_t pulse_length_us = PULSE_LENGTH_US;
  uint32_t pulse_length_samps = int(std::floor(float(pulse_length_us) / rx_rate));
  uint32_t tau_spacing_us = TAU_SPACING_US;

  // Get the start and end sample of each pulse
  for (int i=0; i<pulse_list.size(); i++) {
    auto pulse_start_us = pulse_list[i] * tau_spacing_us;
    auto pulse_start_samps = int(std::floor(float(pulse_start_us) / rx_rate));
    pulse_starts_in_samps.push_back(pulse_start_samps + extra_samples);
    pulse_ends_in_samps.push_back(pulse_start_samps + pulse_length_samps + extra_samples);
  }

  // Create samples for a single pulse
  std::vector<std::complex<float>> single_pulse_samps(pulse_length_samps, std::complex<float>(0.0, 0.0));
  for (int f=0; f<rx_freqs.size(); f++) {
    auto sampling_freq = 2 * M_PI * rx_freqs[f] / rx_rate;
    for (int i=0; i<pulse_length_samps; i++) {
      auto radians = fmod(sampling_freq * i, 2 * M_PI);
      single_pulse_samps[i] += std::complex<float>(cos(radians), sin(radians));
    }
  }

  // Now we make a vector of samples for a single antenna
  std::vector<std::complex<float>> single_antenna_samples;
  for (int i=0; i<pulse_list.size(); i++) {
    for (uint32_t j=pulse_starts_in_samps[i]; j<pulse_ends_in_samps[i]; j++) {
      single_antenna_samples[j] = single_pulse_samps[j - pulse_starts_in_samps[i]];
    }
  }

  // Now we make a flat array of (identical) samples for each channel/antenna, with all data for the first
  // channel coming before all data for the second channel, and so on.
  std::vector<std::complex<float>> all_samps;
  for (int i=0; i<num_channels; i++) {
    for (int j=0; j<single_antenna_samples.size(); j++)
      all_samps.push_back(single_antenna_samples[j]);
    }
  }

  return all_samps;
}

/* This is set up to be as close as possible to rx_signal_processing/rx_dsp_chain.cu, with differences only to
 * remove dependencies on the other Borealis modules. The purpose of this method is to test the CUDA kernel
 * processing of some sample data.
 */
int main(int argc, char** argv) {

  // TODO(Remington): Figure out how to load these from a file (use argv?)
  std::vector<std::vector<float>> filter_taps;
  std::ifstream tapfile;

  int num_stages = 4;

  // Get the filter taps for each stage from file.
  for (int i=0; i<num_stages; i++) {
    char filter_stage[sizeof(char)];
    std::sprintf(filter_stage, "%d", i);
    std::vector<float> taps;
    float real, imag;
    char newline_eater;
    tapfile.open("/home/remington/pulse_interfacing/normalscan_taps_" << filter_stage << ".txt"); // TODO(Remington): Put these files somewhere useful
    while (tapfile >> real >> imag >> newline_eater) {
      if ((real != 0.0) || (imag != 0.0))
      taps.push_back(real);
    }
    filter_taps.push_back(taps);
  }
  Filtering filters;

  std::vector<uint32_t> dm_rates = DM_RATES;
  double rx_rate = RX_RATE;
  uint32_t total_antennas = NUM_CHANNELS;
  std::vector<double> rx_freqs = FREQS;

  filters = Filtering(filter_taps);

  // Create the data for this test
  auto in_samps = make_samples(dm_rates, rx_rate, total_antennas, rx_freqs, filter_taps);
  auto samples_needed = NUM_SAMPS;

  // We are not testing beamforming and correlating here, so we omit the sections from rx_dsp_chain.cu pertaining
  // to them. These can be tested by running the radar and comparing with borealis_postprocessors.

  filters.mix_first_stage_to_bandpass(rx_freqs, rx_rate);

  auto complex_taps = filters.get_mixed_filter_taps();

  uint32_t total_dm_rate = 1;
  for (int i=0; i<dm_rates.size(); i++) {
    total_dm_rate *= dm_rates[i];
  }
  double output_sample_rate = rx_rate / total_dm_rate;

  std::vector<rx_slice_test> slice_info;
  for (uint32_t i=0; i<rx_freqs.size(); i++) {
    slice_info.push_back(rx_slice_test(rx_freqs[i], i));
  }

  DSPCoreTesting *dp = new DSPCoreTesting(rx_rate, output_sample_rate, filter_taps, dm_rates, slice_info);

  dp->allocate_and_copy_frequencies(rx_freqs.data(), rx_freqs.size());

  dp->allocate_and_copy_rf_samples(total_antennas, samples_needed, in_samps.data());

  dp->allocate_and_copy_bandpass_filters(complex_taps[0].data(), complex_taps[0].size());

  auto num_output_samples_per_antenna = NUM_SAMPS / dm_rates[0];
  auto total_output_samples_1 = rx_freqs.size() * num_output_samples_per_antenna * total_antennas;

  dp->allocate_output(total_output_samples_1);

  dp->initial_memcpy_callback();

  auto last_filter_output = dp->get_last_filter_output_d();

  call_decimate<DecimationType::bandpass>(dp->get_rf_samples_p(),
      last_filter_output, dp->get_bp_filters_p(), dm_rates[0],
      samples_needed, complex_taps[0].size()/rx_freqs.size(),
      rx_freqs.size(), total_antennas, rx_rate, dp->get_frequencies_p(),
      "Bandpass stage of decimation", dp->get_cuda_stream());

  std::vector<uint32_t> samples_per_antenna(complex_taps.size());
  std::vector<uint32_t> total_output_samples(complex_taps.size());

  samples_per_antenna[0] = num_output_samples_per_antenna;
  total_output_samples[0] = total_output_samples_1;

  // When decimating, we go from one set of samples for each antenna in the first stage
  // to multiple sets of reduced samples for each frequency in further stages. Output samples are
  // grouped by frequency with all samples for each antenna following each other
  // before samples of another frequency start. In the first stage need a filter for each
  // frequency, but in the next stages we only need one filter for all data sets.
  cuComplex* prev_output = last_filter_output;
  for (uint32_t i=1; i<complex_taps.size(); i++) {
    samples_per_antenna[i] = samples_per_antenna[i-1]/dm_rates[i];
    total_output_samples[i] = rx_freqs.size() * samples_per_antenna[i] * total_antennas;

    dp->allocate_and_copy_lowpass_filter(complex_taps[i].data(), complex_taps[i].size());
    dp->allocate_output(total_output_samples[i]);

    auto allocated_lp_filter = dp->get_last_lowpass_filter_d();
    last_filter_output = dp->get_last_filter_output_d();

    call_decimate<DecimationType::lowpass>(prev_output, last_filter_output, allocated_lp_filter,
      dm_rates[i], samples_per_antenna[i-1], complex_taps[i].size(), rx_freqs.size(),
      total_antennas, rx_rate, dp->get_frequencies_p(), " stage of decimation",
      dp->get_cuda_stream());

    prev_output = last_filter_output;
  }

  dp->cuda_postprocessing_callback(total_antennas, samples_needed, samples_per_antenna,
                                   total_output_samples);
}