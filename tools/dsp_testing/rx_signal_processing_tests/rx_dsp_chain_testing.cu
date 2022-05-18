/*
Copyright 2022 SuperDARN Canada

See LICENSE for details

This file contains C++ code to test the CUDA kernels defined in rx_signal_processing/decimate.cu
*/

#include <cuComplex.h>  // cuComplex type and all cuCmulf/cuCaddf functions.
#include <iostream>
#include <stdint.h>
#include <complex>
#include <cmath>
#include <vector>
#include "decimate_testing.hpp"
#include "rx_signal_processing/filtering.hpp"


#define FREQS {-1.25e6, 1.25e6}
#define RX_RATE (5.0e6)
#define DM_RATES {10, 5, 6, 5}
#define NUM_SAMPS 451500
#define NUM_CHANNELS 20

/* This is set up to be as close as possible to rx_signal_processing/rx_dsp_chain.cu, with differences only to
 * remove dependencies on the other Borealis modules. The purpose of this method is to test the CUDA kernel
 * processing of some sample data.
 */
int main(int argc, char** argv) {

  std::vector<std::vector<float>> filter_taps;
  Filtering filters;

  std::vector<uint32_t> dm_rates = DM_RATES;
  double rx_rate = RX_RATE;
  uint32_t total_antennas = NUM_CHANNELS;
  double output_sample_rate;

  std::vector<std::complex<float>> in_samps(NUM_CHANNELS*NUM_SAMPS, std::complex<float>(1.0, 1.0));

  filters = Filtering(filter_taps);

  // Create the data for this test
  // TODO(Remington): Figure out how to do this

  // We are not testing beamforming and correlating here, so this is a large deviation from rx_dsp_chain.cu

  std::vector<double> rx_freqs = FREQS;
  filters.mix_first_stage_to_bandpass(rx_freqs, rx_rate)

  auto complex_taps = filters.get_mixed_filter_taps();

  // TODO(Remington): Make this class and figure out which parameters are actually needed.
  DSPCoreTesting *dp = new DSPCoreTesting(std::ref(context), sig_options, sp_packet.sequence_num(),
                                      rx_rate, output_sample_rate, filter_taps, initialization_time,
                                      sequence_start_time, dm_rates, slice_info);

  auto total_dm_rate = std::accumulate(dm_rates.begin(), dm_rates.end(), 1, std::multiplies<int64_t>());

  dp->allocate_and_copy_frequencies(rx_freqs.data(), rx_freqs.size());

  dp->allocate_and_copy_rf_samples();

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
  std::vector<uint32_t> total_output_samples(compex_taps.size());

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