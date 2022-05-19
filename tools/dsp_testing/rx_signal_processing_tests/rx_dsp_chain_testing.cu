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

// Default decimation scheme parameters from Borealis
#define STAGE0_TAPS {1.52515E-08,2.16921E-08,2.94632E-08,3.87368E-08,4.96984E-08,6.25476E-08,7.74977E-08,9.47768E-08,1.14627E-07,1.37306E-07,1.63085E-07,1.9225E-07,2.251E-07,2.61951E-07,3.03128E-07,3.48972E-07,3.99835E-07,4.56079E-07,5.18076E-07,5.86206E-07,6.60859E-07,7.42425E-07,8.31301E-07,9.27884E-07,1.03257E-06,1.14574E-06,1.2678E-06,1.3991E-06,1.54001E-06,1.69087E-06,1.85199E-06,2.02368E-06,2.20618E-06,2.39973E-06,2.6045E-06,2.82062E-06,3.04818E-06,3.28719E-06,3.5376E-06,3.79927E-06,4.07201E-06,4.3555E-06,4.64934E-06,4.95301E-06,5.26589E-06,5.58722E-06,5.91608E-06,6.25144E-06,6.59208E-06,6.93663E-06,7.28352E-06,7.63098E-06,7.97707E-06,8.31958E-06,8.65611E-06,8.984E-06,9.30031E-06,9.60188E-06,9.88522E-06,1.01466E-05,1.03818E-05,1.05866E-05,1.07561E-05,1.08852E-05,1.09686E-05,1.10003E-05,1.09741E-05,1.08834E-05,1.07211E-05,1.04796E-05,1.01512E-05,9.72738E-06,9.1993E-06,8.55765E-06,7.79263E-06,6.89394E-06,5.85079E-06,4.65189E-06,3.28543E-06,1.73908E-06,-1.79301E-20,-1.94517E-06,-4.11031E-06,-6.50983E-06,-9.15865E-06,-1.20722E-05,-1.52665E-05,-1.87578E-05,-2.25633E-05,-2.67002E-05,-3.11866E-05,-3.60407E-05,-4.12813E-05,-4.69276E-05,-5.29991E-05,-5.9516E-05,-6.64983E-05,-7.39668E-05,-8.19424E-05,-9.04461E-05,-9.94993E-05,-0.000109123,-0.00011934,-0.000130171,-0.000141638,-0.000153763,-0.000166567,-0.000180072,-0.000194299,-0.000209269,-0.000225002,-0.000241519,-0.000258839,-0.000276982,-0.000295967,-0.00031581,-0.000336529,-0.00035814,-0.000380658,-0.000404096,-0.000428468,-0.000453785,-0.000480057,-0.000507293,-0.000535498,-0.000564679,-0.000594839,-0.000625978,-0.000658096,-0.00069119,-0.000725255,-0.000760283,-0.000796263,-0.000833182,-0.000871025,-0.000909771,-0.000949401,-0.000989887,-0.0010312,-0.00107331,-0.00111619,-0.00115978,-0.00120405,-0.00124895,-0.00129444,-0.00134045,-0.00138692,-0.00143379,-0.00148099,-0.00152846,-0.0015761,-0.00162384,-0.00167159,-0.00171926,-0.00176674,-0.00181394,-0.00186074,-0.00190704,-0.00195271,-0.00199763,-0.00204167,-0.00208469,-0.00212656,-0.00216712,-0.00220622,-0.0022437,-0.00227941,-0.00231317,-0.00234481,-0.00237414,-0.00240099,-0.00242516,-0.00244645,-0.00246466,-0.0024796,-0.00249104,-0.00249876,-0.00250256,-0.00250219,-0.00249744,-0.00248806,-0.00247381,-0.00245446,-0.00242976,-0.00239945,-0.00236328,-0.00232099,-0.00227233,-0.00221703,-0.00215483,-0.00208545,-0.00200864,-0.00192412,-0.00183161,-0.00173085,-0.00162156,-0.00150347,-0.00137631,-0.0012398,-0.00109368,-0.000937667,-0.000771501,-0.000594913,-0.00040764,-0.000209421,1.04816E-18,0.000220878,0.00045346,0.000697993,0.000954715,0.00122386,0.00150565,0.00180032,0.00210807,0.00242912,0.00276365,0.00311187,0.00347395,0.00385006,0.00424038,0.00464504,0.0050642,0.00549799,0.00594652,0.0064099,0.00688822,0.00738158,0.00789004,0.00841366,0.00895247,0.00950651,0.0100758,0.0106603,0.0112601,0.011875,0.0125051,0.0131502,0.0138104,0.0144855,0.0151753,0.0158799,0.0165989,0.0173323,0.0180798,0.0188413,0.0196166,0.0204054,0.0212074,0.0220224,0.0228502,0.0236903,0.0245425,0.0254064,0.0262817,0.027168,0.0280649,0.028972,0.0298889,0.030815,0.0317501,0.0326935,0.0336448,0.0346035,0.035569,0.0365409,0.0375186,0.0385014,0.0394889,0.0404804,0.0414754,0.0424732,0.0434731,0.0444747,0.0454771,0.0464798,0.047482,0.0484832,0.0494826,0.0504795,0.0514733,0.0524632,0.0534486,0.0544287,0.0554028,0.0563702,0.0573302,0.0582821,0.0592252,0.0601586,0.0610819,0.0619941,0.0628947,0.0637828,0.0646579,0.0655192,0.066366,0.0671977,0.0680135,0.0688129,0.0695951,0.0703596,0.0711057,0.0718327,0.0725401,0.0732273,0.0738937,0.0745388,0.075162,0.0757627,0.0763405,0.0768949,0.0774253,0.0779314,0.0784127,0.0788688,0.0792993,0.0797038,0.0800819,0.0804334,0.080758,0.0810552,0.081325,0.081567,0.081781,0.0819669,0.0821244,0.0822535,0.0823541,0.0824259,0.0824691,0.0824835,0.0824691,0.0824259,0.0823541,0.0822535,0.0821244,0.0819669,0.081781,0.081567,0.081325,0.0810552,0.080758,0.0804334,0.0800819,0.0797038,0.0792993,0.0788688,0.0784127,0.0779314,0.0774253,0.0768949,0.0763405,0.0757627,0.075162,0.0745388,0.0738937,0.0732273,0.0725401,0.0718327,0.0711057,0.0703596,0.0695951,0.0688129,0.0680135,0.0671977,0.066366,0.0655192,0.0646579,0.0637828,0.0628947,0.0619941,0.0610819,0.0601586,0.0592252,0.0582821,0.0573302,0.0563702,0.0554028,0.0544287,0.0534486,0.0524632,0.0514733,0.0504795,0.0494826,0.0484832,0.047482,0.0464798,0.0454771,0.0444747,0.0434731,0.0424732,0.0414754,0.0404804,0.0394889,0.0385014,0.0375186,0.0365409,0.035569,0.0346035,0.0336448,0.0326935,0.0317501,0.030815,0.0298889,0.028972,0.0280649,0.027168,0.0262817,0.0254064,0.0245425,0.0236903,0.0228502,0.0220224,0.0212074,0.0204054,0.0196166,0.0188413,0.0180798,0.0173323,0.0165989,0.0158799,0.0151753,0.0144855,0.0138104,0.0131502,0.0125051,0.011875,0.0112601,0.0106603,0.0100758,0.00950651,0.00895247,0.00841366,0.00789004,0.00738158,0.00688822,0.0064099,0.00594652,0.00549799,0.0050642,0.00464504,0.00424038,0.00385006,0.00347395,0.00311187,0.00276365,0.00242912,0.00210807,0.00180032,0.00150565,0.00122386,0.000954715,0.000697993,0.00045346,0.000220878,1.04816E-18,-0.000209421,-0.00040764,-0.000594913,-0.000771501,-0.000937667,-0.00109368,-0.0012398,-0.00137631,-0.00150347,-0.00162156,-0.00173085,-0.00183161,-0.00192412,-0.00200864,-0.00208545,-0.00215483,-0.00221703,-0.00227233,-0.00232099,-0.00236328,-0.00239945,-0.00242976,-0.00245446,-0.00247381,-0.00248806,-0.00249744,-0.00250219,-0.00250256,-0.00249876,-0.00249104,-0.0024796,-0.00246466,-0.00244645,-0.00242516,-0.00240099,-0.00237414,-0.00234481,-0.00231317,-0.00227941,-0.0022437,-0.00220622,-0.00216712,-0.00212656,-0.00208469,-0.00204167,-0.00199763,-0.00195271,-0.00190704,-0.00186074,-0.00181394,-0.00176674,-0.00171926,-0.00167159,-0.00162384,-0.0015761,-0.00152846,-0.00148099,-0.00143379,-0.00138692,-0.00134045,-0.00129444,-0.00124895,-0.00120405,-0.00115978,-0.00111619,-0.00107331,-0.0010312,-0.000989887,-0.000949401,-0.000909771,-0.000871025,-0.000833182,-0.000796263,-0.000760283,-0.000725255,-0.00069119,-0.000658096,-0.000625978,-0.000594839,-0.000564679,-0.000535498,-0.000507293,-0.000480057,-0.000453785,-0.000428468,-0.000404096,-0.000380658,-0.00035814,-0.000336529,-0.00031581,-0.000295967,-0.000276982,-0.000258839,-0.000241519,-0.000225002,-0.000209269,-0.000194299,-0.000180072,-0.000166567,-0.000153763,-0.000141638,-0.000130171,-0.00011934,-0.000109123,-9.94993E-05,-9.04461E-05,-8.19424E-05,-7.39668E-05,-6.64983E-05,-5.9516E-05,-5.29991E-05,-4.69276E-05,-4.12813E-05,-3.60407E-05,-3.11866E-05,-2.67002E-05,-2.25633E-05,-1.87578E-05,-1.52665E-05,-1.20722E-05,-9.15865E-06,-6.50983E-06,-4.11031E-06,-1.94517E-06,-1.79301E-20,1.73908E-06,3.28543E-06,4.65189E-06,5.85079E-06,6.89394E-06,7.79263E-06,8.55765E-06,9.1993E-06,9.72738E-06,1.01512E-05,1.04796E-05,1.07211E-05,1.08834E-05,1.09741E-05,1.10003E-05,1.09686E-05,1.08852E-05,1.07561E-05,1.05866E-05,1.03818E-05,1.01466E-05,9.88522E-06,9.60188E-06,9.30031E-06,8.984E-06,8.65611E-06,8.31958E-06,7.97707E-06,7.63098E-06,7.28352E-06,6.93663E-06,6.59208E-06,6.25144E-06,5.91608E-06,5.58722E-06,5.26589E-06,4.95301E-06,4.64934E-06,4.3555E-06,4.07201E-06,3.79927E-06,3.5376E-06,3.28719E-06,3.04818E-06,2.82062E-06,2.6045E-06,2.39973E-06,2.20618E-06,2.02368E-06,1.85199E-06,1.69087E-06,1.54001E-06,1.3991E-06,1.2678E-06,1.14574E-06,1.03257E-06,9.27884E-07,8.31301E-07,7.42425E-07,6.60859E-07,5.86206E-07,5.18076E-07,4.56079E-07,3.99835E-07,3.48972E-07,3.03128E-07,2.61951E-07,2.251E-07,1.9225E-07,1.63085E-07,1.37306E-07,1.14627E-07,9.47768E-08,7.74977E-08,6.25476E-08,4.96984E-08,3.87368E-08,2.94632E-08,2.16921E-08,1.52515E-08}
#define STAGE1_TAPS {0.00134784,0.00212072,0.00304621,0.00409921,0.00523508,0.00638685,0.00746302,0.00834627,0.00889316,0.00893523,0.00828156,0.00672294,0.00403781,-9.33393E-18,-0.00561187,-0.0130033,-0.0223514,-0.0337915,-0.0474023,-0.0631919,-0.0810828,-0.100898,-0.122351,-0.145032,-0.168402,-0.191789,-0.214388,-0.235261,-0.25335,-0.267485,-0.276407,-0.278786,-0.273247,-0.258409,-0.232909,-0.195447,-0.14482,-0.0799664,8.55312E-17,0.0957479,0.207696,0.335979,0.480427,0.640541,0.81548,1.00406,1.20476,1.41571,1.63477,1.85949,2.08718,2.31499,2.53989,2.75879,2.96856,3.16612,3.34851,3.51291,3.65676,3.77776,3.87399,3.94386,3.98625,4.00046,3.98625,3.94386,3.87399,3.77776,3.65676,3.51291,3.34851,3.16612,2.96856,2.75879,2.53989,2.31499,2.08718,1.85949,1.63477,1.41571,1.20476,1.00406,0.81548,0.640541,0.480427,0.335979,0.207696,0.0957479,8.55312E-17,-0.0799664,-0.14482,-0.195447,-0.232909,-0.258409,-0.273247,-0.278786,-0.276407,-0.267485,-0.25335,-0.235261,-0.214388,-0.191789,-0.168402,-0.145032,-0.122351,-0.100898,-0.0810828,-0.0631919,-0.0474023,-0.0337915,-0.0223514,-0.0130033,-0.00561187,-9.33393E-18,0.00403781,0.00672294,0.00828156,0.00893523,0.00889316,0.00834627,0.00746302,0.00638685,0.00523508,0.00409921,0.00304621,0.00212072,0.00134784}
#define STAGE2_TAPS {0.567877,0.804425,0.67621,-3.72854E-16,-1.16223,-2.41482,-3.08583,-2.44275,6.59452E-16,4.20682,9.51148,14.747,18.5905,20.0027,18.5905,14.747,9.51148,4.20682,6.59452E-16,-2.44275,-3.08583,-2.41482,-1.16223,-3.72854E-16,0.67621,0.804425,0.567877}
#define STAGE3_TAPS {5.6471,44.3529,44.3529,5.6471}
#define DM_RATES {10, 5, 6, 5}

// Other default Borealis parameters
#define FREQS {-1.25e6, 1.25e6}
#define RX_RATE (5.0e6)
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
 * @param[in]  num_samples        Number of samples per channel.
 */
std::vector<std::complex<float>> make_samples(std::vector<uint32_t> dm_rates, double rx_rate, uint32_t num_channels,
                                              std::vector<double> rx_freqs, std::vector<std::vector<float>> filter_taps,
                                              uint32_t num_samples)
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
  std::vector<std::complex<float>> single_antenna_samples(num_samples, std::complex<float>(0.0, 0.0));
  for (int i=0; i<pulse_list.size(); i++) {
    for (uint32_t j=pulse_starts_in_samps[i]; j<pulse_ends_in_samps[i]; j++) {
      single_antenna_samples[j] = single_pulse_samps[j - pulse_starts_in_samps[i]];
    }
  }

  // Now we make a flat array of (identical) samples for each channel/antenna, with all data for the first
  // channel coming before all data for the second channel, and so on.
  std::vector<std::complex<float>> all_samps;
  for (int i=0; i<num_channels; i++) {
    for (int j=0; j<single_antenna_samples.size(); j++) {
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

  std::vector<std::vector<float>> filter_taps;

  std::vector<float> taps0 = STAGE0_TAPS;
  std::vector<float> taps1 = STAGE1_TAPS;
  std::vector<float> taps2 = STAGE2_TAPS;
  std::vector<float> taps3 = STAGE3_TAPS;

  filter_taps.push_back(taps0);
  filter_taps.push_back(taps1);
  filter_taps.push_back(taps2);
  filter_taps.push_back(taps3);

  Filtering filters;

  std::vector<uint32_t> dm_rates = DM_RATES;
  double rx_rate = RX_RATE;
  uint32_t total_antennas = NUM_CHANNELS;
  std::vector<double> rx_freqs = FREQS;

  filters = Filtering(filter_taps);

  // Create the data for this test
  auto samples_needed = NUM_SAMPS;
  auto in_samps = make_samples(dm_rates, rx_rate, total_antennas, rx_freqs, filter_taps, samples_needed);

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