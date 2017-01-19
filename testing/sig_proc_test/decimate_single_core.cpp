#include <vector>
#include <cmath>
#include <complex>
#include <iostream>
#include <chrono>


#define TAPS {0.00915239, 0.00343574, 0.00224751, 0.000106589, -0.00214693, -0.00349026, -0.00317333, -0.00109762, 0.00205945, 0.00506022, 0.00660177, 0.00589334, 0.00304782, -0.000877342, -0.00426933, -0.00560879, -0.00412003, -0.000146325, 0.00495041, 0.00926453, 0.0111033, 0.00964994, 0.00533226, -0.000269668, -0.00495705, -0.00663219, -0.00392405, 0.00341625, 0.0144696, 0.02761, 0.0409102, 0.0527519, 0.0621208, 0.0687165, 0.0728072, 0.0749315, 0.075576, 0.0749315, 0.0728072, 0.0687165, 0.0621208, 0.0527519, 0.0409102, 0.02761, 0.0144696, 0.00341625, -0.00392405, -0.00663219, -0.00495705, -0.000269668, 0.00533226, 0.00964994, 0.0111033, 0.00926453, 0.00495041, -0.000146325, -0.00412003, -0.00560879, -0.00426933, -0.000877342, 0.00304782, 0.00589334, 0.00660177, 0.00506022, 0.00205945, -0.00109762, -0.00317333, -0.00349026, -0.00214693, 0.000106589, 0.00224751, 0.00343574, 0.00915239,}
#define FREQS {1.0,1.0,1.0}
#define RX_RATE (10.0e6)
#define DM_RATE 12
#define NUM_SAMPS 1000000
#define NUM_CHANNELS 20




int main(int argc, char** argv) {

    std::vector<float> filter_taps = TAPS;
    std::vector<float> freqs = FREQS;
    std::vector<std::complex<float>> in_samps(NUM_CHANNELS*NUM_SAMPS,std::complex<float>(1.0,1.0));
    std::vector<std::complex<float>> out_samps(freqs.size()*NUM_CHANNELS*NUM_SAMPS/DM_RATE,std::complex<float>(0.0,0.0));

    std::vector<std::complex<float>> filter_taps_bp(freqs.size() * filter_taps.size());
    for (int i=0; i<freqs.size(); i++) {
        auto sampling_freq = 2 * M_PI * freqs[i]/RX_RATE;

        for (int j=0;j < filter_taps.size(); j++) {
            auto radians = fmod(sampling_freq * j,2 * M_PI);
            auto I = filter_taps[j] * cos(radians);
            auto Q = filter_taps[j] * sin(radians);
            filter_taps_bp[i*filter_taps.size() + j] = std::complex<float>(I,Q);
        }
    }

    std::chrono::steady_clock::time_point timing_start = std::chrono::steady_clock::now();

    for (int m=0; m<freqs.size(); m++){
        for (int n=0; n<NUM_CHANNELS; n++){
            for (int i=0; i<NUM_SAMPS/DM_RATE; i+=DM_RATE){
                for (int j=0; j<filter_taps_bp.size();j++) {
                    if ((i+j) < NUM_SAMPS) {
                        auto channel_off = m * NUM_SAMPS/DM_RATE;
                        auto freq_off = m * NUM_CHANNELS;
                        out_samps[freq_off + channel_off + i] += filter_taps_bp[j] * in_samps[i+j];
                    }
                }
            }
        }
    }

    std::chrono::steady_clock::time_point timing_end = std::chrono::steady_clock::now();
    std::cout << "Time to decimate " << NUM_SAMPS << " by " <<DM_RATE << " on CPU: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (timing_end - timing_start).count()
      << "us" << std::endl;

}












