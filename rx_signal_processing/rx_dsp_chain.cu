#include <vector>
#include <string>
#include <zmq.hpp> // REVIEW #4 Need to explain what we use from this lib in our general documentation
#include <thread>
#include <complex>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdint.h>
#include <signal.h>
#include <cstdlib>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <cuda_profiler_api.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>
#include "utils/protobuf/computationpacket.pb.h"
#include "utils/protobuf/sigprocpacket.pb.h"
#include "utils/driver_options/driveroptions.hpp"
#include "utils/signal_processing_options/signalprocessingoptions.hpp"
#include "utils/shared_memory/shared_memory.hpp"

#include "dsp.hpp"

extern "C" {
    #include "remez.h"
}

// REVIEW #30 all filter-building functions could be placed in a separate file?
// REPLY could be. Could potentially make a filter building class.

std::vector<double> create_normalized_lowpass_filter_bands(float cutoff, float transition_band,
                        float Fs) { //REVIEW #5 units for params //REPLY this should prolly go in the doc string
    std::vector<double> filterbands; //REVIEW #3 describe band choices and how this works //REPLY Should this description be here or in our written documentation?
    filterbands.push_back(0.0);
    filterbands.push_back(cutoff/Fs);
    filterbands.push_back((cutoff + transition_band)/Fs);
    filterbands.push_back(0.5);

    return filterbands;
}

// REVIEW #30 Is this in the right file? Should it be in dsp.cu along with get_gpu_properties?
// REPLY Perhaps. I like to accurately build functions in places they are being used. This method
// is only being used in this file which is why I didnt build an interface for it.

// REVIEW #4 not sure where cudaDeviceProp is found...
// REPLY NVCC doesn't need headers to know about common CUDA constructs. You basically
// have to study to cuda programming/library guides to know whats available.
// http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#multi-device-system
void print_gpu_properties(std::vector<cudaDeviceProp> gpu_properties) {
    for(auto i : gpu_properties) { // REVIEW #28 does this need to be "auto&" stackoverflow says use this way "auto" if you aren't changing anything, but use auto& if you are.?
                                   // REPLY auto& is only needed if need to modify your iterator. We dont need that feature here.
        std::cout << "Device name: " << i.name << std::endl;
        std::cout << "  Max grid size x: " << i.maxGridSize[0] << std::endl;
        std::cout << "  Max grid size y: " << i.maxGridSize[1] << std::endl;
        std::cout << "  Max grid size z: " << i.maxGridSize[2] << std::endl;
        std::cout << "  Max threads per block: " << i.maxThreadsPerBlock
            << std::endl;
        std::cout << "  Max size of block dimension x: " << i.maxThreadsDim[0]
                << std::endl;
        std::cout << "  Max size of block dimension y: " << i.maxThreadsDim[1]
                << std::endl;
        std::cout << "  Max size of block dimension z: " << i.maxThreadsDim[2]
                << std::endl;
        std::cout << "  Memory Clock Rate (GHz): " << i.memoryClockRate/1e6
            << std::endl;
        std::cout << "  Memory Bus Width (bits): " << i.memoryBusWidth
            << std::endl;
        std::cout << "  Peak Memory Bandwidth (GB/s): " <<
           2.0*i.memoryClockRate*(i.memoryBusWidth/8)/1.0e6 << std::endl; // REVIEW #29 magic calculation with magic numbers?
        std::cout << "  Max shared memory per block: " << i.sharedMemPerBlock
            << std::endl;
    }// REVIEW other properties: asyncEngineCount, totalGlobalMem, regsPerBlock, regsPerMultiprocessor, warpSize, memPitch, clockRate, totalConstMem, major, minor, textureAlignment, deviceOverlap, multiProcessorCount, kernelExecTimeoutEnabled, integrated, canMapHostMemory, computeMode, concurrentKernels, concurrentManagedAccess, ECCEnabled, pciBusID, pciDeviceID, pciDomainID, tccDriver, globalL1CacheSupported, hostNativeAtomicSupported, isMultiGpuBoard, l2CacheSize, localL1CacheSupported, managedMemory, maxSurface[1,2,3]D, maxSurface[1,2]DLayered, maxSurfaceCubemap, maxSurfaceCubemapLayered, maxTexture[1,2]D[Layered,Linear,Mipmap], maxTexture2DGather, maxTexture3D[Alt],maxThreadsPerMultiprocessor, multiGpuBoardGroupID, pageableMemoryAccess, sharedMemPerMultiprocessor, singleToDoublePrecisionPerfRatio, streamPrioritiesSupported, surfaceAlignment, texturePitchAlignment, unifiedAddressing,
}   //REPLY I feel like these properties are somewhat low level and dont offer much to be listed at this point.

uint32_t calculate_num_filter_taps(float rate, float transition_width) { // REVIEW #26 transition_width and transition_band are both used?
    auto k = 3; //from formula 7-6 of Lyons text REVIEW #1 and #7 add some explanation for design of choice of k=3
                // REPLY this feels well suited for our written documentation.
    auto num_taps = k * (rate/transition_width); //REVIEW #5 provide units for rate & transition width
                                                 //REPLY This can be added to the docstring

    return num_taps; // REVIEW #33
    //REVIEW #28 this function is returning a uint32; will this always round down? comment/document this
}   //REPLY conversion from float to int never rounds, it always truncates. I'm unsure this needs to
    //commented unless we need to choose and describe a particular choice of floor, ceil, or round

std::vector<std::complex<float>> create_filter(uint32_t num_taps, float filter_cutoff,
                                                float transition_width, // REVIEW #31 this line is over 80 characters, need to be consistent. #31
                                                                        //REPLY I've been trying to stick to 100 lines. 80 is tough with all the templating
                                    float rate) { // REVIEW #5 filter_cutoff, transition_width, rate
                                                  //REPLY docstring
    // REVIEW #3 explain algorithm including weight
    // REPLY to be done in written documentation?
    std::vector<double> desired_band_gain = {1.0,0.0};
    std::vector<double> weight = {1.0,1.0};

    auto filter_bands = create_normalized_lowpass_filter_bands(filter_cutoff,transition_width,
                            rate);

    /*remez returns number of taps + 1. Should we use num_taps + 1
      or should we pass num_taps - 1 to remez?
    */ //REVIEW #6 make this a TODO - does this depend on num_taps being odd or even?
    //TODO(Keith): Investigate number of taps behaviour
    std::vector<double> filter_taps(num_taps + 1,0.0);
    auto converges = remez(filter_taps.data(), num_taps + 1, (filter_bands.size()/2), // REVIEW #32 declare and initialize number of bands for better readability
                                                                                      //REPLY I am unsure what you mean as filterbands is both declared and initialized. Do you just not want the use of auto here?
        filter_bands.data(),desired_band_gain.data(),weight.data(),BANDPASS,GRIDDENSITY); // REVIEW #15 are we passing the right values, should error check before passing these params
                                                                                          //REPLY We should maybe error check the function inputs. we should have a strong error case in the failure to converge, or verify the filters saved to file.
    if (converges < 0){ // REVIEW #10 remez returns True or False is this check correct?
        std::cerr << "Filter failed to converge with cutoff of " << filter_cutoff // REVIEW #5 & #34 print units
            << ", transition width " << transition_width << ", and rate "
            << rate << std::endl;
        //TODO(keith): throw error
    }

    std::vector<std::complex<float>> complex_taps;
    for (auto &i : filter_taps) {
        auto complex_tap = std::complex<float>(i,0.0);
        complex_taps.push_back(complex_tap); // REVIEW #0 does this work?
                                             // REPLY This is just converting real numbers to complex numbers with 0 in the imag part.
                                              //  I dont think this needs to be here anymore. It can be done during the mixing.
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

void save_filter_to_file(std::vector<std::complex<float>> filter_taps, const char* name) {
    std::ofstream filter;
    filter.open(name); // REVIEW #15 Should check that name isn't null, also - can we use c++ string here?
    for (auto &i : filter_taps){ // REVIEW #15 Should filter_taps be error checked? What if it's an empty vector?
        filter << i << std::endl; // REVIEW #9 Does the initialization of filter always succeed? What if not?
    }
    filter.close();
}


int main(int argc, char **argv){
    // REVIEW #35 main is > 200 lines, kind of large maybe the entire filter setup process could be moved out?
    GOOGLE_PROTOBUF_VERIFY_VERSION; // REVIEW #4 state what this does? macro to verify headers and lib are same version.

    auto driver_options = DriverOptions(); // REVIEW #15 Should do a quick sanity check on all config options for driver, sig
    auto sig_options = SignalProcessingOptions(); // #26 REVIEW Should the naming be updated along with DSP ?
    auto rx_rate = driver_options.get_rx_rate(); // #5 REVIEW What units is rx_rate in?
    zmq::context_t sig_proc_context(1); // REVIEW #4 - what is "1"?

    zmq::socket_t driver_socket(sig_proc_context, ZMQ_PAIR);
    driver_socket.bind("ipc:///tmp/feeds/1"); // REVIEW #29 Should this be in a config file? Sort of a magic string right now

    // REVIEW #1 Need a comment here to explain which blocks these 3 sockets talk to, the driver socket is obvious, but these three may not be
    zmq::socket_t radarctrl_socket(sig_proc_context, ZMQ_PAIR); // REVIEW #26 Name of radarctrl may need to be updated to be consistent with our discussion on Friday March 10th
    radarctrl_socket.bind("ipc:///tmp/feeds/2");

    zmq::socket_t ack_socket(sig_proc_context, ZMQ_PAIR);
    ack_socket.bind("ipc:///tmp/feeds/3"); // REVIEW #37 check the errno on all binds

    zmq::socket_t timing_socket(sig_proc_context, ZMQ_PAIR);
    timing_socket.bind("ipc:///tmp/feeds/4");

    auto gpu_properties = get_gpu_properties();
    print_gpu_properties(gpu_properties);

    uint32_t first_stage_dm_rate = 0, second_stage_dm_rate = 0, third_stage_dm_rate = 0; // REVIEW #32 this line only initializes third stage, the others are only declared. bad style?
                                                                                         // REVIEW just a mistake. This doesnt work like python does
    //Check for non integer dm rates
    if (fmod(rx_rate,sig_options.get_first_stage_sample_rate()) > 0.0) {
        //TODO(keith): handle error
    }
    else if (fmod(sig_options.get_first_stage_sample_rate(),
                    sig_options.get_second_stage_sample_rate()) > 0.0) {
        //TODO(keith): handle error
    }
    else if(fmod(sig_options.get_second_stage_sample_rate(),
                sig_options.get_third_stage_sample_rate()) > 0.0) { // REVIEW #1 or #2 perhaps a comment needed to clarify what this error means (dm rate needs to be int)
        //TODO(keith): handle error                                     // REPLY okay
    }
    else{
        auto float_dm_rate = rx_rate/sig_options.get_first_stage_sample_rate(); // REVIEW #26 don't like the name rate_f, not obvious why it's called that, seems like a temp variable, so perhaps indicate that?
        first_stage_dm_rate = static_cast<uint32_t>(float_dm_rate);             // REPLY okay

        float_dm_rate = sig_options.get_first_stage_sample_rate()/
                    sig_options.get_second_stage_sample_rate();
        second_stage_dm_rate = static_cast<uint32_t>(float_dm_rate);

        float_dm_rate = sig_options.get_second_stage_sample_rate()/
                    sig_options.get_third_stage_sample_rate();
        third_stage_dm_rate = static_cast<uint32_t>(float_dm_rate);
    }
    // REVIEW #15 Even though the second and third stage sample rates are set in config, the same error check should be done on the 2nd and 3rd stage dm rates as for first stage dm rate.
    // REPLY okay
    std::cout << "1st stage dm rate: " << first_stage_dm_rate << std::endl
        << "2nd stage dm rate: " << second_stage_dm_rate << std::endl
        << "3rd stage dm rate: " << third_stage_dm_rate << std::endl;


    auto S_lowpass1 = calculate_num_filter_taps(rx_rate, //REVIEW #26 perhaps name of S_lowpass1 could be more self-explanatory as lowpass1_numtaps ?
                                                         //REPLY I kept the naming consistent with the nomclature of the textbook.
                                    sig_options.get_first_stage_filter_transition());
    auto S_lowpass2 = calculate_num_filter_taps(sig_options.get_first_stage_sample_rate(),
                                    sig_options.get_second_stage_filter_transition());
    auto S_lowpass3 = calculate_num_filter_taps(sig_options.get_second_stage_sample_rate(),
                                    sig_options.get_third_stage_filter_transition());


    std::chrono::steady_clock::time_point timing_start = std::chrono::steady_clock::now();


    auto filtertaps_1 = create_filter(S_lowpass1, sig_options.get_first_stage_filter_cutoff(),
                        sig_options.get_first_stage_filter_transition(), rx_rate);
    auto filtertaps_2 = create_filter(S_lowpass2,sig_options.get_second_stage_filter_cutoff(),
                        sig_options.get_second_stage_filter_transition(),
                        sig_options.get_first_stage_sample_rate());
    auto filtertaps_3 = create_filter(S_lowpass3,sig_options.get_third_stage_filter_cutoff(),
                        sig_options.get_third_stage_filter_transition(),
                        sig_options.get_second_stage_sample_rate());

    std::cout << "Number of 1st stage taps: " << S_lowpass1 << std::endl
        << "Number of 2nd stage taps: " << S_lowpass2 << std::endl // REVIEW #34 mention that it's the number of taps?
        << "Number of 3rd stage taps: " << S_lowpass3 <<std::endl
        << "Number of 1st stage taps after padding: " << filtertaps_1.size() <<std::endl
        << "Number of 2nd stage taps after padding: " << filtertaps_2.size() <<std::endl
        << "Number of 3rd stage taps after padding: " << filtertaps_3.size() <<std::endl;
    std::chrono::steady_clock::time_point timing_end = std::chrono::steady_clock::now();
    std::cout << "Time to create 3 filters: "
      << std::chrono::duration_cast<std::chrono::microseconds>
                                                  (timing_end - timing_start).count() // REVIEW #31 fix this whitespace up
                                                                                      // REPLY how to fix this very long line?
      << "us" << std::endl;

    save_filter_to_file(filtertaps_1,"filter1coefficients.dat");
    save_filter_to_file(filtertaps_2,"filter2coefficients.dat"); // REVIEW #36 where should the coefficient files be placed? Currently this will put them in the calling dir I think
                                                                 // REPLY It does, and I do not know. Up to us I guess.
    save_filter_to_file(filtertaps_3,"filter3coefficients.dat");

    while(1){
        //Receive packet from radar control
        zmq::message_t radctl_request;
        radarctrl_socket.recv(&radctl_request);
        sigprocpacket::SigProcPacket sp;  // REVIEW #26 'sp' and 'r_msg_str' var names need re-thinking (same below with 'cp' and 'c_msg_str')
                                          //REPLY okay. open to suggestions.
        std::string r_msg_str(static_cast<char*>(radctl_request.data()), radctl_request.size());
        if (sp.ParseFromString(r_msg_str) == false){
            //TODO(keith): handle error
        }
        // REVIEW #37 need to check the boolean return value from each ParseFromString call http://stackoverflow.com/questions/22121922/how-can-i-get-more-details-about-errors-generated-during-protobuf-parsing-c
        // REPLY Should this just halt the program? The only feasible way this goes wrong is if someone sends
        //non protobufs

	    // REVIEW #15 all protobuf fields are optional, check all fields you require are filled (similarly when using computationpacket)
        // REPLY I think this is only possible if default fields are not also valid values
        std::cout << "Got radar_control request" << std::endl; // REVIEW #34 Change to radar_control to indicate where the request actually came from? (same with 'drive' below, maybe name 'usrp_driver' to be consistent with naming of dir structure)
                                                           // REPLY okay. I was thinking of removing this though.
        //Then receive packet from driver
        zmq::message_t driver_request;
        driver_socket.recv(&driver_request);
        computationpacket::ComputationPacket cp;
        std::string c_msg_str(static_cast<char*>(driver_request.data()), driver_request.size());
        if (cp.ParseFromString(c_msg_str) == false) {
            //TODO(keith): handle error
        }

        std::cout << "Got driver request" << std::endl;

        //Verify driver and radar control packets align
        if (sp.sequence_num() != cp.sequence_num()) {
            //TODO(keith): handle error
            std::cout << "SEQUENCE NUMBER mismatch radar_control: " << sp.sequence_num() // REVIEW #34 - debug output 'rctl' should be renamed to radar_control and driver to 'usrp_driver'
                << " usrp_driver: " << cp.sequence_num();                                // REPLY this might be removed depending on how we handle errors
        }

        //Parse needed packet values now
        if (sp.rxchannel_size() == 0) {
            //TODO(keith): handle error
        }
        std::vector<double> rx_freqs; // REVIEW #28 Shouldn't this just be a vector of integers? I don't think sub-hz resolution available or necessary? Careful of division elsewhere if it's changed
                                      //REPLY probably. I just used what was in the protobuffer.
        for(int i=0; i<sp.rxchannel_size(); i++) {
            rx_freqs.push_back(sp.rxchannel(i).rxfreq());
        }

        timing_start = std::chrono::steady_clock::now();

        // REVIEW #1 since filtertaps_1_bp_h has a filter in it for each rx freq, it should be indicated/documented somehow
        //REPLY could be renamed. Its just a flat buffer instead of a 2d array
        std::vector<std::complex<float>> filtertaps_1_bp_h(rx_freqs.size()*filtertaps_1.size()); // REVIEW #26 - what does 'bp_h' mean?
                                                                                                 //REPLY bandpass_host
        for (int i=0; i<rx_freqs.size(); i++) { // REVIEW #10 What is rx_freq? Is it actual freq or offset from centre freq?
                                                // REPLY This should be commented in the protobuffer I think
            auto sampling_freq = 2 * M_PI * rx_freqs[i]/rx_rate; // REVIEW #5 radians per sample
                                                                 // trailing comment for this?

            for(int j=0;j < filtertaps_1.size(); j++) { // REVIEW #0 and #8 If filter taps had imaginary part, this would not break, but it wouldn't work properly and it would be a mystery as to why. Place full calculation of I and Q
                                                        // REPLY This filter should be purely real before mixing. I will likely rework it to not be in complex real form.
                auto radians = fmod(sampling_freq * j,2 * M_PI);
                auto I = filtertaps_1[j].real() * cos(radians);
                auto Q = filtertaps_1[j].real() * sin(radians);
                filtertaps_1_bp_h[i*filtertaps_1.size() + j] = std::complex<float>(I,Q);
            }
        }

        timing_end = std::chrono::steady_clock::now();

        std::cout << "NCO mix timing: "
          << std::chrono::duration_cast<std::chrono::microseconds>(timing_end - timing_start).count()
          << "us" << std::endl;

        std::vector<std::complex<float>> filtertaps_2_h(filtertaps_2.size()); // REVIEW #26 what does _h mean?
                                                                              // REPLY cuda examples will often denote a distinction between host and device with d and h. I followed the same scheme. could rename
        std::vector<std::complex<float>> filtertaps_3_h(filtertaps_3.size()); // REVIEW #0 need filtertaps_3 and filtertaps_2.size() * rx_freqs.size()
                                                                              // REPLY I actually think this is wrong to supply a size here at all.
        for (uint32_t i=0; i< rx_freqs.size(); i++){ // REVIEW #28 iterator type not consistent with previous for loop, should it be of type size_t?
                                                     // REPLY I need more clarifying here
            filtertaps_2_h.insert(filtertaps_2_h.end(),filtertaps_2.begin(),filtertaps_2.end()); // REVIEW #22 is this duplication necessary?
                                                                                                 // REPLY Maybe not. I now realize from reviewing that we go from one set of samples and multiple filters
                                                                                                 // to multiple sets of samples and one filter
            filtertaps_3_h.insert(filtertaps_3_h.end(),filtertaps_3.begin(),filtertaps_3.end());
        }

        DSPCore *dp = new DSPCore(&ack_socket, &timing_socket,
                                                         sp.sequence_num(), cp.name().c_str());


        auto total_antennas = sig_options.get_main_antenna_count() +
                                sig_options.get_interferometer_antenna_count();
        auto total_samples = cp.numberofreceivesamples() * total_antennas;

        std::cout << "Total samples in data message: " << total_samples // REVIEW #34 change 'elements' to 'samples' to be clear
            << std::endl;                                               // REPLY okay

        dp->allocate_and_copy_rf_samples(total_samples);
        dp->allocate_and_copy_first_stage_filters(filtertaps_1_bp_h.data(), filtertaps_1_bp_h.size());


// REVIEW #10 Should num of output samples also be based on length of filter to reduce edge effects? See overlap-save algorithm
// REPLY I dont know the answer to this.
        auto num_output_samples_1 = rx_freqs.size() * cp.numberofreceivesamples()/first_stage_dm_rate // REVIEW #28 - What if the division has a remainder?
                                        * total_antennas;

        dp->allocate_first_stage_output(num_output_samples_1);

        gpuErrchk(cudaStreamAddCallback(dp->get_cuda_stream(),
                                    DSPCore::initial_memcpy_callback, dp, 0)); // REVIEW #3 explain how the timing is done, explain what this does (events, callback, kernel call) a diagram would probably be good in the user-facing documentation
                                                                               // REPLY agreed. I already started one.

        dp->call_decimate(dp->get_rf_samples_p(),
            dp->get_first_stage_output_p(),
            dp->get_first_stage_bp_filters_p(), first_stage_dm_rate,
            cp.numberofreceivesamples(), filtertaps_1.size(), rx_freqs.size(),
            total_antennas, "First stage of decimation");



        dp->allocate_and_copy_second_stage_filters(filtertaps_2_h.data(), filtertaps_2_h.size());
        auto num_output_samples_2 = num_output_samples_1 / second_stage_dm_rate; //REVIEW #28 what if there is a remainder? Should this be declared as some form of int if it's necessary for it to be an int?
                                                                                 //REPLY could be. All values used at this point are uint32 so it should follow truncation rules.
        dp->allocate_second_stage_output(num_output_samples_2);
        auto samples_per_antenna_2 = cp.numberofreceivesamples()/first_stage_dm_rate; // REVIEW #26 num_samps_2 could be better named, samples2_per_antenna ?
                                                                            // REPLY okay
        dp->call_decimate(dp->get_first_stage_output_p(),
            dp->get_second_stage_output_p(),
            dp->get_second_stage_filters_p(), second_stage_dm_rate,
            samples_per_antenna_2, filtertaps_2.size(), rx_freqs.size(),
            total_antennas, "Second stage of decimation"); // REVIEW #0 first_stage_output_p points to array of data that has separate datapoints for different frequencies, is this taken into account?
// REVIEW #0 should we pass rx_freqs.size() or should we just pass = 1 because there is only one lowpass filter in this stage? Perhaps should write another decimate function for this.
//REPLY might need a redesign here

// REVIEW #0 current call_decimate function is built for the bandpass filters (to separate data by frequency) - for the second stage, we don't need this so a separate function would be better
// REPLY Might need a redesign here, but perhaps not. I find it more advantageous to minimize the amount of cuda to have to debug

        dp->allocate_and_copy_third_stage_filters(filtertaps_3_h.data(), filtertaps_3_h.size());
        auto num_output_samples_3 = num_output_samples_2 / third_stage_dm_rate;
        dp->allocate_third_stage_output(num_output_samples_3);
        auto num_samps_3 = num_samps_2/second_stage_dm_rate;
        dp->call_decimate(dp->get_second_stage_output_p(),
            dp->get_third_stage_output_p(),
            dp->get_third_stage_filters_p(), third_stage_dm_rate,
            num_samps_3, filtertaps_3.size(), rx_freqs.size(),
            total_antennas, "Third stage of decimation");

        dp->allocate_and_copy_host_output(num_output_samples_3);

        // New in CUDA 5.0: Add a CPU callback which is called once all currently pending operations in the CUDA stream have finished
        gpuErrchk(cudaStreamAddCallback(dp->get_cuda_stream(),
                                            DSPCore::cuda_postprocessing_callback, dp, 0));


    }


}
