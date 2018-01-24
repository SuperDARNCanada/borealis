/*

Copyright 2017 SuperDARN Canada

See LICENSE for details

  \file dsp.cu
  This file contains the implementation for the all the needed GPU DSP work.
*/

#include "dsp.hpp"
#include "utils/protobuf/sigprocpacket.pb.h"
#include "utils/protobuf/processeddata.pb.h"
#include "utils/shared_macros/shared_macros.hpp"
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <string>
#include <sstream>
#include <cuComplex.h>
#include <chrono>
#include <thread>

//TODO(keith): decide on handing gpu errors
//TODO(keith): potentially add multigpu support

//This keep postprocess local to this file.
namespace {
  /**
   * @brief      Sends an acknowledgment to the radar control and starts the timing after the
   *             RF samples have been copied.
   *
   * @param[in]  stream           CUDA stream this callback is associated with.
   * @param[in]  status           Error status of CUDA work in the stream.
   * @param[in]  processing_data  A pointer to the DSPCore associated with this CUDA stream.
   */
  void CUDART_CB initial_memcpy_callback_handler(cudaStream_t stream, cudaError_t status,
                          void *processing_data)
  {
    gpuErrchk(status);

    auto imc = [processing_data]()
    {
      auto dp = static_cast<DSPCore*>(processing_data);
      dp->send_ack();
      dp->start_decimate_timing();
      DEBUG_MSG(COLOR_RED("Finished initial memcpy handler for sequence #" 
                 << dp->get_sequence_num() << ". Thread should exit here"));
    };

    std::thread start_imc(imc);
    start_imc.join();
  }

  void create_processed_data_packet(processeddata::ProcessedData &pd, DSPCore* dp)
  {

    for(uint32_t i=0; i<dp->get_rx_freqs().size(); i++) {
      auto dataset = pd.add_outputdataset();
      #ifdef DEBUG
        auto add_debug_data = [dataset,i](std::string stage_name, cuComplex *output_p,
                                            uint32_t num_antennas, uint32_t num_samps_per_antenna)
        {
          auto debug_samples = dataset->add_debugsamples();

          debug_samples->set_stagename(stage_name);
          auto stage_output = output_p;
          auto stage_samps_per_set = num_antennas * num_samps_per_antenna;

          for (uint32_t j=0; j<num_antennas; j++){
            auto antenna_data = debug_samples->add_antennadata();
            for(uint32_t k=0; k<num_samps_per_antenna; k++) {
              auto idx = i * stage_samps_per_set + j * num_samps_per_antenna + k;
              auto antenna_samp = antenna_data->add_antennasamples();
              antenna_samp->set_real(stage_output[idx].x);
              antenna_samp->set_imag(stage_output[idx].y);
            }
          }
        };

        add_debug_data("stage_1",dp->get_first_stage_output_h(),dp->get_num_antennas(),
                    dp->get_num_first_stage_samples_per_antenna());
        add_debug_data("stage_2",dp->get_second_stage_output_h(),dp->get_num_antennas(),
                    dp->get_num_second_stage_samples_per_antenna());
        add_debug_data("stage_3",dp->get_third_stage_output_h(),dp->get_num_antennas(),
                    dp->get_num_third_stage_samples_per_antenna());

      #endif
      DEBUG_MSG("Created dataset for sequence #" << COLOR_RED(dp->get_sequence_num()));
    }

  }

  /**
   * @brief      Spawns the postprocessing work after all work in the CUDA stream is completed.
   *
   * @param[in]  stream           CUDA stream this callback is associated with.
   * @param[in]  status           Error status of CUDA work in the stream.
   * @param[in]  processing_data  A pointer to the DSPCore associated with this CUDA stream.
   *
   * The callback itself cannot call anything CUDA related as it may deadlock. It can, however
   * spawn a new thread and then exit gracefully, allowing the thread to do the work.
   */
  void CUDART_CB postprocess(cudaStream_t stream, cudaError_t status, void *processing_data)
  {
    gpuErrchk(status);

    auto pp = [processing_data]()
    {
      auto dp = static_cast<DSPCore*>(processing_data);
      dp->stop_timing();
      dp->send_timing();

      processeddata::ProcessedData pd;

      TIMEIT_IF_DEBUG("Fill + send processed data time ",
        [&]() {
          create_processed_data_packet(pd,dp);
          //dp->send_processed_data(pd);
        }
      );
      DEBUG_MSG("Cuda kernel timing: " << COLOR_GREEN(dp->get_decimate_timing()) << "ms");
      DEBUG_MSG("Complete process timing: " << COLOR_GREEN(dp->get_total_timing()) << "ms");
      auto sq_num = dp->get_sequence_num();
      delete dp;

      DEBUG_MSG(COLOR_RED("Deleted DP in postprocess for sequence #" << sq_num
                  << ". Thread should terminate here."));
    };

    std::thread start_pp(pp);
    start_pp.detach();

    //TODO(keith): add copy to host and final process details
  }

}


/**
 * @brief      Gets the properties of each GPU in the system.
 *
 * @return     The gpu properties.
 */
std::vector<cudaDeviceProp> get_gpu_properties()
{
  std::vector<cudaDeviceProp> gpu_properties;
  int num_devices = 0;

  gpuErrchk(cudaGetDeviceCount(&num_devices));

  for(int i=0; i< num_devices; i++) {
      cudaDeviceProp properties;
      gpuErrchk(cudaGetDeviceProperties(&properties, i));
      gpu_properties.push_back(properties);
  }

  return gpu_properties;
}

/**
 * @brief      Prints the properties of each cudaDeviceProp in the vector.
 *
 * @param[in]  gpu_properties  A vector of cudaDeviceProp structs.
 *
 * More info on properties and calculations here:
 * https://devblogs.nvidia.com/parallelforall/how-query-device-properties-and-handle-errors-cuda-cc/
 */
void print_gpu_properties(std::vector<cudaDeviceProp> gpu_properties) {
  for(auto i : gpu_properties) {
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
       2.0*i.memoryClockRate*(i.memoryBusWidth/8)/1.0e6 << std::endl;
    std::cout << "  Max shared memory per block: " << i.sharedMemPerBlock
      << std::endl;
    std::cout << "  Warpsize: " << i.warpSize << std::endl;
  }
}


/**
 * @brief      Initializes the parameters needed in order to do asynchronous DSP processing.
 *
 * @param      ack_s         A pointer to the socket used for acknowledging when the transfer of RF
 *                           samples has completed.
 * @param[in]  timing_s      A pointer to the socket used for reporting GPU kernel timing.
 * @param[in]  sq_num        The pulse sequence number for which will be acknowledged.
 * @param[in]  shr_mem_name  The char string used to open a section of shared memory with RF
 *                           samples.
 *
 * The constructor creates a new CUDA stream and initializes the timing events. It then opens
 * the shared memory with the received RF samples for a pulse sequence.
 */
DSPCore::DSPCore(zmq::socket_t *ack_s, zmq::socket_t *timing_s, zmq::socket_t *data_s,
                    uint32_t sq_num, std::string shr_mem_name, std::vector<double> freqs)
{

  sequence_num = sq_num;
  ack_socket = ack_s;
  timing_socket = timing_s;
  data_write_socket = data_s;
  rx_freqs = freqs;
  //https://devblogs.nvidia.com/parallelforall/gpu-pro-tip-cuda-7-streams-simplify-concurrency/
  gpuErrchk(cudaStreamCreate(&stream));
  gpuErrchk(cudaEventCreate(&initial_start));
  gpuErrchk(cudaEventCreate(&kernel_start));
  gpuErrchk(cudaEventCreate(&stop));
  gpuErrchk(cudaEventCreate(&mem_transfer_end));
  gpuErrchk(cudaEventRecord(initial_start, stream));

  shr_mem = SharedMemoryHandler(shr_mem_name);
  shr_mem.open_shr_mem();

}

/**
 * @brief      Frees all associated pointers, events, and streams. Removes and deletes shared
 *             memory.
 */
DSPCore::~DSPCore()
{
  gpuErrchk(cudaFree(rf_samples_d));
  gpuErrchk(cudaFree(first_stage_bp_filters_d));
  gpuErrchk(cudaFree(second_stage_filter_d));
  gpuErrchk(cudaFree(third_stage_filter_d));
  gpuErrchk(cudaFree(first_stage_output_d));
  gpuErrchk(cudaFree(second_stage_output_d));
  gpuErrchk(cudaFree(third_stage_output_d));
  gpuErrchk(cudaFreeHost(host_output_h));
  #ifdef DEBUG
    gpuErrchk(cudaFreeHost(first_stage_output_h));
    gpuErrchk(cudaFreeHost(second_stage_output_h));
    gpuErrchk(cudaFreeHost(third_stage_output_h));
  #endif
  gpuErrchk(cudaEventDestroy(initial_start));
  gpuErrchk(cudaEventDestroy(kernel_start));
  gpuErrchk(cudaEventDestroy(stop));
  gpuErrchk(cudaStreamDestroy(stream));

  shr_mem.remove_shr_mem();

  DEBUG_MSG(COLOR_RED("Running deconstructor for sequence #" << sequence_num));

}

/**
 * @brief      Allocates device memory for the RF samples and then copies them to device.
 *
 * @param[in]  total_samples  Total number of samples to copy.
 */
void DSPCore::allocate_and_copy_rf_samples(uint32_t total_samples)
{
  size_t rf_samples_size = total_samples * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&rf_samples_d, rf_samples_size));
  gpuErrchk(cudaMemcpyAsync(rf_samples_d,shr_mem.get_shrmem_addr(), rf_samples_size,
    cudaMemcpyHostToDevice, stream));

}

/**
 * @brief      Allocates device memory for the first stage filters and then copies them to the
 *             device.
 *
 * @param[in]  taps        A pointer to the first stage filter taps.
 * @param[in]  total_taps  The total number of taps for all filters.
 */
void DSPCore::allocate_and_copy_first_stage_filters(void *taps, uint32_t total_taps)
{
  size_t first_stage_bp_filters_size = total_taps * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&first_stage_bp_filters_d, first_stage_bp_filters_size));
  gpuErrchk(cudaMemcpyAsync(first_stage_bp_filters_d, taps,
        first_stage_bp_filters_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocates device memory for the second stage filter and then copies it to the
 *             device.
 *
 * @param[in]  taps        A pointer to the second stage filter taps.
 * @param[in]  total_taps  The total number of taps for all filters.
 */
void DSPCore::allocate_and_copy_second_stage_filter(void *taps, uint32_t total_taps)
{
  size_t second_stage_filter_size = total_taps * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&second_stage_filter_d, second_stage_filter_size));
  gpuErrchk(cudaMemcpyAsync(second_stage_filter_d, taps,
         second_stage_filter_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocates device memory for the third stage filter and then copies it to the
 *             device.
 *
 * @param[in]  taps        A pointer to the third stage filters.
 * @param[in]  total_taps  The total number of taps for all filters.
 */
void DSPCore::allocate_and_copy_third_stage_filter(void *taps, uint32_t total_taps)
{
  size_t third_stage_filter_size = total_taps * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&third_stage_filter_d, third_stage_filter_size));
  gpuErrchk(cudaMemcpyAsync(third_stage_filter_d, taps,
        third_stage_filter_size, cudaMemcpyHostToDevice, stream));
}

/**
 * @brief      Allocates device memory for the output of the first stage filters.
 *
 * @param[in]  num_first_stage_output_samples  The total number of output samples from first
 *                                             stage.
 */
void DSPCore::allocate_first_stage_output(uint32_t num_first_stage_output_samples)
{
  size_t first_stage_output_size = num_first_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&first_stage_output_d, first_stage_output_size));
}

/**
 * @brief      Allocates device memory for the output of the second stage filters.
 *
 * @param[in]  num_second_stage_output_samples  The total number of output samples from second
 *             stage.
 */
void DSPCore::allocate_second_stage_output(uint32_t num_second_stage_output_samples)
{
  size_t second_stage_output_size = num_second_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&second_stage_output_d, second_stage_output_size));
}

/**
 * @brief      Allocates device memory for the output of the third stage filters.
 *
 * @param[in]  num_third_stage_output_samples  The total number of output samples from third
 *                                             stage.
 */
void DSPCore::allocate_third_stage_output(uint32_t num_third_stage_output_samples)
{
  size_t third_stage_output_size = num_third_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaMalloc(&third_stage_output_d, third_stage_output_size));
}

/**
 * @brief      Allocates host memory for final decimated samples and copies from device to host.
 *
 * @param[in]  num_host_samples  Number of host samples to copy back from device.
 */
void DSPCore::allocate_and_copy_host_output(uint32_t num_host_samples)
{
  size_t host_output_size = num_host_samples * sizeof(cuComplex);
  gpuErrchk(cudaHostAlloc(&host_output_h, host_output_size, cudaHostAllocDefault));
  gpuErrchk(cudaMemcpyAsync(host_output_h, third_stage_output_d,
        host_output_size, cudaMemcpyDeviceToHost,stream));
}


void DSPCore::allocate_and_copy_first_stage_host(uint32_t num_first_stage_output_samples)
{
  size_t host_output_size = num_first_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaHostAlloc(&first_stage_output_h, host_output_size, cudaHostAllocDefault));
  gpuErrchk(cudaMemcpyAsync(first_stage_output_h, first_stage_output_d,
        host_output_size, cudaMemcpyDeviceToHost,stream));
}

void DSPCore::allocate_and_copy_second_stage_host(uint32_t num_second_stage_output_samples)
{
  size_t host_output_size = num_second_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaHostAlloc(&second_stage_output_h, host_output_size, cudaHostAllocDefault));
  gpuErrchk(cudaMemcpyAsync(second_stage_output_h, second_stage_output_d,
        host_output_size, cudaMemcpyDeviceToHost,stream));
}

void DSPCore::allocate_and_copy_third_stage_host(uint32_t num_third_stage_output_samples)
{
  size_t host_output_size = num_third_stage_output_samples * sizeof(cuComplex);
  gpuErrchk(cudaHostAlloc(&third_stage_output_h, host_output_size, cudaHostAllocDefault));
  gpuErrchk(cudaMemcpyAsync(third_stage_output_h, third_stage_output_d,
        host_output_size, cudaMemcpyDeviceToHost,stream));
}

/**
 * @brief      Stops the timers that the constructor starts.
 */
void DSPCore::stop_timing()
{
  gpuErrchk(cudaEventRecord(stop, stream));
  gpuErrchk(cudaEventSynchronize(stop));

  gpuErrchk(cudaEventElapsedTime(&total_process_timing_ms, initial_start, stop));
  gpuErrchk(cudaEventElapsedTime(&decimate_kernel_timing_ms, kernel_start, stop));
  gpuErrchk(cudaEventElapsedTime(&mem_time_ms, initial_start, mem_transfer_end));
  DEBUG_MSG("Cuda memcpy time: " << COLOR_GREEN(mem_time_ms) << "ms");

}

/**
 * @brief      Sends the GPU kernel timing to the radar control.
 *
 * The timing here is used as a rate limiter, so that the GPU doesn't become backlogged with data.
 * If the GPU is overburdened, this will result in less averages, but the system wont crash.
 */
void DSPCore::send_timing()
{
  sigprocpacket::SigProcPacket sp;
  sp.set_kerneltime(decimate_kernel_timing_ms);
  sp.set_sequence_num(sequence_num);

  std::string s_msg_str;
  sp.SerializeToString(&s_msg_str);
  zmq::message_t s_msg(s_msg_str.size());
  memcpy ((void *) s_msg.data (), s_msg_str.c_str(), s_msg_str.size());

  timing_socket->send(s_msg);
  DEBUG_MSG(COLOR_RED("Sent timing after processing with sequence #" << sequence_num));

}


/**
 * @brief      Add the postprocessing callback to the stream.
 *
 */
void DSPCore::cuda_postprocessing_callback(std::vector<double> freqs, uint32_t total_antennas,
                                            uint32_t num_output_samples_per_antenna_1,
                                            uint32_t num_output_samples_per_antenna_2,
                                            uint32_t num_output_samples_per_antenna_3)
{
    #ifdef DEBUG
      auto total_output_samples_1 = num_output_samples_per_antenna_1 * rx_freqs.size() *
                                      total_antennas;
      auto total_output_samples_2 = num_output_samples_per_antenna_2 * rx_freqs.size() *
                                      total_antennas;
      auto total_output_samples_3 = num_output_samples_per_antenna_3 * rx_freqs.size() *
                                      total_antennas;

      allocate_and_copy_first_stage_host(total_output_samples_1);
      allocate_and_copy_second_stage_host(total_output_samples_2);
      allocate_and_copy_third_stage_host(total_output_samples_3);

      num_first_stage_samples_per_antenna = num_output_samples_per_antenna_1;
      num_second_stage_samples_per_antenna = num_output_samples_per_antenna_2;
    #endif

    rx_freqs = freqs;
    num_antennas = total_antennas;
    num_third_stage_samples_per_antenna = num_output_samples_per_antenna_3;

    gpuErrchk(cudaStreamAddCallback(stream, postprocess, this, 0));

    DEBUG_MSG(COLOR_RED("Added stream callback for sequence #" << sequence_num));
}

/**
 * @brief      Sends the acknowledgment to the radar control that the RF samples have been
 *             transfered.
 *
 * RF samples of one pulse sequence can be transfered asynchronously while samples of another are
 * being processed. This means that it is possible to start running a new pulse sequence in the
 * driver as soon as the samples are copied. The asynchronous nature means only timing constraint
 * is the time needed to run the GPU kernels for decimation.
 */
void DSPCore::send_ack()
{
  sigprocpacket::SigProcPacket sp;
  sp.set_sequence_num(sequence_num);

  std::string s_msg_str;
  sp.SerializeToString(&s_msg_str);
  zmq::message_t s_msg(s_msg_str.size());
  memcpy ((void *) s_msg.data(), s_msg_str.c_str(), s_msg_str.size());
  ack_socket->send(s_msg);
  DEBUG_MSG(COLOR_RED("Sent ack after copy for sequence_num #" << sequence_num));
}

void DSPCore::send_processed_data(processeddata::ProcessedData &pd)
{
  std::string p_msg_str;
  pd.SerializeToString(&p_msg_str);
  zmq::message_t p_msg(p_msg_str.size());
  memcpy ((void *) p_msg.data(), p_msg_str.c_str(), p_msg_str.size());
  data_write_socket->send(p_msg);
  DEBUG_MSG(COLOR_RED("Send processed data to data_write for sequence #" << sequence_num));
}


/**
 * @brief      Starts the timing before the GPU kernels execute.
 *
 */
void DSPCore::start_decimate_timing()
{
  gpuErrchk(cudaEventRecord(kernel_start, stream));
  gpuErrchk(cudaEventRecord(mem_transfer_end,stream));
}

/**
 * @brief      Adds the callback to the CUDA stream to acknowledge the RF samples have been copied.
 *
 */
void DSPCore::initial_memcpy_callback()
{
  gpuErrchk(cudaStreamAddCallback(stream, initial_memcpy_callback_handler, this, 0));
}


/**
 * @brief      Gets the device pointer to the RF samples.
 *
 * @return     The RF samples device pointer.
 */
cuComplex* DSPCore::get_rf_samples_p(){
  return rf_samples_d;
}

/**
 * @brief      Gets the device pointer to the first stage bandpass filters.
 *
 * @return     The first stage bandpass filters device pointer.
 */
cuComplex* DSPCore::get_first_stage_bp_filters_p(){
  return first_stage_bp_filters_d;
}

/**
 * @brief      Gets the device pointer to the second stage filters.
 *
 * @return     The second stage filter device pointer.
 */
cuComplex* DSPCore::get_second_stage_filter_p(){
  return second_stage_filter_d;
}

/**
 * @brief      Gets the device pointer to the third stage filters.
 *
 * @return     The third stage filter device pointer.
 */
cuComplex* DSPCore::get_third_stage_filter_p(){
  return third_stage_filter_d;
}

/**
 * @brief      Gets the device pointer to output of the first stage decimation.
 *
 * @return     The first stage output device pointer.
 */
cuComplex* DSPCore::get_first_stage_output_p(){
  return first_stage_output_d;
}

/**
 * @brief      Gets the device pointer to output of the second stage decimation.
 *
 * @return     The second stage output device pointer.
 */
cuComplex* DSPCore::get_second_stage_output_p(){
  return second_stage_output_d;
}

/**
 * @brief      Gets the device pointer to output of the third stage decimation.
 *
 * @return     The third stage output device pointer.
 */
cuComplex* DSPCore::get_third_stage_output_p(){
  return third_stage_output_d;
}

std::vector<double> DSPCore::get_rx_freqs()
{
  return rx_freqs;
}
/**
 * @brief      Gets the CUDA stream this DSPCore's work is associated to.
 *
 * @return     The CUDA stream.
 */
cudaStream_t DSPCore::get_cuda_stream(){
  return stream;
}

/**
 * @brief      Gets the total GPU process timing in milliseconds.
 *
 * @return     The total process timing.
 */
float DSPCore::get_total_timing()
{
  return total_process_timing_ms;
}

/**
 * @brief      Gets the total decimation timing in milliseconds.
 *
 * @return     The decimation timing.
 */
float DSPCore::get_decimate_timing()
{
  return decimate_kernel_timing_ms;
}

cuComplex* DSPCore::get_first_stage_output_h()
{
  return first_stage_output_h;
}

cuComplex* DSPCore::get_second_stage_output_h()
{
  return second_stage_output_h;
}

cuComplex* DSPCore::get_third_stage_output_h()
{
  return third_stage_output_h;
}

uint32_t DSPCore::get_num_antennas()
{
  return num_antennas;
}

uint32_t DSPCore::get_num_first_stage_samples_per_antenna()
{
  return num_first_stage_samples_per_antenna;
}

uint32_t DSPCore::get_num_second_stage_samples_per_antenna()
{
  return num_second_stage_samples_per_antenna;
}

uint32_t DSPCore::get_num_third_stage_samples_per_antenna()
{
  return num_third_stage_samples_per_antenna;
}

uint32_t DSPCore::get_sequence_num()
{
  return sequence_num;
}



