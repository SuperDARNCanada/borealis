/*Copyright 2016 SuperDARN Canada*/
#ifndef SRC_USRP_DRIVERS_UTILS_SHARED_MEMORY_HPP_
#define SRC_USRP_DRIVERS_UTILS_SHARED_MEMORY_HPP_

#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <string>

std::string random_string(size_t length);

class SharedMemoryHandler {
 public:
  explicit SharedMemoryHandler(std::string name);
  SharedMemoryHandler();
  void create_shr_mem(size_t mem_size);
  void open_shr_mem();
  void* get_shrmem_addr();
  void remove_shr_mem();
  std::string get_region_name();

 private:
  std::string region_name;
  boost::interprocess::mapped_region shr_region;
};

#endif  // SRC_USRP_DRIVERS_UTILS_SHARED_MEMORY_HPP_
