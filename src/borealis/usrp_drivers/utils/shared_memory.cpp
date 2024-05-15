#include "shared_memory.hpp"

#include <algorithm>
#include <iostream>
/**
 * @brief      Generates a string of random characters
 *
 * @param[in]  length  The length of desired string.
 *
 * @return     A string of random characters.
 *
 * This string is used for creation of named shared memory.
 */
std::string random_string(size_t length) {
  // Lambda expression to return a random character.
  auto randchar = []() -> char {
    const char charset[] =
        "0123456789"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz";
    const size_t max_index = (sizeof(charset) - 1);
    return charset[rand() % max_index];
  };
  std::string str(length, 0);
  std::generate_n(str.begin(), length, randchar);
  return str;
}

boost::interprocess::mapped_region shr_mem_create(std::string name,
                                                  size_t size) {
  boost::interprocess::shared_memory_object::remove(name.c_str());

  // Create a shared memory object with read/write privilege.
  auto create_mode = boost::interprocess::create_only;
  auto access_mode = boost::interprocess::read_write;
  boost::interprocess::shared_memory_object shm(create_mode, name.c_str(),
                                                access_mode);

  // Set size
  shm.truncate(size);

  // Map the whole shared memory in this process
  boost::interprocess::mapped_region region(shm, access_mode);

  return region;
}

boost::interprocess::mapped_region shr_mem_open(std::string name) {
  // Open a shared memory object with read privilege.
  auto create_mode = boost::interprocess::open_only;
  auto access_mode = boost::interprocess::read_only;
  boost::interprocess::shared_memory_object shm(create_mode, name.c_str(),
                                                access_mode);

  // Map the whole shared memory in this process
  boost::interprocess::mapped_region region(shm, access_mode);

  return region;
}

SharedMemoryHandler::SharedMemoryHandler() {}

SharedMemoryHandler::SharedMemoryHandler(std::string name) {
  region_name = name;
}

void SharedMemoryHandler::create_shr_mem(size_t mem_size) {
  shr_region = shr_mem_create(region_name, mem_size);
}

void SharedMemoryHandler::open_shr_mem() {
  shr_region = shr_mem_open(region_name);
}

void SharedMemoryHandler::remove_shr_mem() {
  boost::interprocess::shared_memory_object::remove(region_name.c_str());
}

void* SharedMemoryHandler::get_shrmem_addr() {
  return shr_region.get_address();
}

std::string SharedMemoryHandler::get_region_name() { return region_name; }
