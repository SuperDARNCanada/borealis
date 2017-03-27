#ifndef SHARED_MEM_H
#define SHARED_MEM_H

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <string>


enum shr_mem_switcher {FIRST, SECOND};

class SharedMemoryHandler {


    public:
        explicit SharedMemoryHandler(std::string name);
        void create_shr_mem(size_t mem_size);
        void open_shr_mem();
        void* get_shrmem_addr();
        void remove_shr_mem();

    private:
        std::string region_name;
        boost::interprocess::mapped_region shr_region;

};




#endif
