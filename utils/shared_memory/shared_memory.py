# Copyright SuperDARN Canada 2021
# Converted shared_memory.cpp(.hpp) to Python on Dec. 13, 2021 by Remington Rohel
from numpy.random import randint


def random_string(length: int) -> str:
    """Generates a string of random characters.

    This string is used for creation of named shared memory.

    :param  length: The length of the desired string.
    :type   length: int

    :return:    A string of random characters.
    """
    def randchar() -> str:
        # Lambda expression to return a random character
        charset = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return charset[randint(len(charset))]

    result = ""
    for i in range(length):
        result += randchar()

    return result


class SharedMemoryHandler(object):
    """Class for handling shared memory attributes and functionality."""
    def __init__(self, name: str = None):
        self._region_name = name
        self._shr_region = None

    def create_shr_mem(self, mem_size):
        # shr_region = shr_mem_create(region_name, mem_size);
        pass

    def open_shr_mem(self):
        # shr_region = shr_mem_open(region_name);
        pass

    def get_shrmem_addr(self):
        # return shr_region.get_address();
        pass

    def remove_shr_mem(self):
        # boost::interprocess::shared_memory_object::remove(region_name.c_str());
        pass

    @property
    def get_region_name(self):
        return self._region_name
