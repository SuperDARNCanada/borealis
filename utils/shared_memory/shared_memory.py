# Copyright SuperDARN Canada 2021
# Converted shared_memory.cpp(.hpp) to Python on Dec. 13, 2021 by Remington Rohel
from mmap import mmap
import posix_ipc as ipc
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
        self._semaphore = None

    def create_shr_mem(self, mem_size):
        """Creates a new shared memory file."""
        self._shr_region = ipc.SharedMemory(name=self._region_name, flags=ipc.O_CREAT, size=mem_size)
        self._semaphore = ipc.Semaphore(name=None, flags=ipc.O_CREX)

    def open_shr_mem(self):
        """Returns an mmap object pointing to the shared memory file."""
        return mmap(self._shr_region.fd, self._shr_region.size)

    def get_shrmem_addr(self):
        # return shr_region.get_address();
        pass

    def remove_shr_mem(self):
        """Deletes a shared memory file."""
        # boost::interprocess::shared_memory_object::remove(region_name.c_str());
        self._shr_region.close_fd()
        self._shr_region.unlink()

    @property
    def get_region_name(self):
        return self._region_name


class SemaphoreHandler(object):
    """Class for handling semaphores."""
    def __init__(self, name: str = None):
        self._name = name
        self._semaphore = None

    def create_semaphore(self):
        """Creates a semaphore."""
        try:
            self._semaphore = ipc.Semaphore(name=self._name, flags=ipc.O_CREX)
        except ipc.ExistentialError:
            # Semaphore already exists, link to it
            self._semaphore = ipc.Semaphore(name=self._name)

    def get_semaphore(self):
        """Returns the semaphore object."""
        return self._semaphore

    def remove_semaphore(self):
        """Unlinks the semaphore."""
        self._semaphore.release()
        self._semaphore.unlink()
