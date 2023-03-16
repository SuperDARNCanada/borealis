#!/usr/bin/python3

"""
    realtime package
    ~~~~~~~~~~~~~~~~
    Sends data to realtime applications.

    :copyright: 2019 SuperDARN Canada
"""

import zmq
import threading
import os
import queue
import re
import json
import zlib
import pydarnio
import numpy as np
from backscatter import fitacf

from utils.options import Options
from utils import socket_operations as so


class RealtimeServer:
    """
    Realtime server with two threads. One of which expects RAWACF temp files from the
    datawrite module, which it then converts to FITACF format and enqueues.
    The second thread dequeues the converted fitacf data and sends to any listening clients.
    Usage:
    rt = RealtimeServer()
    rt.start_threads()
    """
    def __init__(self, logger):
        """
        :param logger: Logging object
        """
        self.options = Options()

        self.borealis_sockets = so.create_sockets([self.options.rt_to_dw_identity],
                                                  self.options.router_address)
        self.data_write_to_realtime = self.borealis_sockets[0]

        self.context = zmq.Context().instance()
        self.realtime_socket = self.context.socket(zmq.PUB)
        self.realtime_socket.bind(self.options.realtime_address)

        # By default, the size of this queue is 'infinite'
        self.q = queue.Queue()

        # Keep track of the timestamp of the last file, so we only convert 1st slice
        # of simultaneous multi-slice operations
        self.last_file_time = None

        # The logging object
        self.log = logger

    def start_threads(self):
        """
        Start the threads of the realtime server
        """
        threads = [threading.Thread(target=self.get_temp_file_from_datawrite),
                   threading.Thread(target=self.handle_remote_connection)]

        self.log.debug("Starting threads")

        for thread in threads:
            thread.daemon = True
            thread.start()

        self.log.debug("Threads started")

        for thread in threads:
            thread.join()

    def read_and_convert_file_to_fitacf(self, filename):
        """
        Reads and converts a borealis temp rawacf site file, returning fitacf data.
        Updates self.last_file_time.
        It expects a filename with format: YYYYMMDD.HHMM.ss.uuuuuu.[rad].[slice_id].rawacf.hdf5
        :param filename: string, filename of the borealis temp rawacf site file to convert
        :return fitacf data in python dict, or None on error or if we don't need to convert
        """
        # Error check the filename format
        temp_rawacf_regex = '\d{8}\.\d{4}\.\d{2}\.\d{6}\.[a-z]{3}\.[0-9]\.rawacf\.hdf5'
        base_filename = os.path.basename(filename)
        match = re.fullmatch(temp_rawacf_regex, base_filename)
        if not match:
            # There is an antennas_iq temp file also sent every time, so this log msg is debug
            # instead of a warning
            self.log.debug("temp file did not match regex", filename=filename)
            os.remove(filename)
            return None

        # Read and convert data
        fields = base_filename.split(".")
        file_time = fields[0] + fields[1] + fields[2] + fields[3]

        # Make sure we only process the first slice for simultaneous multi-slice data for now
        if file_time == self.last_file_time:
            self.log.debug("not processing multi-slice data for slices > 0", filename=base_filename)
            os.remove(filename)
            return None

        self.last_file_time = file_time

        slice_num = int(fields[5])

        try:
            self.log.info("using pydarnio to convert", filename=filename)
            converted = pydarnio.BorealisConvert(filename, "rawacf", "/dev/null", slice_num, "site")
            os.remove(filename)
        except pydarnio.exceptions.borealis_exceptions.BorealisConvert2RawacfError as e:
            self.log.warn("error converting", filename=filename)
            os.remove(filename)
            return None

        return fitacf._fit(converted.sdarn_dict[0])

    def fitacf_data_to_queue(self, fitacf_data):
        """
        Makes a copy of the fitacf data, and converts to native python type.
        Adds converted fitacf data to queue.
        :param fitacf_data: fitacf data as output from backscatter
        """
        tmp = fitacf_data.copy()

        # Can't jsonify numpy, so we convert to native types for rt purposes.
        for k, v in fitacf_data.items():
            if hasattr(v, 'dtype'):
                if isinstance(v, np.ndarray):
                    tmp[k] = v.tolist()
                else:
                    tmp[k] = v.item()
        # q.put(item) by default blocks until a slot is available in the queue
        self.q.put(tmp)

    def get_temp_file_from_datawrite(self):
        """
        Runs infinite loop waiting for a file name received on socket from datawrite module.
        If the file is a rawacf file, then it converts the rawacf to fitacf data, and adds
        the data to the queue.
        """
        while True:
            filename = so.recv_data(self.data_write_to_realtime, self.options.dw_to_rt_identity,
                                    self.log)

            fitacf_data = self.read_and_convert_file_to_fitacf(filename)
            if fitacf_data:
                self.fitacf_data_to_queue(fitacf_data)

    def handle_remote_connection(self):
        """
        Compresses and serializes the data to send to the client.
        """
        while True:
            # q.get() default blocks until an item is available
            data_dict = self.q.get()
            serialized = json.dumps(data_dict)
            compressed = zlib.compress(serialized.encode())
            self.realtime_socket.send(compressed)


if __name__ == '__main__':
    from utils import log_config
    log = log_config.log()
    log.info(f"REALTIME BOOTED")
    try:
        rt = RealtimeServer(log)
        rt.start_threads()
        log.info(f"REALTIME EXITED")
    except Exception as main_exception:
        log.critical("REALTIME CRASHED", error=main_exception)
        log.exception("REALTIME CRASHED", exception=main_exception)
