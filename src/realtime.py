#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
#
# realtime.py

# Sends data to realtime applications

import zmq
import threading
import os
import queue
import json
import zlib
import re
import numpy as np
import pydarnio
from backscatter import fitacf

import utils.options.realtime_options as rto
import utils.socket_operations as so
import utils.shared_macros as sm

rt_print = sm.MODULE_PRINT("Realtime", "green")


class RealtimeServer:
    """
    Realtime server with two threads. One of which expects RAWACF temp files from the
    datawrite module, which it then converts to FITACF format and enqueues.
    The second thread dequeues the converted fitacf data and sends to any listening clients.

    Usage:
    rt = RealtimeServer()
    rt.start_threads()

    """
    def __init__(self):
        self.opts = rto.RealtimeOptions()

        self.borealis_sockets = so.create_sockets([self.opts.rt_to_dw_identity],
                                                  self.opts.router_address)
        self.data_write_to_realtime = self.borealis_sockets[0]

        self.context = zmq.Context().instance()
        self.realtime_socket = self.context.socket(zmq.PUB)
        self.realtime_socket.bind(self.opts.rt_address)

        # By default, the size of this queue is 'infinite'
        self.q = queue.Queue()

        # Keep track of the timestamp of the last file, so we only convert 1st slice
        # of simultaneous multi-slice operations
        self.last_file_time = None

    def start_threads(self):
        """
        Start the threads of the realtime server
        """
        threads = [threading.Thread(target=self.get_temp_file_from_datawrite),
                   threading.Thread(target=self.handle_remote_connection)]

        for thread in threads:
            thread.daemon = True
            thread.start()

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
        temp_rawacf_regex = '\d{8}\.\d{4}\.\d{2}\.\d{6}\.[a-z]{3}\.[a-z]\.rawacf\.hdf5'
        match = re.fullmatch(temp_rawacf_regex, filename)
        if not match:
            os.remove(filename)
            return None

        # Read and convert data
        fields = filename.split(".")
        file_time = fields[0] + fields[1] + fields[2] + fields[3]

        # Make sure we only process the first slice for simultaneous multislice data for now
        if file_time == self.last_file_time:
            os.remove(filename)
            return None

        self.last_file_time = file_time

        slice_num = int(fields[5])
        try:
            rt_print("Using pyDARNio to convert {}".format(filename))
            converted = pydarnio.BorealisConvert(filename, "rawacf", "/dev/null",
                                                 slice_num, "site")
            os.remove(filename)
        except pydarnio.exceptions.borealis_exceptions.BorealisConvert2RawacfError as e:
            rt_print("Error converting {}".format(filename))
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
            filename = so.recv_data(self.data_write_to_realtime, self.opts.dw_to_rt_identity,
                                    rt_print)

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
    rt = RealtimeServer()
    rt.start_threads()
