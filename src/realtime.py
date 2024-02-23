#!/usr/bin/python3

"""
    realtime package
    ~~~~~~~~~~~~~~~~
    Sends data to realtime applications.

    :copyright: 2019 SuperDARN Canada
"""

import zmq
import threading
import pickle
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

    def fitacf_data_to_queue(self, rawacf_data):
        """
        Makes a copy of the fitacf data, and converts to native python type.
        Adds converted fitacf data to queue.
        :param fitacf_data: fitacf data as output from backscatter
        """
        # TODO: Make sure we only process the first slice for simultaneous multi-slice data for now
        try:
            record = sorted(list(rawacf_data.keys()))[0]
            log.info("using pydarnio to convert", record=record)
            converted = pydarnio.BorealisConvert. \
                _BorealisConvert__convert_rawacf_record(0, (record, rawacf_data[record]), "")
        except pydarnio.borealis_exceptions.BorealisConvert2RawacfError:
            log.info("error converting")
            return None

        for rec in converted:
            fit_data = fitacf._fit(rec)
            tmp = fit_data.copy()

            # Can't jsonify numpy so we convert to native types for realtime purposes.
            for k, v in fit_data.items():
                if hasattr(v, 'dtype'):
                    if isinstance(v, np.ndarray):
                        tmp[k] = v.tolist()
                    else:
                        tmp[k] = v.item()

            self.q.put(tmp)

    def get_temp_file_from_datawrite(self):
        """
        Runs infinite loop waiting for a file name received on socket from datawrite module.
        If the file is a rawacf file, then it converts the rawacf to fitacf data, and adds
        the data to the queue.
        """
        while True:
            rawacf_pickled = so.recv_bytes(self.data_write_to_realtime, self.options.dw_to_rt_identity,
                                     self.log)

            rawacf_data = pickle.loads(rawacf_pickled)
            if rawacf_data:
                self.fitacf_data_to_queue(rawacf_data)

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
