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
import json
import zlib
import pydarnio
import numpy as np
from backscatter import fitacf

import utils.options.realtime_options as rto
from utils import socket_operations as so


def main():
    opts = rto.RealtimeOptions()

    borealis_sockets = so.create_sockets([opts.rt_to_dw_identity], opts.router_address)
    data_write_to_realtime = borealis_sockets[0]

    context = zmq.Context().instance()
    realtime_socket = context.socket(zmq.PUB)
    realtime_socket.bind(opts.rt_address)

    q = queue.Queue()

    def get_temp_file_from_datawrite():
        last_file_time = None
        while True:
            filename = so.recv_data(data_write_to_realtime, opts.dw_to_rt_identity, log)

            if "rawacf" in filename:
                # Read and convert data
                fields = filename.split(".")
                file_time = fields[0] + fields[1] + fields[2] + fields[3]

                # Make sure we only process the first slice for simultaneous multi-slice data for now
                if file_time == last_file_time:
                    os.remove(filename)
                    continue

                last_file_time = file_time

                slice_num = int(fields[5])
                try:
                    log.info("using pydarnio to convert", filename=filename)
                    converted = pydarnio.BorealisConvert(filename, "rawacf", "/dev/null", slice_num, "site")
                    os.remove(filename)
                except pydarnio.exceptions.borealis_exceptions.BorealisConvert2RawacfError as e:
                    log.info("error converting", filename=filename)
                    os.remove(filename)
                    continue

                data = converted.sdarn_dict

                fit_data = fitacf._fit(data[0])
                tmp = fit_data.copy()

                # Can't jsonify numpy so we convert to native types for realtime purposes.
                for k, v in fit_data.items():
                    if hasattr(v, 'dtype'):
                        if isinstance(v, np.ndarray):
                            tmp[k] = v.tolist()
                        else:
                            tmp[k] = v.item()

                q.put(tmp)
            else:
                os.remove(filename)

    def handle_remote_connection():
        """
        Compresses and serializes the data to send to the server.
        """

        while True:
            data_dict = q.get()
            serialized = json.dumps(data_dict)
            compressed = zlib.compress(serialized.encode('utf-8'))
            realtime_socket.send(compressed)

    threads = [threading.Thread(target=get_temp_file_from_datawrite),
               threading.Thread(target=handle_remote_connection)]

    for thread in threads:
        thread.daemon = True
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    from utils import log_config
    log = log_config.log()
    log.info(f"REALTIME BOOTED")
    try:
        main()
        log.info(f"REALTIME EXITED")
    except Exception as main_exception:
        log.critical("REALTIME CRASHED", error=main_exception)
        log.exception("REALTIME CRASHED", exception=main_exception)
