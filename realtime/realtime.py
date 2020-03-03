#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
#
# realtime.py

# Sends data to realtime applications

import zmq
import threading
import os
import sys
import queue
import json
import zlib
import pydarn
import numpy as np
from backscatter import fitacf

borealis_path = os.environ['BOREALISPATH']
if not borealis_path:
    raise ValueError("BOREALISPATH env variable not set")

sys.path.append(borealis_path + '/utils/')
import realtime_options.realtime_options as rto
from zmq_borealis_helpers import socket_operations as so
import shared_macros.shared_macros as sm

rt_print = sm.MODULE_PRINT("Realtime", "green")

def _main():
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
            filename = so.recv_data(data_write_to_realtime, opts.dw_to_rt_identity, rt_print)

            if "rawacf" in filename:
                #Read and convert data
                fields = filename.split(".")
                file_time = fields[0] + fields[1] + fields[2] + fields[3]


                # Make sure we only process the first slice for simulatenous multislice data for now
                if file_time == last_file_time:
                    continue

                last_file_time = file_time

                slice_num = int(fields[5])
                try:
                    converted = pydarn.BorealisConvert(filename, "rawacf", "/dev/null", slice_num,
                                                    "site")
                except:
                    rt_print("Error converting {}".format(filename))
                    os.remove(filename)
                    continue

                data = converted.sdarn_dict

                fit_data = fitacf._fit(data[0])
                tmp = fit_data.copy()

                # Can't jsonify numpy so we convert to native types for rt purposes.
                for k,v in fit_data.items():
                    if hasattr(v, 'dtype'):
                        if isinstance(v, np.ndarray):
                            tmp[k] = v.tolist()
                        else:
                            tmp[k] = v.item()

                q.put(tmp)

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
    _main()

