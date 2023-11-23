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
import json
import zlib
import pydarnio
import numpy as np
from backscatter import fitacf

from utils.options import Options
from utils import socket_operations as so


def main():
    options = Options()

    borealis_sockets = so.create_sockets([options.rt_to_dw_identity], options.router_address)
    data_write_to_realtime = borealis_sockets[0]

    context = zmq.Context().instance()
    realtime_socket = context.socket(zmq.PUB)
    realtime_socket.bind(options.realtime_address)

    q = queue.Queue()

    def get_temp_file_from_datawrite():
        while True:
            rawacf_pickled = so.recv_bytes(data_write_to_realtime, options.dw_to_rt_identity, log)
            rawacf_data = pickle.loads(rawacf_pickled)

            # TODO: Make sure we only process the first slice for simultaneous multi-slice data for now
            try:
                record = sorted(list(rawacf_data.keys()))[0]
                log.info("using pydarnio to convert", record=record)
                converted = pydarnio.BorealisConvert.\
                    _BorealisConvert__convert_rawacf_record(0, (record, rawacf_data[record]), "")
            except pydarnio.borealis_exceptions.BorealisConvert2RawacfError:
                log.info("error converting")
                continue

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

                q.put(tmp)

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
