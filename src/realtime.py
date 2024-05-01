#!/usr/bin/python3

"""
    realtime package
    ~~~~~~~~~~~~~~~~
    Sends data to realtime applications.

    :copyright: 2019 SuperDARN Canada
"""

import json
import pickle
import zlib

from backscatter import fitacf
import numpy as np
import pydarnio
import zmq

from utils.options import Options
from utils import socket_operations as so


def convert_and_fit_record(rawacf_record):
    """Converts a rawacf record to DMAP format and fits using backscatter, returning the results"""
    # We only grab the first slice. This means the first slice of any that are SEQUENCE or CONCURRENT interfaced.
    first_key = sorted(list(rawacf_record.keys()))[0]
    log.info("converting record", record=first_key)
    converted = (pydarnio.BorealisConvert.
                 _BorealisConvert__convert_rawacf_record(0, (first_key, rawacf_record[first_key]), ""))

    fitted_records = []
    for rec in converted:
        fit_data = fitacf._fit(rec)
        fitted_records.append(fit_data.copy())

    return fitted_records


def realtime_server():
    """Receives data from data_write, converts to fitacf, then serves over a web socket."""
    options = Options()
    context = zmq.Context().instance()

    # Socket for receiving data from data_write
    data_write_socket = so.create_sockets([options.rt_to_dw_identity], options.router_address)[0]

    # Socket for serving data over the web
    publish_socket = context.socket(zmq.PUB)
    publish_socket.bind(options.realtime_address)
    publish_socket.setsockopt(zmq.LINGER, 500)  # milliseconds to wait for message to send when closing socket

    try:
        while True:
            rawacf_pickled = so.recv_bytes(data_write_socket, options.dw_to_rt_identity, log)  # This is blocking
            rawacf_data = pickle.loads(rawacf_pickled)

            try:
                fitted_recs = convert_and_fit_record(rawacf_data)
            except Exception as e:
                log.critical("error processing record", exception=e)
                continue

            for rec in fitted_recs:
                # Can't jsonify numpy, so we convert to native types for serving over the web
                for k, v in rec.items():
                    if hasattr(v, 'dtype'):
                        if isinstance(v, np.ndarray):
                            rec[k] = v.tolist()
                        else:
                            rec[k] = v.item()

                publishable_data = zlib.compress(json.dumps(rec).encode('utf-8'))
                publish_socket.send(publishable_data)  # Serve the data over the websocket. This is non-blocking in a background thread

    except KeyboardInterrupt:
        log.critical('Interrupt received')
    finally:
        # Clean up the sockets
        data_write_socket.close()
        publish_socket.close()
        context.term()


if __name__ == '__main__':
    from utils import log_config

    log = log_config.log()
    log.info(f"REALTIME BOOTED")
    try:
        realtime_server()
        log.info(f"REALTIME EXITED")
    except Exception as main_exception:
        log.critical("REALTIME CRASHED", error=main_exception)
        log.exception("REALTIME CRASHED", exception=main_exception)
