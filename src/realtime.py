#!/usr/bin/python3

"""
    realtime package
    ~~~~~~~~~~~~~~~~
    Sends data to realtime applications.

    :copyright: 2019 SuperDARN Canada
"""

import inspect
import json
from pathlib import Path
import pickle
import zlib

from backscatter import fitacf
import numpy as np
import pydarnio
import structlog
import zmq


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


def realtime_server(recv_socket, server_socket):
    """Receives data from a socket, converts to fitacf, then serves over another socket.

    :param   recv_socket: Socket to receive data over. Must be an appropriate zmq socket type for receiving.
    :type    recv_socket: zmq.Socket
    :param server_socket: Socket to serve fitted data over. Must be an appropriate zmq socket type for sending.
    :type  server_socket: zmq.Socket
    """
    try:
        while True:
            rawacf_pickled = so.recv_bytes_from_any_iden(recv_socket)  # This is blocking
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
                server_socket.send(publishable_data)  # Serve the data over the websocket. This is non-blocking in a background thread
    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.critical("Aborting", error=e)
    finally:
        recv_socket.close()
        server_socket.close()


if __name__ == '__main__':
    from utils import log_config, socket_operations as so
    from utils.options import Options

    log = log_config.log()
    log.info(f"REALTIME BOOTED")

    options = Options()
    context = zmq.Context().instance()

    # Socket for receiving data from data_write
    data_write_socket = so.create_sockets([options.rt_to_dw_identity], options.router_address)[0]

    # Socket for serving data over the web
    publish_socket = context.socket(zmq.PUB)
    publish_socket.bind(options.realtime_address)
    publish_socket.setsockopt(zmq.LINGER, 500)  # milliseconds to wait for message to send when closing socket

    try:
        realtime_server(data_write_socket, publish_socket)
        log.info(f"REALTIME EXITED")
    except KeyboardInterrupt:
        log.critical("REALTIME INTERRUPTED")
    except Exception as main_exception:
        log.critical("REALTIME CRASHED", error=main_exception)
        log.exception("REALTIME CRASHED", exception=main_exception)
    finally:
        # Clean up the sockets
        context.term()

else:
    from .utils import socket_operations as so
    from .utils.options import Options

    caller = Path(inspect.stack()[-1].filename)
    module_name = caller.name.split('.')[0]
    log = structlog.getLogger(module_name)
