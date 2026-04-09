#!/usr/bin/python3

"""
realtime package
~~~~~~~~~~~~~~~~
Sends data to realtime applications.

:copyright: 2019 SuperDARN Canada
"""

import bz2
import datetime as dt
import inspect
from pathlib import Path
import pickle

from procdarn import fitacf3_recs
import pydarnio
import structlog
import zmq


def fit_record(rawacf_records):
    """Fits a list of DMAP-formatted rawacf records using backscatter, returning the results"""
    first_rec = rawacf_records[0]
    timestamp = dt.datetime(
        first_rec["time.yr"],
        first_rec["time.mo"],
        first_rec["time.dy"],
        first_rec["time.hr"],
        first_rec["time.mt"],
        first_rec["time.sc"],
        first_rec["time.us"],
    )
    log.info("fitting record", timestamp=timestamp)

    fitted_records = fitacf3_recs(rawacf_records)

    return fitted_records


def realtime_server(recv_socket, server_socket):
    """Receives data from a socket, converts to fitacf, then serves over another socket.

    :param   recv_socket: Socket to receive data over. Must be an appropriate zmq socket type for receiving.
    :type    recv_socket: zmq.Socket
    :param server_socket: Socket to serve fitted data over. Must be an appropriate zmq socket type for sending.
    :type  server_socket: zmq.Socket
    """
    while True:
        try:
            rawacf_pickled = so.recv_bytes_from_any_iden(
                recv_socket
            )  # This is blocking
        except (zmq.ContextTerminated, zmq.ZMQError):  # No way to recover from this
            recv_socket.close()
            server_socket.close()
            return

        rawacf_data = pickle.loads(rawacf_pickled)
        # this will be a dict keyed by slice ID, values are dicts of (timestamp, [DMAP data dicts])

        slice_ids = sorted(list(rawacf_data.keys()))
        try:
            fitted_recs = fit_record(rawacf_data[slice_ids[0]])
            # Only fit the first slice of any that are SEQUENCE or CONCURRENT interfaced.
        except Exception as err:
            log.critical("error processing record", exception=err)
            continue

        data_to_send = pydarnio.write_fitacf(fitted_recs)
        publishable_data = bz2.compress(data_to_send)
        try:
            # Serve the data over the websocket. This is non-blocking in a background thread that zmq handles
            server_socket.send(publishable_data)
        except (zmq.ContextTerminated, zmq.ZMQError):  # No way to recover from this
            recv_socket.close()
            server_socket.close()
            return


if __name__ == "__main__":
    from utils import log_config, socket_operations as so
    from utils.options import Options

    log = log_config.log()
    log.info("REALTIME BOOTED")

    options = Options()
    context = zmq.Context().instance()

    # Socket for receiving data from data_write
    data_write_socket = so.create_sockets(
        options.router_address, options.rt_to_dw_identity
    )

    # Socket for serving data over the web
    publish_socket = context.socket(zmq.PUB)
    publish_socket.setsockopt(
        zmq.LINGER, 0
    )  # milliseconds to wait for message to send when closing socket

    try:
        publish_socket.bind(options.realtime_address)
    except (
        zmq.ZMQError
    ) as e:  # Raised if the address is invalid (e.g. if device doesn't exist)
        log.critical("REALTIME CRASHED", error=e)
        log.exception("REALTIME CRASHED", exception=e)
        data_write_socket.close()
        publish_socket.close()

    try:
        realtime_server(data_write_socket, publish_socket)
        log.info("REALTIME EXITED")
    except KeyboardInterrupt:
        log.critical("REALTIME INTERRUPTED")
    except Exception as main_exception:
        log.critical("REALTIME CRASHED", error=main_exception)
        log.exception("REALTIME CRASHED", exception=main_exception)

else:
    from .utils import socket_operations as so
    from .utils.options import Options

    caller = Path(inspect.stack()[-1].filename)
    module_name = caller.name.split(".")[0]
    log = structlog.getLogger(module_name)
