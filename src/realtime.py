#!/usr/bin/python3

"""
realtime package
~~~~~~~~~~~~~~~~
Sends data to realtime applications.

:copyright: 2019 SuperDARN Canada
"""

import argparse
import bz2
import datetime as dt
import inspect
from pathlib import Path
import pickle
import sys

from backscatter import fitacf
import numpy as np
import pydarnio
import structlog
import zmq

sys.path.append(str(Path(__file__).resolve().parent))
from utils.file_formats import SliceData


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

    fitted_records = []
    for rec in rawacf_records:
        fit_data = fitacf._fit(rec)
        fit_data["pwr0"] = np.array(
            fit_data["pwr0"], dtype=np.float32
        )  # backscatter returns float64, need float32
        fitted_records.append(fit_data.copy())

    return fitted_records


def realtime_server(
    recv_socket,
    server_socket,
    stream_antennas_iq: bool = True,
    stream_bfiq: bool = True,
    stream_rawacf: bool = True,
):
    """Receives data from a socket, dispatches to the appropriate handler, and serves over another socket.

    :param         recv_socket: Socket to receive data over. Must be an appropriate zmq socket type for receiving.
    :type          recv_socket: zmq.Socket
    :param       server_socket: Socket to serve data over. Must be appropriate zmq socket type for sending.
    :type        server_socket: zmq.Socket
    :param  stream_antennas_iq: Flag to enable streaming antennas_iq data
    :type   stream_antennas_iq: bool
    :param         stream_bfiq: Flag to enable streaming bfiq data
    :type          stream_bfiq: bool
    :param       stream_rawacf: Flag to enable streaming rawacf data
    :type        stream_rawacf: bool
    """
    while True:
        try:
            msg_header, bytes_packet = so.recv_bytes_from_any_iden(
                recv_socket, with_header=True
            )  # This is blocking
        except (zmq.ContextTerminated, zmq.ZMQError):  # No way to recover from this
            recv_socket.close()
            server_socket.close()
            return
        slice_data = pickle.loads(bytes_packet)
        if type(slice_data).__name__ != "SliceData":
            log.error(
                "Expected SliceData message",
                received_type=type(slice_data),
                size=len(slice_data),
            )
            raise ValueError(
                f"Message data has type {type(slice_data).__name__}, expected `SliceData`"
            )

        if msg_header == b"rawacf":
            log.info("Received rawacf slice")
            serve_fitacf(slice_data, server_socket)
            if stream_rawacf:
                serve_slice(slice_data, server_socket, "rawacf")
        elif msg_header == b"bfiq" and stream_bfiq:
            log.info("Received bfiq slice")
            serve_slice(slice_data, server_socket, "bfiq")
        elif msg_header == b"antennas_iq" and stream_antennas_iq:
            log.info("Received antennas_iq slice")
            serve_slice(slice_data, server_socket, "antennas_iq")
        else:
            log.critical(
                "Error receiving slice data",
                header=msg_header,
                slice_data=slice_data,
            )
            raise ValueError(f"Received data with unexpected header {msg_header}")


def serve_fitacf(rawacf_data, server_socket):
    """Converts to fitacf, then serves over a zmq socket.

    :param   rawacf_data: Rawacf data in SliceData object form.
    :type    rawacf_data: SliceData
    :param server_socket: Socket to serve fitted data over. Must be an appropriate zmq socket type for sending.
    :type  server_socket: zmq.Socket
    """

    slice_id = rawacf_data.slice_id
    dmap_data = rawacf_data.to_dmap()
    try:
        fitted_recs = fit_record(dmap_data)
    except Exception as err:
        log.critical("error processing record", exception=err)
        return

    data_to_send = pydarnio.write_fitacf(fitted_recs)
    publishable_data = bz2.compress(data_to_send)

    msg_topic = f"fitacf/{slice_id}".lower()
    send_bytes(publishable_data, server_socket, msg_topic)


def serve_slice(slice_data: SliceData, server_socket: zmq.Socket, file_type: str):
    """Serves SliceData over a zmq Publish socket.

    :param    slice_data: Data for the slice
    :type     slice_data: SliceData
    :param server_socket: Socket to serve fitted data over. Must be an appropriate zmq socket type for sending.
    :type  server_socket: zmq.Socket
    :param     file_type: Type of data [antennas_iq, bfiq, rawacf] stored in the SliceData object
    :type      file_type: str
    """

    # Use a message topic of FILE_TYPE/SLICE_ID to facilitate message filtering on the receiver end
    msg_topic = f"{file_type}/{slice_data.slice_id}".lower()
    data_bytes = pickle.dumps(slice_data)
    send_bytes(data_bytes, server_socket, msg_topic)


def send_bytes(data: bytes, sock: zmq.Socket, topic: str):
    try:
        # Serve the data over the socket. This is non-blocking in a background thread that zmq handles
        sock.send_multipart((topic.encode("utf-8"), data))
    except (zmq.ContextTerminated, zmq.ZMQError):  # No way to recover from this
        sock.close()
        raise


if __name__ == "__main__":
    from utils import log_config, socket_operations as so
    from utils.options import Options

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--disable-antennas-iq",
        help="Disable antennas_iq data streaming",
        action="store_false",
    )
    parser.add_argument(
        "--disable-bfiq",
        help="Disable bfiq data streaming",
        action="store_false",
    )
    parser.add_argument(
        "--disable-rawacf",
        help="Disable rawacf streaming",
        action="store_false",
    )
    args = parser.parse_args()

    log = log_config.log()
    log.info("REALTIME BOOTED")

    options = Options()
    context = zmq.Context().instance()

    # Socket for receiving data from data_write
    data_write_socket = so.create_sockets(
        options.router_address,
        options.rt_to_dw_identity,
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
        realtime_server(
            data_write_socket,
            publish_socket,
            args.disable_antennas_iq,
            args.disable_bfiq,
            args.disable_rawacf,
        )
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
