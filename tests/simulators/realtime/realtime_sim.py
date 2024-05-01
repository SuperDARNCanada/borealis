"""
realtime_sim.py
~~~~~~~~~~~~~~~

Simulator for testing realtime.py. This script serves mock data to ``realtime.realtime_server()``
and verifies that fitted data is received over a corresponding socket.
"""

import json
from pathlib import Path
import pickle
import sys
import threading
import zlib

import zmq

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.realtime import realtime_server
from src.utils import socket_operations as so


def realtime_sim(ctx: zmq.Context):
    """Wrapper around realtime_server() that allows us to configure the sockets"""
    rawacf_recv_socket = ctx.socket(zmq.PAIR)
    rawacf_recv_socket.connect("inproc://rt_simulator")

    fitacf_server = ctx.socket(zmq.PAIR)
    fitacf_server.bind("inproc://fitacf_server")

    rawacf_recv_socket.send_string("READY")     # Tell the main thread that this is ready
    rawacf_recv_socket.recv()   # Block until the main thread is ready

    try:
        realtime_server(rawacf_recv_socket, fitacf_server)
    except KeyboardInterrupt:
        rawacf_recv_socket.close()
        fitacf_server.close()

if __name__ == '__main__':
    from src.utils import log_config
    log = log_config.log(console=True, console_log_level="DEBUG", logfile=False, aggregator=False)

    context = zmq.Context().instance()

    rawacf_send_socket = context.socket(zmq.PAIR)
    rawacf_send_socket.bind("inproc://rt_simulator")

    log.info("Starting simulator thread...")
    thread = threading.Thread(target=realtime_sim, args=(context,), daemon=True)
    thread.start()

    rawacf_send_socket.recv_string()    # Wait for READY from realtime_sim()
    log.info("Simulator thread ready")

    # This socket is for getting the fitacf data back from realtime_sim()
    fitacf_sink = context.socket(zmq.PAIR)
    fitacf_sink.connect("inproc://fitacf_server")

    # Tell realtime_sim() that we're ready
    rawacf_send_socket.send_string("READY")

    # Load in a record of data
    infile = open("rawacf_record.pkl", "rb")
    rawacf_data = pickle.load(infile)

    # Send rawacf data to the realtime_sim thread
    log.info("Sending rawacf data", **rawacf_data)
    so.send_bytes(rawacf_send_socket, "sim", pickle.dumps(rawacf_data))

    # Get the fitacf data back from realtime_sim
    recvd_data = fitacf_sink.recv()
    fitacf_data = json.loads(zlib.decompress(recvd_data).decode('utf-8'))

    # Log the data to the console
    log.info("fitacf data received", **fitacf_data)

    rawacf_send_socket.close()
    fitacf_sink.close()
    context.term()