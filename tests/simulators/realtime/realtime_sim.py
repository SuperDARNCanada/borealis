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
import time
import zlib

import zmq

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.utils import log_config, socket_operations as so
from src.realtime import realtime_server


def realtime_sim(ctx: zmq.Context):
    """Wrapper around realtime_server() that allows us to configure the sockets"""
    rawacf_recv_socket = ctx.socket(zmq.PAIR)
    rawacf_recv_socket.connect("inproc://rt_simulator")

    fitacf_server = ctx.socket(zmq.PUB)
    fitacf_server.bind("inproc://fitacf_server")
    fitacf_server.setsockopt(zmq.LINGER, 0)

    realtime_server(rawacf_recv_socket, fitacf_server)  # Runs indefinitely


if __name__ == "__main__":
    log = log_config.log(
        console=True, console_log_level="DEBUG", logfile=False, aggregator=False
    )

    context = zmq.Context().instance()

    # This socket is for sending rawacf data to the simulator
    rawacf_send_socket = context.socket(zmq.PAIR)
    rawacf_send_socket.bind("inproc://rt_simulator")

    # This socket is for getting the fitacf data back from realtime_sim()
    fitacf_sink = context.socket(zmq.SUB)
    fitacf_sink.connect("inproc://fitacf_server")
    fitacf_sink.setsockopt(zmq.SUBSCRIBE, b"")  # Receive all messages

    log.info("Starting simulator thread...")
    thread = threading.Thread(target=realtime_sim, args=(context,), daemon=True)
    thread.start()

    # Load in a record of data
    infile = open(str(Path(__file__).resolve().parent) + "/rawacf_record.pkl", "rb")
    rawacf_data = pickle.load(infile)

    for i in range(
        5
    ):  # Change this loop if you want to simulate sending multiple data packets
        # Send rawacf data to the realtime_sim thread
        log.info("Sending rawacf data")
        so.send_bytes(rawacf_send_socket, "sim", pickle.dumps(rawacf_data))

        # Get the fitacf data back from realtime_sim
        recvd_data = fitacf_sink.recv()
        fitacf_data = json.loads(zlib.decompress(recvd_data).decode("utf-8"))

        # Log the data to the console
        log.info("fitacf data received")
        time.sleep(1)

    rawacf_send_socket.close()
    fitacf_sink.close()
    context.term()  # This will kill the thread
