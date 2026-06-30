"""
realtime_sim.py
~~~~~~~~~~~~~~~

Simulator for testing realtime.py. This script serves mock data to ``realtime.realtime_server()``
and verifies that fitted data is received over a corresponding socket.
"""

from pathlib import Path
import pickle
import sys
import threading
import time

import pydarnio
import zmq

sys.path.append(str(Path(__file__).resolve().parents[3]))
from src.utils import log_config, socket_operations as so
from src.realtime import realtime_server


def realtime_sim(ctx: zmq.Context):
    """Wrapper around realtime_server() that allows us to configure the sockets"""
    rawacf_recv_socket = ctx.socket(zmq.PAIR)
    rawacf_recv_socket.connect("inproc://rt_simulator")

    data_server = ctx.socket(zmq.PUB)
    data_server.bind("inproc://data_server")
    data_server.setsockopt(zmq.LINGER, 0)

    realtime_server(rawacf_recv_socket, data_server)  # Runs indefinitely


if __name__ == "__main__":
    log = log_config.log(
        console=True, console_log_level="DEBUG", logfile=False, aggregator=False
    )

    context = zmq.Context().instance()

    # This socket is for sending data to the simulator
    realtime_sock = context.socket(zmq.PAIR)
    realtime_sock.bind("inproc://rt_simulator")

    # This socket is for getting the data back from realtime_sim()
    sink = context.socket(zmq.SUB)
    sink.connect("inproc://data_server")
    sink.setsockopt(zmq.SUBSCRIBE, b"")  # Receive all messages
    sink.setsockopt(zmq.RCVTIMEO, 2000)  # timeout after 2000 ms

    log.info("Starting simulator thread...")
    thread = threading.Thread(target=realtime_sim, args=(context,), daemon=True)
    thread.start()

    # Load in a record of data
    data_dir = str(Path(__file__).resolve().parent)
    with open(data_dir + "/antennas_iq-0.pkl", "rb") as f:
        antennas_iq_data = pickle.load(f)
    with open(data_dir + "/bfiq-0.pkl", "rb") as f:
        bfiq_data = pickle.load(f)
    with open(data_dir + "/rawacf-0.pkl", "rb") as f:
        rawacf_data = pickle.load(f)

    log.info("Subscribing to all messages")
    for i in range(
        3
    ):  # Change this loop if you want to simulate sending multiple data packets
        log.info("Sending rawacf data")
        so.send_pyobj(realtime_sock, "sim", rawacf_data, header="rawacf")

        fitacf_recvd = sink.recv_multipart(copy=True)
        try:
            fitacf_data = pydarnio.read_fitacf(fitacf_recvd[1], mode="strict")
        except Exception as e:
            log.error("Could not interpret as fitacf", error=e)
            raise
        log.info("fitacf data received")

        rawacf_recvd = sink.recv_multipart(copy=True)
        log.info("rawacf data received")

        log.info("Sending bfiq data")
        so.send_pyobj(realtime_sock, "sim", bfiq_data, header="bfiq")
        bfiq_recvd = sink.recv_multipart(copy=True)
        log.info("bfiq data received")

        log.info("Sending antennas_iq data")
        so.send_pyobj(realtime_sock, "sim", antennas_iq_data, header="antennas_iq")
        antiq_recvd = sink.recv_multipart(copy=True)
        log.info("antennas_iq data received")

        time.sleep(1)

    sink.setsockopt(zmq.UNSUBSCRIBE, b"")
    sink.setsockopt(zmq.SUBSCRIBE, b"antennas_iq")  # only receive antennas_iq packets
    log.info("Subscribing to antennas_iq")
    time.sleep(1)

    for i in range(
        3
    ):  # Change this loop if you want to simulate sending multiple data packets
        log.info("Sending rawacf data")
        so.send_pyobj(realtime_sock, "sim", rawacf_data, header="rawacf")

        try:
            packet_recvd = sink.recv_multipart()
        except zmq.ZMQError as err:
            if err.errno != zmq.EAGAIN:
                log.warning(
                    "Received unexpected error when subscribed to antennas_iq",
                    error=err,
                )
        else:
            log.warning("Received unexpected packet when subscribed to antennas_iq")

        log.info("Sending bfiq data")
        so.send_pyobj(realtime_sock, "sim", bfiq_data, header="bfiq")
        try:
            packet_recvd = sink.recv_multipart()
        except zmq.ZMQError as err:
            if err.errno != zmq.EAGAIN:
                log.warning(
                    "Received unexpected error when subscribed to antennas_iq",
                    error=err,
                )
        else:
            log.warning("Received unexpected packet when subscribed to antennas_iq")

        log.info("Sending antennas_iq data")
        so.send_pyobj(realtime_sock, "sim", antennas_iq_data, header="antennas_iq")
        try:
            antiq_recvd = sink.recv_multipart(copy=True)
        except zmq.ZMQError as err:
            log.warning("Unexpected error when subscribed to antennas_iq", error=err)
            raise
        log.info("antennas_iq data received")

        time.sleep(1)

    # Rebuild the socket to reset the subscription filter
    sink.setsockopt(zmq.UNSUBSCRIBE, b"antennas_iq")
    sink.setsockopt(zmq.SUBSCRIBE, b"fitacf")  # Receive all messages
    log.info("Subscribing to fitacf")
    time.sleep(1)

    for i in range(
        3
    ):  # Change this loop if you want to simulate sending multiple data packets
        log.info("Sending bfiq data")
        so.send_pyobj(realtime_sock, "sim", bfiq_data, header="bfiq")
        try:
            packet_recvd = sink.recv_multipart()
        except zmq.ZMQError as err:
            if err.errno != zmq.EAGAIN:
                log.warning(
                    "Received unexpected error when subscribed to fitacf", error=err
                )
        else:
            log.warning("Received unexpected packet when subscribed to fitacf")

        log.info("Sending antennas_iq data")
        so.send_pyobj(realtime_sock, "sim", antennas_iq_data, header="antennas_iq")
        try:
            packet_recvd = sink.recv_multipart()
        except zmq.ZMQError as err:
            if err.errno != zmq.EAGAIN:
                log.warning(
                    "Received unexpected error when subscribed to fitacf", error=err
                )
        else:
            log.warning("Received unexpected packet when subscribed to fitacf")

        log.info("Sending rawacf data")
        so.send_pyobj(realtime_sock, "sim", rawacf_data, header="rawacf")
        try:
            fitacf_recvd = sink.recv_multipart(copy=True)
        except zmq.ZMQError as err:
            log.warning("Unexpected error when subscribed to fitacf", error=err)
            raise
        fitacf_data = pydarnio.read_fitacf(fitacf_recvd[1], mode="strict")
        log.info("fitacf data received")

        # Assert that a rawacf packet is not also received (only subscribed to fitacf)
        try:
            packet_recvd = sink.recv_multipart()
        except zmq.ZMQError as err:
            if err.errno != zmq.EAGAIN:
                log.warning(
                    "Received unexpected error when subscribed to fitacf", error=err
                )
        else:
            log.warning("Received unexpected packet when subscribed to fitacf")

        time.sleep(1)

    realtime_sock.close()
    sink.close()
    context.term()  # This will kill the thread
