"""
This script collects realtime data from Borealis radars and serves it over a specified address.
The data is served as a pickled `Borealis.SliceData` object.
"""

import argparse
import datetime as dt
import dmap
from pathlib import Path
import pickle
import sys
import zmq

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.utils.file_formats import SliceData
from src.utils.writers import DMAPWriter, HDF5Writer


def decode_msg(msg):
    """
    Takes a message of realtime data from a Borealis radar and converts it to a SliceData object and topic.
    """
    topic = msg[0].decode("utf-8")
    fmt = topic.split("/")[0]
    print(topic)

    if fmt == "fitacf":
        data = dmap.read_fitacf(msg[-1], mode="strict")
        return fmt, data

    data = pickle.loads(msg[-1])
    slice_data = SliceData()
    for k, v in data.items():
        setattr(slice_data, k, v)

    return fmt, slice_data


def connect_socket(ctx: zmq.Context, port: int, topic: str):
    """
    Connects a SUBSCRIBE socket to `port` on localhost
    """
    socket = ctx.socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, topic.encode("utf-8"))
    socket.setsockopt(zmq.RCVTIMEO, 100)  # time out after 100 ms
    socket.connect(f"tcp://127.0.0.1:{port}")
    return socket


def handler(in_ports: list[int], topic: str, write_to: str = ""):
    """
    This function sets up sockets to ingest Borealis realtime data.

    in_ports: list[int]
        List of ports to accept Borealis data from.
    topic: str
        Topic (prefix of message) to subscribe to. Only messages that start with `topic` are ingested.
    write_to: str
        Path to write the data to.
    """
    context = zmq.Context.instance()
    poller = zmq.Poller()

    # Create sockets to ingest the radar data
    sockets = {}
    for port in in_ports:
        sock = connect_socket(context, port, topic)
        poller.register(sock, zmq.POLLIN)
        sockets[port] = [sock, dt.datetime.now(dt.timezone.utc)]

    while True:
        # Refresh any sockets that are no longer working
        for port in sockets.keys():
            sock, last_used = sockets[port]
            if dt.datetime.now(dt.timezone.utc) - last_used > dt.timedelta(seconds=10):
                poller.unregister(sock)
                sock.close()
                new_sock = connect_socket(context, port, topic)
                poller.register(new_sock, zmq.POLLIN)
                sockets[port] = [new_sock, dt.datetime.now(dt.timezone.utc)]

        # Get new messages from sockets
        socks = dict(poller.poll(10000))
        for sock, status in socks.items():
            if status != zmq.POLLIN:
                continue
            try:
                msg = sock.recv_multipart()

                # Update the time that the socket was last received from
                port, last_used = [
                    (k, v[1]) for k, v in sockets.items() if v[0] == sock
                ][0]
                sockets[port] = (sock, dt.datetime.now(dt.timezone.utc))

                fmt, data = decode_msg(msg)
                if fmt == "fitacf":
                    station = data[0]["stid"]
                    writer = DMAPWriter
                else:
                    station = data.station
                    writer = HDF5Writer
                print(station, last_used.strftime("%Y-%m-%d %H:%M:%S.%f"), len(msg[-1]))

                # TODO: Change this if you would like to do something else with the data!
                if write_to is not None and write_to != "":
                    writer.write_record(write_to, data, fmt)

            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break  # interrupted
                else:
                    raise e
            except ValueError as e:
                print(
                    f"{dt.datetime.now(dt.timezone.utc)}: Error decoding message\n{e}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "in_ports", nargs="+", type=int, help="Ports to accept Borealis data from"
    )
    parser.add_argument("--topic", default="bfiq", help="Topic to subscribe to")
    parser.add_argument("--write_to", help="Path to write data to")
    args = parser.parse_args()

    handler(args.in_ports, args.topic, args.write_to)
