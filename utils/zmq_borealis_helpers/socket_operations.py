#!/usr/bin/env python3

# Copyright 2017 SuperDARN Canada
#
# socket_operations.py
# 2018-03-26
# socket operations utility file, recv, send, setup, etc...

import sys
import os
import zmq
import time
from datetime import datetime, timedelta

sys.path.append(os.environ["BOREALISPATH"])


def create_sockets(identities, router_addr):
    """Creates a DEALER socket for each identity in the list argument. Each socket is then connected
    to the router

    :param identities: Unique identities to give to sockets.
    :type identities: List
    :param router_addr: Address of the router socket
    :type router_addr: string
    :returns: Newly created and connected sockets.
    :rtype: List
    """
    context = zmq.Context().instance()
    num_sockets = len(identities)
    sockets = [context.socket(zmq.DEALER) for _ in range(num_sockets)]
    for sk, iden in zip(sockets, identities):
        sk.setsockopt_string(zmq.IDENTITY, iden)
        sk.connect(router_addr)

    return sockets


def recv_data(socket, sender_iden, pprint):
    """Receives data from a socket and verifies it comes from the correct sender.

    :param socket: Socket to recv from.
    :type socket: Zmq socket
    :param sender_iden: Identity of the expected sender.
    :type sender_iden: String
    :param pprint: A function to pretty print the message
    :type pprint: function
    :returns: Received data
    :rtype: String or Protobuf or None
    """
    recv_identity, empty, data = socket.recv_multipart()
    if recv_identity != sender_iden.encode('utf-8'):
        err_msg = "Expected identity {}, received from identity {}."
        err_msg = err_msg.format(sender_iden, recv_identity)
        pprint(err_msg)
        return None
    else:
        return data.decode('utf-8')


def send_data(socket, recv_iden, msg):
    """Sends data to another identity.

    :param socket: Socket to send from.
    :type socket: Zmq socket.
    :param recv_iden: The identity to send to.
    :type recv_iden: String
    :param msg: The data message to send.
    :type msg: String
    """
    frames = [recv_iden.encode('utf-8'), b"", msg.encode('utf-8')]
    socket.send_multipart(frames)

# Aliases for sending to a socket
send_reply = send_request = send_data

# Aliases for receiving from a socket 
recv_reply = recv_request = recv_data

def recv_bytes(socket, sender_iden, pprint):
    """Receives data from a socket and verifies it comes from the correct sender.

    :param socket: Socket to recv from.
    :type socket: Zmq socket
    :param sender_iden: Identity of the expected sender.
    :type sender_iden: String
    :param pprint: A function to pretty print the message
    :type pprint: function
    :returns: Received data
    :rtype: String or Protobuf or None
    """
    recv_identity, empty, bytes_object = socket.recv_multipart()
    if recv_identity != sender_iden.encode('utf-8'):
        err_msg = "Expected identity {}, received from identity {}."
        err_msg = err_msg.format(sender_iden, recv_identity)
        pprint(err_msg)
        return None
    else:
        return bytes_object


def send_bytes(socket, recv_iden, bytes_object):
    """Sends experiment to another identity.

    :param socket: Socket to send from.
    :type socket: Zmq socket.
    :param recv_iden: The identity to send to.
    :type recv_iden: String
    :param pickled_exp: The experiment to send.
    :type msg: bytes object or object encoded using highest pickle protocol available.
    """
    frames = [recv_iden.encode('utf-8'), b"", bytes_object]
    socket.send_multipart(frames)

send_pulse = send_obj = send_exp = send_bytes

recv_pulse = recv_obj = recv_exp = recv_bytes


