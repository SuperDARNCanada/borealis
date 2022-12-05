#!/usr/bin/env python3

"""
    socket_operations.py
    ~~~~~~~~~~~~~~~~~~~~~~~~~~
    Socket operations utility file, recv, send, setup, etc...

    :copyright: 2017 SuperDARN Canada
"""

import zmq


def create_sockets(identities, router_addr):
    """
    Creates a DEALER socket for each identity in the list argument. Each socket is then connected
    to the router

    :param      identities:     Unique identities to give to sockets.
    :type       identities:     list
    :param      router_addr:    Address of the router socket
    :type       router_addr:    str

    :returns:   Newly created and connected sockets.
    :rtype:     list
    """
    context = zmq.Context().instance()
    num_sockets = len(identities)
    sockets = [context.socket(zmq.DEALER) for _ in range(num_sockets)]
    for sk, iden in zip(sockets, identities):
        sk.setsockopt_string(zmq.IDENTITY, iden)
        sk.connect(router_addr)

    return sockets


def recv_data(socket, sender_iden, pprint):
    """
    Receives data from a socket and verifies it comes from the correct sender.

    :param      socket:         Socket to recv from.
    :type       socket:         ZMQ socket
    :param      sender_iden:    Identity of the expected sender.
    :type       sender_iden:    str
    :param      pprint:         A function to pretty print the message
    :type       pprint:         function

    :returns:   Received data
    :rtype:     String or Protobuf or None
    """
    recv_identity, _, data = socket.recv_multipart()
    if recv_identity != sender_iden.encode('utf-8'):
        err_msg = f"Expected identity {sender_iden}, received from identity {recv_identity}."
        pprint(err_msg)
        return None
    else:
        return data.decode('utf-8')

def send_data(socket, recv_iden, msg):
    """
    Sends data to another identity.

    :param  socket:     Socket to send from.
    :type   socket:     ZMQ socket.
    :param  recv_iden:  The identity to send to.
    :type   recv_iden:  str
    :param  msg:        The data message to send.
    :type   msg:        str
    """
    frames = [recv_iden.encode('utf-8'), b"", msg.encode('utf-8')]
    socket.send_multipart(frames)

# Aliases for sending to a socket
send_reply = send_request = send_data

# Aliases for receiving from a socket
recv_reply = recv_request = recv_data

def recv_bytes(socket, sender_iden, pprint):
    """
    Receives data from a socket and verifies it comes from the correct sender.

    :param      socket:         Socket to recv from.
    :type       socket:         ZMQ socket
    :param      sender_iden:    Identity of the expected sender.
    :type       sender_iden:    str
    :param      pprint:         A function to pretty print the message
    :type       pprint:         function

    :returns:   Received data
    :rtype:     String or Protobuf or None
    """
    recv_identity, _, bytes_object = socket.recv_multipart()
    if recv_identity != sender_iden.encode('utf-8'):
        err_msg = f"Expected identity {sender_iden}, received from identity {recv_identity}."
        pprint(err_msg)
        return None
    else:
        return bytes_object

def recv_bytes_from_any_iden(socket):
    """
    Receives data from a socket, returns just the data and strips off the identity

    :param      socket: Socket to recv from.
    :type       socket: ZMQ socket

    :returns:   Received data
    :rtype:     String or Protobuf or None
    """

    _, _, bytes_object = socket.recv_multipart()
    return bytes_object


def send_bytes(socket, recv_iden, bytes_object):
    """Sends experiment to another identity.

    :param  socket:         Socket to send from.
    :type   socket:         ZMQ socket.
    :param  recv_iden:      The identity to send to.
    :type   recv_iden:      str
    :param  bytes_object:   The bytes to send, or object encoded using highest pickle protocol
                            available.
    :type   bytes_object:   bytes
    """
    frames = [recv_iden.encode('utf-8'), b"", bytes_object]
    socket.send_multipart(frames)

send_pulse = send_obj = send_exp = send_bytes

recv_pulse = recv_obj = recv_exp = recv_bytes
