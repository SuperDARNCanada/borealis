#!/usr/bin/env python3

"""
    socket_operations.py
    ~~~~~~~~~~~~~~~~~~~~
    Socket operations utility file, recv, send, setup, etc...

    :copyright: 2017 SuperDARN Canada

    :todo: log.debug all functions
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


def recv_data(socket, sender_identity, log):
    """
    Receives data from a socket and verifies it comes from the correct sender.

    :param      socket:             Socket to recv from.
    :type       socket:             ZMQ socket
    :param      sender_identity:    Identity of the expected sender.
    :type       sender_identity:    str
    :param      log:                A logging object.
    :type       log:                Class

    :returns:   Received data
    :rtype:     String or Protobuf or None
    """

    receiver_identity, _, data = socket.recv_multipart()
    if receiver_identity != sender_identity.encode("utf-8"):
        log.error(
            "sender_identity != receiver_identity",
            sender_identity=sender_identity,
            receiver_identity=receiver_identity,
        )
        return None
    else:
        return data.decode("utf-8")


def send_data(socket, receiver_identity, msg):
    """
    Sends data to another identity.

    :param  socket:             Socket to send from.
    :type   socket:             ZMQ socket.
    :param  receiver_identity:  The identity to send to.
    :type   receiver_identity:  str
    :param  msg:                The data message to send.
    :type   msg:                str
    """
    frames = [receiver_identity.encode("utf-8"), b"", msg.encode("utf-8")]
    socket.send_multipart(frames)


# Aliases for sending to a socket
send_reply = send_request = send_data

# Aliases for receiving from a socket
recv_reply = recv_request = recv_data


def recv_bytes(socket, sender_identity, log):
    """
    Receives data from a socket and verifies it comes from the correct sender.

    :param      socket:         Socket to recv from.
    :type       socket:         ZMQ socket
    :param      sender_identity:    Identity of the expected sender.
    :type       sender_identity:    str
    :param      log:            A logging object.
    :type       log:            Class

    :returns:   Received data
    :rtype:     String or Protobuf or None
    """
    receiver_identity, _, bytes_object = socket.recv_multipart()
    if receiver_identity != sender_identity.encode("utf-8"):
        log.error(
            "sender_identity != receiver_identity",
            sender_identity=sender_identity,
            receiver_identity=receiver_identity,
        )
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


def send_bytes(socket, receiver_identity, bytes_object):
    """Sends experiment to another identity.

    :param  socket:         Socket to send from.
    :type   socket:         ZMQ socket.
    :param  receiver_identity:      The identity to send to.
    :type   receiver_identity:      str
    :param  bytes_object:   The bytes to send, or object encoded using highest pickle protocol
                            available.
    :type   bytes_object:   bytes
    """
    frames = [receiver_identity.encode("utf-8"), b"", bytes_object]
    socket.send_multipart(frames)


send_pulse = send_obj = send_exp = send_bytes

recv_pulse = recv_obj = recv_exp = recv_bytes
