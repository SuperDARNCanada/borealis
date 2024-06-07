#!/usr/bin/env python3

"""
    socket_operations.py
    ~~~~~~~~~~~~~~~~~~~~
    Socket operations utility file, recv, send, setup, etc...

    :copyright: 2017 SuperDARN Canada

    :todo: log.debug all functions
"""

import zmq
import pickle


def create_sockets(router_addr, *identities):
    """
    Creates a DEALER socket for each identity in the list argument. Each socket is then connected
    to the router

    :param      router_addr:    Address of the router socket
    :type       router_addr:    str
    :param      identities:     Unique identities to give to sockets.
    :type       identities:     tuple

    :returns:   Newly created and connected sockets.
    :rtype:     list or str
    """
    context = zmq.Context().instance()
    num_sockets = len(identities)
    sockets = [context.socket(zmq.DEALER) for _ in range(num_sockets)]
    for sk, iden in zip(sockets, identities):
        sk.setsockopt_string(zmq.IDENTITY, iden)
        sk.connect(router_addr)

    if num_sockets == 1:
        return sockets[0]

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


def send_bytes(socket, receiver_identity, bytes_object, log=None):
    """Sends experiment to another identity.

    :param  socket:         Socket to send from.
    :type   socket:         ZMQ socket.
    :param  receiver_identity:      The identity to send to.
    :type   receiver_identity:      str
    :param  bytes_object:   The bytes to send, or object encoded using highest pickle protocol
                            available.
    :type   bytes_object:   bytes
    """
    if log:
        log.debug(
            "Sending message",
            sender=socket.get(zmq.IDENTITY),
            receiver=receiver_identity,
        )
    frames = [receiver_identity.encode("utf-8"), b"", bytes_object]
    socket.send_multipart(frames)


def recv_pyobj(socket, sender_identity, log, expected_type=None):
    """Un-packs a pickled object received from recv_bytes. Can be used
    to check if the received object is of the expected type.

    Args:
        socket:
        expected_type:
        log:
    """
    bytes_packet = recv_bytes(socket, sender_identity, log)
    message = pickle.loads(bytes_packet)
    if expected_type:
        if not isinstance(message, expected_type):
            log.error(
                "received message != expected message",
                received_message=type(message),
                expected_message=expected_type,
            )
            return None
    return message


def send_pyobj(socket, receiver_identity, message, log=None):
    """Pickles the message and passes it to send bytes to be communicated
    over the router.

    Args:
        socket:
        receiver_identity:
        message:
        log:
    """
    bytes_packet = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    send_bytes(socket, receiver_identity, bytes_packet, log)
