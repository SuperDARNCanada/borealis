"""
socket_operations.py
~~~~~~~~~~~~~~~~~~~~

Functions for creating and using sockets for inter-process communication (IPC). The code below shows a minimal example.

In one process::

    import socket_operations as so
    from options import Options
    options = Options()
    sock = so.create_sockets(
        options.router_address,
        "BOREALIS_USER"
    )
    so.send_string(sock, "OTHER_BOREALIS_USER", "Good choice")

In another process::

    import socket_operations as so
    from options import Options
    options = Options()
    sock = so.create_sockets(
        options.router_address,
        "OTHER_BOREALIS_USER"
    )
    msg = so.recv_string(sock, "BOREALIS_USER")
    assert msg == "Good choice"

:todo: log.debug all functions
"""

from typing import Any, Optional, Union, Tuple
import zmq
import pickle


def create_sockets(
    router_addr: str, *identities: str
) -> Union[zmq.Socket, list[zmq.Socket]]:
    """
    Creates a ``DEALER`` socket for each identity in the list argument. Each socket is then connected
    to the router.

    :param      router_addr:    Address of the router socket
    :type       router_addr:    str
    :param      identities:     Unique identities to give to sockets
    :type       identities:     str

    :returns:   Newly created and connected sockets.
    :rtype:     Union[zmq.Socket, list[zmq.Socket]]
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


def recv_string(
    socket: zmq.Socket,
    sender_identity: str,
    log: Optional[Any] = None,
    with_header: bool = False,
) -> Union[str, Tuple[bytes, str]]:
    """
    Receives data from a socket and verifies it comes from the correct sender.

    :param      socket:             Socket to recv from
    :type       socket:             zmq.Socket
    :param      sender_identity:    Identity of the expected sender
    :type       sender_identity:    str
    :param      log:                A logging object
    :type       log:                Optional[Any]
    :param      with_header:        Flag to return message header
    :type       with_header:        bool

    :returns:   Received data, possibly with a header
    :rtype:     Union[str, Tuple[bytes, str]]
    """

    receiver_identity, header, data = socket.recv_multipart()
    if receiver_identity != sender_identity.encode("utf-8"):
        if log is not None:
            log.error(
                "sender_identity != receiver_identity",
                sender_identity=sender_identity,
                receiver_identity=receiver_identity,
            )
        raise ValueError(
            f"Message did not come from expected source. Origin: {receiver_identity}\tExpected: {sender_identity}"
        )
    data = data.decode("utf-8")

    if with_header:
        return header, data
    else:
        return data


def send_string(
    socket: zmq.Socket, receiver_identity: str, msg: str, header: Optional[str] = None
):
    """
    Sends data to another identity.

    :param  socket:             Socket to send from
    :type   socket:             zmq.Socket
    :param  receiver_identity:  The identity to send to
    :type   receiver_identity:  str
    :param  msg:                The data message to send
    :type   msg:                str
    :param  header:             An optional header for the message
    :type   header:             str
    """
    if header is None:
        header = b""
    else:
        header = header.encode("utf-8")
    frames = [receiver_identity.encode("utf-8"), header, msg.encode("utf-8")]
    socket.send_multipart(frames)


def recv_bytes(
    socket: zmq.Socket,
    sender_identity: str,
    log: Optional[Any] = None,
    with_header: bool = False,
) -> Optional[Union[bytes, Tuple[bytes, bytes]]]:
    """
    Receives data from a socket and verifies it comes from the correct sender.

    :param      socket:             Socket to recv from
    :type       socket:             zmq.Socket
    :param      sender_identity:    Identity of the expected sender
    :type       sender_identity:    str
    :param      log:                A logging object
    :type       log:                Optional[Any]
    :param      with_header:        Flag to return message header
    :type       with_header:        bool

    :returns:   Received data
    :rtype:     Optional[bytes]
    """
    receiver_identity, header, bytes_object = socket.recv_multipart()
    if receiver_identity != sender_identity.encode("utf-8"):
        log.error(
            "sender_identity != receiver_identity",
            sender_identity=sender_identity,
            receiver_identity=receiver_identity,
        )
        return None
    else:
        if with_header:
            return header, bytes_object
        else:
            return bytes_object


def recv_bytes_from_any_iden(
    socket: zmq.Socket, with_header: bool = False
) -> Union[bytes, Tuple[bytes, bytes]]:
    """
    Receives data from a socket, returns just the data and strips off the identity

    :param      socket: Socket to recv from
    :type       socket: zmq.Socket
    :param with_header: Flag to return message header
    :type  with_header: bool

    :returns:   Received data
    :rtype:     Optional[bytes]
    """

    _, header, bytes_object = socket.recv_multipart()
    if with_header:
        return header, bytes_object
    else:
        return bytes_object


def send_bytes(
    socket: zmq.Socket,
    receiver_identity: str,
    bytes_object: bytes,
    log: Optional[Any] = None,
    header: Optional[str] = None,
):
    """Sends experiment to another identity.

    :param  socket:             Socket to send from
    :type   socket:             zmq.Socket
    :param  receiver_identity:  The identity to send to
    :type   receiver_identity:  str
    :param  bytes_object:       The bytes to send
    :type   bytes_object:       bytes
    :param  log:                A logging object
    :type   log:                Optional[Any]
    :param  header:             An optional header for the message
    :type   header:             str
    """
    if log:
        log.debug(
            "Sending message",
            sender=socket.get(zmq.IDENTITY),
            receiver=receiver_identity,
        )
    if header is None:
        header = b""
    else:
        header = header.encode("utf-8")
    frames = [receiver_identity.encode("utf-8"), header, bytes_object]
    socket.send_multipart(frames)


def recv_pyobj(
    socket: zmq.Socket,
    sender_identity: str,
    log: Optional[Any] = None,
    expected_type: Optional[Any] = None,
    with_header: bool = False,
) -> Any:
    """
    Receive a pickled Python object.

    Can be used to check if the received object is of ``expected_type``.

    :param  socket:             Socket to receive from
    :type   socket:             zmq.Socket
    :param  sender_identity:    The identity of the sender
    :type   sender_identity:    str
    :param  log:                A logging object
    :type   log:                Optional[Any]
    :param  expected_type:      The data type expected when receiving
    :type   expected_type:      Optional[Any]
    :param    with_header:      Flag to return message header
    :type     with_header:      bool

    :returns: an object
    :rtype: Any
    """
    # TODO: Account for multiple expected types
    header, bytes_packet = recv_bytes(socket, sender_identity, log, with_header=True)
    message = pickle.loads(bytes_packet)
    if expected_type is not None and not isinstance(message, expected_type):
        if log is not None:
            log.error(
                "received message != expected message",
                received_message=type(message),
                expected_message=expected_type,
            )
        raise ValueError(
            f"Message data has type {type(message)}, expected {expected_type}"
        )
    if with_header:
        return header, message
    else:
        return message


def send_pyobj(
    socket: zmq.Socket,
    receiver_identity: str,
    message: Any,
    log: Optional[Any] = None,
    header: Optional[str] = None,
):
    """Pickles the message and passes it to send bytes to be communicated
    over the router.

    :param  socket:             Socket to send from
    :type   socket:             zmq.Socket
    :param  receiver_identity:  The identity of the receiver
    :type   receiver_identity:  str
    :param  message:            The object to send
    :type   message:            Any
    :param  log:                A logging object
    :type   log:                Optional[Any]
    :param  header:             Message header
    :type   header:             Optional[str]
    """
    bytes_packet = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
    send_bytes(socket, receiver_identity, bytes_packet, log, header=header)
