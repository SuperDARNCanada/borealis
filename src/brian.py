#!/usr/bin/python

"""
brian process
~~~~~~~~~~~~~
This program communicates with all processes to administrate the Borealis software

:copyright: 2017 SuperDARN Canada
"""

import os
import sys
import time
import threading
import argparse
import zmq
import pickle
from utils import socket_operations as so
from utils.options import Options
from utils.message_formats import SequenceMetadataMessage

sys.path.append(os.environ["BOREALISPATH"])
from utils.message_formats import RxSamplesMetadata

TIME_PROFILE = True


def router(options, realtime_off):
    """
    The router is responsible for moving traffic between modules by routing traffic using named sockets.

    :param  options: Options parsed from config file
    :type   options: Options class
    :param realtime_off: Flag indicating if realtime is disabled
    :type  realtime_off: bool
    """

    context = zmq.Context().instance()
    router = context.socket(zmq.ROUTER)
    router.setsockopt(zmq.ROUTER_MANDATORY, 1)
    router.bind(options.router_address)

    log.info("booting router")
    frames_to_send = []
    while True:
        events = router.poll(timeout=1)

        if events:
            dd = router.recv_multipart()
            sender, receiver, empty, data = dd
            log.debug(
                "router input: sender->receiver", sender=sender, receiver=receiver
            )
            frames_received = [receiver, sender, empty, data]
            frames_to_send.append(frames_received)
            log.debug(
                "router output: receiver->sender", sender=sender, receiver=receiver
            )

        non_sent = []
        retry_logs = []
        for frames in frames_to_send:
            try:
                router.send_multipart(frames)
            except zmq.ZMQError as e:
                sender = frames[1]
                receiver = frames[0]

                log_dict = {"sender": sender, "receiver": receiver, "error": str(e)}

                # Check if message was intended for realtime, and drop the message if so
                if sender.decode("utf-8") == options.dw_to_rt_identity:
                    if realtime_off:
                        log.debug("dropping message", **log_dict)
                    else:
                        log.warning("dropping message", **log_dict)

                # Otherwise, try to resend the message
                else:
                    retry_logs.append(log_dict)
                    non_sent.append(frames)
        if len(non_sent) > 0:
            log.debug("Retrying to send frames", frames=retry_logs)

        frames_to_send = non_sent


def sequence_timing(options):
    """
    Thread function for sequence timing

    This function controls the flow of data between brian's sequence timing and other parts of the
    radar system. This function serves to check whether the sequence timing is working as expected
    and to rate control the system to make sure the processing can handle data rates.

    :param  context: zmq context, if None, then this method will get one
    :type   context: zmq context, optional
    """

    ids = [
        options.brian_to_radctrl_identity,
        options.brian_to_driver_identity,
        options.brian_to_dspbegin_identity,
        options.brian_to_dspend_identity,
    ]

    sockets_list = so.create_sockets(options.router_address, *ids)

    brian_to_radar_control = sockets_list[0]
    brian_to_driver = sockets_list[1]
    brian_to_dsp_begin = sockets_list[2]
    brian_to_dsp_end = sockets_list[3]

    sequence_poller = zmq.Poller()
    sequence_poller.register(brian_to_radar_control, zmq.POLLIN)
    sequence_poller.register(brian_to_dsp_begin, zmq.POLLIN)
    sequence_poller.register(brian_to_dsp_end, zmq.POLLIN)
    sequence_poller.register(brian_to_driver, zmq.POLLIN)

    context = zmq.Context().instance()

    start_new_sock = context.socket(zmq.PAIR)
    start_new_sock.bind("inproc://rate_limiter")

    def rate_limiter():
        """
        This function serves to rate control the system. If processing is faster than the
        sequence time then the speed of the driver is the limiting factor. If processing takes
        longer than sequence time, then the dsp unit limits the speed of the system.

        To ensure that the system does not overallocate resources, some checks are conducted.
        We keep track of the state of the worker (the driver module) and resource (ringbuffer)
        that are critical. When the ringbuffer is emptied, the driver module can start filling it.
        """

        rate_limiter_socket = context.socket(zmq.PAIR)
        rate_limiter_socket.connect("inproc://rate_limiter")

        driver_busy = True
        ringbuffer_filled = False
        time_now = time.perf_counter()

        while True:
            if not driver_busy and not ringbuffer_filled:
                # Acknowledge new sequence can begin to Radar Control by requesting new sequence metadata
                log.debug("requesting metadata from radar_control")
                so.send_string(
                    brian_to_radar_control,
                    options.radctrl_to_brian_identity,
                    "Requesting metadata",
                )
                driver_busy = True

            message = rate_limiter_socket.recv_string()

            if TIME_PROFILE:
                time_mark = time.perf_counter() - time_now
                time_now = time.perf_counter()
                log.verbose(message, time=time_mark)

            if message == "driver collected sequence, ready for another":
                ringbuffer_filled = True
                driver_busy = False
            if message == "ringbuffer free":
                ringbuffer_filled = False

    thread = threading.Thread(target=rate_limiter)
    thread.daemon = True
    thread.start()

    time.sleep(1)  # todo: investigate the need for this sleep

    last_processing_time = 0

    first_time = True
    late_counter = 0
    while True:
        if first_time:
            # Request new sequence metadata
            log.debug("requesting metadata from radar control")
            so.send_string(
                brian_to_radar_control,
                options.radctrl_to_brian_identity,
                "Requesting metadata",
            )
            first_time = False

        socks = dict(sequence_poller.poll())

        if brian_to_driver in socks and socks[brian_to_driver] == zmq.POLLIN:
            # Receive metadata of completed sequence from driver such as timing
            reply = so.recv_bytes(
                brian_to_driver, options.driver_to_brian_identity, log
            )
            meta = RxSamplesMetadata.parse(reply.decode("utf-8"))

            log.debug(
                "driver sent",
                sequence_time=meta.sequence_time * 1e3,
                sequence_time_unit="ms",
                sequence_num=meta.sequence_num,
            )

            # Requesting acknowledgement of work begins from DSP
            log.debug("requesting work begins from dsp")
            iden = options.dspbegin_to_brian_identity + str(meta.sequence_num)
            so.send_string(brian_to_dsp_begin, iden, "Requesting work begins")

            start_new_sock.send_string("driver collected sequence, ready for another")

        if (
            brian_to_radar_control in socks
            and socks[brian_to_radar_control] == zmq.POLLIN
        ):
            # Get new sequence metadata from radar control
            sigp = so.recv_pyobj(
                brian_to_radar_control,
                options.radctrl_to_brian_identity,
                log,
                expected_type=SequenceMetadataMessage,
            )

            log.debug(
                "radar control sent",
                sequence_time=sigp.sequence_time,
                sequence_time_unit="ms",
                sequence_num=sigp.sequence_num,
            )

            # Request acknowledgement of sequence from driver
            log.debug("requesting ack from driver")
            so.send_string(
                brian_to_driver, options.driver_to_brian_identity, "Requesting ack"
            )

        if brian_to_dsp_begin in socks and socks[brian_to_dsp_begin] == zmq.POLLIN:
            # Get acknowledgement that work began in processing.
            reply = so.recv_bytes_from_any_iden(brian_to_dsp_begin)
            sig_p = pickle.loads(reply)
            log.debug("dsp began", sequence_num=sig_p["sequence_num"])

            # Requesting acknowledgement of work ends from DSP
            log.debug("requesting work end from dsp")
            iden = options.dspend_to_brian_identity + str(sig_p["sequence_num"])
            so.send_string(brian_to_dsp_end, iden, "Requesting work ends")

            # Acknowledge we want to start something new.
            start_new_sock.send_string("ringbuffer free")

        if brian_to_dsp_end in socks and socks[brian_to_dsp_end] == zmq.POLLIN:
            # Receive ack that work finished on previous sequence.
            reply = so.recv_bytes_from_any_iden(brian_to_dsp_end)
            sig_p = pickle.loads(reply)

            if sig_p["sequence_num"] != 0:
                if sig_p["kerneltime"] > last_processing_time:
                    late_counter += 1
                else:
                    late_counter = 0
            last_processing_time = sig_p["kerneltime"]

            log.debug(
                "brian to dsp",
                kernel_time=sig_p["kerneltime"],
                sequence_time_unit="ms",
                sequence_num=sig_p["sequence_num"],
                late_counter=late_counter,
            )


def main():
    parser = argparse.ArgumentParser()
    help_msg = "Run only the router. Do not run any of the other threads or functions."
    parser.add_argument("--router-only", action="store_true", help=help_msg)
    parser.add_argument(
        "--realtime-off", action="store_true", help="Flag if realtime is disabled"
    )
    args = parser.parse_args()

    options = Options()
    threads = [
        threading.Thread(
            target=router, args=(options,), kwargs={"realtime_off": args.realtime_off}
        )
    ]

    if not args.router_only:
        threads.append(threading.Thread(target=sequence_timing, args=(options,)))

    for thread in threads:
        thread.daemon = True
        thread.start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    from utils import log_config

    log = log_config.log()
    log.info("BRIAN BOOTED")
    try:
        main()
        log.info("BRIAN EXITED")
    except Exception as main_exception:
        log.critical("BRIAN CRASHED", error=main_exception)
        log.exception("BRIAN CRASHED", exception=main_exception)
