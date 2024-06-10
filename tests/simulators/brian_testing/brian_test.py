#!/usr/bin/python

# Copyright 2017 SuperDARN Canada
#
# brian.py
# 2018-01-30
# Communicate with all processes to administrate the borealis software

import sys
import os
import zmq
import time

sys.path.append(os.environ["BOREALISPATH"])

if (
    __debug__
):  # TODO need to get build flavour from scons environment, 'release' may be 'debug'
    sys.path.append(os.environ["BOREALISPATH"] + "/build/debug/borealis/utils/protobuf")
else:
    sys.path.append(
        os.environ["BOREALISPATH"] + "/build/release/borealis/utils/protobuf"
    )

import sigprocpacket_pb2
import rxsamplesmetadata_pb2
import processeddata_pb2

# TODO: Socket options to look at: IDENTITY, AFFINITY, LINGER
# TODO: USE send_multipart, with human-readable identities, then the router knows how to handle the
# response to the request, see chapter 3, figures 29/30

RADCTRL_EXPHAN_IDEN = b"RADCTRL_EXPHAN_IDEN"
RADCTRL_DSP_IDEN = b"RADCTRL_DSP_IDEN"
RADCTRL_DRIVER_IDEN = b"RADCTRL_DRIVER_IDEN"
RADCTRL_BRIAN_IDEN = b"RADCTRL_BRIAN_IDEN"

DRIVER_RADCTRL_IDEN = b"DRIVER_RADCTRL_IDEN"
DRIVER_DSP_IDEN = b"DRIVER_DSP_IDEN"
DRIVER_BRIAN_IDEN = b"DRIVER_BRIAN_IDEN"

EXPHAN_RADCTRL_IDEN = b"EXPHAN_RADCTRL_IDEN"
EXPHAN_DSP_IDEN = b"EXPHAN_DSP_IDEN"

DSP_RADCTRL_IDEN = b"DSP_RADCTRL_IDEN"
DSP_DRIVER_IDEN = b"DSP_DRIVER_IDEN"
DSP_EXPHAN_IDEN = b"DSP_EXPHAN_IDEN"
DSP_DW_IDEN = b"DSP_DW_IDEN"
DSPBEGIN_BRIAN_IDEN = b"DSPBEGIN_BRIAN_IDEN"
DSPEND_BRIAN_IDEN = b"DSPEND_BRIAN_IDEN"

DW_DSP_IDEN = b"DW_DSP_IDEN"

BRIAN_RADCTRL_IDEN = b"BRIAN_RADCTRL_IDEN"
BRIAN_DRIVER_IDEN = b"BRIAN_DRIVER_IDEN"
BRIAN_DSPBEGIN_IDEN = b"BRIAN_DSPBEGIN_IDEN"
BRIAN_DSPEND_IDEN = b"BRIAN_DSPEND_IDEN"

ROUTER_ADDRESS = "tcp://127.0.0.1:7878"

TIME = 0.087  # 0.069


def create_sockets(identities, router_addr=ROUTER_ADDRESS):
    """Creates a DEALER socket for each identity in the list argument. Each socket is then connected
    to the router

    :param identities: Unique identities to give to sockets.
    :type identities: List
    :param router_addr: Address of the router socket, defaults to ROUTER_ADDRESS
    :type router_addr: string, optional
    :returns: Newly created and connected sockets.
    :rtype: List
    """

    context = zmq.Context().instance()
    num_sockets = len(identities)
    sockets = [context.socket(zmq.DEALER) for _ in range(num_sockets)]
    for sk, iden in zip(sockets, identities):
        sk.setsockopt(zmq.IDENTITY, iden)
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
    if recv_identity != sender_iden:
        err_msg = "Expected identity {}, received from identity {}."
        err_msg = err_msg.format(sender_iden, recv_identity)
        pprint(err_msg)
        return None
    else:
        return data


recv_reply = recv_request = recv_pulse = recv_data


def send_data(socket, recv_iden, msg):
    """Sends data to another identity.

    :param socket: Socket to send from.
    :type socket: Zmq socket.
    :param recv_iden: The identity to send to.
    :type recv_iden: String
    :param msg: The data message to send.
    :type msg: String
    """
    frames = [recv_iden, b"", b"{}".format(msg)]
    socket.send_multipart(frames)


send_reply = send_request = send_pulse = send_data


def radar_control(context=None):
    """Thread function for radar control

    This function simulates the flow of data between radar control and other parts of the radar
    system.
    :param context: zmq context, if None, then this method will get one
    :type context: zmq context, optional
    """

    ids = [
        RADCTRL_EXPHAN_IDEN,
        RADCTRL_DSP_IDEN,
        RADCTRL_DRIVER_IDEN,
        RADCTRL_BRIAN_IDEN,
    ]
    sockets_list = create_sockets(ids)

    radar_control_to_exp_handler = sockets_list[0]
    radar_control_to_dsp = sockets_list[1]
    radar_control_to_driver = sockets_list[2]
    radar_control_to_brian = sockets_list[3]

    def printing(msg):
        RADAR_CONTROL = "\033[33m" + "RADAR_CONTROL: " + "\033[0m"
        sys.stdout.write(RADAR_CONTROL + msg + "\n")

    time.sleep(1)
    count = 0
    time_counter = 0
    time_inc = 0.0
    while True:

        # radar_control sends a request for an experiment to experiment_handler
        printing("Requesting experiment")
        send_request(
            radar_control_to_exp_handler, EXPHAN_RADCTRL_IDEN, "Requesting Experiment"
        )

        # radar_control receives new experiment
        reply = recv_reply(radar_control_to_exp_handler, EXPHAN_RADCTRL_IDEN, printing)
        reply_output = "Experiment handler sent -> {}".format(reply)
        printing(reply_output)

        start = time.time()
        sigp = sigprocpacket_pb2.SigProcPacket()
        sigp.sequence_time = TIME
        sigp.sequence_num = count
        count += 1

        # Brian requests sequence metadata for timeouts
        request = recv_request(radar_control_to_brian, BRIAN_RADCTRL_IDEN, printing)
        request_output = "Brian requested -> {}".format(request)
        printing(request_output)

        send_reply(radar_control_to_brian, BRIAN_RADCTRL_IDEN, sigp.SerializeToString())

        middle = time.time()
        printing("brian time {}".format(middle - start))
        # Radar control receives request for metadata from DSP

        send_reply(radar_control_to_dsp, DSP_RADCTRL_IDEN, sigp.SerializeToString())

        middle2 = time.time()
        printing("dsp time {}".format(middle2 - middle))

        # sending pulses to driver
        printing("Sending pulses")
        num_pulses = range(8)
        for i in num_pulses:
            if i == num_pulses[0]:
                pulse = "sob_pulse"
            elif i == num_pulses[-1]:
                pulse = "eob_pulse"
            else:
                pulse = str(i)

            send_pulse(radar_control_to_driver, DRIVER_RADCTRL_IDEN, pulse)

        end = time.time()

        printing("Time {}".format(end - start))

        time_counter += 1
        time_inc += end - start
        print("time_inc {}".format(time_inc))
        if time_inc > 3.0:
            printing("Number of averages: {}".format(time_counter))
            time_counter = 0
            time_inc = 0.0


def experiment_handler(context=None):
    """Thread function for experiment handler

    This function simulates the flow of data between experiment handler and other parts of the radar
    system.
    :param context: zmq context, if None, then this method will get one
    :type context: zmq context, optional
    """

    ids = [EXPHAN_RADCTRL_IDEN, EXPHAN_DSP_IDEN]
    sockets_list = create_sockets(ids)

    exp_handler_to_radar_control = sockets_list[0]
    exp_handler_to_dsp = sockets_list[1]

    def printing(msg):
        EXPERIMENT_HANDLER = "\033[34m" + "EXPERIMENT HANDLER: " + "\033[0m"
        sys.stdout.write(EXPERIMENT_HANDLER + msg + "\n")

    def update_experiment():
        # Recv complete processed data from DSP
        send_request(exp_handler_to_dsp, DSP_EXPHAN_IDEN, "Need completed data")

        data = recv_data(exp_handler_to_dsp, DSP_EXPHAN_IDEN, printing)
        data_output = "Dsp sent -> {}".format(data)
        printing(data_output)

    thread = threading.Thread(target=update_experiment)
    thread.daemon = True
    thread.start()

    time.sleep(1)
    while True:
        # experiment_handler replies with an experiment to radar_control
        request = recv_request(
            exp_handler_to_radar_control, RADCTRL_EXPHAN_IDEN, printing
        )
        request_msg = "Radar control made request -> {}.".format(request)
        printing(request_msg)

        # sending experiment back to radar control
        printing("Sending experiment")
        send_reply(
            exp_handler_to_radar_control, RADCTRL_EXPHAN_IDEN, "Giving experiment"
        )


def driver(context=None):
    """Thread function for driver

    This function simulates the flow of data between the driver and other parts of the radar
    system.
    :param context: zmq context, if None, then this method will get one
    :type context: zmq context, optional
    """

    ids = [DRIVER_DSP_IDEN, DRIVER_RADCTRL_IDEN, DRIVER_BRIAN_IDEN]
    sockets_list = create_sockets(ids)

    driver_to_dsp = sockets_list[0]
    driver_to_radar_control = sockets_list[1]
    driver_to_brian = sockets_list[2]

    def printing(msg):
        DRIVER = "\033[34m" + "DRIVER: " + "\033[0m"
        sys.stdout.write(DRIVER + msg + "\n")

    time.sleep(1)
    sq = 0
    while True:

        # getting pulses from radar control
        while True:
            pulse = recv_pulse(driver_to_radar_control, RADCTRL_DRIVER_IDEN, printing)
            printing("Received pulse {}".format(pulse))
            if pulse == "eob_pulse":
                break

        start = time.time()
        time.sleep(TIME)
        end = time.time()

        samps_meta = rxsamplesmetadata_pb2.RxSamplesMetadata()
        samps_meta.sequence_time = end - start
        samps_meta.sequence_num = sq
        sq += 1

        # sending sequence data to dsp
        request = recv_request(driver_to_dsp, DSP_DRIVER_IDEN, printing)
        request_output = "Dsp sent -> {}".format(request)
        printing(request_output)

        send_reply(driver_to_dsp, DSP_DRIVER_IDEN, samps_meta.SerializeToString())

        # sending collected data to brian
        request = recv_request(driver_to_brian, BRIAN_DRIVER_IDEN, printing)
        request_output = "Brian sent -> {}".format(request)
        printing(request_output)

        send_reply(driver_to_brian, BRIAN_DRIVER_IDEN, samps_meta.SerializeToString())


def dsp(context=None):
    """Thread function for dsp

    This function simulates the flow of data between dsp and other parts of the radar
    system.
    :param context: zmq context, if None, then this method will get one
    :type context: zmq context, optional
    """

    ids = [
        DSP_RADCTRL_IDEN,
        DSP_DRIVER_IDEN,
        DSP_EXPHAN_IDEN,
        DSP_DW_IDEN,
        DSPBEGIN_BRIAN_IDEN,
        DSPEND_BRIAN_IDEN,
    ]
    sockets_list = create_sockets(ids)

    dsp_to_radar_control = sockets_list[0]
    dsp_to_driver = sockets_list[1]
    dsp_to_experiment_handler = sockets_list[2]
    dsp_to_data_write = sockets_list[3]
    dsp_to_brian_begin = sockets_list[4]
    dsp_to_brian_end = sockets_list[5]

    def printing(msg):
        DSP = "\033[35m" + "DSP: " + "\033[0m"
        sys.stdout.write(DSP + msg + "\n")

    time.sleep(1)
    first_time = True
    while True:

        printing("Requesting metadata from radar control")

        # # DSP makes a request for new sequence processing metadata to radar control
        # send_request(dsp_to_radar_control, RADCTRL_DSP_IDEN,"Need metadata")

        # Radar control sends back metadata
        reply = recv_reply(dsp_to_radar_control, RADCTRL_DSP_IDEN, printing)

        sigp = sigprocpacket_pb2.SigProcPacket()
        sigp.ParseFromString(reply)
        reply_output = "Radar control sent -> sequence {} time {}".format(
            sigp.sequence_num, sigp.sequence_time
        )
        printing(reply_output)

        # request data from driver
        send_request(dsp_to_driver, DRIVER_DSP_IDEN, "Need data to process")
        reply = recv_reply(dsp_to_driver, DRIVER_DSP_IDEN, printing)

        meta = rxsamplesmetadata_pb2.RxSamplesMetadata()
        meta.ParseFromString(reply)
        reply_output = "Driver sent -> time {}".format(meta.sequence_time)
        printing(reply_output)

        # Copy samples to device
        time.sleep(TIME * 0.50)

        # acknowledge start of work
        request = recv_request(dsp_to_brian_begin, BRIAN_DSPBEGIN_IDEN, printing)
        request_output = "Brian sent -> {}".format(request)
        printing(request_output)
        send_data(
            dsp_to_brian_begin,
            BRIAN_DSPBEGIN_IDEN,
            "Ack start of work, " "sqnum {}".format(sigp.sequence_num),
        )

        # doing work!
        def do_work(sqn_num):
            sequence_num = sqn_num

            start = time.time()
            time.sleep(TIME * 0.9)
            end = time.time()

            proc_data = processeddata_pb2.ProcessedData()
            proc_data.processing_time = end - start
            proc_data.sequence_num = sequence_num

            # acknowledge end of work
            request = recv_request(dsp_to_brian_end, BRIAN_DSPEND_IDEN, printing)
            request_output = "Brian sent -> {}".format(request)
            printing(request_output)
            send_data(
                dsp_to_brian_end, BRIAN_DSPEND_IDEN, proc_data.SerializeToString()
            )

            # send data to experiment handler
            request = recv_request(dsp_to_experiment_handler, EXPHAN_DSP_IDEN, printing)
            request_output = "Experiment handler sent -> {}".format(request)
            printing(request_output)

            send_data(dsp_to_experiment_handler, EXPHAN_DSP_IDEN, "All the datas")

            # send data to data write
            request = recv_request(dsp_to_data_write, DW_DSP_IDEN, printing)
            request_output = "Data write sent -> {}".format(request)
            printing(request_output)

            send_data(dsp_to_data_write, DW_DSP_IDEN, "All the datas")

        thread = threading.Thread(target=do_work, args=(sigp.sequence_num,))
        thread.daemon = True
        thread.start()


def data_write(context=None):
    """
    Thread for data_write sockets testing
    :param context: zmq context, if None, then this method will get one
    :return:
    """
    # Request socket to dsp (processed samples)

    ids = [DW_DSP_IDEN]
    sockets_list = create_sockets(ids)

    data_write_to_dsp = sockets_list[0]

    def printing(msg):
        DATA_WRITE = "\033[32m" + "DATA WRITE: " + "\033[0m"
        sys.stdout.write(DATA_WRITE + msg + "\n")

    time.sleep(1)
    while True:
        # Request processed data
        send_request(data_write_to_dsp, DSP_DW_IDEN, "Requesting processed data")

        data = recv_data(data_write_to_dsp, DSP_DW_IDEN, printing)
        data_output = "Dsp sent -> {}".format(data)
        printing(data_output)


late_counter = 0


def sequence_timing():
    """Thread function for sequence timing

    This function simulates the flow of data between brian's sequence timing and other parts of the
    radar system. This function serves to check whether the sequence timing is working as expected
    and to rate control the system to make sure the processing can handle data rates.
    :param context: zmq context, if None, then this method will get one
    :type context: zmq context, optional
    """

    ids = [
        BRIAN_RADCTRL_IDEN,
        BRIAN_DRIVER_IDEN,
        BRIAN_DSPBEGIN_IDEN,
        BRIAN_DSPEND_IDEN,
    ]
    sockets_list = create_sockets(ids)

    brian_to_radar_control = sockets_list[0]
    brian_to_driver = sockets_list[1]
    brian_to_dsp_begin = sockets_list[2]
    brian_to_dsp_end = sockets_list[3]

    sequence_poller = zmq.Poller()
    sequence_poller.register(brian_to_radar_control, zmq.POLLIN)
    sequence_poller.register(brian_to_dsp_begin, zmq.POLLIN)
    sequence_poller.register(brian_to_dsp_end, zmq.POLLIN)
    sequence_poller.register(brian_to_driver, zmq.POLLIN)

    def printing(msg):
        SEQUENCE_TIMING = "\033[31m" + "SEQUENCE TIMING: " + "\033[0m"
        sys.stdout.write(SEQUENCE_TIMING + msg + "\n")

    context = zmq.Context().instance()

    start_new_sock = context.socket(zmq.PAIR)
    start_new_sock.bind("inproc://start_new")

    def start_new():
        """This function serves to rate control the system. If processing is faster than the
        sequence time than the speed of the driver is the limiting factor. If processing takes
        longer than sequence time, than the dsp unit limits the speed of the system.
        """
        start_new = context.socket(zmq.PAIR)
        start_new.connect("inproc://start_new")

        want_to_start = False
        good_to_start = True
        extra_good_to_start = False
        dsp_finish_counter = 2
        while True:

            if want_to_start and good_to_start and dsp_finish_counter:
                # Acknowledge new sequence can begin to Radar Control by requesting new sequence
                # metadata
                printing("Requesting metadata from Radar control")
                send_request(
                    brian_to_radar_control, RADCTRL_BRIAN_IDEN, "Requesting metadata"
                )
                want_to_start = good_to_start = False
                dsp_finish_counter -= 1

            message = start_new.recv()
            if message == "want_to_start":
                want_to_start = True

            if message == "good_to_start":
                good_to_start = True

            if message == "extra_good_to_start":
                dsp_finish_counter = 1

            print(
                "WTS {}, GTS {}, EGTS {}".format(
                    want_to_start, good_to_start, extra_good_to_start
                )
            )

    thread = threading.Thread(target=start_new)
    thread.daemon = True
    thread.start()

    time.sleep(1)

    pulse_seq_times = {}
    driver_times = {}
    processing_times = {}

    first_time = True
    processing_done = True

    while True:

        if first_time:
            # Request new sequence metadata
            printing("Requesting metadata from Radar control")
            send_request(
                brian_to_radar_control, RADCTRL_BRIAN_IDEN, "Requesting metadata"
            )
            first_time = False

        start = time.time()
        socks = dict(sequence_poller.poll())
        end = time.time()
        printing("Poller time {}".format(end - start))

        if brian_to_driver in socks and socks[brian_to_driver] == zmq.POLLIN:

            # Receive metadata of completed sequence from driver such as timing
            reply = recv_reply(brian_to_driver, DRIVER_BRIAN_IDEN, printing)
            meta = rxsamplesmetadata_pb2.RxSamplesMetadata()
            meta.ParseFromString(reply)
            reply_output = "Driver sent -> time {}, sqnum {}".format(
                meta.sequence_time, meta.sequence_num
            )
            printing(reply_output)

            driver_times[meta.sequence_num] = meta.sequence_time

            # Requesting acknowledgement of work begins from DSP
            printing("Requesting work begins from DSP")
            send_request(
                brian_to_dsp_begin, DSPBEGIN_BRIAN_IDEN, "Requesting work begins"
            )
            # acknowledge we want to start something new
            start_new_sock.send("want_to_start")

        if (
            brian_to_radar_control in socks
            and socks[brian_to_radar_control] == zmq.POLLIN
        ):

            # Get new sequence metadata from radar control
            reply = recv_reply(brian_to_radar_control, RADCTRL_BRIAN_IDEN, printing)

            sigp = sigprocpacket_pb2.SigProcPacket()
            sigp.ParseFromString(reply)
            reply_output = "Radar control sent -> sequence {} time {}".format(
                sigp.sequence_num, sigp.sequence_time
            )
            printing(reply_output)

            pulse_seq_times[sigp.sequence_num] = sigp.sequence_time

            # Request acknowledgement of sequence from driver
            printing("Requesting ack from driver")
            send_request(brian_to_driver, DRIVER_BRIAN_IDEN, "Requesting ack")

        if brian_to_dsp_begin in socks and socks[brian_to_dsp_begin] == zmq.POLLIN:

            # Get acknowledgement that work began in processing.
            reply = recv_reply(brian_to_dsp_begin, DSPBEGIN_BRIAN_IDEN, printing)
            reply_output = "Dsp sent -> {}".format(reply)
            printing(reply_output)

            # Requesting acknowledgement of work ends from DSP
            printing("Requesting work end from DSP")
            send_request(brian_to_dsp_end, DSPEND_BRIAN_IDEN, "Requesting work ends")

            # acknowledge that we are good and able to start something new
            start_new_sock.send("good_to_start")

        if brian_to_dsp_end in socks and socks[brian_to_dsp_end] == zmq.POLLIN:

            global late_counter
            # Receive ack that work finished on previous sequence.
            reply = recv_reply(brian_to_dsp_end, DSPEND_BRIAN_IDEN, printing)

            proc_d = processeddata_pb2.ProcessedData()
            proc_d.ParseFromString(reply)
            reply_output = "Dsp sent -> time {}, sqnum {}".format(
                proc_d.processing_time, proc_d.sequence_num
            )
            printing(reply_output)

            print(proc_d.sequence_num)
            processing_times[proc_d.sequence_num] = proc_d.processing_time
            if proc_d.sequence_num != 0:
                if proc_d.processing_time > processing_times[proc_d.sequence_num - 1]:
                    late_counter += 1
                else:
                    late_counter = 0
            printing("Late counter {}".format(late_counter))

            # acknowledge that we are good and able to start something new
            start_new_sock.send("extra_good_to_start")


def router():
    context = zmq.Context().instance()
    router = context.socket(zmq.ROUTER)
    router.bind(ROUTER_ADDRESS)

    sys.stdout.write("Starting router!\n")
    while True:
        dd = router.recv_multipart()
        # sys.stdout.write(dd)
        sender, receiver, empty, data = dd
        output = (
            "Router input/// Sender -> {}: Receiver -> {}: empty: Data -> {}\n".format(
                *dd
            )
        )
        # sys.stdout.write(output)
        frames = [receiver, sender, empty, data]
        output = (
            "Router output/// Receiver -> {}: Sender -> {}: empty: Data -> {}\n".format(
                *frames
            )
        )
        # sys.stdout.write(output)
        router.send_multipart(frames)


if __name__ == "__main__":
    import threading

    sys.stdout.write("BRIAN_TESTER: Main\n")

    threads = []

    threads.append(threading.Thread(target=router))
    time.sleep(0.5)
    threads.append(threading.Thread(target=radar_control))
    threads.append(threading.Thread(target=experiment_handler))
    threads.append(threading.Thread(target=sequence_timing))
    threads.append(threading.Thread(target=driver))
    threads.append(threading.Thread(target=dsp))
    threads.append(threading.Thread(target=data_write))

    for thread in threads:
        thread.daemon = True
        thread.start()

    sys.stdout.write("BRIAN_TESTER: Threads all set up\n")
    while True:
        time.sleep(1)
