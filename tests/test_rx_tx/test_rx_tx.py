import numpy as np
import uhd.usrp as usrp
import uhd
import math
import sys
import time
import threading

# Define constants
DEV_ARGS = "num_recv_frames=512,num_send_frames=256,send_buff_size=2304000"
CLKSRC = "internal"

RXSUBDEV = "A:A A:B"
TXSUBDEV = "A:A"

TXCHAN = np.array([0])
RXCHAN = np.array([0, 1])

TXRATE = 250.0e3
RXRATE = 250.0e3
TXFREQ = 11e6
RXFREQ = 11e6
DELAY = 10e-3

PULSETIMES = [0.0, 13500e-3, 18000e-6, 30000e-6, 33000e-6, 39000e-6, 40500e-6]

start_tx = False
start_time = uhd.types.TimeSpec(0)
tx_ringbuffer_size = 10000
test_mode = "0"


# Make ramped pulses
def make_ramped_pulse(tx_rate: float):
    """
    Purpose:
            Create a ramped pulse to be used during testing
    Pre-conditions:
            tx_rate: the data transfer rate of the device, double format
    Return:
            a list of complex numbers representing the generated pulse
    """
    # define amplitude and pulse length
    amp = 1.0 / math.sqrt(2.0)
    pulse_len = 300.0e-6

    tr_start_pad = 60
    tr_end_pad = 60
    num_samps_per_antenna = int(
        np.ceil(pulse_len * (tx_rate + tr_start_pad + tr_end_pad))
    )
    tx_freqs = [1e6]

    # initialize samples list
    samples = np.zeros(num_samps_per_antenna, dtype=np.complex64)

    # build the pulse one sample at a time
    for i in np.arange(tr_start_pad, num_samps_per_antenna - tr_end_pad):
        nco_point = np.complex64(0.0)
        for freq in tx_freqs:
            sampling_freq = 2 * math.pi * freq / tx_rate

            radians = math.fmod(sampling_freq * i, 2 * math.pi)
            I = amp * math.cos(radians)
            Q = amp * math.sin(radians)

            nco_point += complex(I, Q)

        samples[i] = nco_point

    ramp_size = int(10e-6 * tx_rate)

    k = 0
    for i in np.arange(tr_start_pad, tr_start_pad + ramp_size):
        a = k / ramp_size
        samples[i] *= complex(a, 0.0)
        k += 1

    k = 0
    for j in np.arange(
        num_samps_per_antenna - tr_end_pad - 1,
        num_samps_per_antenna - tr_end_pad - 1 - ramp_size,
        -1,
    ):
        a = k / ramp_size
        samples[j] *= complex(a, 0.0)

    return samples


# RX THREAD
def recv(usrp_d, rx_chans):
    """
    Function defines the operation of the recieve thread for the USRP
    :param usrp_d: The MultiUSRP object
    :param rx_chans: A list specifying the receiver channels
    """
    # create the stream args object
    rx_stream_args = usrp.StreamArgs("fc32", "sc16")
    # set channel list
    rx_stream_args.channels = rx_chans
    # set rx streamer
    rx_stream = usrp_d.get_rx_stream(rx_stream_args)

    # create buffer
    usrp_buffer_size = 100 * rx_stream.get_max_num_samps()
    ringbuffer_size = (
        sys.getsizeof(500.0e6) / np.dtype(np.complex128).itemsize / usrp_buffer_size
    ) * usrp_buffer_size
    recv_buffer = np.zeros((int(rx_chans.size), int(np.ceil(ringbuffer_size))))

    # make stream
    rx_stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
    rx_stream_cmd.stream_now = False
    rx_stream_cmd.num_samps = 0
    rx_stream_cmd.time_spec = usrp_d.get_time_now() + uhd.types.TimeSpec(DELAY)

    # timing stuff
    stream_start = time.time()
    rx_stream.issue_stream_cmd(rx_stream_cmd)
    stream_end = time.time()

    # set up metadata and statistics counters
    meta = uhd.types.RXMetadata()

    buffer_inc = 0
    timeout_count = 0
    overflow_count = 0
    overflow_oos_count = 0
    late_count = 0
    bchain_count = 0
    align_count = 0
    badp_count = 0
    first_time = True

    # execute testing
    test_trials = 0
    test_while = True
    while test_while:
        num_rx_samples = rx_stream.recv(recv_buffer, meta, 3.0)
        print("Recv", num_rx_samples, "samples", "\n")
        print("On ringbuffer idx", usrp_buffer_size * buffer_inc, "\n")

        error_code = meta.error_code
        print("RX TIME:", meta.time_spec.get_real_secs(), "\n")
        if first_time:
            start_time = meta.time_spec
            start_tx = True
            first_time = False

        # handle errors
        if error_code == uhd.types.RXMetadataErrorCode.none:
            pass
        elif error_code == uhd.types.RXMetadataErrorCode.timeout:
            print("Timed out!\n")
            timeout_count += 1
        elif error_code == uhd.types.RXMetadataErrorCode.overflow:
            print("Overflow!\n")
            print("OOS:", meta.out_of_sequence, "\n")
            if meta.out_of_sequence:
                overflow_oos_count += 1
            overflow_count += 1
        elif error_code == uhd.types.RXMetadataErrorCode.late:
            print("Late!\n")
            late_count += 1
        elif error_code == uhd.types.RXMetadataErrorCode.broken_chain:
            print("Broken Chain!\n")
            bchain_count += 1
        elif error_code == uhd.types.RXMetadataErrorCode.alignment:
            print("Alignment!\n")
            align_count += 1
        elif error_code == uhd.types.RXMetadataErrorCode.bad_packet:
            print("Bad Packet!")
            badp_count += 1

        test_trials += 1

        if test_trials >= 1000000:
            test_while = False
            test_trials = 0
    # Print results
    print("Timeout count:", timeout_count, "\n")
    print("Overflow count:", overflow_count, "\n")
    print("Overflow OOS count:", overflow_oos_count, "\n")
    print("Late count:", late_count, "\n")
    print("Broken chain count", bchain_count, "\n")
    print("Alignment count", align_count, "\n")
    print("Bad packet count", badp_count, "\n")


# TX THREAD
def tx(usrp_d: usrp.MultiUSRP, tx_chans):
    """
    Defines operation of transfer thread for USRP testing
    :param usrp_d: The MultiUSRP object
    :param tx_chans: A list of channels for transmit testing
    """
    tx_stream_args = usrp.StreamArgs("fc32", "sc16")
    tx_stream_args.channels = tx_chans
    tx_stream = usrp_d.get_tx_stream(tx_stream_args)

    # create tx samples
    pulse = make_ramped_pulse(TXRATE)
    tx_samples = pulse
    try:
        for chan in tx_chans[1:]:
            np.stack((tx_samples, pulse))
    except IndexError:
        print("Only one tx channel, continuing")

    # set up transmit test
    time_per_pulse = PULSETIMES

    count = 1
    test_trials = 0
    test_while = True

    # run tests
    while test_while:
        u_time_now = usrp_d.get_time_now()
        time_zero = u_time_now + uhd.types.TimeSpec(DELAY)

        print("Starting tx #", count, "\n")

        def send(start_time):
            """
            Helper function for sending signals to TXIO board
            :param start_time: the start time to send a signal to the board
            """
            # create metadata for transmission
            meta = uhd.types.TXMetadata()
            meta.has_time_spec = True
            time_to_send_pulse = uhd.types.TimeSpec(start_time)
            pulse_start_time = time_zero + time_to_send_pulse
            meta.time_spec = pulse_start_time

            meta.start_of_burst = True

            # set up loop controls
            num_samps_sent = 0
            samples_per_buff = np.size(tx_samples[0])

            # send samples
            while num_samps_sent < samples_per_buff:
                num_samps_to_send = samples_per_buff - num_samps_sent
                num_samps_sent = tx_stream.send(tx_samples, meta)
                meta.start_of_burst = False
                meta.has_time_spec = False

            # finish tx stream
            meta.end_of_burst = True
            tx_stream.send(np.zeros((np.size(tx_samples[0]), 0)), meta)

        # send samples to board
        for i in np.arange(np.size(time_per_pulse)):
            send(time_per_pulse[i])

        seq_time = time_per_pulse[-1] + 23.5e-3
        start_sample = (
            np.uint32((time_zero.get_real_secs() - start_time.get_real_secs()) * RXRATE)
            % tx_ringbuffer_size
        )

        time.sleep(seq_time + 2 * DELAY)
        if (start_sample + (seq_time * RXRATE)) > tx_ringbuffer_size:
            end_sample = (
                np.uint32(start_sample + (seq_time * RXRATE)) - tx_ringbuffer_size
            )

            print(
                "Tx #",
                count,
                "needs sample",
                start_sample,
                "to",
                tx_ringbuffer_size - 1,
                "and 0 to",
                end_sample,
                "\n",
            )
        else:
            end_sample = np.uint32(start_sample + (seq_time * RXRATE))

            print("Tx #", count, "needs sample", start_sample, "to", end_sample, "\n")

        usrp_d.clear_command_time()
        count += 1

        test_trials += 1
        if test_trials >= 100:
            test_while = False
            test_trials = 0
    return


# MAIN LOOP
def UHD_SAFE_MAIN():
    """
    Main program loop
    """
    argv = sys.argv
    test_mode = argv[2]
    # Throw error for incomplete arguments
    if not (len(argv) == 4):
        print("TXIO Board tsting requires address and testing mode arguments.\n")
        print("Test modes: txrx, txo, rxo, idle, full")
    # output heading
    print("\n")
    print(
        "-----------------------------------------TXIO BOARD TESTING SCRIPT--------------------------------------------------"
    )
    print("Version:", argv[1], "\n")
    print("Unit IP Address:", argv[2], "\n")
    print("Test mode:", argv[3], "\n")
    print("\n")

    # Setup a usrp device

    usrp_d = usrp.MultiUSRP(DEV_ARGS + ",addr=" + argv[2])
    usrp_d.set_clock_source(CLKSRC)

    rx_subdev = usrp.SubdevSpec(RXSUBDEV)
    usrp_d.set_rx_subdev_spec(rx_subdev)

    usrp_d.set_rx_rate(RXRATE)

    rx_chans = RXCHAN

    rx_tune_request = uhd.types.TuneRequest(RXFREQ)
    for channel in rx_chans:
        usrp_d.set_rx_freq(rx_tune_request, channel)
        actual_freq = usrp_d.get_rx_freq(channel)
        if not (actual_freq == RXFREQ):
            print("requested rx ctr freq", RXFREQ, "actual_freq", actual_freq, "\n")

    # set usrp time
    res = time.clock_getres(time.CLOCK_REALTIME)
    tt_sc = time.time()
    while (((tt_sc) - np.floor(tt_sc)) < 0.2) or (((tt_sc) - np.floor(tt_sc)) > 0.3):
        tt_sc = time.time()
        time.sleep(0.01)

    curr_time = uhd.types.TimeSpec(tt_sc)
    usrp_d.set_time_now(curr_time)

    # configure usrp motherboards
    for i in np.arange(usrp_d.get_num_mboards()):
        usrp_d.set_gpio_attr("RXA", "CTRL", 0xFFFF, 0b11111111, i)
        usrp_d.set_gpio_attr("RXA", "DDR", 0xFFFF, 0b11111111, i)

        # Mirror pins along bank for easier scoping
        usrp_d.set_gpio_attr("RXA", "ATR_RX", 0xFFFF, 0b000000010, i)
        usrp_d.set_gpio_attr("RXA", "ATR_RX", 0xFFFF, 0b000000100, i)

        usrp_d.set_gpio_attr("RXA", "ATR_TX", 0xFFFF, 0b000001000, i)
        usrp_d.set_gpio_attr("RXA", "ATR_TX", 0xFFFF, 0b000010000, i)

        # XX is the actual TR signal
        usrp_d.set_gpio_attr("RXA", "ATR_XX", 0xFFFF, 0b000100000, i)
        usrp_d.set_gpio_attr("RXA", "ATR_XX", 0xFFFF, 0b001000000, i)

        # 0X acts as 'scope sync'
        usrp_d.set_gpio_attr("RXA", "ATR_0X", 0xFFFF, 0b010000000, i)
        usrp_d.set_gpio_attr("RXA", "ATR_0X", 0xFFFF, 0b100000000, i)

    # tx config
    tx_chans = TXCHAN

    tx_subdev = usrp.SubdevSpec(TXSUBDEV)
    usrp_d.set_tx_subdev_spec(tx_subdev)
    usrp_d.set_tx_rate(TXRATE)

    tx_tune_request = uhd.types.TuneRequest(TXFREQ)
    for channel in tx_chans:
        usrp_d.set_tx_freq(tx_tune_request, channel)
        actual_freq = usrp_d.get_tx_freq(channel)
        if not (actual_freq == RXFREQ):
            print("requested tx ctr freq", TXFREQ, "actual_freq", actual_freq, "\n")

    # Select the test sequence to run
    if argv[3] == "txrx":
        rx_thread = threading.Thread(target=recv, args=(usrp_d, rx_chans))
        tx_thread = threading.Thread(target=tx, args=(usrp_d, tx_chans))
        rx_thread.start()
        rx_thread.join()
        tx_thread.start()
        tx_thread.join()
    elif argv[3] == "txo":
        tx_thread = threading.Thread(target=tx, args=(usrp_d, tx_chans))
        tx_thread.start()
        tx_thread.join()
    elif argv[3] == "rxo":
        rx_thread = threading.Thread(target=recv, args=(usrp_d, rx_chans))
        rx_thread.start()
        rx_thread.join()
    elif argv[3] == "idle":
        print("IDLE...\n")
        while 1:
            continue
    elif argv[3] == "full":
        print("Not yet implemented\n")
        # TODO: Implement this
    else:
        print("Invalid testing mode provided.\n")
    return


if __name__ == "__main__":
    UHD_SAFE_MAIN()
