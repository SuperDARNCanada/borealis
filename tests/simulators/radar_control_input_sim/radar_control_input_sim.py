#!/usr/bin/python
import sys
import time
import threading
import os


BOREALISPATH = os.environ["BOREALISPATH"]
sys.path.append(BOREALISPATH)
sys.path.append(os.environ["BOREALISPATH"] + "/build/release/borealis/utils/protobuf")

from borealis.utils import socket_operations as so
from borealis.utils.options.experimentoptions import ExperimentOptions

from driverpacket_pb2 import DriverPacket
from sigprocpacket_pb2 import SigProcPacket
from rxsamplesmetadata_pb2 import RxSamplesMetadata
from processeddata_pb2 import ProcessedData


options = ExperimentOptions()

ids = [
    options.dsp_to_radctrl_identity,
    options.driver_to_radctrl_identity,
    options.driver_to_dsp_identity,
    options.driver_to_brian_identity,
    options.dsp_to_driver_identity,
    options.dspbegin_to_brian_identity,
    options.dspend_to_brian_identity,
]

sockets_list = so.create_sockets(ids, options.router_address)

dsp_to_radctrl = sockets_list[0]
driver_to_radctrl = sockets_list[1]
driver_to_dsp = sockets_list[2]
driver_to_brian = sockets_list[3]
dsp_to_driver = sockets_list[4]
dsp_to_brian_begin = sockets_list[5]
dsp_to_brian_end = sockets_list[6]


def printing(err_msg):
    print(err_msg)


def dsp():
    while True:
        # so.send_request(dsp_to_radctrl, options.radctrl_to_dsp_identity ,"Need metadata")
        reply = so.recv_bytes(dsp_to_radctrl, options.radctrl_to_dsp_identity, printing)

        sigp = SigProcPacket()
        sigp.ParseFromString(reply)

        so.send_request(
            dsp_to_driver, options.driver_to_dsp_identity, "Need data to process"
        )
        reply = so.recv_bytes(dsp_to_driver, options.driver_to_dsp_identity, printing)

        meta = RxSamplesMetadata()
        meta.ParseFromString(reply)

        time.sleep(0.030)

        request = so.recv_request(
            dsp_to_brian_begin, options.brian_to_dspbegin_identity, printing
        )
        so.send_bytes(
            dsp_to_brian_begin,
            options.brian_to_dspbegin_identity,
            sigp.SerializeToString(),
        )

        def do_work(sqn_num):
            sequence_num = sqn_num

            start = time.time()
            time.sleep(0.05)
            end = time.time()

            proc_data = ProcessedData()
            proc_data.processing_time = end - start
            proc_data.sequence_num = sequence_num

            # acknowledge end of work
            request = so.recv_request(
                dsp_to_brian_end, options.brian_to_dspend_identity, printing
            )
            so.send_bytes(
                dsp_to_brian_end,
                options.brian_to_dspend_identity,
                proc_data.SerializeToString(),
            )

        thread = threading.Thread(target=do_work, args=(sigp.sequence_num,))
        thread.daemon = True
        thread.start()


def driver():
    while True:
        eob = False
        sqn_time = 0.0

        while not eob:
            data = so.recv_bytes(
                driver_to_radctrl, options.radctrl_to_driver_identity, printing
            )

            pulse = DriverPacket()
            pulse.ParseFromString(data)

            if pulse.numberofreceivesamples > 0:
                sqn_time = pulse.numberofreceivesamples / options.tx_sample_rate

            eob = pulse.EOB

        start = time.time()
        time.sleep(sqn_time)

        end = time.time()

        samps_meta = RxSamplesMetadata()
        samps_meta.sequence_time = end - start
        samps_meta.sequence_num = pulse.sequence_num

        # sending sequence data to dsp
        request = so.recv_request(
            driver_to_dsp, options.dsp_to_driver_identity, printing
        )
        so.send_bytes(
            driver_to_dsp,
            options.dsp_to_driver_identity,
            samps_meta.SerializeToString(),
        )

        # sending collected data to brian
        request = so.recv_request(
            driver_to_brian, options.brian_to_driver_identity, printing
        )
        so.send_bytes(
            driver_to_brian,
            options.brian_to_driver_identity,
            samps_meta.SerializeToString(),
        )


if __name__ == "__main__":
    dsp_thread = threading.Thread(target=dsp)
    driver_thread = threading.Thread(target=driver)

    dsp_thread.daemon = True
    driver_thread.daemon = True

    dsp_thread.start()
    driver_thread.start()

    while True:
        time.sleep(1)
