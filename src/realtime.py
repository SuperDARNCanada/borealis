#!/usr/bin/python3

"""
    realtime package
    ~~~~~~~~~~~~~~~~
    Sends data to realtime applications.

    :copyright: 2019 SuperDARN Canada
"""

import zmq
import threading
import pickle
import json
import time
import zlib
import pydarnio
import numpy as np
from backscatter import fitacf

from utils.options import Options
from utils import socket_operations as so


def convert_and_fit_record(rawacf_record):
    """Converts a rawacf record to DMAP format and fits using backscatter, returning the results"""
    # We only grab the first slice. This means the first slice of any that are SEQUENCE or CONCURRENT
    # interfaced.
    first_key = sorted(list(rawacf_record.keys()))[0]
    log.info("converting record", record=first_key)
    converted = (pydarnio.BorealisConvert.
                 _BorealisConvert__convert_rawacf_record(0, (first_key, rawacf_record[first_key]), ""))

    fitted_records = []
    for rec in converted:
        fit_data = fitacf._fit(rec)
        fitted_records.append(fit_data.copy())

    return fitted_records


def process_rawacf_data(options, ctx):
    """Receives rawacf data from data_write and fits using backscatter, then sends the result to a server thread.

    :param options: Options object for getting socket identities and addresses
    :type  options: Options
    :param     ctx: zmq context instance for socket creation
    :type      ctx: zmq.Context
    """
    data_write_to_realtime = so.create_sockets([options.rt_to_dw_identity], options.router_address)[0]

    # This socket is for sending fitted data to the other thread, for serving over the web
    realtime_socket = ctx.socket(zmq.PAIR)
    realtime_socket.bind("inproc://realtime")

    while True:
        rawacf_pickled = so.recv_bytes(data_write_to_realtime, options.dw_to_rt_identity, log)
        rawacf_data = pickle.loads(rawacf_pickled)

        try:
            fitted_recs = convert_and_fit_record(rawacf_data)
        except Exception as e:
            log.exception("error processing record", exception=e)
            continue

        for rec in fitted_recs:
            # Can't jsonify numpy, so we convert to native types for serving over the web
            for k, v in rec.items():
                if hasattr(v, 'dtype'):
                    if isinstance(v, np.ndarray):
                        rec[k] = v.tolist()
                    else:
                        rec[k] = v.item()

            realtime_socket.send(json.dumps(rec))  # Send the record to serve_data()


def serve_data(options, ctx):
    """Serves fitted data as JSON over the realtime address.

    :param options: Options object for getting socket identities and addresses
    :type  options: Options
    :param     ctx: zmq context instance for socket creation
    :type      ctx: zmq.Context
    """
    publish_socket = ctx.socket(zmq.PUB)
    publish_socket.bind(options.realtime_address)

    realtime_socket = ctx.socket(zmq.PAIR)
    realtime_socket.connect("inproc://realtime")

    while True:
        data_dict_json = realtime_socket.recv()  # Get a record from process_rawacf_data()
        compressed = zlib.compress(data_dict_json.encode('utf-8'))
        publish_socket.send(compressed)


def main():
    options = Options()
    context = zmq.Context().instance()

    threads = [threading.Thread(target=process_rawacf_data, args=(options, context,)),
               threading.Thread(target=serve_data, args=(options, context,))]

    for thread in threads:
        thread.daemon = True
        thread.start()
        time.sleep(1)   # Wait so that the in-process socket can be set up correctly

    for thread in threads:
        thread.join()


if __name__ == '__main__':
    from utils import log_config

    log = log_config.log()
    log.info(f"REALTIME BOOTED")
    try:
        main()
        log.info(f"REALTIME EXITED")
    except Exception as main_exception:
        log.critical("REALTIME CRASHED", error=main_exception)
        log.exception("REALTIME CRASHED", exception=main_exception)
