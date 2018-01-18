import zmq
import sys
import datetime
import os

if not os.environ["BOREALISPATH"]:
    raise ValueError("BOREALISPATH env variable not set")

if __debug__:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')
else:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/release/utils/protobuf')
import processeddata_pb2

sys.path.append(os.environ["BOREALISPATH"] + '/utils/data_write_options')
import data_write_options

if __name__== '__main__':
    options = data_write_options.DataWriteOptions()
    context = zmq.Context()
    processed_data_socket = context.socket(zmq.PAIR)
    processed_data_socket.bind(options.rx_dsp_to_data_write_address)

    while(True):
        try:
            data = processed_data_socket.recv()
        except KeyboardInterrupt:
            processed_data_socket.close()
            context.term()
            sys.exit()

        start = datetime.datetime.now()

        pd = processeddata_pb2.ProcessedData()
        pd.ParseFromString(data)

        end = datetime.datetime.now()

	del data

        if __debug__:
            diff = end - start
            time = diff.total_seconds() * 1000
            print("Time to parse + delete: {0} ms".format(time))



















