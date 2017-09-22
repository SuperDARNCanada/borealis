import zmq
import sys
import datetime
import json
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


class DataWrite(object):
    """This class contains the functions used to write out processed data to files.

    """

    def __init__(self, processed_data, options):
        super(DataWrite, self).__init__()
        self.debug_file = options.debug_file
        self.processed_data = processed_data

    def output_debug_data(self):
        """
        Writes out to file a JSON representation of each stage of filtering for debug analysis.

        """
        debug_data = {}
        for set_num,data_set in enumerate(self.processed_data.outputdataset):
            set_str = "dataset_{0}".format(set_num)
            debug_data[set_str] = {}
            for stage_num,debug_samples in enumerate(data_set.debugsamples):
                real = []
                imag = []
                stage_str = debug_samples.stagename
                debug_data[set_str][stage_str] = {}
                for antenna_num,antenna_data in enumerate(debug_samples.antennadata):
                    ant_str = "antenna_{0}".format(antenna_num)
                    debug_data[set_str][stage_str][ant_str] = {"real":[],"imag":[]}
                    for antenna_sample in antenna_data.antennasamples:
                        debug_data[set_str][stage_str][ant_str]["real"].append(antenna_sample.real)
                        debug_data[set_str][stage_str][ant_str]["imag"].append(antenna_sample.imag)

        with open(self.debug_file,'w') as f:
            f.write(json.dumps(debug_data))

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



        dw = DataWrite(pd)

        if __debug__:
            dw.output_debug_data()


        end = datetime.datetime.now()

        if __debug__:
            diff = end - start
            time = diff.total_seconds() * 1000
            print("Time to parse + write: {0} ms".format(time))



















