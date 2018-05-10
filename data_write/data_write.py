import zmq
import sys
import datetime
import json
import os
import h5py
from utils.zmq_borealis_helpser import socket_operations as so

if not os.environ["BOREALISPATH"]:
    raise ValueError("BOREALISPATH env variable not set")

if __debug__:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')
else:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/release/utils/protobuf')
import processeddata_pb2

sys.path.append(os.environ["BOREALISPATH"] + '/utils/data_write_options')
import data_write_options
borealis_path = os.environ['BOREALISPATH']


def printing(msg):
    DATA_WRITE = "\033[96m" + "DATA WRITE: " + "\033[0m"
    sys.stdout.write(DATA_WRITE + msg + "\n")


class DataWrite(object):
    """This class contains the functions used to write out processed data to files.

    """
    def __init__(self, processed_data, data_write_options):
        super(DataWrite, self).__init__()
        self.options = data_write_options
        self.processed_data = processed_data

    def output_debug_data(self):
        """
        Writes out to file a JSON representation of each stage of filtering for debug analysis.

        """
        debug_data = {}
        for set_num, data_set in enumerate(self.processed_data.outputdataset):
            set_str = "dataset_{0}".format(set_num)
            debug_data[set_str] = {}
            for stage_num, debug_samples in enumerate(data_set.debugsamples):
                stage_str = debug_samples.stagename
                debug_data[set_str][stage_str] = {}
                for antenna_num, antenna_data in enumerate(debug_samples.antennadata):
                    ant_str = "antenna_{0}".format(antenna_num)
                    debug_data[set_str][stage_str][ant_str] = {"real": [], "imag": []}
                    for antenna_sample in antenna_data.antennasamples:
                        debug_data[set_str][stage_str][ant_str]["real"].append(antenna_sample.real)
                        debug_data[set_str][stage_str][ant_str]["imag"].append(antenna_sample.imag)

        self.write_json_file(self.options.debug_file, debug_data)

    def write_json_file(self, filename, data_dict):
        """
        Write out data to a json file

        """
        with open(filename, 'w') as f:
            f.write(json.dumps(data_dict))

    def write_hdf5_file(self, filename, data_dict):
        """
        Write out data to an hdf5 file

        """
        hdf5_file = h5py.File(filename, "w")

    def write_dmap_file(self, filename, data_dict):
        """
        Write out data to a dmap file

        """
        pass

    def output_data(self, write_rawacf=True, write_iq=False, write_pre_bf_iq=False,
                    hdf5=False, json=True, dmap=False):
        """
        Write out data to a file
 
        """
        file_format_string = None
        if hdf5:
            file_format_string = 'hdf5'
        elif json:
            file_format_string = 'json'
        elif dmap:
            file_format_string = 'dmap'

        if not file_format_string:
            raise ValueError("File format selection required (hdf5, json, dmap), none given")

        # Iterate over every data set, one data set per frequency
        iq_available = False
        rawacf_available = False
        pre_bf_iq_available = False
        data_format_string = None
        iq_pre_bf_data_dict = {}
        rawacf_data_dict = {}
        iq_data_dict = {}
        final_data_dict = {}
        for freq_num, data_set in enumerate(self.processed_data.outputdataset):
            freq_str = "frequency_{0}".format(freq_num)
            # Find out what is available in the data to determine what to write out

            # Main acfs were calculated
            if len(data_set.mainacf) > 0:
                rawacf_available = True
                rawacf_data_dict[freq_str]['mainacf'] = {'real': [], 'imag': []}
                for complex_sample in data_set.mainacf:
                    rawacf_data_dict[freq_str]['mainacf']['real'].append(complex_sample.real)
                    rawacf_data_dict[freq_str]['mainacf']['imag'].append(complex_sample.imag)

            # Interferometer acfs were calculated
            if len(data_set.intacf) > 0:
                rawacf_available = True
                rawacf_data_dict[freq_str]['intacf'] = {'real': [], 'imag': []}
                for complex_sample in data_set.intacf:
                    rawacf_data_dict[freq_str]['intacf']['real'].append(complex_sample.real)
                    rawacf_data_dict[freq_str]['intacf']['imag'].append(complex_sample.imag)

            # Cross correlations were calculated
            if len(data_set.xcf) > 0:
                rawacf_available = True
                rawacf_data_dict[freq_str]['xcf'] = {'real': [], 'imag': []}
                for complex_sample in data_set.xcf:
                    rawacf_data_dict[freq_str]['xcf']['real'].append(complex_sample.real)
                    rawacf_data_dict[freq_str]['xcf']['imag'].append(complex_sample.imag)

            # IQ samples were beamformed
            if len(data_set.beamformediqsamples) > 0:
                iq_available = True
                iq_data_dict[freq_str] = {'real': [], 'imag': []}
                for complex_sample in data_set.beamformediqsamples:
                    iq_data_dict[freq_str]['real'].append(complex_sample.real)
                    iq_data_dict[freq_str]['imag'].append(complex_sample.imag)

            # Debug samples are available
            if len(data_set.debugsamples) > 0:
                for stage_num, debug_samples in enumerate(data_set.debugsamples):
                    if debug_samples.stagename == 'stage_3':
                        # Final stage, so write these samples only to file
                        pre_bf_iq_available = True
                        for ant_num, ant_data in enumerate(debug_samples.antennadata):
                            ant_str = "antenna_{0}".format(ant_num)
                            iq_pre_bf_data_dict[freq_str][ant_str] = {'real': [], 'imag': []}
                            for ant_samp in ant_data.antennasamples:
                                iq_pre_bf_data_dict[freq_str][ant_str]['real'].append(ant_samp.real)
                                iq_pre_bf_data_dict[freq_str][ant_str]['imag'].append(ant_samp.imag)
                    else:
                        continue

        if write_rawacf and rawacf_available:
            data_format_string = "rawacf"
            final_data_dict = rawacf_data_dict
        if write_iq and iq_available:
            data_format_string = "bfiq"
            final_data_dict = iq_data_dict
        if write_pre_bf_iq and pre_bf_iq_available:
            data_format_string = "iq"
            final_data_dict = iq_pre_bf_data_dict

        # What is the name and location for the dataset?
        today_string = datetime.datetime.today().strftime("%Y%m%d")
        datetime_string = datetime.datetime.today().strftime("%Y%m%d.%H%M.%S")
        dataset_name = "{0}.{1}.{2}.{3}".format(datetime_string, self.options.site_id,
                                                data_format_string, file_format_string)
        dataset_location = "{0}/{1}/{2}".format(self.data_directory, today_string, dataset_name)

        if hdf5:
            self.write_hdf5_file(dataset_location, final_data_dict)
        elif json:
            self.write_json_file(dataset_location, final_data_dict)
        elif dmap:
            self.write_dmap_file(dataset_location, final_data_dict)


if __name__ == '__main__':
    options = data_write_options.DataWriteOptions()
    ids = [options.dsp_to_dw_identity]
    dsp_to_data_write = so.create_sockets(ids, options.router_address)

    while True:
        try:
            data = socket_operations.recv_data(dsp_to_data_write, ids[0], printing)
        except KeyboardInterrupt:
            processed_data_socket.close()
            context.term()
            sys.exit()

        start = datetime.datetime.now()

        pd = processeddata_pb2.ProcessedData()
        pd.ParseFromString(data)

        dw = DataWrite(pd, options)

        if __debug__:
            dw.output_debug_data()
        else:
            dw.output_data()

        end = datetime.datetime.now()

        if __debug__:
            diff = end - start
            time = diff.total_seconds() * 1000
            print("Sequence number: {0}".format(pd.sequence_num))
            print("Time to process samples: {0} s".format(pd.processing_time))
            print("Time to parse + write: {0} ms".format(time))
