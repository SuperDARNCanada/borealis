import mmap
import sys
import zmq
sys.path.append('../../build/debug/utils/protobuf')
import computationpacket_pb2
import numpy as np
import time
import datetime

if __name__ == "__main__":
    context = zmq.Context()
    driver_socket = context.socket(zmq.PAIR)
    driver_socket.bind("ipc:///tmp/feeds/1")

    computation_packet = computationpacket_pb2.ComputationPacket()

    mmapped_files = []
    while True:
        
        data = driver_socket.recv()
        print("Received data",str(datetime.datetime.utcnow()))
        start = time.time()
        numeric_data = np.fromstring(data,dtype=np.complex64)
        end = time.time()
        print("len ",len(data)," time ",end - start)
        print(numeric_data[0:5])
        
        # computation_packet.ParseFromString(data)

        # print(computation_packet.region_name,computation_packet.size)

        # mmap_name = computation_packet.region_name

        # result = filter(lambda mm: mm['region_name'] == mmap_name, mmapped_files)

        # current_dict = {}

        # import time

        # start = time.time()

        # if not result:
        #     f = open(mmap_name,'r+b')
        #     mm = mmap.mmap(f.fileno(), 0)
        #     f.close()
        #     new_mmap_dict = {
        #         'region_name' : mmap_name,
        #         'size' : computation_packet.size,
        #         #'fd' : f,
        #         'mm' : mm
        #     }

        #     mmapped_files.append(new_mmap_dict)
        #     current_dict = new_mmap_dict
        # else:
        #     current_dict = result[0]
        #     if current_dict['size'] != computation_packet.size:
        #         current_dict['mm'].close()

        #         f = open(mmap_name,'r+b')
        #         mm = mmap.mmap(f.fileno(), 0)
        #         f.close()

        #         current_dict['size'] = computation_packet.size
        #         current_dict['mm'] = mm


        # end = time.time()
        
        # print("time ",end - start)
        
        # result = filter(lambda mm: mm['region_name'] == mmap_name, mmapped_files)
        # print(len(mmapped_files))
        # print(len(mmapped_files[0]['mm']))
        # rec_data = np.fromstring(result[0]['mm'][:-1],dtype=np.float32)
        # print(rec_data[0:25])







