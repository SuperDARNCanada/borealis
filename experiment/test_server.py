#!/usr/bin/python

import time
import zmq
import sys
sys.path.append('../utils/protobuf')
import driverpacket_pb2

context=zmq.Context()
socket=context.socket(zmq.PAIR)
socket.bind("tcp://10.65.0.25:33033")

while True:
	message=socket.recv()
	driverpacket=driverpacket_pb2.DriverPacket()
	print "Received a message"
	driverpacket.ParseFromString(message)
	
	#print driverpacket.channels
	#print driverpacket.centerfreq
	print len(driverpacket.samples)
	#time.sleep(1)
	socket.send('K')
