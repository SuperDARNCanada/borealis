#!/usr/bin/python

import time
import zmq
import sys
sys.path.append('../utils/protobuf')
import driverpacket_pb2

context=zmq.Context()
socket=context.socket(zmq.REP)
socket.bind("tcp://10.65.0.25:33033")

while True:
	message=socket.recv()
	driverpacket=driverpacket_pb2.DriverPacket()
	print "REceived a message"
	driverpacket.ParseFromString(message)
	
	print driverpacket.channels

	#time.sleep(1)
	socket.send('K')
