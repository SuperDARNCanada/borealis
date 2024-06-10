#!/usr/bin/python

import sys
import zmq
import time

sys.path.append('../build/release/utils/protobuf')
import sigprocpacket_pb2
import driverpacket_pb2


def setup_driver_socket(): # to send pulses to driver.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.bind("ipc:///tmp/feeds/0")
    return cpsocket


def setup_sigproc_params_socket(): #to send data to receive code.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.bind("ipc:///tmp/feeds/2")
    return cpsocket


def setup_sigproc_cpack_socket(): #to send data to receive code.
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.bind("ipc:///tmp/feeds/3")
    return cpsocket


def setup_sigproc_timing_ack_socket():
    context=zmq.Context()
    cpsocket=context.socket(zmq.PAIR)
    cpsocket.bind("ipc:///tmp/feeds/4")
    return cpsocket


def data_back_to_runradar(packet,procsocket, seqnum, kerneltime):
    """ Place data in the receiver packet and send it via zeromq to the
        receiver unit.
    """

    packet.Clear()
    packet.sequence_num=seqnum
    if kerneltime != None:
        packet.kerneltime=kerneltime
#    for num, cpo in enumerate(cpos):
#        channel_add = packet.rxchannel.add()
#        packet.rxchannel[num].rxfreq = cpo_list[cpo].rxfreq
#        packet.rxchannel[num].nrang = cpo_list[cpo].nrang
#        packet.rxchannel[num].frang = cpo_list[cpo].frang
#        for bnum, beamdir in enumerate(beam_dict[cpo]):
#            beam_add = packet.rxchannel[num].beam_directions.add()
#            # beamdir is a list 20-long with phase for each antenna for that beam direction.
#            for pnum, phi in enumerate(beamdir):
#                #print phi
#                phase = cmath.exp(phi*1j)
#                phase_add = packet.rxchannel[num].beam_directions[bnum].phase.add()
#                packet.rxchannel[num].beam_directions[bnum].phase[pnum].real_phase = phase.real
#                packet.rxchannel[num].beam_directions[bnum].phase[pnum].imag_phase = phase.imag


    # Don't need channel numbers, always send 20 beam directions
    #for chan in channels:
    #    receiverpacket.channels.append(chan)
    # Beam directions will be formated e^i*phi so that a 0 will indicate not
    # to receive on that channel.

#    for i in range(0,len(rxfreqs)):
#        beam_array_add=receiverpacket.BeamDirections.add()
#        for phi in beamdirs[i,:]:
#            phase = math.exp(phi*1j)
#            receiverpacket.BeamDirections[i].phase.append(phase)
#
    # get response TODO
    procsocket.send(packet.SerializeToString());
    return


def main():

    procsocket=setup_sigproc_params_socket()
    proccpsocket=setup_sigproc_cpack_socket()
    proctimesocket=setup_sigproc_timing_ack_socket()
    sigprocpacket=sigprocpacket_pb2.SigProcPacket()
    driversocket=setup_driver_socket()
    driverpacket=driverpacket_pb2.DriverPacket()        
    databackpacket=sigprocpacket_pb2.SigProcPacket()

    while True:
        try:
            rxdata = procsocket.recv(flags=zmq.NOBLOCK)
            sigprocpacket.ParseFromString(rxdata)
            seqnum = sigprocpacket.sequence_num
            time.sleep(0.030)
            print "CP Ack back"
            data_back_to_runradar(databackpacket,proccpsocket,seqnum,None)
            time.sleep(0.02)
            print "Timing Ack back"
            data_back_to_runradar(databackpacket,proctimesocket,seqnum,80)
        except zmq.Again:
            print "NOTHING RX"
            pass

        time.sleep(1)

        EOB = False 
        while not EOB:
            try:
                txdata=driversocket.recv(flags=zmq.NOBLOCK)
                driverpacket.ParseFromString(txdata)
                EOB = driverpacket.EOB
            except zmq.Again:
                time.sleep(0.05)
                print "NOTHING TX"
                pass
        seqnum = driverpacket.sqnnum
        print "Driver Ack back"
        data_back_to_runradar(databackpacket,driversocket,seqnum,None)

main()

