import subprocess as sp
import os
import itertools as it
import sys
import

sys.path.append(os.environ["BOREALISPATH"])

if __debug__:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')  # TODO need to get this from scons environment, 'release' may be 'debug'
else:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/release/utils/protobuf')

sys.path.append(os.environ["BOREALISPATH"] + '/utils/zmq_borealis_helpers')
import socket_operations as so


def get_tids(process_name):
    p1 = sp.Popen(['bash','/root/get_tids.sh', process_name], stdout=sp.PIPE)
    #p2 = sp.Popen(['awk', "'{printf '%s\n', $1}'"], stdin=p1.stdout, stdout=sp.PIPE)


    stdout = p1.communicate()
    p1.stdout.close()
    #print(cpus[0].splitlines())

    tids = []

    for line in stdout[0].splitlines()[2:]:
    	split_line = line.split()

    	tid = split_line[1]
    	tids.append(tid)

    return tids

def set_affinity(tids,already_configured,cpus):
    new_tids = list(set(tids) - set(already_configured))

    for tid in zip(new_tids,it.cycle(cpus)):
        print(tid,cpu)
        cmd = "taskset -c -p {0} {1}".format(cpu,tid)
        sp.call(cmd.split())

    return already_configured + new_tids



def printing(msg):
    set_affinity = "\033[31m" + "SET AFFINITTY: " + "\033[0m"
    sys.stdout.write(set_affinity + msg + "\n")

if __name__ == "__main__":
    process_name = sys.argv[1]
    cpus = range(17)
    ids = [MAINAFFINITY_DRIVER_IDEN, TXAFFINITY_DRIVER_IDEN, RXAFFINITY_DRIVER_IDEN]

    sockets_list = so.create_sockets(ids, )
    mainaffinity_to_driver = sockets_list[0]
    txaffinity_to_driver = sockets_list[1]
    rxaffinity_to_driver = sockets_list[2]

    already_configured = []
    set_non_uhd = so.recv_data(mainaffinity_to_driver, driver_mainaffinity_iden, printing)

    set_affinity(get_tids(process_name), already_configured, cpus[0:3])

    so.send_reply(mainaffinity_to_driver, driver_mainaffinity_iden, "ALL SET")


    set_rx = so.recv_data(rxaffinity_to_driver, driver_rxaffinity_iden, printing)

    set_affinity(get_tids(process_name), already_configured, cpus[3:14])

    so.send_reply(rxaffinity_to_driver, driver_rxaffinity_iden, printing)





