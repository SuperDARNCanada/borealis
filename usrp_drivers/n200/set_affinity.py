import subprocess as sp
import os
import itertools as it
import sys

sys.path.append(os.environ["BOREALISPATH"])

if __debug__:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/debug/utils/protobuf')  # TODO need to get this from scons environment, 'release' may be 'debug'
else:
    sys.path.append(os.environ["BOREALISPATH"] + '/build/release/utils/protobuf')

sys.path.append(os.environ["BOREALISPATH"] + '/utils/zmq_borealis_helpers')
sys.path.append(os.environ["BOREALISPATH"] + '/utils/driver_options')
import socket_operations as so
import set_affinity_options as op

def get_tids(process_name):

    bash_cmd = "for i in $(pgrep {0}); do ps -mo pid,tid,fname,user,psr -p $i;done"
    bash_cmd = bash_cmd.format((process_name))
    p1 = sp.Popen(bash_cmd, stdout=sp.PIPE, shell=True)
    #p2 = sp.Popen(['awk', "'{printf '%s\n', $1}'"], stdin=p1.stdout, stdout=sp.PIPE)


    stdout = p1.communicate()
    p1.stdout.close()
    print(stdout)

    tids = []
    for line in stdout[0].splitlines()[2:]:
    	split_line = line.split()

    	tid = split_line[1]
    	tids.append(tid)

    print tids
    return tids

def set_affinity(tids,already_configured,cpus):
    new_tids = list(set(tids) - set(already_configured))

    for tid,cpu in zip(new_tids,it.cycle(cpus)):
        print(tid,cpu)
        cmd = "taskset -c -p {0} {1}".format(cpu,tid)
        sp.call(cmd.split())

    return already_configured + new_tids



def printing(msg):
    set_affinity = "\033[31m" + "SET AFFINITTY: " + "\033[0m"
    sys.stdout.write(set_affinity + msg + "\n")

if __name__ == "__main__":
    process_name = sys.argv[1]
    options = op.SetAffinityOptions()
    cpus = range(12)

    ids = [options.mainaffinity_to_driver_identity, options.txaffinity_to_driver_identity,
            options.rxaffinity_to_driver_identity]

    sockets_list = so.create_sockets(ids, options.router_address)
    mainaffinity_to_driver = sockets_list[0]
    txaffinity_to_driver = sockets_list[1]
    rxaffinity_to_driver = sockets_list[2]

    already_configured = []
    set_main_uhd = so.recv_data(mainaffinity_to_driver, options.driver_to_mainaffinity_identity,
                                printing)

    already_configured = set_affinity(get_tids(process_name), already_configured, [cpus[0]])

    so.send_reply(mainaffinity_to_driver, options.driver_to_mainaffinity_identity, "ALL SET")


    set_tx_rx = so.recv_data(mainaffinity_to_driver, options.driver_to_mainaffinity_identity,
                                printing)

    already_configured = set_affinity(get_tids(process_name), already_configured, cpus[1:3])

    so.send_reply(mainaffinity_to_driver, options.driver_to_mainaffinity_identity, "ALL SET")


    set_rx_uhd = so.recv_data(rxaffinity_to_driver, options.driver_to_rxaffinity_identity, printing)

    already_configured = set_affinity(get_tids(process_name), already_configured, cpus[3:10])

    so.send_reply(rxaffinity_to_driver, options.driver_to_rxaffinity_identity, "ALL SET")


    set_tx_uhd = so.recv_data(txaffinity_to_driver, options.driver_to_txaffinity_identity, printing)

    already_configured = set_affinity(get_tids(process_name), already_configured, cpus[10:12])

    so.send_reply(txaffinity_to_driver, options.driver_to_txaffinity_identity, "ALL SET")



