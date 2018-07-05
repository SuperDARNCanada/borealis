import subprocess as sp
import os
import itertools as it
import sys
import time

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

    tids = []
    for line in stdout[0].splitlines()[2:]:
    	split_line = line.split()

    	tid = split_line[1]
    	tids.append(tid)

    return tids

def set_affinity(tids, cpus, num_boxes):
    cmd = "taskset -c -p {0} {1}"


    #main, uhd control, tx and rx threads
    print zip(tids[:3]+tids[-4:],cpus[0:7])
    for tid,cpu in zip(tids[:3]+tids[-4:],cpus[0:7]):
        sp.call(cmd.format(cpu,tid).split())
        #time.sleep(0.02)

    #uhd tx threads
    # for tid in tids[4:4+num_boxes]:
    #     sp.call(cmd.format(cpus[7],tid).split())

    for tid,cpu in zip(tids[3:3+num_boxes],it.cycle(cpus[9:])):
        sp.call(cmd.format(cpu,tid).split())
        #time.sleep(0.02)


    #uhd rx threads
    for tid,cpu in zip(tids[3+num_boxes:-3],it.cycle(cpus[8:9])):
        sp.call(cmd.format(cpu,tid).split())
        #time.sleep(0.02)



def printing(msg):
    set_affinity = "\033[31m" + "SET AFFINITTY: " + "\033[0m"
    sys.stdout.write(set_affinity + msg + "\n")

if __name__ == "__main__":
    process_name = "n200_driver"
    options = op.SetAffinityOptions()

    num_boxes = options.device_str.count("addr")
    cpus = range(10)

    ids = [options.mainaffinity_to_driver_identity, options.txaffinity_to_driver_identity,
            options.rxaffinity_to_driver_identity]

    sockets_list = so.create_sockets(ids, options.router_address)
    mainaffinity_to_driver = sockets_list[0]


    set_main_uhd = so.recv_data(mainaffinity_to_driver, options.driver_to_mainaffinity_identity,
                                printing)


    so.send_reply(mainaffinity_to_driver, options.driver_to_mainaffinity_identity, "ALL SET")

    #time.sleep(5)
    while True:
        set_affinity(get_tids(process_name), cpus, num_boxes)
        time.sleep(1)


