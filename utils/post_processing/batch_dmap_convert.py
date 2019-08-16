# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

import os
from multiprocessing import Process
import subprocess as sp


def run_dmap(date):
    script_command = '/home/dataman/borealis/utils/post_processing/dmap_convert.sh {date} > /home/dataman/borealis/utils/post_processing/logs/{date}_dmap.log'
    cmd = script_command.format(date=str(date))
    print(cmd)
    sp.call(cmd.split())


if __name__ == "__main__":

    dates = [#20190523,
            20190524,
            20190525,
            20190526,
            20190527,
            20190528,
            20190529,
            20190530,
            20190531,
            20190601,
            20190602,
            20190603,
            20190604,
            20190605,
            20190607,
            20190608,
            20190609,
            20190610,
            20190611,
            20190612,
            20190613,
            20190614,
            20190615,
            20190616,
            20190617,
            20190618,
            20190619,
            20190625,
            20190626,
            20190627,
            20190628,
            20190629,
            20190630,
            20190701,
            20190702,
            20190705,
            20190706,
            20190707,
            20190708,
            20190709,
            20190710,
            20190711,
            20190712,
            20190713,
            20190714,
            20190715,
            20190716,
            20190717,
            20190718,
            20190719,
            20190720,
            20190721,
            20190722,
            20190723,
            20190724,
            20190725,
            20190726,
            20190727,
            20190728,
            20190729,
            20190730,
            20190731,
            20190801,
            20190802,
            20190803,
            20190804,
            20190805,
            20190806]

    jobs = []

    dates_left = True
    date_index = 0
    num_processes = 2

    while dates_left:
        for procnum in range(num_processes):
            try:
                date = dates[date_index + procnum]
                print('Dmapping: ' + str(date))
            except IndexError:
                if date_index + procnum == 0:
                    print('No files found to check!')
                    raise
                files_left = False
                break
            p = Process(target=run_dmap, args=(date,))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        date_index += num_processes
  
