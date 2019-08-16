import os
from multiprocessing import Process
import subprocess as sp


def run_fix_and_dmap(date):
    script_command = '/home/dataman/borealis/utils/post_processing/fix_and_dmap.sh {date} > /home/dataman/borealis/utils/post_processing/logs/{date}_fix_and_dmap.log'
    cmd = script_command.format(date=str(date))
    print(cmd)
    sp.call(cmd.split())


if __name__ == "__main__":

    dates = [20190401,
            20190402,
            20190403,
            20190404,
            20190405,
            20190406,
            20190407,
            20190408,
            20190409,
            20190410,
            20190411,
            20190412,
            20190413,
            20190414,
            20190415,
            20190416,
            20190417,
            20190418,
            20190419,
            20190420,
            20190421,
            20190422,
            20190423,
            20190424,
            20190425,
            20190426,
            20190427,
            20190428,
            20190429,
            20190430,
            20190501,
            20190502,
            20190503,
            20190504,
            20190505,
            20190506,
            20190507,
            20190508,
            20190509,
            #20190510,
            20190511,
            20190512,
            20190513,
            20190514,
            20190515,
            20190516,
            20190517,
            20190518,
            20190519,
            20190520,
            20190521,
            20190522]

    jobs = []

    dates_left = True
    date_index = 0
    num_processes = 2

    while dates_left:
        for procnum in range(num_processes):
            try:
                date = dates[date_index + procnum]
                print('Fixing and Dmapping: ' + str(date))
            except IndexError:
                if date_index + procnum == 0:
                    print('No files found to check!')
                    raise
                files_left = False
                break
            p = Process(target=run_fix_and_dmap, args=(date,))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        date_index += num_processes
  
