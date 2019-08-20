# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

import sys
import os
import math
import numpy as np
import datetime
import deepdish
import bz2
import glob
import time
from multiprocessing import Pool, Queue, Manager, Process
import argparse

def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
     
    :returns: the usage message
    """

    usage_message = """ borealis_gaps.py [-h] data_dir start_day end_day
    
    Pass in the data directory that you want to check for borealis gaps. This script uses 
    multiprocessing to check for gaps in the files and gaps between files. 

    The data directory passed in should have within it multiple directories named YYYYMMDD
    for a day's worth of data that is held there.
    """

    return usage_message


def borealis_gaps_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("data_dir", help="Path to the directory that holds the day directories of "
                                         "*[filetype].hdf5.site.bz2 files.")
    parser.add_argument("start_day", help="First day directory to check, given as YYYYMMDD.")
    parser.add_argument("end_day", help="Last day directory to check, given as YYYYMMDD.")
    parser.add_argument("--filetype", help="The filetype that you want to check gaps in (bfiq or "
                                           "rawacf typical). Default 'rawacf'")
    parser.add_argument("--gap_spacing", help="The gap spacing that you wish to check the file"
                                              " for, in seconds. Default 7s.")

    return parser


def decompress_bz2(filename):
    basename = os.path.basename(filename)
    newfilepath = os.path.dirname(filename) + '/' + '.'.join(basename.split('.')[0:-1]) # all but bz2

    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filename, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)

    return newfilepath


def check_for_gaps_in_file(rawacf_file, gap_spacing, gaps_dict, file_duration_dict):
    """

    :param gap_spacing: minimum spacing allowed between integration periods, in seconds.
    """
    print('Checking for gaps in : ' + rawacf_file)
    if os.path.basename(filename).split('.')[-1] in ['bz2', 'bzip2']:
        borealis_hdf5_file = decompress_bz2(filename)
        bzip2 = True
    else:
        borealis_hdf5_file = filename
        bzip2 = False

    records = sorted(deepdish.io.load(borealis_hdf5_file).keys())
    first_record = records[0]
    last_record = records[-1]

    inner_dict1 = file_duration_dict 
    inner_dict1[rawacf_file] = (first_record, last_record)
    file_duration_dict = inner_dict1 # reassign

    # records are first sequence timestamp, since epoch, in ms. 
    # expected difference in timestamps of no more than 7s. 
    # (6s integrations are default for 2min scans on some modes)
    # however gap spacing is passed in to allow different uses of the function

    inner_dict2 = gaps_dict
    if rawacf_file not in inner_dict2.keys():
        inner_dict2[rawacf_file] = [] # need to mutate and reassign to notify proxy
    # check all spacings within file
    for record_num, record in enumerate(records):
        if record_num == len(records) - 1:
            continue
        this_record = datetime.datetime.utcfromtimestamp(float(record)/1000)
        expected_next_record = this_record + datetime.timedelta(seconds=float(gap_spacing))
        if datetime.datetime.utcfromtimestamp(float(records[record_num + 1])/1000) > expected_next_record:
            # append the gap to the dictionary list where key = filename,
            # value = list of gaps. Gaps are lists of (gap_start, gap_end)
            #print('A Gap found in file!')
            #print(inner_dict2[rawacf_file])
            inner_dict2[rawacf_file] = inner_dict2[rawacf_file] + [(record, records[record_num + 1])]
            #print(inner_dict2[rawacf_file])
            gaps_dict = inner_dict2

    if bzip2:
        if borealis_hdf5_file.split('.')[-1] in ['bz2', 'bzip2']:
            print('Warning: attempted remove of original bzip file {}'.format(borealis_hdf5_file))
        else:
            os.remove(borealis_hdf5_file)


def check_for_gaps_between_files(file_duration_dict, gap_spacing, gaps_dict):
    """
    :param gap_spacing: minimum spacing allowed between integration periods, in seconds. 
    """
    # print(file_duration_dict)
    sorted_filenames = sorted(file_duration_dict.keys())
    previous_last_record = file_duration_dict[sorted_filenames[0]][1]
    new_dict = gaps_dict
    for file_num, filename in enumerate(sorted_filenames):
        if file_num == 0:
            continue # skip first one
        previous_end_time = datetime.datetime.utcfromtimestamp(float(previous_last_record)/1000) # last record integration start time in the first file.
        (first_record, last_record) = file_duration_dict[filename]
        start_time = datetime.datetime.utcfromtimestamp(float(first_record)/1000)
        end_time = datetime.datetime.utcfromtimestamp(float(last_record)/1000)
        if start_time > previous_end_time + datetime.timedelta(seconds=float(gap_spacing)):
            # append gap to this filename's list of gaps. Dict key is filename, list is list of (gap_start, gap_end)
            #print('A GAP!')
            if filename not in new_dict.keys():
                new_dict[filename] = []
                print('Adding to dict') 
            #print(new_dict[filename])
            new_dict[filename] = [(previous_last_record, first_record)] + new_dict[filename]
            # new_dict[filename].insert(0, (previous_last_record, first_record)) # insert at front
            #print(new_dict[filename])
        previous_last_record = last_record
    return new_dict


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + datetime.timedelta(n)


def print_gaps(gaps_dict):

    strf_format = '%Y%m%d %H:%M:%S'
    for filename in sorted(gaps_dict.keys()):
        gaps = gaps_dict[filename]
        if gaps:  # not empty
            for (gap_start, gap_end) in gaps:
                gap_start_time = datetime.datetime.utcfromtimestamp(float(gap_start)/1000)
                gap_end_time = datetime.datetime.utcfromtimestamp(float(gap_end)/1000)
                gap_duration = gap_end_time - gap_start_time
                print(filename)
                print('GAP: ' + gap_start_time.strftime(strf_format) + ' - ' + gap_end_time.strftime(strf_format))
                duration = gap_duration.total_seconds()
                duration_min = duration/60.0
                print('Duration: ' + str(duration) + ' s, ' + str(duration_min) + ' min')
        else:
            print(filename + ': NONE')

if __name__ == '__main__':
    parser = borealis_gaps_parser()
    args = parser.parse_args()

    if args.gap_spacing is None:
        gap_spacing = 7 # s
    else:
        gap_spacing = float(args.gap_spacing)

    if args.filetype is None:
        filetype = 'rawacf'
    else:
        filetype = args.filetype

    files = []
    
    data_dir = args.data_dir
    if data_dir[-1] != '/':
        data_dir += '/'

    start_day = datetime.datetime(year=int(args.start_day[0:4]), month=int(args.start_day[4:6]), day=int(args.start_day[6:8]))
    end_day = datetime.datetime(year=int(args.end_day[0:4]), month=int(args.end_day[4:6]), day=int(args.end_day[6:8]))    

    for one_day in daterange(start_day, end_day):
        print(one_day.strftime("%Y%m%d"))
        files.extend(glob.glob(data_dir + one_day.strftime("%Y%m%d") + '/*.' + filetype + '.hdf5.site.bz2'))
    
    manager1 = Manager()
    gaps_dict = manager1.dict()
    manager2 = Manager()
    file_duration_dict = manager2.dict()
    jobs = []

    for filename in files:
        gaps_dict[filename] = []
    
    # print(rawacf_files)

    files_left = True
    filename_index = 0
    num_processes = 4
    while files_left:
        for procnum in range(num_processes):
            try:
                filename = files[filename_index + procnum]
            except IndexError:
                if filename_index + procnum == 0:
                    print('No files found to check!')
                    raise
                files_left = False
                break
            p = Process(target=check_for_gaps_in_file, args=(filename, gap_spacing, gaps_dict, file_duration_dict))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        filename_index += num_processes

    all_gaps_dict = check_for_gaps_between_files(file_duration_dict, gap_spacing, gaps_dict)

    print_gaps(all_gaps_dict)
    sorted_filenames = sorted(file_duration_dict.keys())
    print('The first timestamp in this file set is: {}'.format(datetime.datetime.utcfromtimestamp(float(file_duration_dict[sorted_filenames[0]][0]/1000))))
    print('The last timestamp in this file set is: {}'.format(datetime.datetime.utcfromtimestamp(float(file_duration_dict[sorted_filenames[-1]][1]/1000))))

#    pool = Pool(processes=8)  # 8 worker processes
#    pool.map(plot_bfiq_file_power, iq_files)


