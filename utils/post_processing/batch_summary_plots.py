import sys
import os
import argparse
import glob
import pydarn
import matplotlib.pyplot as plt
from multiprocessing import Pool, Process
import bz2


def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
     
    :returns: the usage message
    """

    usage_message = """ batch_summary_plots.py [-h] directory_to_place path_regex
    
    Pass in the directory you wish to convert. Filenames with .fitacf and .fitacf.dmap will 
    be attempted to be plotted to a summary plot. The plots will be written to the given 
    directory with the added extension .summaryplot.png.

    """

    return usage_message


def summary_plot_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("directory_to_place", help="Where to place summary plots")
    parser.add_argument("path_regex", nargs="+", help="Path regex to the files you wish to plot"
                                                    "summary files from, or list of all files")
    return parser


def single_rtp_plot(fitacf_file):
    """
    Plots beam zero param p_l
    """
    pydarn_reader = pydarn.DarnRead(fitacf_file)
    fitacf_data = pydarn_reader.read_fitacf()
    plt.figure(figsize=(12, 8))
    im, cb, cmap, time_axis, elev_axis, z_data = pydarn.RTP.plot_range_time(fitacf_data, parameter='p_l', beam_num=0, groundscatter=False)
    #plt.gcf().set_size_inches(8, 12)
    plt.savefig(fitacf_file + '.p_l.png', dpi=500)
    plt.close()


def plot_fitacf_summary(fitacf_file, directory_to_place):
    """
    Plots beam 0 fitacf file summary plot.
    """

    pydarn_reader = pydarn.DarnRead(fitacf_file)
    fitacf_data = pydarn_reader.read_fitacf()

    pydarn.RTP.plot_summary(fitacf_data, beam_num=0, groundscatter=True, boundary={'nave': (0,40)})
    
    if directory_to_place[-1] != '/':
        directory_to_place = directory_to_place + '/'
    plt.savefig(directory_to_place + os.path.basename(fitacf_file) + '.summaryplot.png', dpi=500)
    plt.close()


if __name__ == "__main__":
    parser = summary_plot_parser()
    args = parser.parse_args()

    # summary_pool = Pool(processes=4)

    fitacf_files = args.path_regex

    jobs = []
    files_left = True
    filename_index = 0
    num_processes = 4

    while files_left:
        for procnum in range(num_processes):
            try:
                filename = fitacf_files[filename_index + procnum]
                print('Plotting: ' + filename)
            except IndexError:
                if filename_index + procnum == 0:
                    print('No files found to check!')
                    raise
                files_left = False
                break
            p = Process(target=plot_fitacf_summary, args=(filename, args.directory_to_place,))
            jobs.append(p)
            p.start()

        for proc in jobs:
            proc.join()

        filename_index += num_processes

    # summary_pool.map(plot_fitacf_summary, fitacf_files)
    # summary_pool.map(single_rtp_plot, fitacf_files)
