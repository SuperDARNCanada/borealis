import sys
import os
import argparse
import glob
import pydarn
import matplotlib.pyplot as plt
from multiprocessing import Pool
import bz2


def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
     
    :returns: the usage message
    """

    usage_message = """ batch_summary_plots.py [-h] directory_to_convert 
    
    Pass in the directory you wish to convert. Filenames with .fitacf and .fitacf.dmap will 
    be attempted to be plotted to a summary plot. The plots will be written to the same 
    directory with the added extension .summaryplot.png.

    """

    return usage_message


def summary_plot_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("directory_to_convert", help="Path to the files you wish to plot"
                                                    "summary files from/to.")
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


def plot_fitacf_summary(fitacf_file):
    """
    Plots beam 0 fitacf file summary plot.
    """

    pydarn_reader = pydarn.DarnRead(fitacf_file)
    fitacf_data = pydarn_reader.read_fitacf()

    pydarn.RTP.plot_summary(fitacf_data, beam_num=0, groundscatter=True, boundary={'nave': (0,40)})
    
    plt.savefig(fitacf_file + '.summaryplot.png', dpi=500)
    plt.close()


if __name__ == "__main__":
    parser = summary_plot_parser()
    args = parser.parse_args()

    summary_pool = Pool(processes=4)

    fitacf_files = glob.glob(args.directory_to_convert + '*.fitacf')
    fitacf_files.extend(glob.glob(args.directory_to_convert + '*.fitacf.dmap'))

    summary_pool.map(plot_fitacf_summary, fitacf_files)
    # summary_pool.map(single_rtp_plot, fitacf_files)
