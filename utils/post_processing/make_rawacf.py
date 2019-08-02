# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

"""
Script is used to convert bfiq files into rawacf files (all hdf5), without 
the need to run Borealis' signal processing module or datawrite.
"""
import argparse
import bz2
import os
import sys

from bfiq_to_rawacf import bfiq_to_rawacf_postprocessing

def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
    """

    usage_message = """ convert_bfiq_to_rawacf.py [-h] borealis_bfiq_file rawacf_directory
    
    Process a bfiq file to a rawacf Borealis file and place in the given directory. If the 
    input file is bzip2 compressed, the output file will also be bzip2 compressed before 
    exiting."""

    return usage_message


def script_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("borealis_bfiq_file", help="Path to the file that you wish to process to a "
                                                   "Borealis rawacf. (e.g. 20190327.2210.38.sas.0.bfiq.hdf5.bz2)")
    parser.add_argument("rawacf_directory", help="Path to place the processed file in.")

    return parser


def create_rawacf_filename(filename_to_convert, rawacf_directory):
    """
    Creates a rawacf filename in the rawacf_directory, 
    to write the file to. 

    Parameters
    ----------
    filename_to_convert
    	must not have a bz2 extension, should be decompressed.
    rawacf_directory
    	directory of new rawacf file. 
    """
    if rawacf_directory[-1] != '/':
    	rawacf_directory += '/'
    basename = os.path.basename(filename_to_convert)
    basename_without_ext = '.'.join(basename.split('.')[0:-2]) # all but .bfiq.hdf5
    rawacf_filename = rawacf_directory + basename_without_ext + '.rawacf.hdf5'
    return rawacf_filename


def decompress_bz2(filename):
    basename = os.path.basename(filename) 
    newfilepath = os.path.dirname(filename) + '/' + '.'.join(basename.split('.')[0:-1]) # all but bz2

    with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filename, 'rb') as file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            new_file.write(data)    

    return newfilepath


def compress_bz2(filename):
    bz2_filename = filename + '.bz2'

    with open(filename, 'rb') as file, bz2.BZ2File(bz2_filename, 'wb') as bz2_file:
        for data in iter(lambda : file.read(100 * 1024), b''):
            bz2_file.write(data)   

    return bz2_filename


def rawacf_processor(filename, rawacf_directory):
    """
    Checks if the file is bz2, decompresses if necessary, and 
    writes to a rawacf file. If the file was bz2, then the resulting rawacf
    file will also be compressed to bz2.

    Parameters
    ----------
    filename
    	filename to convert, can be bz2 compressed
    rawacf_directory
    	pathname to put the new file into
    """

    if os.path.basename(filename).split('.')[-1] in ['bz2', 'bzip2']:
        borealis_bfiq_file = decompress_bz2(filename)
        bzip2 = True
    else:
        borealis_bfiq_file = filename
        bzip2 = False

    rawacf_filename = create_rawacf_filename(borealis_bfiq_file, rawacf_directory)

    bfiq_to_rawacf_postprocessing(borealis_bfiq_file, rawacf_filename)

    if bzip2:
        # remove the input file from the directory because it was generated.
        os.remove(borealis_bfiq_file)
        # compress the rawacf file to bz2 if the input file was given as bz2.
        bz2_filename = compress_bz2(rawacf_filename)
        os.remove(rawacf_filename)
        rawacf_filename = bz2_filename

    return rawacf_filename


def main():
    parser = script_parser()
    args = parser.parse_args()

    rawacf_processor(args.borealis_bfiq_file, args.rawacf_directory)


if __name__ == "__main__":
    main()
