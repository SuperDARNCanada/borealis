# Copyright SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

import argparse
import bz2
import sys
import os
import copy
import itertools
import subprocess as sp
import numpy as np
import warnings
import tables
warnings.filterwarnings('ignore', category=tables.NaturalNameWarning)
import deepdish as dd

def usage_msg():
    """
    Return the usage message for this process.
     
    This is used if a -h flag or invalid arguments are provided.
    """

    usage_message = """ borealis_fixer.py [-h] filename fixed_dat_dir
    
    **** NOT TO BE USED IN PRODUCTION ****
    **** USE WITH CAUTION ****

    Modify a borealis file with updated data fields. Modify the script where
    indicated to update the file. Used in commissioning phase of Borealis when 
    data fields were not finalized."""

    return usage_message


def script_parser():
    parser = argparse.ArgumentParser(usage=usage_msg())
    parser.add_argument("filename", help="Path to the file that you wish to modify")
    parser.add_argument("fixed_data_dir", help="Path to place the updated file in.")

    return parser


def update_file(filename, out_file):

    recs = dd.io.load(filename)
    sorted_keys = sorted(list(recs.keys()))


    tmp_file = out_file + ".tmp"


    write_dict = {}
    
    def convert_to_numpy(data):
        """Converts lists stored in dict into numpy array. Recursive.

        Args:
            data (Python dictionary): Dictionary with lists to convert to numpy arrays.
        """
        for k, v in data.items():
            if isinstance(v, dict):
                convert_to_numpy(v)
            elif isinstance(v, list):
                data[k] = np.array(v)
            else:
                continue
        return data

    for key_num, group_name in enumerate(sorted_keys):

        # APPLY CHANGE HERE
        #recs[group_name]['data_dimensions'][0] = 2
        if 'noise_at_freq' not in recs[group_name].keys():
            recs[group_name]['noise_at_freq'] = np.array([0.0] * int(recs[group_name]['num_sequences']), dtype=np.float64)
            if key_num == 0:
                print('noise_at_freq added')
        if 'data_normalization_factor' not in recs[group_name].keys():
            recs[group_name]['data_normalization_factor'] = np.float64(9999999.999999996)
            if key_num == 0:
                print('data_normalization_factor added')
        if 'comment' in recs[group_name].keys():
            recs[group_name]['experiment_comment'] = recs[group_name]['comment']
            del recs[group_name]['comment']
            if key_num == 0:
                print('experiment_comment added')
        if 'slice_comment' not in recs[group_name].keys():
            recs[group_name]['slice_comment'] = np.unicode_('')
            if key_num == 0:
                print('slice_comment added')
        if 'experiment_string' in recs[group_name].keys():
            recs[group_name]['experiment_name'] = recs[group_name]['experiment_string']
            del recs[group_name]['experiment_string']
            if key_num == 0:
                print('experiment_name added')
        if 'num_slices' not in recs[group_name].keys():
            recs[group_name]['num_slices'] = np.int64(1)
            if key_num == 0:
                print('num_slices added')
        if 'range_sep' not in recs[group_name].keys():
            recs[group_name]['range_sep'] = np.float32(44.96887)
            if key_num == 0:
                print('range_sep added')
        if 'num_ranges' not in recs[group_name].keys():
            recs[group_name]['num_ranges'] = np.uint32(75)
            if key_num == 0:
                print('num_ranges added')
        if 'timestamp_of_write' in recs[group_name].keys():
            del recs[group_name]['timestamp_of_write']
            if key_num == 0:
                print('timestamp_of_write removed')
        if not isinstance(recs[group_name]['experiment_id'], np.int64):
            recs[group_name]['experiment_id'] = np.int64(recs[group_name]['experiment_id'])
            if key_num == 0:
                print('experiment id type changed')
        if len(recs[group_name]['lags']) == 0: # empty - issue in April, generate from pulses
            if key_num == 0:
                print('lagtable generated')
            lag_table = list(itertools.combinations(recs[group_name]['pulses'], 2))
            lag_table.append([recs[group_name]['pulses'][0], recs[group_name][
                'pulses'][0]])  # lag 0
            # sort by lag number
            lag_table = sorted(lag_table, key=lambda x: x[1] - x[0])
            lag_table.append([recs[group_name]['pulses'][-1], recs[group_name][
                'pulses'][-1]])  # alternate lag 0
            recs[group_name]['lags'] = lag_table

        # if recs[group_name]['correlation_dimensions'].shape[0] == 2:
        #     recs[group_name]['correlation_dimensions'] = np.array([1] + list(recs[group_name]['correlation_dimensions']), dtype=np.uint32)
        #     # assuming num_beams = 1 here. Giving three dimensions as required     
        #     recs[group_name]['correlation_descriptors'] = np.array(['num_beams', 'num_ranges', 'num_lags', dtype=np.unicode_])
        # if not isinstance(recs[group_name]['correlation_dimensions'][0], np.uint32):
        #     recs[group_name]['correlation_dimensions'] = np.array(recs[group_name]['correlation_dimensions'], dtype=np.uint32)
        # if recs[group_name]['correlation_dimensions'][2] == 0:
        #     recs[group_name]['correlation_dimensions'][2] = np.uint32(recs[group_name]['lags'].shape[0])

        write_dict = {}
        write_dict[group_name] = convert_to_numpy(recs[group_name])
        dd.io.save(tmp_file, write_dict, compression=None)

        # use external h5copy utility to move new record into 2hr file.
        cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
        cmd = cmd.format(newfile=tmp_file, twohr=out_file, dtstr=group_name)

        # TODO(keith): improve call to subprocess.
        sp.call(cmd.split())
        os.remove(tmp_file)


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


def file_updater(filename, fixed_data_dir):
    """
    Checks if the file is bz2, decompresses if necessary, and 
    writes to a fixed data directory. If the file was bz2, then the resulting
    file will also be compressed to bz2.

    Parameters
    ----------
    filename
        filename to update, can be bz2 compressed
    fixed_data_dir
        pathname to put the new file into
    """

    if os.path.basename(filename).split('.')[-1] in ['bz2', 'bzip2']:
        hdf5_file = decompress_bz2(filename)
        bzip2 = True
    else:
        hdf5_file = filename
        bzip2 = False
    
    if fixed_data_dir[-1] == '/':
        out_file = fixed_data_dir + os.path.basename(hdf5_file)
    else:
        out_file = fixed_data_dir + "/" + os.path.basename(hdf5_file)

    update_file(hdf5_file, out_file)

    if bzip2:
        # remove the input file from the directory because it was generated.
        os.remove(hdf5_file)
        # compress the updated file to bz2 if the input file was given as bz2.
        bz2_filename = compress_bz2(out_file)
        os.remove(out_file)
        out_file = bz2_filename

    return out_file


if __name__ == "__main__":
    parser = script_parser()
    args = parser.parse_args()

    file_updater(args.filename, args.fixed_data_dir)
