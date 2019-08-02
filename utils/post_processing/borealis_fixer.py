# Copyright SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

import argparse
import sys
import os
import copy
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


def update_file(filename, fixed_data_dir):

    recs = dd.io.load(filename)
    sorted_keys = sorted(list(recs.keys()))

    out_file = fixed_data_dir + "/" + os.path.basename(filename)
    tmp_file = fixed_data_dir + "/" + os.path.basename(filename) + ".tmp"


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

    for group_name in sorted_keys:

        # APPLY CHANGE HERE
        # recs[group_name]['data_dimensions'][0] = 2
        recs[group_name]['noise_at_freq'] = np.array([0.0] * int(recs[group_name]['num_sequences']), dtype=np.float64)
        recs[group_name]['data_normalization_factor'] = np.float64(9999999.999999996)
        recs[group_name]['experiment_comment'] = recs[group_name]['comment']
        del recs[group_name]['comment']
        recs[group_name]['slice_comment'] = np.unicode_('')
        recs[group_name]['experiment_name'] = recs[group_name]['experiment_string']
        del recs[group_name]['experiment_string']
        recs[group_name]['num_slices'] = np.int64(1)
        recs[group_name]['range_sep'] = np.float32(44.96887) 
        recs[group_name]['num_ranges'] = np.uint32(75)
        # recs[group_name]['xcfs'] = recs[group_name]['xcfs'] * -1

        write_dict = {}
        write_dict[group_name] = convert_to_numpy(recs[group_name])
        dd.io.save(tmp_file, write_dict, compression=None)

        # use external h5copy utility to move new record into 2hr file.
        cmd = 'h5copy -i {newfile} -o {twohr} -s {dtstr} -d {dtstr}'
        cmd = cmd.format(newfile=tmp_file, twohr=out_file, dtstr=group_name)

        # TODO(keith): improve call to subprocess.
        sp.call(cmd.split())
        os.remove(tmp_file)


if __name__ == "__main__":
    parser = script_parser()
    args = parser.parse_args()

    update_file(args.filename, args.fixed_data_dir)
