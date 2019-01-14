import deepdish as dd
import sys
import os

def main():
    filename = sys.argv[1]
    fixed_data_dir = sys.argv[2]

    recs = dd.io.load(filename)

    for k, v in recs.items():
        v['num_ranges'] = 75
        v['range_separation'] = 45

    out_file = fixed_data_dir + "/" + os.path.basename(filename)

    dd.io.save(out_file, recs, compression=None)

if __name__ == "__main__":
    main()