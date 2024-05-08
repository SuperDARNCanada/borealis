#
# Filter written rawrf data using remai filter.
#
# Then beamform and produce output_samples_iq

import matplotlib

matplotlib.use("TkAgg")
import sys
import os
import deepdish
import argparse

sys.path.append(os.environ["BOREALISPATH"])

borealis_path = os.environ["BOREALISPATH"]
config_file = borealis_path + "/config.ini"


def testing_parser():
    """
    Creates the parser for this script.

    :returns: parser, the argument parser for the testing script.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filename",
        help="The name of the rawrf file to filter and create output_samples from.",
    )

    return parser


def main():
    parser = testing_parser()
    args = parser.parse_args()
    rawrf_file = args.filename

    data_file_ext = rawrf_file.split(".")[-2]
    if data_file_ext != "rawrf":
        raise Exception("Please provide a rawrf file.")

    data = deepdish.io.load(rawrf_file)

    # order the keys and get the first record.


if __name__ == "__main__":
    main()
