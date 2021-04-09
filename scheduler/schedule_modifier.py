#!/usr/bin/python3

# Copyright 2019 SuperDARN Canada
#
# schedule_modifier.py
# 2019-05-28
# Safetly adds or removes events from the desired SCD file.
#
import scd_utils
import argparse

def main():
    parser = argparse.ArgumentParser(description="Add or remove lines in the schedule")
    parser.add_argument('--scd-dir', required=True, help='The scd working directory')
    parser.add_argument('--site', required=True, help='The site at which to modify')
    group = parser.add_mutually_exclusive_group(required=True, )
    group.add_argument('--add', action='store_true')
    group.add_argument('--remove', action='store_true')
    parser.add_argument('--date', metavar='YYYYMMDD', required=True, help='The date to run')
    parser.add_argument('--time', metavar='hh:mm', required=True, help='The time to run')
    parser.add_argument('--experiment', required=True, help='The experiment at the line')
    parser.add_argument('--mode-type', required=True, help='The scheduling mode type for this time period ie common, special, discretionary')
    parser.add_argument('--prio', default=0, help='The priority of the line')
    parser.add_argument('--duration', default='-', help='The duration of the line')
    parser.add_argument('--kwargs', default='', help='String of text containing kwargs for experiment')

    args = parser.parse_args()

    scd_dir = args.scd_dir
    site_id =args.site

    scd_file = '{}/{}.scd'.format(scd_dir, site_id)

    scd_util = scd_utils.SCDUtils(scd_file)

    if args.add:
        scd_util.add_line(args.date, args.time, args.experiment, args.mode_type, args.prio, args.duration, args.kwargs)

    if args.remove:
        scd_util.remove_line(args.date, args.time, args.experiment, args.mode_type, args.prio, args.duration, args.kwargs)    

if __name__ == '__main__':
    main()

