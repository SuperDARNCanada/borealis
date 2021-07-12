#!/bin/bash

# Merges log files from Borealis based on timestamps.

PROGRAMNAME=$0
LOGDIR=/data/borealis_logs

usage() {
  echo "usage: $PROGRAMNAME [-d DIR] year month day hour minute"
  echo " "
  echo "  $PROGRAMNAME will search $LOGDIR for borealis logfiles"
  echo "  of the format YYYY.MM.DD.HH:MM-[module] and merge the"
  echo "  brian, radar_control, data_write, and signal_processing"
  echo "  files into one logfile YYYY.MM.DD.HH:MM-log, sorted by time."
  echo " "
  echo "  -d DIR    search in directory DIR for logfiles"
  exit 1
}

if [ $# == 0 ]; then
  usage;
fi

# If option is given, then update directory of logfiles
# Also, capture all the variables for finding files
if [ $1 == "-d" ]; then
  LOGDIR=$2
  YEAR=$3
  MONTH=$4
  DAY=$5
  HOUR=$6
  MINUTE=$7
else
  YEAR=$1
  MONTH=$2
  DAY=$3
  HOUR=$4
  MINUTE=$5
fi

# Path to files plus the prefix for each file
PREFIX="$LOGDIR/$YEAR.$MONTH.$DAY.$HOUR:$MINUTE"

# Temporary file for writing to
TMPFILE="$PREFIX-tmp"

touch $TMPFILE
cat "$PREFIX-radar_control" | grep -E "^[[:digit:]]{6}" >> $TMPFILE
cat "$PREFIX-brian" | grep -E "^[[:digit:]]{6}" >> $TMPFILE
cat "$PREFIX-data_write" | grep -E "^[[:digit:]]{6}" >> $TMPFILE
cat "$PREFIX-signal_processing" | grep -E "^[[:digit:]]{6}" >> $TMPFILE

sort $TMPFILE > "$PREFIX-log"
rm $TMPFILE

exit 0
