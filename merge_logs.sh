#!/bin/bash

# Merges log files from Borealis based on timestamps.

PROGRAMNAME=$0
LOGDIR=/data/borealis_logs

function usage={
  echo "usage: $PROGRAMNAME [-d DIR] year month day hour minute"
  echo "  $PROGRAMNAME will search $LOGDIR for borealis logfiles"
  echo "  of the format YYYY.MM.DD.HH:MM-[module] and merge them"
  echo "  into one logfile YYYY.MM.DD.HH:MM-log, sorted by time."
  echo " "
  echo "  -d DIR    search in directory DIR for logfiles"
  exit 1
}
