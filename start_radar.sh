#!/bin/bash

killall screen
sleep 1
cd $BOREALISPATH
./steamed_hams.sh $1 release

