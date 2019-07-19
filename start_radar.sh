#!/bin/bash

screen -X -S borealis quit
sleep 1
cd $BOREALISPATH
./steamed_hams.sh $1 release

