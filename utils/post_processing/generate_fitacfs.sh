#!/bin/bash

# Copyright 2018 SuperDARN
# Description: Gennerates relevant fitacf files from relevant rawacf files that are in the working directory.
#              If a problem occurs an email is sent to Kevin and Marina. 
#
# Authors: 
# Kevin Krieger 
# Marina Schmidt 
# Marci Detwiller


script_name=$0
workingdir_raw=$1$3/
workingdir_fitacf=$2$3/
yyyymmdd=$3
#logfile=$4

# Current standard for previous fitacf generation is 2.5, however 3.0 is now 
# available - for Borealis data we will use 3.0 Marci Detwiller
version=3.0

echo 'Going through each rawfile and making fitacf files'

rawacfFiles=0
# looping through every relevant rawacf file in the working directory and preforming the following tasks:
#                   - convert to the specified fitacf version (algorithm)
#                   - check if any errors occured during make_fit call, if so remove fitacf file (if created)
#                           and add to the email for potential rawacf review (maybe it is currupt!)
#                   - check if the file is empty, same process as above
#                   - check if their is dmap error using backscatter, same process 

for rawfile in $workingdir_raw*.rawacf.dmap
do
    echo $rawfile
    basefilename=`basename ${rawfile}`
    echo "make_fit -fitacf-version ${version} ${rawfile} > ${workingdir_fitacf}/${basefilename%.rawacf.dmap}.fitacf.dmap" 
    make_fit -fitacf-version ${version} ${rawfile} > ${workingdir_fitacf}/${basefilename%.rawacf.dmap}.fitacf.dmap 
    returnvalue=$?
    # Check if make_fit succeeded, if not then log it, remove it, and email the peeps
    if [ ${returnvalue} -ne 0 ]
    then
        echo ${workingdir_fitacf}/${basefilename%.rawacf.dmap}.fitacf.dmap # >> ${logfile}.failed_fitacfs
        message="Error: make_fit returned ${returnvalue} on ${rawfile}"
        echo ${message} ${script_name}
        message=$( (rm -v  ${workingdir_fitacf}/${basefilename%.rawacf.dmap}.fitacf.dmap) 2>&1)
        echo ${message} ${script_name}
        continue
    fi
    # Check if make_fit succeeded, if not then log it, remove it, and email the peeps 
    if [ ! -s ${workingdir_fitacf}/${basefilename%.rawacf.dmap}.fitacf.dmap ] 
    then
        echo ${workingdir_fitacf}/${basefilename%.rawacf.dmap}.fitacf.dmap >> ${logfile}.failed_fitacfs
        message="Error: make_fit generated a empty fitacf file ${rawfile%.rawacf.dmap}.fitacf.dmap"
        echo ${message} ${script_name}
        message=$( (rm -v ${workingdir_fitacf}/${basefilename%.rawacf.dmap}.fitacf.dmap) 2>&1)
        echo ${message} ${script_name}
        continue
    fi
done
