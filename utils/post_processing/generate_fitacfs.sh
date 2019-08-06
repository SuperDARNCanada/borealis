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
workingdir_raw=$1
workingdir_fitacf=$2
yyyymm=$3
logfile=$4


# Current standard for previous fitacf generation is 2.5, however 3.0 is now 
# available - for Borealis data we will use 3.0 Marci Detwiller
version=3.0

./logger.sh 'Going through each rawfile and making fitacf files' ${logfile} ${script_name}

rawacfFiles=0
# looping through every relevant rawacf file in the working directory and preforming the following tasks:
#                   - convert to the specified fitacf version (algorithm)
#                   - check if any errors occured during make_fit call, if so remove fitacf file (if created)
#                           and add to the email for potential rawacf review (maybe it is currupt!)
#                   - check if the file is empty, same process as above
#                   - check if their is dmap error using backscatter, same process 
for rawfile in $(ls ${workingdir_raw} | grep ${yyyymm}.*.rawacf)
do
    echo "make_fit -fitacf-version ${version} ${workingdir_raw}/${rawfile} > ${workingdir_fitacf}/${rawfile%.*}.fitacf" >> ${logfile} 
    make_fit -fitacf-version ${version} ${workingdir_raw}/${rawfile} > ${workingdir_fitacf}/${rawfile%.*}.fitacf 
    returnvalue=$?
    # Check if make_fit succeeded, if not then log it, remove it, and email the peeps
    if [ ${returnvalue} -ne 0 ]
    then
        echo ${workingdir_fitacf}/${rawfile%.*}.fitacf >> ${logfile}.failed_fitacfs
        message="Error: make_fit returned ${returnvalue} on ${workingdir_raw}/${rawfile}"
        ./logger.sh ${message} ${logfile} ${script_name}
        message=$( (rm -v  ${workingdir_fitacf}/${rawfile%.*}.fitacf >> ${logfile}) 2>&1)
        ./logger.sh ${message} ${logfile} ${script_name}
         message=$( (rm -v  ${workingdir_raw}/${rawfile} >> ${logfile}) 2>&1)
        ./logger.sh ${message} ${logfile} ${script_name}
        continue
    fi
    # Check if make_fit succeeded, if not then log it, remove it, and email the peeps 
    if [ ! -s ${workingdir_fitacf}/${rawfile%.*}.fitacf ] 
    then
        echo ${workingdir_fitacf}/${rawfile%.*}.fitacf >> ${logfile}.failed_fitacfs
        message="Error: make_fit generated a empty fitacf file ${rawfile%.*}.fitacf"
        ./logger.sh ${message} ${logfile} ${script_name}
        message=$( (rm -v ${workingdir_fitacf}/${rawfile%.*}.fitacf >> ${logfile}) 2>&1)
        ./logger.sh ${message} ${logfile} ${script_name}
         message=$( (rm -v  ${workingdir_raw}/${rawfile} >> ${logfile}) 2>&1)
        ./logger.sh ${message} ${logfile} ${script_name}
        continue
    fi
done
