#!/bin/bash

# Copyright 2019 SuperDARN Canada, University of Saskatchewan
# Author: Marci Detwiller

DAY_TO_DMAP=$1

DIRECTORY_TO_SOURCE="/borealis_site_data/sas_rawacf_unfixed/$DAY_TO_DMAP/*"
NEW_DIR="/borealis_site_data/sas_rawacf_temp/$DAY_TO_DMAP/"
NEW_DIR_REGEX="$NEW_DIR*.rawacf.hdf5.bz2"
NEW_DIR_DMAP="$NEW_DIR*.dmap.bz2"
DMAP_DIR="/borealis_site_data/sas_rawacf_dmap/$DAY_TO_DMAP/"

mkdir $NEW_DIR
mkdir $DMAP_DIR

source /home/dataman/borealis_env/bin/activate
cd /home/dataman/pydarn
# python3 setup.py install

cd /home/dataman/borealis/utils/post_processing

echo "python3 ./batch_rawacf_fixer.py $NEW_DIR $DIRECTORY_TO_SOURCE"
python3 ./batch_rawacf_fixer.py $NEW_DIR $DIRECTORY_TO_SOURCE

echo "python3 ./batch_borealis_convert.py $NEW_DIR_REGEX"
python3 ./batch_borealis_convert.py $NEW_DIR_REGEX

mv $NEW_DIR_DMAP $DMAP_DIR

