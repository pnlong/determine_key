#!/bin/bash
#SBATCH -A tdlong_lab                   ## account to charge
#SBATCH -p standard                     ## run on the standard partition
#SBATCH --job-name=gunzip_key_data      ## job name
#SBATCH --error=/dfs7/adl/pnlong/artificial_dj/determine_key/gunzip_key_data.err            ## error log file
#SBATCH --output=/dfs7/adl/pnlong/artificial_dj/determine_key/gunzip_key_data.out           ## output log file

# README
# Phillip Long
# August 11, 2023
# ungzips and untars the directory created by key_dataset.py on the cluster

key_data="/dfs7/adl/pnlong/artificial_dj/data/key_data"

# tar gzip
# tar -zcvf /Volumes/Seagate/artificial_dj_data/key_data.tar.gz /Volumes/Seagate/artificial_dj_data/key_data

tar -xvzf "${key_data}.tar.gz" -C "${key_data}"

# rename key_data 
# mv "${key_data}" "${key_data}_temp"
# move unzipped tarball to correct location
# mv "${key_data}_temp/Volumes/Seagate/artificial_dj_data/key_data" "${key_data}"
# rm -rf "${key_data}_temp"

