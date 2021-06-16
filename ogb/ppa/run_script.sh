#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONIOENCODING=utf-8
echo "Device: $CUDA_VISIBLE_DEVICES"

dataset="ogbg-ppa"
output_dir="save"
config_file="./"$dataset".json"

time_stamp=`date '+%s'`

mkdir -p $output_dir

python -u ./main.py --config=$config_file --id=$commit_id --ts=$time_stamp --dir=$output_dir
