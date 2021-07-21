#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=$1
export PYTHONIOENCODING=utf-8

echo "Device: $CUDA_VISIBLE_DEVICES"
dataset="ogbg-mol"
output_dir="../../../epcb_results/ogbg/"
config_file="./"$dataset".json"

time_stamp=`date '+%s'`
commit_id=`git rev-parse HEAD`
std_file=${output_dir}"stdout/"${time_stamp}_${commit_id}".txt"

mkdir -p $output_dir"stdout/"

python -u ./main.py --config=$config_file --id=$commit_id --ts=$time_stamp --dir=$output_dir"board/"

#pid=$!

#echo "Stdout dir:   $std_file"
#echo "Start time:   `date -d @$time_stamp  '+%Y-%m-%d %H:%M:%S'`"
#echo "CUDA DEVICES: $CUDA_VISIBLE_DEVICES"
#echo "pid:          $pid"
#cat $config_file

#sleep 1

#tail -f $std_file
