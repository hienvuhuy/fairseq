#!/bin/bash

input_file=$1
mode=$2
input_directory=$(dirname "$input_file")

filename=$(basename -- "$input_file")
extension="${filename##*.}"
filename="${filename%.*}"
if [[ $mode == "bos" ]]; then
  awk '{gsub("_bos ", ""); print}' $input_file > $input_directory"/$filename.removed.$extension"
elif [[ $mode == "eos" ]]; then
  awk '{gsub("_eos ", ""); print}' $input_file > $input_directory"/$filename.removed.$extension"
else
  awk '{gsub("_bos ", ""); gsub("_eos ", ""); print}' $input_file > $input_directory"/$filename.removed.$extension"
fi