#!/bin/bash

#Usage: bash separate_data.sh file_name.extension number_of_small_files
#Output: file_name.01.extension

NUMBER_OF_SMALL_FILES=3
SOURCE_INPUT_FILE=''
TARGET_INPUT_FILE

LENGTH=$(wc -l < $1)
echo $LENGTH

# split -l 3000 valid.en valid.en

