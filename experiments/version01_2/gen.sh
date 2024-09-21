#!/bin/bash

# Read the file name from the console
# filename=$1
# echo "File name is $filename"
# Define the arguments seeds
# args_list=(2381 5351 6047 6829 9221 9859 8053 1097 8677 2521)
args_list=(8677 2521)
# Run the file 10 times with the arguments
for arg in "${args_list[@]}"; do
    echo "Running $filename with argument $arg"
    python moving_dismiss.py "$arg"
    python moving_dismiss2.py "$arg"
    python cook.py "$arg"
done