#!/bin/bash

# Define the arguments seeds
# args_list=(2381 5351 6047 6829 9221 9859 8053 1097 8677 2521)
# args_list=(80   70   70   20   70   80   80   85   95   85)

args_list=(2521)
argsb_list=(95)

# Ensure both lists have the same length
if [ ${#args_list[@]} -ne ${#argsb_list[@]} ]; then
    echo "Error: args_list and argsb_list must have the same length"
    exit 1
fi

# Run the file with the arguments
export DATAROOT=../../data 
for i in "${!args_list[@]}"; do
    arg=${args_list[$i]}
    argb=${argsb_list[$i]}
    echo "Running with arguments $arg and $argb"
    python moving_dismiss.py "$arg" "$argb" 
    python moving_dismiss2.py "$arg" "$argb" 
    python cook.py "$arg"
    huawei-eval -f "$arg.json" -s "$arg" -v
done