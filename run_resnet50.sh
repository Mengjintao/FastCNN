#!/bin/bash

# usage: ./winograd_dev [in_channels] [out_channels] [image dim] [kernel dim] [paddings] [strides] [#threads]

algo=$2
tuning=$4
iterations=$6
num_threads=$8

# Resnet-50
./winograd_dev -ic 3    -oc 64   -is 224 224 -ks 7 7 -ip 3 -s 2 2 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 64   -oc 64   -is 56 56   -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 64   -oc 64   -is 56 56   -ks 3 3 -ip 1 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 64   -oc 256  -is 56 56   -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 256  -oc 64   -is 56 56   -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 256  -oc 128  -is 56 56   -ks 1 1 -ip 0 -s 2 2 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 128  -oc 128  -is 28 28   -ks 3 3 -ip 1 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 128  -oc 512  -is 28 28   -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 256  -oc 512  -is 56 56   -ks 1 1 -ip 0 -s 2 2 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 512  -oc 128  -is 28 28   -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 128  -oc 512  -is 28 28   -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 512  -oc 256  -is 28 28   -ks 1 1 -ip 0 -s 2 2 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 256  -oc 256  -is 14 14   -ks 3 3 -ip 1 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 256  -oc 1024 -is 14 14   -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 512  -oc 1024 -is 28 28   -ks 1 1 -ip 0 -s 2 2 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 1024 -oc 256  -is 14 14   -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 1024 -oc 512  -is 14 14   -ks 1 1 -ip 0 -s 2 2 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 512  -oc 512  -is 7 7     -ks 3 3 -ip 1 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 512  -oc 2048 -is 7 7     -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 1024 -oc 2048 -is 14 14   -ks 1 1 -ip 0 -s 2 2 -a $algo -tn $tuning -i $iterations -t $num_threads
./winograd_dev -ic 2048 -oc 512  -is 7 7     -ks 1 1 -ip 0 -s 1 1 -a $algo -tn $tuning -i $iterations -t $num_threads
