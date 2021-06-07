algo=$2
tuning=$4
iterations=$6

./winograd_dev -ic 3 -oc 64 -is 224 224 -ks 3 3 -ip 1 -a $algo -tn $tuning -i $iterations
./winograd_dev -ic 64 -oc 64 -is 224 224 -ks 3 3 -ip 1 -a $algo -tn $tuning -i $iterations
./winograd_dev -ic 64 -oc 128 -is 112 112 -ks 3 3 -ip 1 -a $algo -tn $tuning -i $iterations
./winograd_dev -ic 128 -oc 128 -is 112 112 -ks 3 3 -ip 1 -a $algo -tn $tuning -i $iterations
./winograd_dev -ic 128 -oc 256 -is 56 56 -ks 3 3 -ip 1 -a $algo -tn $tuning -i $iterations
./winograd_dev -ic 256 -oc 256 -is 56 56 -ks 3 3 -ip 1 -a $algo -tn $tuning -i $iterations
./winograd_dev -ic 256 -oc 512 -is 28 28 -ks 3 3 -ip 1 -a $algo -tn $tuning -i $iterations
./winograd_dev -ic 512 -oc 512 -is 28 28 -ks 3 3 -ip 1 -a $algo -tn $tuning -i $iterations
./winograd_dev -ic 512 -oc 512 -is 14 14 -ks 3 3 -ip 1 -a $algo -tn $tuning -i $iterations
