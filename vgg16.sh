echo "conv1_1"
./winograd_dev   3  64 224 1
echo "conv1_2"
./winograd_dev  64  64 224 1


echo "conv2_1"
./winograd_dev  64 128 112 1
echo "conv2_2"
./winograd_dev 128 128 112 1

echo "conv3_1"
./winograd_dev 128 256  56 1
echo "conv3_2"
./winograd_dev 256 256  56 1


echo "conv4_1"
./winograd_dev 256 512  28 1
echo "conv4_2"
./winograd_dev 512 512  28 1


echo "conv5_1" 
./winograd_dev 512 512  14 1
