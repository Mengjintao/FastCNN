# FastConv
We can run FastConv on android mobile phones with Snapdragon 835, Snapdragon 855, Snapdragon888, ARM server with Kunpeng 920 and MacBook with Apple M1. This description sketches how to obtain the corresponding software package needed, source code, and how to compile source code to get the reported performance.

## Hardware dependencies
This program requires hardwares support armv8 aarch64 architecture.
* MacBook with Apple M1 processor
* ARM Server with Kunpeng 920
* X86 Server usb-connected with mobile phones with Snapdragon 835, 855 or 888

## Software dependencies
* Operating system: 64-bit Linux, macOS and Android
* Compiler: Apple Clang++/GNU G++, and Android Clang++
* Other development software: If used on Android, Android NDK and ADB are necessary. Android NDK is used to cross-compile the program on the server, and Android ADB is used to upload the binary executable file to the mobile device and remotely debug the program on the server.

## Installation
#### Android NDK: 
Please refer to Android NDK Document (https://developer.android.com/ndk/guides) for installation.
#### Android ADB:
Please refer to Android ADB Document
(https://developer.android.com/studio/command-line/adb) for installation.
#### This software:
```bash
git clone 
cd FastConv/
make
```

## Experiment workflow
### Running this software on Linux or MacOS:
-a is used to select algorithm \
-tn is used to decide whether to autotune \
-i is used to decide the number of iterations
##### Layer-wise Evaluation on VGG-16:
```bash
# Algorithm is automatically selected, no tuning, use default parameters, No. iterations is 10
./run_vgg16.sh -a auto -tn no_tuning -i 10
# Algorithm is winograd, no tuning, use default parameters, No. iterations is 10
./run_vgg16.sh -a winograd -tn no_tuning -i 10
# Algorithm is Im2col, no tuning, use default parameters, No. iterations is 10
./run_vgg16.sh -a im2col -tn no_tuning -i 10
# Algorithm is automatically selected, tuning, No. iterations is 10
./run_vgg16.sh -a auto -tn tuning -i 10
# Algorithm is winograd, tuning, No. iterations is 10
./run_vgg16.sh -a winograd -tn tuning -i 10
# Algorithm is Im2col, tuning, No. iterations is 10
./run_vgg16.sh -a im2col -tn tuning -i 10
```
##### Layer-wise Evaluation on ResNet-50:
```bash
# Algorithm is automatically selected, no tuning, use default parameters, No. iterations is 10
./run_resnet50.sh -a auto -tn no_tuning -i 10
# Algorithm is winograd, no tuning, use default parameters, No. iterations is 10
./run_resnet50.sh -a winograd -tn no_tuning -i 10
# Algorithm is Im2col, no tuning, use default parameters, No. iterations is 10
./run_resnet50.sh -a im2col -tn no_tuning -i 10
# Algorithm is automatically selected, tuning, No. iterations is 10
./run_resnet50.sh -a auto -tn tuning -i 10
# Algorithm is winograd, tuning, No. iterations is 10
./run_resnet50.sh -a winograd -tn tuning -i 10
# Algorithm is Im2col, tuning, No. iterations is 10
./run_resnet50.sh -a im2col -tn tuning -i 10
```
### Running this software on Android:
First, upload the binary executable file and running scripts to the Android device.
```bash
# Upload the binary executable file
adb -s (Device_id) push ./winograd_dev $(PathinAndroiddevice)
# Upload the running scripts
adb -s (Device_id) push run_vgg16.sh $(PathinAndroiddevice)
adb -s (Device_id) push run_resnet50.sh $(PathinAndroiddevice)
# Connect to android device
adb -s (Device_id) shell
```
Then, the method of running this software is similar to running on Linux.


