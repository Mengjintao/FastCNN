#CXX = /tmp/ndk-jtmeng/aarch64-linux-android/bin/clang++  
#CXX = ~/software/arm/bin/clang++ 
#CXX ?= ~/software/arm/bin/clang++ 
CXX = g++

#CXX_SRC=winograd_kernels.cpp  winograd_kernels_F63.cpp winoF63.cpp helper.cpp TensorGEMM.cpp ConvTest.cpp ConvKernels.cpp
CXX_SRC=./winoF63/winoF63.cpp ./utility/helper.cpp ./utility/common.cpp ./winoF63/TensorGEMM.cpp ConvTest.cpp ./ncnn/ConvKernels.cpp
CXX_SRC+=./im2col/gemm_kernel.cpp ./im2col/gemm_pack.cpp ./convLayer/im2colConv.cpp

CXX_FLAGS = -std=c++11 $(DEFS) -Wno-format -I$(PWD) -O3 -fopenmp
LD_FLAGS= -pie -pthread -std=c++11 -fopenmp
CXX_FLAGS+=-march=armv8-a

#LD_FLAGS= -pie -pthread -fopenmp -static-libstdc++ -static-openmp -O3
#CXX_FLAGS+=--target=aarch64-none-linux-android21 --gcc-toolchain=/home/jtmeng/software/Android-NDK/android-ndk-r21b/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/jtmeng/software/Android-NDK/android-ndk-r21b/toolchains/llvm/prebuilt/linux-x86_64/sysroot -mcpu=cortex-a55
#CXX_FLAGS+=--target=aarch64-none-linux-android21 --gcc-toolchain=/home/jtmeng/software/Android-NDK/android-ndk-r21b/toolchains/llvm/prebuilt/linux-x86_64 --sysroot=/home/jtmeng/software/Android-NDK/android-ndk-r21b/toolchains/llvm/prebuilt/linux-x86_64/sysroot -march=armv8-a
EXEC=winograd_dev

OBJ_DIR=./obj
OBJ = $(patsubst %.cpp, $(OBJ_DIR)/%.cpp.o, $(CXX_SRC))

all:dir $(OBJ) 
	$(CXX) $(DEFS) $(LD_FLAGS) $(OBJ_DIR)/*.o $(OBJ_DIR)/utility/*.o $(OBJ_DIR)/winoF63/*.o $(OBJ_DIR)/ncnn/*.o  -o $(EXEC)

$(OBJ_DIR)/%.cpp.o:%.cpp
	$(CXX) -c $(CXX_FLAGS) $(DEFS) $< -o $@
dir: 
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/utility
	mkdir -p $(OBJ_DIR)/winoF63
	mkdir -p $(OBJ_DIR)/ncnn
clean :
	rm -rf $(EXEC) $(OBJ_DIR)
