#ifndef WINOF63ZC_H_
#define WINOF63ZC_H_
#include "../utility/helper.h"

int winoF63_v1_single(float *baseResult, float *testInput, float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int padWidth, int padHeight, int strideWidth, int strideHeight, int tileBlock, float *buf, int ocBlock, float *kernelBuf, int icBlock, float *inputBuf, int tileRegBlock, int ocRegBlock, int scheduling, int num_threads,
                    Timer& inputTran, Timer& kernelTran, Timer& GEMM, Timer& outputTran);

int winoF63_v1_multi(float *baseResult, float *testInput, float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int padWidth, int padHeight, int strideWidth, int strideHeight, int tileBlock, float *buf, int ocBlock, float *kernelBuf, int icBlock, float *inputBuf, int tileRegBlock, int ocRegBlock, int scheduling, int num_threads,
                    Timer& inputTran, Timer& kernelTran, Timer& GEMM, Timer& outputTran);

int winoF63_v1_multi_v1(float *baseResult, float *testInput, float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int padWidth, int padHeight, int strideWidth, int strideHeight, int tileBlock, float *buf, int ocBlock, float *kernelBuf, int icBlock, float *inputBuf, int tileRegBlock, int ocRegBlock, int scheduling, int num_threads,
                    Timer& inputTran, Timer& kernelTran, Timer& GEMM, Timer& outputTran);

void offline_kernel_transform_v1(float *offlineKernel, float* testKernel, int outputChannels, int inputChannels, int ocBlock, int ocRegBlock, int icBlock);

#endif