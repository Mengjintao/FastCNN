#ifndef WINOF63_H_
#define WINOF63_H_

#pragma once

#include <stdio.h>



int winoF63_naive(float *baseResult, float *testInput, float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int padWidth, int padHeight, int num_threads);

void retransformKernel2(float* kernel, int outputChannels, int inputChannels, int ocBlock, int icBlock, int ocRegBlock);
void retransformKernel(float *testKernel, int oc, int ic, int ocBlock);

void offlineKernelTransform(float *offlineKernel, float* testKernel, int outputChannels, int inputChannels, int ocBlock, int ocRegBlock);

int winoF63(float *baseResult, float *testInput, const float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int pad_width, int pad_height, int stride_width, int stride_height, int tileBlock, float *bufBlock, int ocBlock, float *kernelBlock, int icBlock, float *inputBuf, int tileRegBlock, int ocRegBlock, int enableOffKernel, int num_threads);

#endif
