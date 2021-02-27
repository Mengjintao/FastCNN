#ifndef TensorGEMM_H_
#define TensorGEMM_H_

#pragma once

#include <stdio.h>

void TensorGEMMInnerKernel4x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel4x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel4x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel4x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel4x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);

void TensorGEMMInnerKernel5x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel5x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel5x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel5x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);

void TensorGEMMInnerKernel7x6x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel7x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel7x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel7x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel7x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel7x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);

void TensorGEMMInnerKernel6x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel6x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel6x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel6x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel6x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);

void TensorGEMMInnerKernel3x7x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel3x6x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel3x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel3x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel3x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel3x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel3x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);

void TensorGEMMInnerKernel2x10x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);
void TensorGEMMInnerKernel2x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride);

#endif
