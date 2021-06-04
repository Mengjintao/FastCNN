#ifndef TENSORGEMMZC_H_
#define TENSORGEMMZC_H_

void TensorGEMMInnerKernel4x4x4_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);
void TensorGEMMInnerKernel4x3x4_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);
void TensorGEMMInnerKernel4x2x4_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);
void TensorGEMMInnerKernel4x1x4_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);

void TensorGEMMInnerKernel4x4x4_v2(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);
void TensorGEMMInnerKernel4x4x4_v3(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);

void TensorGEMMInnerKernel4x4x5_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);
void TensorGEMMInnerKernel4x3x5_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);
void TensorGEMMInnerKernel4x2x5_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);
void TensorGEMMInnerKernel4x1x5_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);

void TensorGEMMInnerKernel4x4x5_v2(float* WTp, const float* UTp, float* vp, int inChannels, int tileN);
#endif