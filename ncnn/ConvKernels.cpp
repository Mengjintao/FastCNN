//
//  BasicConv.cpp
//  WinogradConvolution
//
//

#include "ConvKernels.hpp"
#include <string.h>
#include <stdlib.h>
#include <arm_neon.h>
#include <assert.h>

void conv3x3s1_neon(float *rawInput, int input_channels, struct nnp_size inputDim,
                    float* kernel, struct nnp_size kernelDim,
                    float* output, int output_channels, struct nnp_size outputDim,
                    struct nnp_padding paddingSize, struct nnp_size subsampling)
{
    int paddedWidth  = (int)(inputDim.width + paddingSize.left + paddingSize.right);
    int paddedHeight = (int)(inputDim.height + paddingSize.top + paddingSize.bottom);
    
    float* paddedInput = (float *) malloc(paddedWidth * paddedHeight * sizeof(float) * input_channels);
    memset(paddedInput, 0, paddedWidth * paddedHeight * sizeof(float) * input_channels);
    int channelId = 0;
    for(; channelId < input_channels; ++channelId){
        //Beginning position in each input channel.
        float* padPtr = paddedInput + channelId * paddedWidth * paddedHeight + paddedWidth * paddingSize.top + paddingSize.left;
        float* inPtr  = rawInput    + channelId * inputDim.height * inputDim.width;
        for(int i = 0; i < inputDim.height; ++i){
            memcpy(padPtr, inPtr, sizeof(float) * inputDim.width);
            padPtr += paddedWidth;
            inPtr  += inputDim.width;
        }
    }
    int w    = paddedWidth;
    //    int h    = paddedHeight;
    int inch = (int) input_channels;
    
    int outw  = (int) outputDim.width;
    int outh  = (int) outputDim.height;
    int outch = (int) output_channels;
    
    //    const float* bias = _bias;
    
    //#pragma omp parallel for
    for (int p=0; p<outch; p++)
    {
        //        Mat out = top_blob.channel(p);
        //        const float bias0 = bias ? bias[p] : 0.f;
        //        out.fill(bias0);
        float* out = output + p * outw * outh;
        memset(out, 0, sizeof(float) * outputDim.height * outputDim.width);
        
        const float* kernel0 = kernel + p * inch * 9;
        for (int q=0; q<inch; q++)
        {
            float* outptr = out;
            float* outptr2 = outptr + outw;
            
            //            const float* img0 = bottom_blob.channel(q);
            float* img0 = paddedInput + paddedHeight * paddedWidth * q;
            
            const float* r0 = img0;
            const float* r1 = img0 + w;
            const float* r2 = img0 + w * 2;
            const float* r3 = img0 + w * 3;
            
            const float* k0 = kernel0;
            const float* k1 = kernel0 + 3;
            const float* k2 = kernel0 + 6;
            
#if __ARM_NEON
            float32x4_t _k0123 = vld1q_f32(kernel0);
            float32x4_t _k3456 = vld1q_f32(kernel0+3);
            float32x4_t _k6789 = vld1q_f32(kernel0+6);
#endif // __ARM_NEON
            
            int i = 0;
            
            for (; i+1 < outh; i+=2)
            {
                
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
#endif // __ARM_NEON
                
#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum1 = vld1q_f32(outptr);
                    float32x4_t _sum2 = vdupq_n_f32(0.f);
                    float32x4_t _sum3 = vld1q_f32(outptr2);
                    float32x4_t _sum4 = vdupq_n_f32(0.f);
                    
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);
                    
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);
                    
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);
                    
                    float32x4_t _r30 = vld1q_f32(r3);
                    float32x4_t _r30n = vld1q_f32(r3 + 4);
                    float32x4_t _r31 = vextq_f32(_r30, _r30n, 1);
                    float32x4_t _r32 = vextq_f32(_r30, _r30n, 2);
                    
                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k0123, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r01, _k0123, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k0123, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k3456, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k3456, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k3456, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k6789, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k6789, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k6789, 2);
                    
                    _sum3 = vfmaq_laneq_f32(_sum3, _r10, _k0123, 0);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r11, _k0123, 1);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r12, _k0123, 2);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r20, _k3456, 0);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r21, _k3456, 1);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r22, _k3456, 2);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r30, _k6789, 0);
                    _sum4 = vfmaq_laneq_f32(_sum4, _r31, _k6789, 1);
                    _sum3 = vfmaq_laneq_f32(_sum3, _r32, _k6789, 2);
                    
                    _sum1 = vaddq_f32(_sum1, _sum2);
                    _sum3 = vaddq_f32(_sum3, _sum4);
                    
                    vst1q_f32(outptr, _sum1);
                    vst1q_f32(outptr2, _sum3);
                    
                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    r3 += 4;
                    outptr += 4;
                    outptr2 += 4;
                }
#endif // __aarch64__
#endif // __ARM_NEON
                
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r30 = vld1q_f32(r3);
                    
                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);
                    
                    float32x4_t _sum2 = vmulq_f32(_r10, _k0123);
                    _sum2 = vmlaq_f32(_sum2, _r20, _k3456);
                    _sum2 = vmlaq_f32(_sum2, _r30, _k6789);
                    
                    _sum = vsetq_lane_f32(*outptr, _sum, 3);
                    _sum2 = vsetq_lane_f32(*outptr2, _sum2, 3);
                    
#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
                    *outptr2 = vaddvq_f32(_sum2);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    float32x2_t _ss2 = vadd_f32(vget_low_f32(_sum2), vget_high_f32(_sum2));
                    
                    float32x2_t _sss2 = vpadd_f32(_ss, _ss2);
                    
                    *outptr = vget_lane_f32(_sss2, 0);
                    *outptr2 = vget_lane_f32(_sss2, 1);
#endif // __aarch64__
#else
                    //                    int remain = outw;
                    //                    float sum = 0;
                    //                    float sum2 = 0;
                    
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    
                    sum2 += r1[0] * k0[0];
                    sum2 += r1[1] * k0[1];
                    sum2 += r1[2] * k0[2];
                    sum2 += r2[0] * k1[0];
                    sum2 += r2[1] * k1[1];
                    sum2 += r2[2] * k1[2];
                    sum2 += r3[0] * k2[0];
                    sum2 += r3[1] * k2[1];
                    sum2 += r3[2] * k2[2];
                    
                    *outptr += sum;
                    *outptr2 += sum2;
#endif
                    r0++;
                    r1++;
                    r2++;
                    r3++;
                    outptr++;
                    outptr2++;
                }
                
                r0 += 2 + w;
                r1 += 2 + w;
                r2 += 2 + w;
                r3 += 2 + w;
                
                outptr += outw;
                outptr2 += outw;
            }
            
            //r + 1 is outh.
            //odd row.
            for (; i < outh; i++)
            {
                
#if __ARM_NEON
                int nn = outw >> 2;
                int remain = outw & 3;
#else
                int remain = outw;
#endif // __ARM_NEON
                
#if __ARM_NEON
#if __aarch64__
                for (; nn>0; nn--)
                {
                    float32x4_t _sum1 = vld1q_f32(outptr);
                    float32x4_t _sum2 = vdupq_n_f32(0.f);
                    
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r00n = vld1q_f32(r0 + 4);
                    float32x4_t _r01 = vextq_f32(_r00, _r00n, 1);
                    float32x4_t _r02 = vextq_f32(_r00, _r00n, 2);
                    
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r10n = vld1q_f32(r1 + 4);
                    float32x4_t _r11 = vextq_f32(_r10, _r10n, 1);
                    float32x4_t _r12 = vextq_f32(_r10, _r10n, 2);
                    
                    float32x4_t _r20 = vld1q_f32(r2);
                    float32x4_t _r20n = vld1q_f32(r2 + 4);
                    float32x4_t _r21 = vextq_f32(_r20, _r20n, 1);
                    float32x4_t _r22 = vextq_f32(_r20, _r20n, 2);
                    
                    _sum1 = vfmaq_laneq_f32(_sum1, _r00, _k0123, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r01, _k0123, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r02, _k0123, 2);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r10, _k3456, 0);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r11, _k3456, 1);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r12, _k3456, 2);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r20, _k6789, 0);
                    _sum2 = vfmaq_laneq_f32(_sum2, _r21, _k6789, 1);
                    _sum1 = vfmaq_laneq_f32(_sum1, _r22, _k6789, 2);
                    
                    _sum1 = vaddq_f32(_sum1, _sum2);
                    
                    vst1q_f32(outptr, _sum1);
                    
                    r0 += 4;
                    r1 += 4;
                    r2 += 4;
                    outptr += 4;
                }
#endif
#endif // __ARM_NEON
                for (; remain>0; remain--)
                {
#if __ARM_NEON
                    float32x4_t _r00 = vld1q_f32(r0);
                    float32x4_t _r10 = vld1q_f32(r1);
                    float32x4_t _r20 = vld1q_f32(r2);
                    
                    float32x4_t _sum = vmulq_f32(_r00, _k0123);
                    _sum = vmlaq_f32(_sum, _r10, _k3456);
                    _sum = vmlaq_f32(_sum, _r20, _k6789);
                    
                    _sum = vsetq_lane_f32(*outptr, _sum, 3);
                    
#if __aarch64__
                    *outptr = vaddvq_f32(_sum);
#else
                    float32x2_t _ss = vadd_f32(vget_low_f32(_sum), vget_high_f32(_sum));
                    _ss = vpadd_f32(_ss, _ss);
                    
                    *outptr = vget_lane_f32(_ss, 0);
#endif // __aarch64__
#else
                    float sum = 0;
                    
                    sum += r0[0] * k0[0];
                    sum += r0[1] * k0[1];
                    sum += r0[2] * k0[2];
                    sum += r1[0] * k1[0];
                    sum += r1[1] * k1[1];
                    sum += r1[2] * k1[2];
                    sum += r2[0] * k2[0];
                    sum += r2[1] * k2[1];
                    sum += r2[2] * k2[2];
                    
                    *outptr += sum;
#endif
                    r0++;
                    r1++;
                    r2++;
                    outptr++;
                }
                
                r0 += 2;
                r1 += 2;
                r2 += 2;
            }
            
            kernel0 += 9;
        }
    }
    free(paddedInput);
}
