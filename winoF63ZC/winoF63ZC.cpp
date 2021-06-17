#include "./winoF63ZC.h"
#include <stdlib.h>
#include <arm_neon.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "../utility/helper.h"
#include "../utility/common.h"
#include "../winoF63/TensorGEMM.h"
#include "./TensorGEMMZC.h"

static inline void winograd_f6k3_kernel_transform_v1(
    const float32x4_t g0, const float32x4_t g1, const float32x4_t g2,
    float32x4_t &w0,
    float32x4_t &w1,
    float32x4_t &w2,
    float32x4_t &w3,
    float32x4_t &w4,
    float32x4_t &w5,
    float32x4_t &w6,
    float32x4_t &w7,
    float32x4_t f4,
    float32x4_t f2)
{
    /*
     * w0 = g0
     * w1 = ((g0 + g2) + g1) * (-2.0 / 9)
     * w2 = ((g0 + g2) - g1) * (-2.0 / 9)
     * w3 = ((g0 + 4 * g2) + 2 * g1) * (1.0 / 90)
     * w4 = ((g0 + 4 * g2) - 2 * g1) * (1.0 / 90)
     * w5 = ((g2 + 4 * g0) + 2 * g1) * (1.0 / 180)
     * w6 = ((g2 + 4 * g0) - 2 * g1) * (1.0 / 180)
     * w7 = g2
     */

    /*
     * Compute
     *   w2 := g0 + g2
     *   w4 := g0 + 4 * g2
     *   w6 := g2 + 4 * g0
     */
    w0 = g0;
    w7 = g2;
    w2 = g0 + g2;
    w4 = g0 + f4*g2;
    w6 = g2 + f4*g0;

    /*
     * Compute
     *   w1 = (g0 + g2) + g1
     *   w2 = (g0 + g2) - g1
     *   w3 = (g0 + 4 * g2) + 2 * g1
     *   w4 = (g0 + 4 * g2) - 2 * g1
     *   w5 = (g2 + 4 * g0) + 2 * g1
     *   w6 = (g2 + 4 * g0) - 2 * g1
     */
    const float32x4_t two_g1 = g1 *f2;
    w1 = w2 + g1;
    w2 = w2 - g1;
    w3 = w4 + two_g1;
    w4 = w4 - two_g1;
    w5 = w6 + two_g1;
    w6 = w6 - two_g1;

//    if (rescale_coefficients) {
    const float32x4_t minus_2_over_9 = vdupq_n_f32(-2.0/9);
    w1 *= minus_2_over_9;
    w2 *= minus_2_over_9;

    const float32x4_t rcp_90 = vdupq_n_f32(1.0/90);
    w3 *= rcp_90;
    w4 *= rcp_90;

    const float32x4_t rcp_180 = vdupq_n_f32(1.0/180);
    w5 *= rcp_180;
    w6 *= rcp_180;
}

static inline void neon_transpose4x4_inplace_f32_cpp_v1(
    float32x4_t &row0,
    float32x4_t &row1,
    float32x4_t &row2,
    float32x4_t &row3)
{
    /*
     * row0 = ( x00 x01 x02 x03 )
     * row1 = ( x10 x11 x12 x13 )
     * row2 = ( x20 x21 x22 x23 )
     * row3 = ( x30 x31 x32 x33 )
     */
    /*
     * row01 = ( x00 x10 x02 x12 ), ( x01 x11 x03, x13 )
     * row23 = ( x20 x30 x22 x32 ), ( x21 x31 x23, x33 )
     */
    float32x4x2_t row01 = vtrnq_f32(row0, row1);
    float32x4x2_t row23 = vtrnq_f32(row2, row3);

    /*
     * row0 = ( x00 x10 x20 x30 )
     * row1 = ( x01 x11 x21 x31 )
     * row2 = ( x02 x12 x22 x32 )
     * row3 = ( x03 x13 x23 x33 )
     */
    row0 = vcombine_f32(vget_low_f32(row01.val[0]), vget_low_f32(row23.val[0]));
    row1 = vcombine_f32(vget_low_f32(row01.val[1]), vget_low_f32(row23.val[1]));
    row2 = vcombine_f32(vget_high_f32(row01.val[0]), vget_high_f32(row23.val[0]));
    row3 = vcombine_f32(vget_high_f32(row01.val[1]), vget_high_f32(row23.val[1]));
}

static inline void input_transform_v1(
    float32x4_t &r0,
    float32x4_t &r1,
    float32x4_t &r2,
    float32x4_t &r3,
    float32x4_t &r4,
    float32x4_t &r5,
    float32x4_t &r6,
    float32x4_t &r7,
    float32x4_t &t1,
    float32x4_t &t2,
    float32x4_t &s1,
    float32x4_t &s2,
    float32x4_t &p1,
    float32x4_t &p2,
    const float32x4_t &f5_25,
    const float32x4_t &f4_25,
    const float32x4_t &f4,
    const float32x4_t &f2_5,
    const float32x4_t &f2,
    const float32x4_t &f1_25,
    const float32x4_t &f0_5,
    const float32x4_t &f0_25
)
{
    r0 = r0 - r6 + (r4 - r2) * f5_25;
    r7 = r7 - r1 + (r3 - r5) * f5_25;

    //r6 - r4 * f5_25 can be reused
    //r1 - r3 * f5_25 can be reused

    t1 = r2 + r6 - r4 * f4_25;
    t2 = r1 + r5 - r3 * f4_25;

    s1 = r4 * f1_25;
    s2 = r3 * f2_5;


    p1 = r6      + r2*f0_25 - s1;
    p2 = r1*f0_5 - s2       + r5*f2;

    r3 = p1 + p2;
    r4 = p1 - p2;


    //2.5 * (r01 - r03 + r05)

    p1 = r6 + (r2 - s1) * f4;
    p2 = r1 * f2 - s2 + r5 * f0_5;

    r5 = p1 + p2;
    r6 = p1 - p2;

    r1 = t1+t2;
    r2 = t1-t2;
//    r1 = vaddq_f32(t1, t2);
//    r2 = vsubq_f32(t1, t2);
}

static inline void winograd_f6k3_output_transform_inplace_v1(
    float32x4_t &m0,
    float32x4_t &m1,
    float32x4_t &m2,
    float32x4_t &m3,
    float32x4_t &m4,
    float32x4_t &m5,
    float32x4_t &m6,
    float32x4_t &m7)
{
    /*
     * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6)
     * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6)
     * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6)
     * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6)
     * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6)
     * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) + m7
     */

    const float32x4_t m1_add_m2 = m1 + m2;
    const float32x4_t m1_sub_m2 = m1 - m2;
    const float32x4_t m3_add_m4 = m3 + m4;
    const float32x4_t m3_sub_m4 = m3 - m4;
    const float32x4_t m5_add_m6 = m5 + m6;
    const float32x4_t m5_sub_m6 = m5 - m6;

    // Finised with M[0-6] as **inputs** here.
    m0 = m0 + m1_add_m2;
    m5 = m7 + m1_sub_m2;
    // Finised with M[0-7] as **inputs** here.

    const float32x4_t const_16 = vdupq_n_f32(16.0f);
    m1 = vfmaq_f32(m1_sub_m2, const_16, m5_sub_m6);
    m4 = vfmaq_f32(m1_add_m2, const_16, m3_add_m4);

    const float32x4_t const_8 = vdupq_n_f32(8.0f);
    m2 = vfmaq_f32(m1_add_m2, const_8, m5_add_m6);
    m3 = vfmaq_f32(m1_sub_m2, const_8, m3_sub_m4);

    const float32x4_t const_32 = vdupq_n_f32(32.0f);
    m0 = vfmaq_f32(m0, const_32, m5_add_m6);
    m0 += m3_add_m4;

    m5 = vfmaq_f32(m5, const_32, m3_sub_m4);
    m5 += m5_sub_m6;

    const float32x4_t const_2 = vdupq_n_f32(2.0f);
    m1 = vfmaq_f32(m1, m3_sub_m4, const_2);
    m4 = vfmaq_f32(m4, m5_add_m6, const_2);

    const float32x4_t const_4 = vdupq_n_f32(4.0f);
    m2 = vfmaq_f32(m2, m3_add_m4, const_4);
    m3 = vfmaq_f32(m3, m5_sub_m6, const_4);

    const float32x4_t const_0 = vdupq_n_f32(0.0f);
    m6 = const_0;
    m7 = const_0;
}


void printMatrix_v1(float* matrix, int row, int col)
{
	printf("--------------\n");
	for(int i=0;i<row;i++)	
	{
		for(int j=0;j<col;j++)
			printf("%3.3f ", *(matrix+i*col+j));
		printf("\n");
	}
}

void offline_kernel_transform_v1(float *offlineKernel, float* testKernel, int outputChannels, int inputChannels, int ocBlock, int ocRegBlock, int icBlock) {
    float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    float32x4_t m1, m2, s1, s2, t1, t2;//Auxiliary registers
    const float32x4_t vZero = vdupq_n_f32(0.0f);

    const float32x4_t f4 = vdupq_n_f32(4.0f);
    const float32x4_t f2 = vdupq_n_f32(2.0f);

    int ic_block = icBlock;
    int oc_block = ocBlock;

    // memset(offlineKernel, 0, 64 * (outputChannels) * inputChannels * sizeof(float));
    for (int ic_cblock_idx = 0; ic_cblock_idx < inputChannels; ic_cblock_idx += ic_block) {
        // kernelTran.startBench();
        int ic_cblock_step = min(ic_block, inputChannels - ic_cblock_idx);
        float* kernelBuf_copy = offlineKernel + 64 * align_ceil(outputChannels, ocBlock) * ic_cblock_idx;
        // float* kernel_copy = testKernel + 9 * ic_cblock_idx;
        for (int oc_cblock_idx = 0; oc_cblock_idx < outputChannels; oc_cblock_idx += oc_block) {
            // 确定一个panel的高度
            int oc_cblock_step = min(oc_block, outputChannels - oc_cblock_idx);
            for (int oc_rblock_idx = 0; oc_rblock_idx < oc_cblock_step; oc_rblock_idx += ocRegBlock) {
                int oc_rblock_step = min(ocRegBlock, oc_cblock_step - oc_rblock_idx);
                float* kernelBuf_copy_copy = kernelBuf_copy + 64 * oc_cblock_idx * ic_block + 4 * oc_rblock_idx * ic_block;
                float* kernel_copy_copy = testKernel + (oc_cblock_idx + oc_rblock_idx) * inputChannels * 9;
                int tk_step = oc_cblock_step * ic_block * 4;
                float* tk0 = kernelBuf_copy_copy;
                float* tk1  = tk0  + tk_step;
                float* tk2  = tk1  + tk_step;
                float* tk3  = tk2  + tk_step;
                float* tk4  = tk3  + tk_step;
                float* tk5  = tk4  + tk_step;
                float* tk6  = tk5  + tk_step;
                float* tk7  = tk6  + tk_step;
                float* tk8  = tk7  + tk_step;
                float* tk9  = tk8  + tk_step;
                float* tk10 = tk9  + tk_step;
                float* tk11 = tk10 + tk_step;
                float* tk12 = tk11 + tk_step;
                float* tk13 = tk12 + tk_step;
                float* tk14 = tk13 + tk_step;
                float* tk15 = tk14 + tk_step;

                for (int ic = 0; ic < ic_cblock_step; ic++) {
                    for (int oc = 0; oc < oc_rblock_step; oc++) {
                        float* kernel_ptr = kernel_copy_copy + oc * inputChannels * 9 + (ic_cblock_idx + ic) * 9;
                        l0 = vld1q_f32(kernel_ptr);
                        l1 = vld1q_f32(kernel_ptr + 3);
                        l2 = vld1q_f32(kernel_ptr + 6);

                        // kernel transform
                        winograd_f6k3_kernel_transform_v1(l0, l1, l2, m1, m2, s1, l7, t1, t2, s2, r7, f4, f2);
                        neon_transpose4x4_inplace_f32_cpp_v1(m1, m2, s1, l7);
                        neon_transpose4x4_inplace_f32_cpp_v1(t1, t2, s2, r7);
                        winograd_f6k3_kernel_transform_v1(m1, m2, s1,  l0, l1, l2, l3, l4, l5, l6, l7, f4, f2);
                        winograd_f6k3_kernel_transform_v1(t1, t2, s2,  r0, r1, r2, r3, r4, r5, r6, r7, f4, f2);

                        vst1q_f32(tk0,   l0);   tk0  +=4;
                        vst1q_f32(tk1,   r0);   tk1  +=4;
                        vst1q_f32(tk2,   l1);   tk2  +=4;
                        vst1q_f32(tk3,   r1);   tk3  +=4;
                        vst1q_f32(tk4,   l2);   tk4  +=4;
                        vst1q_f32(tk5,   r2);   tk5  +=4;
                        vst1q_f32(tk6,   l3);   tk6  +=4;
                        vst1q_f32(tk7,   r3);   tk7  +=4;
                        vst1q_f32(tk8,   l4);   tk8  +=4;
                        vst1q_f32(tk9,   r4);   tk9  +=4;
                        vst1q_f32(tk10,  l5);   tk10 +=4;
                        vst1q_f32(tk11,  r5);   tk11 +=4;
                        vst1q_f32(tk12,  l6);   tk12 +=4;
                        vst1q_f32(tk13,  r6);   tk13 +=4;
                        vst1q_f32(tk14,  l7);   tk14 +=4;
                        vst1q_f32(tk15,  r7);   tk15 +=4;
                    }
                }
            }
        }
        // kernelTran.accumBench();
    }
    // printMatrix_v1(offlineKernel,  64*outputChannels, inputChannels);
}

int winoF63_v1_single(float *baseResult, float *testInput, float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int padWidth, int padHeight, int strideWidth, int strideHeight, int tileBlock, float *gemmBuf, int ocBlock, float *kernelBuf, int icBlock, float *inputBuf, int tileRegBlock, int ocRegBlock, int scheduling, int num_threads,
                    Timer& inputTran, Timer& kernelTran, Timer& GEMM, Timer& outputTran)
{
    /*
        outputBuf: tileBlock*ocBlock*48
        kernelBuf: icBlock*ocBlock*64
        inputBuf:  tileBlock*icBlock*64
    */
    int enableOffKernel = scheduling%2;
    int loopOrdering    = scheduling/2;
    int outputHeight = inputHeight+ padHeight*2 -2;
    int outputWidth  = inputWidth + padWidth*2  -2;
    int tileH = (inputHeight+ padHeight*2 -2 + 5)/6;
    int tileW = (inputWidth + padWidth*2  -2 + 5)/6;
    int tileN = tileH*tileW;

    // assert(inputChannels%icBlock  == 0);
    assert(tileBlock%tileRegBlock == 0);
    assert(ocBlock%ocRegBlock     == 0);


    // Timer inputTran, kernelTran, GEMM, outputTran, storeBack;

    const float32x4_t f5    = vdupq_n_f32(5.0f);
    const float32x4_t f4    = vdupq_n_f32(4.0f);
    const float32x4_t f2    = vdupq_n_f32(2.0f);
    const float32x4_t f2_5  = vdupq_n_f32(2.5f);
    const float32x4_t f5_25 = vdupq_n_f32(5.25f);
    const float32x4_t f4_25 = vdupq_n_f32(4.25f);
    const float32x4_t f1_25 = vdupq_n_f32(1.25f);
    const float32x4_t f0_5  = vdupq_n_f32(0.5f);
    const float32x4_t f0_25 = vdupq_n_f32(0.25f);
    const float32x4_t vZero = vdupq_n_f32(0.0f);

    float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    float32x4_t m1, m2, s1, s2, t1, t2;//Auxiliary registers

    int ic_block = icBlock;
    int oc_block = ocBlock;

    void (*tensorGEMM)(float *, const float *, float *, int, int); 
    void (*tensorGEMM_for_corner_case)(float *, const float *, float *, int, int);
    if (ocRegBlock == 4) {
        if (tileRegBlock == 4) {
            tensorGEMM = TensorGEMMInnerKernel4x4x4_v3;
            // tensorGEMM = TensorGEMMInnerKernel4x4x4_v1;
            int remain = outputChannels % ocRegBlock;
            if (remain == 3) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x3x4_v1;
            } else if (remain == 2) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x2x4_v1;
            } else if (remain == 1) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x1x4_v1;
            }
        } else if (tileRegBlock == 5) {
            // tensorGEMM = TensorGEMMInnerKernel4x4x5_v1;
            tensorGEMM = TensorGEMMInnerKernel4x4x5_v2;
            int remain = outputChannels % ocRegBlock;
            if (remain == 3) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x3x5_v1;
            } else if (remain == 2) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x2x5_v1;
            } else if (remain == 1) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x1x5_v1;
            }
        }
    }
    

    // float* transform_kernel = static_cast<float*>(_mm_malloc((64 * outputChannels * ic_block + 128) * sizeof(float)));
    // float* transform_input  = static_cast<float*>(_mm_malloc((64 * ic_block * align_ceil(tileBlock, tileRegBlock) + 128) * sizeof(float)));
    // float* gemm_buf         = static_cast<float*>(_mm_malloc((64 * outputChannels * tileN + 128) * sizeof(float)));
    // #pragma omp parallel 
    int oc_cblock_num = (outputChannels - 1) / oc_block + 1;
    int tile_cblock_num = (tileN - 1) / tileBlock + 1; 
    for (int ic_cblock_idx = 0; ic_cblock_idx < inputChannels; ic_cblock_idx += ic_block) {
        int ic_cblock_step = min(ic_block, inputChannels - ic_cblock_idx);
        // printf("ic_cblock_idx = %d\n", ic_cblock_idx); 
        int pre_oc_cblock_idx = -1, pre_tile_cblock_idx = -1;
        bool is_kernel_transform = true, is_input_transform = true;
        for (int cblock_idx = 0; cblock_idx < oc_cblock_num * tile_cblock_num; cblock_idx++) {
            int oc_cblock_idx, tile_cblock_idx;
            if (loopOrdering) { // outer loop is tile
                oc_cblock_idx   = cblock_idx % oc_cblock_num * oc_block;
                tile_cblock_idx = cblock_idx / oc_cblock_num * tileBlock;

                is_kernel_transform = (tile_cblock_idx == 0 ? true: false);
                is_input_transform  = (pre_tile_cblock_idx == tile_cblock_idx ? false: true);
                pre_tile_cblock_idx = tile_cblock_idx;
            } else { // outer loop is oc
                oc_cblock_idx   = cblock_idx / tile_cblock_num * oc_block;
                tile_cblock_idx = cblock_idx % tile_cblock_num * tileBlock;
                
                is_input_transform  = (oc_cblock_idx == 0 ? true: false);
                is_kernel_transform = (pre_oc_cblock_idx == oc_cblock_idx ? false: true);
                pre_oc_cblock_idx = oc_cblock_idx;
            }

            int oc_cblock_end     = min(oc_cblock_idx + oc_block, outputChannels);
            int oc_cblock_begin   = oc_cblock_idx;
            int oc_cblock_step    = oc_cblock_end - oc_cblock_begin;
            int tile_cblock_end   = min(tile_cblock_idx + tileBlock, tileN);
            int tile_cblock_begin = tile_cblock_idx;
            int tile_cblock_step  = tile_cblock_end - tile_cblock_begin;
            int tile_cblock_step_align = align_ceil(tile_cblock_step, tileRegBlock);

            // printf("oc_block_idx = %d, tile_block_idx = %d\n", oc_cblock_idx, tile_cblock_idx);

            if (is_kernel_transform) {
                if (!enableOffKernel) {
                    kernelTran.startBench();
                    // memset(kernelBuf, 0, (64 * outputChannels * ic_block + 128) * sizeof(float));
                    // #pragma omp parallel for 
                    int tk_step = oc_cblock_step * ic_block * 4;
                    // #pragma omp parallel for collapse(2) private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
                    // for (int oc_cblock_idx = 0; oc_cblock_idx < outputChannels; oc_cblock_idx += oc_block) {
                    //     // 确定一个panel的高度
                    //     int oc_cblock_step = min(oc_block, outputChannels - oc_cblock_idx);
                    // #pragma omp parallel for private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
                    for (int oc_rblock_idx = 0; oc_rblock_idx < oc_cblock_step; oc_rblock_idx += ocRegBlock) {
                        int oc_rblock_step = min(ocRegBlock, oc_cblock_step - oc_rblock_idx);
                        float* kernelBuf_copy = kernelBuf + 64 * oc_cblock_idx * ic_block + 4 * oc_rblock_idx * ic_block;
                        float* kernel_copy = testKernel + (oc_cblock_idx + oc_rblock_idx) * inputChannels * 9;
                        float* tk0  = kernelBuf_copy;
                        float* tk1  = tk0  + tk_step;
                        float* tk2  = tk1  + tk_step;
                        float* tk3  = tk2  + tk_step;
                        float* tk4  = tk3  + tk_step;
                        float* tk5  = tk4  + tk_step;
                        float* tk6  = tk5  + tk_step;
                        float* tk7  = tk6  + tk_step;
                        float* tk8  = tk7  + tk_step;
                        float* tk9  = tk8  + tk_step;
                        float* tk10 = tk9  + tk_step;
                        float* tk11 = tk10 + tk_step;
                        float* tk12 = tk11 + tk_step;
                        float* tk13 = tk12 + tk_step;
                        float* tk14 = tk13 + tk_step;
                        float* tk15 = tk14 + tk_step;

                        for (int ic = 0; ic < ic_cblock_step; ic++) {
                            for (int oc = 0; oc < oc_rblock_step; oc++) {
                                float* kernel_ptr = kernel_copy + oc * inputChannels * 9 + (ic_cblock_idx + ic) * 9;
                                l0 = vld1q_f32(kernel_ptr);
                                l1 = vld1q_f32(kernel_ptr + 3);
                                l2 = vld1q_f32(kernel_ptr + 6);

                                // kernel transform
                                winograd_f6k3_kernel_transform_v1(l0, l1, l2, m1, m2, s1, l7, t1, t2, s2, r7, f4, f2);
                                neon_transpose4x4_inplace_f32_cpp_v1(m1, m2, s1, l7);
                                neon_transpose4x4_inplace_f32_cpp_v1(t1, t2, s2, r7);
                                winograd_f6k3_kernel_transform_v1(m1, m2, s1,  l0, l1, l2, l3, l4, l5, l6, l7, f4, f2);
                                winograd_f6k3_kernel_transform_v1(t1, t2, s2,  r0, r1, r2, r3, r4, r5, r6, r7, f4, f2);

                                vst1q_f32(tk0,   l0);   tk0  +=4;
                                vst1q_f32(tk1,   r0);   tk1  +=4;
                                vst1q_f32(tk2,   l1);   tk2  +=4;
                                vst1q_f32(tk3,   r1);   tk3  +=4;
                                vst1q_f32(tk4,   l2);   tk4  +=4;
                                vst1q_f32(tk5,   r2);   tk5  +=4;
                                vst1q_f32(tk6,   l3);   tk6  +=4;
                                vst1q_f32(tk7,   r3);   tk7  +=4;
                                vst1q_f32(tk8,   l4);   tk8  +=4;
                                vst1q_f32(tk9,   r4);   tk9  +=4;
                                vst1q_f32(tk10,  l5);   tk10 +=4;
                                vst1q_f32(tk11,  r5);   tk11 +=4;
                                vst1q_f32(tk12,  l6);   tk12 +=4;
                                vst1q_f32(tk13,  r6);   tk13 +=4;
                                vst1q_f32(tk14,  l7);   tk14 +=4;
                                vst1q_f32(tk15,  r7);   tk15 +=4;
                            }
                        }
                    }
                    // }
                    kernelTran.accumBench();
                    
                } else {
                    // to-do 
                    // if (ic_cblock_idx != 0)
                    //     kernelBuf += 64 * oc_block * ic_block;
                    // printMatrix_v1(kernelBuf, 16*outputChannels / ocRegBlock, 4 * icBlock * ocRegBlock);
                }
                // printf("oc_cblock_idx = %d\n", oc_cblock_idx);
                // printMatrix_v1(transform_kernel, 16*outputChannels / ocRegBlock, 4 * icBlock * ocRegBlock);
                // printf("kernel transform over.\n");
            }

            if (is_input_transform) {
                // for (int tile_cblock_idx = tile_cblock_begin; tile_cblock_idx < tile_cblock_end; tile_cblock_idx += tileBlock) {
                inputTran.startBench();
                // memset(inputBuf, 0, (64 * ic_block * align_ceil(tileBlock, tileRegBlock) + 128) * sizeof(float));
                int tile_rblock_remain_start = tile_cblock_step - tile_cblock_step % tileRegBlock;
                float* inputBuf_copy = inputBuf + 64 * ic_block * tile_cblock_idx;
                int step = 4 * ic_block * tile_cblock_step_align;
                // #pragma omp parallel for collapse(2) private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
                for (int tile_rblock_idx = 0; tile_rblock_idx < tile_rblock_remain_start; tile_rblock_idx += tileRegBlock) {
                    // float *inputBuf_copy = inputBuf + 4 * ic_block * tile_rblock_idx;
                    for (int ic = 0; ic < ic_cblock_step; ic++) {
                        // float *ti0  = inputBuf_copy + 4 * tileRegBlock * ic;
                        float *ti0  = inputBuf_copy + 4 * ic_block * tile_rblock_idx + 4 * tileRegBlock * ic;
                        float *ti1  = ti0  + step;
                        float *ti2  = ti1  + step;
                        float *ti3  = ti2  + step;
                        float *ti4  = ti3  + step;
                        float *ti5  = ti4  + step;
                        float *ti6  = ti5  + step;
                        float *ti7  = ti6  + step;
                        float *ti8  = ti7  + step;
                        float *ti9  = ti8  + step;
                        float *ti10 = ti9  + step;
                        float *ti11 = ti10 + step;
                        float *ti12 = ti11 + step;
                        float *ti13 = ti12 + step;
                        float *ti14 = ti13 + step;
                        float *ti15 = ti14 + step;

                        for (int tb = 0; tb < tileRegBlock; tb++) {
                            int tile = tile_cblock_idx + tile_rblock_idx + tb;
                            int tile_row = (tile/tileW)*6 - padHeight;
                            int tile_col = (tile%tileW)*6 - padWidth;
                            int in_pos   = tile_row * inputWidth + tile_col;

                            float *in_ptr0 = testInput + (ic_cblock_idx + ic) * inputWidth * inputHeight + in_pos;
                            float *in_ptr1 = in_ptr0 + inputWidth;
                            float *in_ptr2 = in_ptr1 + inputWidth;
                            float *in_ptr3 = in_ptr2 + inputWidth;
                            float *in_ptr4 = in_ptr3 + inputWidth;
                            float *in_ptr5 = in_ptr4 + inputWidth;
                            float *in_ptr6 = in_ptr5 + inputWidth;
                            float *in_ptr7 = in_ptr6 + inputWidth;

                            
                            if(tile_row >= 0 && tile_row+8 <= inputHeight && tile_col >= 0 && tile_col+8 <= inputWidth) {
                                l0 = vld1q_f32(in_ptr0);
                                r0 = vld1q_f32(in_ptr0 + 4);
                                l1 = vld1q_f32(in_ptr1);
                                r1 = vld1q_f32(in_ptr1 + 4);
                                l2 = vld1q_f32(in_ptr2);
                                r2 = vld1q_f32(in_ptr2 + 4);
                                l3 = vld1q_f32(in_ptr3);
                                r3 = vld1q_f32(in_ptr3 + 4);
                                l4 = vld1q_f32(in_ptr4);
                                r4 = vld1q_f32(in_ptr4 + 4);
                                l5 = vld1q_f32(in_ptr5);
                                r5 = vld1q_f32(in_ptr5 + 4);
                                l6 = vld1q_f32(in_ptr6);
                                r6 = vld1q_f32(in_ptr6 + 4);
                                l7 = vld1q_f32(in_ptr7);
                                r7 = vld1q_f32(in_ptr7 + 4);
                            } else {
                                float tmp[64];
                                memset(tmp, 0, 256);
                                int XStart = tile_row < 0 ? 0 : tile_row; 
                                int YStart = tile_col < 0 ? 0 : tile_col;
                                XStart     = XStart < inputHeight ? XStart : inputHeight;
                                YStart     = YStart < inputWidth  ? YStart : inputWidth;

                                int XEnd   = tile_row+8 < inputHeight ? tile_row+8 : inputHeight;
                                int YEnd   = tile_col+8 < inputWidth  ? tile_col+8 : inputWidth;
                                XEnd       = XEnd < 0 ? 0 : XEnd;
                                YEnd       = YEnd < 0 ? 0 : YEnd;
                                int YStep  = YEnd - YStart;
                                int XStep  = XEnd - XStart;

                                int len    = YStep * sizeof(float);
                                float *pLocal = testInput + (ic_cblock_idx + ic) * inputHeight * inputWidth + XStart * inputWidth + YStart;
                                float *ptmp   = tmp + (XStart - tile_row) * 8 + YStart - tile_col;
                                for(int ux = 0; ux < XStep; ux++, ptmp += 8, pLocal += inputWidth) {
                                    memcpy(ptmp, pLocal, len);
                                }

                                l0 = vld1q_f32(tmp);
                                r0 = vld1q_f32(tmp + 4);
                                l1 = vld1q_f32(tmp + 8);
                                r1 = vld1q_f32(tmp + 12);
                                l2 = vld1q_f32(tmp + 16);
                                r2 = vld1q_f32(tmp + 20);
                                l3 = vld1q_f32(tmp + 24);
                                r3 = vld1q_f32(tmp + 28);
                                l4 = vld1q_f32(tmp + 32);
                                r4 = vld1q_f32(tmp + 36);
                                l5 = vld1q_f32(tmp + 40);
                                r5 = vld1q_f32(tmp + 44);
                                l6 = vld1q_f32(tmp + 48);
                                r6 = vld1q_f32(tmp + 52);
                                l7 = vld1q_f32(tmp + 56);
                                r7 = vld1q_f32(tmp + 60);
                            }

                            input_transform_v1(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            neon_transpose4x4_inplace_f32_cpp_v1(l0, l1, l2, l3);
                            neon_transpose4x4_inplace_f32_cpp_v1(l4, l5, l6, l7);
                            input_transform_v1(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            neon_transpose4x4_inplace_f32_cpp_v1(r0, r1, r2, r3);
                            neon_transpose4x4_inplace_f32_cpp_v1(r4, r5, r6, r7);

                            input_transform_v1(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            input_transform_v1(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

                            vst1q_f32(ti0,  l0);    ti0  += 4;
                            vst1q_f32(ti1,  l4);    ti1  += 4;
                            vst1q_f32(ti2,  l1);    ti2  += 4;
                            vst1q_f32(ti3,  l5);    ti3  += 4;
                            vst1q_f32(ti4,  l2);    ti4  += 4;
                            vst1q_f32(ti5,  l6);    ti5  += 4;
                            vst1q_f32(ti6,  l3);    ti6  += 4;
                            vst1q_f32(ti7,  l7);    ti7  += 4;
                            vst1q_f32(ti8,  r0);    ti8  += 4;
                            vst1q_f32(ti9,  r4);    ti9  += 4;
                            vst1q_f32(ti10, r1);    ti10 += 4;
                            vst1q_f32(ti11, r5);    ti11 += 4;
                            vst1q_f32(ti12, r2);    ti12 += 4;
                            vst1q_f32(ti13, r6);    ti13 += 4;
                            vst1q_f32(ti14, r3);    ti14 += 4;
                            vst1q_f32(ti15, r7);    ti15 += 4;
                            
                        }
                    }
                }

                if (tile_rblock_remain_start < tile_cblock_step) {
                    // printf("tile_rblock_idx_dddd\n");
                    int tile_rblock_idx = tile_rblock_remain_start;
                    // float *inputBuf_copy = inputBuf + 64 * ic_block * tile_cblock_idx ;
                    int remain = tile_cblock_step - tile_rblock_remain_start;
                    int step = 4 * ic_block * tile_cblock_step_align;

                    for (int ic = 0; ic < ic_cblock_step; ic++) {
                        float *ti0  = inputBuf_copy + 4 * ic_block * tile_rblock_idx + 4 * tileRegBlock * ic;
                        float *ti1  = ti0  + step;
                        float *ti2  = ti1  + step;
                        float *ti3  = ti2  + step;
                        float *ti4  = ti3  + step;
                        float *ti5  = ti4  + step;
                        float *ti6  = ti5  + step;
                        float *ti7  = ti6  + step;
                        float *ti8  = ti7  + step;
                        float *ti9  = ti8  + step;
                        float *ti10 = ti9  + step;
                        float *ti11 = ti10 + step;
                        float *ti12 = ti11 + step;
                        float *ti13 = ti12 + step;
                        float *ti14 = ti13 + step;
                        float *ti15 = ti14 + step; 

                        for (int tb = 0; tb < remain; tb++) {
                            int tile = tile_cblock_idx + tile_rblock_idx + tb;
                            int tile_row = (tile/tileW)*6 - padHeight;
                            int tile_col = (tile%tileW)*6 - padWidth;
                            int in_pos   = tile_row * inputWidth + tile_col;

                            float *in_ptr0 = testInput + (ic_cblock_idx + ic) * inputWidth * inputHeight + in_pos;
                            float *in_ptr1 = in_ptr0 + inputWidth;
                            float *in_ptr2 = in_ptr1 + inputWidth;
                            float *in_ptr3 = in_ptr2 + inputWidth;
                            float *in_ptr4 = in_ptr3 + inputWidth;
                            float *in_ptr5 = in_ptr4 + inputWidth;
                            float *in_ptr6 = in_ptr5 + inputWidth;
                            float *in_ptr7 = in_ptr6 + inputWidth;

                            
                            if(tile_row >= 0 && tile_row+8 <= inputHeight && tile_col >= 0 && tile_col+8 <= inputWidth) {
                                l0 = vld1q_f32(in_ptr0);
                                r0 = vld1q_f32(in_ptr0 + 4);
                                l1 = vld1q_f32(in_ptr1);
                                r1 = vld1q_f32(in_ptr1 + 4);
                                l2 = vld1q_f32(in_ptr2);
                                r2 = vld1q_f32(in_ptr2 + 4);
                                l3 = vld1q_f32(in_ptr3);
                                r3 = vld1q_f32(in_ptr3 + 4);
                                l4 = vld1q_f32(in_ptr4);
                                r4 = vld1q_f32(in_ptr4 + 4);
                                l5 = vld1q_f32(in_ptr5);
                                r5 = vld1q_f32(in_ptr5 + 4);
                                l6 = vld1q_f32(in_ptr6);
                                r6 = vld1q_f32(in_ptr6 + 4);
                                l7 = vld1q_f32(in_ptr7);
                                r7 = vld1q_f32(in_ptr7 + 4);
                            } else {
                                float tmp[64];
                                memset(tmp, 0, 256);
                                int XStart = tile_row < 0 ? 0 : tile_row; 
                                int YStart = tile_col < 0 ? 0 : tile_col;
                                XStart     = XStart < inputHeight ? XStart : inputHeight;
                                YStart     = YStart < inputWidth  ? YStart : inputWidth;

                                int XEnd   = tile_row+8 < inputHeight ? tile_row+8 : inputHeight;
                                int YEnd   = tile_col+8 < inputWidth  ? tile_col+8 : inputWidth;
                                XEnd       = XEnd < 0 ? 0 : XEnd;
                                YEnd       = YEnd < 0 ? 0 : YEnd;
                                int YStep  = YEnd - YStart;
                                int XStep  = XEnd - XStart;

                                int len    = YStep * sizeof(float);
                                float *pLocal = testInput + (ic_cblock_idx + ic) * inputHeight * inputWidth + XStart * inputWidth + YStart;
                                float *ptmp   = tmp + (XStart - tile_row) * 8 + YStart - tile_col;
                                for(int ux = 0; ux < XStep; ux++, ptmp += 8, pLocal += inputWidth) {
                                    memcpy(ptmp, pLocal, len);
                                }

                                l0 = vld1q_f32(tmp);
                                r0 = vld1q_f32(tmp + 4);
                                l1 = vld1q_f32(tmp + 8);
                                r1 = vld1q_f32(tmp + 12);
                                l2 = vld1q_f32(tmp + 16);
                                r2 = vld1q_f32(tmp + 20);
                                l3 = vld1q_f32(tmp + 24);
                                r3 = vld1q_f32(tmp + 28);
                                l4 = vld1q_f32(tmp + 32);
                                r4 = vld1q_f32(tmp + 36);
                                l5 = vld1q_f32(tmp + 40);
                                r5 = vld1q_f32(tmp + 44);
                                l6 = vld1q_f32(tmp + 48);
                                r6 = vld1q_f32(tmp + 52);
                                l7 = vld1q_f32(tmp + 56);
                                r7 = vld1q_f32(tmp + 60);
                            }

                            input_transform_v1(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            neon_transpose4x4_inplace_f32_cpp_v1(l0, l1, l2, l3);
                            neon_transpose4x4_inplace_f32_cpp_v1(l4, l5, l6, l7);
                            input_transform_v1(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            neon_transpose4x4_inplace_f32_cpp_v1(r0, r1, r2, r3);
                            neon_transpose4x4_inplace_f32_cpp_v1(r4, r5, r6, r7);

                            input_transform_v1(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            input_transform_v1(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

                            vst1q_f32(ti0,  l0);    ti0  += 4;
                            vst1q_f32(ti1,  l4);    ti1  += 4;
                            vst1q_f32(ti2,  l1);    ti2  += 4;
                            vst1q_f32(ti3,  l5);    ti3  += 4;
                            vst1q_f32(ti4,  l2);    ti4  += 4;
                            vst1q_f32(ti5,  l6);    ti5  += 4;
                            vst1q_f32(ti6,  l3);    ti6  += 4;
                            vst1q_f32(ti7,  l7);    ti7  += 4;
                            vst1q_f32(ti8,  r0);    ti8  += 4;
                            vst1q_f32(ti9,  r4);    ti9  += 4;
                            vst1q_f32(ti10, r1);    ti10 += 4;
                            vst1q_f32(ti11, r5);    ti11 += 4;
                            vst1q_f32(ti12, r2);    ti12 += 4;
                            vst1q_f32(ti13, r6);    ti13 += 4;
                            vst1q_f32(ti14, r3);    ti14 += 4;
                            vst1q_f32(ti15, r7);    ti15 += 4;
                            
                        }

                        for (int tb = remain; tb < tileRegBlock; tb++) {
                            vst1q_f32(ti0,  vZero);    ti0  += 4;
                            vst1q_f32(ti1,  vZero);    ti1  += 4;
                            vst1q_f32(ti2,  vZero);    ti2  += 4;
                            vst1q_f32(ti3,  vZero);    ti3  += 4;
                            vst1q_f32(ti4,  vZero);    ti4  += 4;
                            vst1q_f32(ti5,  vZero);    ti5  += 4;
                            vst1q_f32(ti6,  vZero);    ti6  += 4;
                            vst1q_f32(ti7,  vZero);    ti7  += 4;
                            vst1q_f32(ti8,  vZero);    ti8  += 4;
                            vst1q_f32(ti9,  vZero);    ti9  += 4;
                            vst1q_f32(ti10, vZero);    ti10 += 4;
                            vst1q_f32(ti11, vZero);    ti11 += 4;
                            vst1q_f32(ti12, vZero);    ti12 += 4;
                            vst1q_f32(ti13, vZero);    ti13 += 4;
                            vst1q_f32(ti14, vZero);    ti14 += 4;
                            vst1q_f32(ti15, vZero);    ti15 += 4;
                        }
                    
                    }
                }
                // }
                inputTran.accumBench();
                // printf("tile_cblock_idx = %d\n", tile_cblock_idx);
                // printf("input transform over.\n");
            }
            
            
            // printMatrix_v1(transform_input, 16*tileBlock / tileRegBlock, 4 * icBlock * tileRegBlock);

            GEMM.startBench();
            // #pragma omp parallel for 
            for (int depth = 0; depth < 16; depth++) {
                // for (int oc_cblock_idx = 0; oc_cblock_idx < outputChannels; oc_cblock_idx += oc_block) {
                    // int oc_cblock_step = min(oc_block, oc_cblock_end - oc_cblock_begin);
                    int oc_rblock_remain_start = oc_cblock_step - oc_cblock_step % ocRegBlock;
                    float *kernel_copy = kernelBuf + 64 * ic_block * oc_cblock_idx   + depth * 4 * oc_cblock_step         * ic_block;
                    float *input_copy  = inputBuf  + 64 * ic_block * tile_cblock_idx + depth * 4 * tile_cblock_step_align * ic_block;
                    float *output_copy = gemmBuf + (depth * outputChannels + oc_cblock_idx) * 4 * tileN + 4 * tile_cblock_idx;
                    // #pragma omp parallel for
                    for (int oc_rblock_idx = 0; oc_rblock_idx < oc_rblock_remain_start; oc_rblock_idx += ocRegBlock) {
                        float *kernel_ptr = kernel_copy + 4 * oc_rblock_idx * ic_block;
                        float *input_ptr  = input_copy;
                        float *output_ptr = output_copy + 4 * oc_rblock_idx * tileN;
                        for (int tile_rblock_idx = 0; tile_rblock_idx < tile_cblock_step_align; tile_rblock_idx += tileRegBlock) {
                            // TensorGEMMInnerKernel4x4x4_v2(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, 16, tileN);
                            // TensorGEMMInnerKernel4x4x4_v1(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, 16, tileN);
                            tensorGEMM(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, tileN);
                            input_ptr  += 4 * tileRegBlock * ic_block;
                            output_ptr += 4 * tileRegBlock;
                        }
                    }

                    if (oc_rblock_remain_start < oc_cblock_step) {
                        // printf("oc_rblock_remian_statrr\n");
                        // printf("oc_rblock_remain_start = %d, oc_block = %d\n", oc_rblock_remain_start, oc_cblock_step);
                        // int remain = oc_cblock_step - oc_rblock_remain_start;
                        float *kernel_ptr = kernel_copy + 4 * oc_rblock_remain_start * ic_block;
                        float *input_ptr  = input_copy;
                        float *output_ptr = output_copy + 4 * oc_rblock_remain_start * tileN;
                        
                        for (int tile_rblock_idx = 0; tile_rblock_idx < tile_cblock_step_align; tile_rblock_idx += tileRegBlock) {
                            // TensorGEMMInnerKernel4x4x4_v2(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, 16, tileN);
                            tensorGEMM_for_corner_case(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, tileN);
                            input_ptr  += 4 * tileRegBlock * ic_block;
                            output_ptr += 4 * tileRegBlock;
                        }
                        
                    }
                    
                // }
            }
            GEMM.accumBench();

            // printf("tensorGEMM over.\n");
        // }
        // printf("ic_cblock_idx = %d.\n", ic_cblock_idx);
            
        }
    }

    outputTran.startBench();
    int step = 4 * outputChannels * tileN;
    int output_size = outputHeight * outputWidth;
    // #pragma omp parallel for private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
    for (int oc = 0; oc < outputChannels; oc++) {
        float *gemmBuf_p0  = gemmBuf + 4 * oc * tileN;
        float *gemmBuf_p1  = gemmBuf_p0  + step;
        float *gemmBuf_p2  = gemmBuf_p1  + step;
        float *gemmBuf_p3  = gemmBuf_p2  + step;
        float *gemmBuf_p4  = gemmBuf_p3  + step;
        float *gemmBuf_p5  = gemmBuf_p4  + step;
        float *gemmBuf_p6  = gemmBuf_p5  + step;
        float *gemmBuf_p7  = gemmBuf_p6  + step;
        float *gemmBuf_p8  = gemmBuf_p7  + step;
        float *gemmBuf_p9  = gemmBuf_p8  + step;
        float *gemmBuf_p10 = gemmBuf_p9  + step;
        float *gemmBuf_p11 = gemmBuf_p10 + step;
        float *gemmBuf_p12 = gemmBuf_p11 + step;
        float *gemmBuf_p13 = gemmBuf_p12 + step;
        float *gemmBuf_p14 = gemmBuf_p13 + step;
        float *gemmBuf_p15 = gemmBuf_p14 + step;

        // #pragma omp parallel for private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2) \
        //         firstprivate(gemmBuf_p0, gemmBuf_p1, gemmBuf_p2, gemmBuf_p3, gemmBuf_p4, gemmBuf_p5, gemmBuf_p6, gemmBuf_p7, gemmBuf_p8, gemmBuf_p9, gemmBuf_p10, gemmBuf_p11, gemmBuf_p12, gemmBuf_p13, gemmBuf_p14, gemmBuf_p15)
        for (int tile = 0; tile < tileN; tile++) {
            l0 = vld1q_f32(gemmBuf_p0); 	gemmBuf_p0+=4;
            r0 = vld1q_f32(gemmBuf_p1);	    gemmBuf_p1+=4;
            l1 = vld1q_f32(gemmBuf_p2);	    gemmBuf_p2+=4;
            r1 = vld1q_f32(gemmBuf_p3);	    gemmBuf_p3+=4;
            l2 = vld1q_f32(gemmBuf_p4);	    gemmBuf_p4+=4;
            r2 = vld1q_f32(gemmBuf_p5);	    gemmBuf_p5+=4;
            l3 = vld1q_f32(gemmBuf_p6);	    gemmBuf_p6+=4;
            r3 = vld1q_f32(gemmBuf_p7);	    gemmBuf_p7+=4;
            l4 = vld1q_f32(gemmBuf_p8);	    gemmBuf_p8+=4;
            r4 = vld1q_f32(gemmBuf_p9);	    gemmBuf_p9+=4;
            l5 = vld1q_f32(gemmBuf_p10);	gemmBuf_p10+=4;
            r5 = vld1q_f32(gemmBuf_p11);	gemmBuf_p11+=4;
            l6 = vld1q_f32(gemmBuf_p12);	gemmBuf_p12+=4;
            r6 = vld1q_f32(gemmBuf_p13);	gemmBuf_p13+=4;
            l7 = vld1q_f32(gemmBuf_p14);	gemmBuf_p14+=4;
            r7 = vld1q_f32(gemmBuf_p15);	gemmBuf_p15+=4;

            winograd_f6k3_output_transform_inplace_v1(l0, l1, l2, l3, l4, l5, l6, l7);
            winograd_f6k3_output_transform_inplace_v1(r0, r1, r2, r3, r4, r5, r6, r7);
            neon_transpose4x4_inplace_f32_cpp_v1(l0, l1, l2, l3);
            neon_transpose4x4_inplace_f32_cpp_v1(r0, r1, r2, r3);
            neon_transpose4x4_inplace_f32_cpp_v1(l4, l5, l6, l7);
            neon_transpose4x4_inplace_f32_cpp_v1(r4, r5, r6, r7);
            winograd_f6k3_output_transform_inplace_v1(l0, l1, l2, l3, r0, r1, r2, r3);
            winograd_f6k3_output_transform_inplace_v1(l4, l5, l6, l7, r4, r5, r6, r7);

            int tileRow = (tile) / tileW * 6;
            int tileCol = (tile) % tileW * 6;

            float* output_p0 = baseResult + oc*output_size + tileRow*outputWidth + tileCol;
            int deltaX = outputHeight - tileRow;
            int deltaY = outputWidth - tileCol;
            // deltaX = deltaX<6?deltaX:6;
            // deltaY = deltaY<6?deltaY:6;
            if (deltaX >= 6 && deltaY >= 6) {
                float* output_p1 = output_p0 + outputWidth;
                float* output_p2 = output_p1 + outputWidth;
                float* output_p3 = output_p2 + outputWidth;
                float* output_p4 = output_p3 + outputWidth;
                float* output_p5 = output_p4 + outputWidth;

                vst1q_f32(output_p0, l0);
                vst1_f32(output_p0 + 4, vget_low_f32(l4));
                vst1q_f32(output_p1,  l1);
                vst1_f32(output_p1 + 4, vget_low_f32(l5));
                vst1q_f32(output_p2, l2);
                vst1_f32(output_p2 + 4, vget_low_f32(l6));
                vst1q_f32(output_p3, l3);
                vst1_f32(output_p3 + 4, vget_low_f32(l7));
                vst1q_f32(output_p4, r0);
                vst1_f32(output_p4 + 4, vget_low_f32(r4));
                vst1q_f32(output_p5, r1);
                vst1_f32(output_p5 + 4, vget_low_f32(r5));
            } else {
                // printf("------------------not entouasd\n");
                deltaX = min(6, deltaX);
                deltaY = min(6, deltaY);
                float tmp[36];
                memset(tmp, 0, 36*sizeof(float));
                vst1q_f32(tmp, l0);
                vst1_f32(tmp   + 4, vget_low_f32(l4));
                vst1q_f32(tmp  + 6, l1);
                vst1_f32(tmp  + 10, vget_low_f32(l5));
                vst1q_f32(tmp + 12, l2);
                vst1_f32(tmp  + 16, vget_low_f32(l6));
                vst1q_f32(tmp + 18, l3);
                vst1_f32(tmp  + 22, vget_low_f32(l7));
                vst1q_f32(tmp + 24, r0);
                vst1_f32(tmp  + 28, vget_low_f32(r4));
                vst1q_f32(tmp + 30, r1);
                vst1_f32(tmp  + 34, vget_low_f32(r5));

                for(int u = 0; u < deltaX; u++) 
                    memcpy(output_p0 + u*outputWidth, tmp + u*6, deltaY*sizeof(float));

            }
            
        }
    }
    outputTran.accumBench();
    
    // printf("outputTransform over.\n");
    // kernelTran.printBench("kernel", 1);
    // inputTran.printBench("inputTran", 1);
    // GEMM.printBench("TensorGEMM", 1);
    // outputTran.printBench("outputTran", 1);
    // _mm_free(transform_kernel);
    // _mm_free(transform_input);
    // _mm_free(gemm_buf);
    // printf("---------------------1 round over---------------.\n");

    return 1;
}

int winoF63_v1_multi(float *baseResult, float *testInput, float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int padWidth, int padHeight, int strideWidth, int strideHeight, int tileBlock, float *gemmBuf, int ocBlock, float *kernelBuf, int icBlock, float *inputBuf, int tileRegBlock, int ocRegBlock, int scheduling, int num_threads,
                Timer& inputTran, Timer& kernelTran, Timer& GEMM, Timer& outputTran)
{
    /*
        outputBuf: tileBlock*ocBlock*48
        kernelBuf: icBlock*ocBlock*64
        inputBuf:  tileBlock*icBlock*64
    */
    int enableOffKernel = scheduling%2;
    int loopOrdering    = scheduling/2;
    int outputHeight = inputHeight+ padHeight*2 -2;
    int outputWidth  = inputWidth + padWidth*2  -2;
    int tileH = (inputHeight+ padHeight*2 -2 + 5)/6;
    int tileW = (inputWidth + padWidth*2  -2 + 5)/6;
    int tileN = tileH*tileW;

    // assert(inputChannels%icBlock  == 0);
    assert(tileBlock%tileRegBlock == 0);
    assert(ocBlock%ocRegBlock     == 0);


    // Timer inputTran, kernelTran, GEMM, outputTran, storeBack;

    const float32x4_t f5    = vdupq_n_f32(5.0f);
    const float32x4_t f4    = vdupq_n_f32(4.0f);
    const float32x4_t f2    = vdupq_n_f32(2.0f);
    const float32x4_t f2_5  = vdupq_n_f32(2.5f);
    const float32x4_t f5_25 = vdupq_n_f32(5.25f);
    const float32x4_t f4_25 = vdupq_n_f32(4.25f);
    const float32x4_t f1_25 = vdupq_n_f32(1.25f);
    const float32x4_t f0_5  = vdupq_n_f32(0.5f);
    const float32x4_t f0_25 = vdupq_n_f32(0.25f);
    const float32x4_t vZero = vdupq_n_f32(0.0f);

    float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    float32x4_t m1, m2, s1, s2, t1, t2;//Auxiliary registers

    int ic_block = icBlock;
    int oc_block = ocBlock;

    void (*tensorGEMM)(float *, const float *, float *, int, int); 
    void (*tensorGEMM_for_corner_case)(float *, const float *, float *, int, int);
    if (ocRegBlock == 4) {
        if (tileRegBlock == 4) {
            tensorGEMM = TensorGEMMInnerKernel4x4x4_v3;
            // tensorGEMM = TensorGEMMInnerKernel4x4x4_v1;
            int remain = outputChannels % ocRegBlock;
            if (remain == 3) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x3x4_v1;
            } else if (remain == 2) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x2x4_v1;
            } else if (remain == 1) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x1x4_v1;
            }
        } else if (tileRegBlock == 5) {
            tensorGEMM = TensorGEMMInnerKernel4x4x5_v1;
            // tensorGEMM = TensorGEMMInnerKernel4x4x5_v2;
            int remain = outputChannels % ocRegBlock;
            if (remain == 3) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x3x5_v1;
            } else if (remain == 2) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x2x5_v1;
            } else if (remain == 1) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x1x5_v1;
            }
        }
    }
    

    // float* transform_kernel = static_cast<float*>(_mm_malloc((64 * outputChannels * ic_block + 128) * sizeof(float)));
    // float* transform_input  = static_cast<float*>(_mm_malloc((64 * ic_block * align_ceil(tileBlock, tileRegBlock) + 128) * sizeof(float)));
    // float* gemm_buf         = static_cast<float*>(_mm_malloc((64 * outputChannels * tileN + 128) * sizeof(float)));
    // #pragma omp parallel 
    int oc_cblock_num = (outputChannels - 1) / oc_block + 1;
    int tile_cblock_num = (tileN - 1) / tileBlock + 1; 
    for (int ic_cblock_idx = 0; ic_cblock_idx < inputChannels; ic_cblock_idx += ic_block) {
        int ic_cblock_step = min(ic_block, inputChannels - ic_cblock_idx);
        // printf("ic_cblock_idx = %d\n", ic_cblock_idx); 
        int pre_oc_cblock_idx = -1, pre_tile_cblock_idx = -1;
        bool is_kernel_transform = true, is_input_transform = true;
        // #pragma omp parallel for
        for (int cblock_idx = 0; cblock_idx < oc_cblock_num * tile_cblock_num; cblock_idx++) {
            int oc_cblock_idx, tile_cblock_idx;
            if (loopOrdering) { // outer loop is tile
                oc_cblock_idx   = cblock_idx % oc_cblock_num * oc_block;
                tile_cblock_idx = cblock_idx / oc_cblock_num * tileBlock;

                is_kernel_transform = (tile_cblock_idx == 0 ? true: false);
                is_input_transform  = (pre_tile_cblock_idx == tile_cblock_idx ? false: true);
                pre_tile_cblock_idx = tile_cblock_idx;
            } else { // outer loop is oc
                oc_cblock_idx   = cblock_idx / tile_cblock_num * oc_block;
                tile_cblock_idx = cblock_idx % tile_cblock_num * tileBlock;
                
                is_input_transform  = (oc_cblock_idx == 0 ? true: false);
                is_kernel_transform = (pre_oc_cblock_idx == oc_cblock_idx ? false: true);
                pre_oc_cblock_idx = oc_cblock_idx;
            }

            int oc_cblock_end     = min(oc_cblock_idx + oc_block, outputChannels);
            int oc_cblock_begin   = oc_cblock_idx;
            int oc_cblock_step    = oc_cblock_end - oc_cblock_begin;
            int tile_cblock_end   = min(tile_cblock_idx + tileBlock, tileN);
            int tile_cblock_begin = tile_cblock_idx;
            int tile_cblock_step  = tile_cblock_end - tile_cblock_begin;
            int tile_cblock_step_align = align_ceil(tile_cblock_step, tileRegBlock);

            // printf("oc_block_idx = %d, tile_block_idx = %d\n", oc_cblock_idx, tile_cblock_idx);

            if (is_kernel_transform) {
                if (!enableOffKernel) {
                    kernelTran.startBench();
                    // memset(kernelBuf, 0, (64 * outputChannels * ic_block + 128) * sizeof(float));
                    // #pragma omp parallel for 
                    int tk_step = oc_cblock_step * ic_block * 4;
                    // #pragma omp parallel for collapse(2) private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
                    // for (int oc_cblock_idx = 0; oc_cblock_idx < outputChannels; oc_cblock_idx += oc_block) {
                    //     // 确定一个panel的高度
                    //     int oc_cblock_step = min(oc_block, outputChannels - oc_cblock_idx);
                    #pragma omp parallel for private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
                    for (int oc_rblock_idx = 0; oc_rblock_idx < oc_cblock_step; oc_rblock_idx += ocRegBlock) {
                        int oc_rblock_step = min(ocRegBlock, oc_cblock_step - oc_rblock_idx);
                        float* kernelBuf_copy = kernelBuf + 64 * oc_cblock_idx * ic_block + 4 * oc_rblock_idx * ic_block;
                        float* kernel_copy = testKernel + (oc_cblock_idx + oc_rblock_idx) * inputChannels * 9;
                        float* tk0  = kernelBuf_copy;
                        float* tk1  = tk0  + tk_step;
                        float* tk2  = tk1  + tk_step;
                        float* tk3  = tk2  + tk_step;
                        float* tk4  = tk3  + tk_step;
                        float* tk5  = tk4  + tk_step;
                        float* tk6  = tk5  + tk_step;
                        float* tk7  = tk6  + tk_step;
                        float* tk8  = tk7  + tk_step;
                        float* tk9  = tk8  + tk_step;
                        float* tk10 = tk9  + tk_step;
                        float* tk11 = tk10 + tk_step;
                        float* tk12 = tk11 + tk_step;
                        float* tk13 = tk12 + tk_step;
                        float* tk14 = tk13 + tk_step;
                        float* tk15 = tk14 + tk_step;

                        // int tid = omp_get_thread_num();
                        // printf("tid = %d, oc_cblock_idx = %d, oc_rblock_idx = %d.\n", tid, oc_cblock_idx, oc_rblock_idx);

                        for (int ic = 0; ic < ic_cblock_step; ic++) {
                            for (int oc = 0; oc < oc_rblock_step; oc++) {
                                float* kernel_ptr = kernel_copy + oc * inputChannels * 9 + (ic_cblock_idx + ic) * 9;
                                l0 = vld1q_f32(kernel_ptr);
                                l1 = vld1q_f32(kernel_ptr + 3);
                                l2 = vld1q_f32(kernel_ptr + 6);

                                // kernel transform
                                winograd_f6k3_kernel_transform_v1(l0, l1, l2, m1, m2, s1, l7, t1, t2, s2, r7, f4, f2);
                                neon_transpose4x4_inplace_f32_cpp_v1(m1, m2, s1, l7);
                                neon_transpose4x4_inplace_f32_cpp_v1(t1, t2, s2, r7);
                                winograd_f6k3_kernel_transform_v1(m1, m2, s1,  l0, l1, l2, l3, l4, l5, l6, l7, f4, f2);
                                winograd_f6k3_kernel_transform_v1(t1, t2, s2,  r0, r1, r2, r3, r4, r5, r6, r7, f4, f2);

                                vst1q_f32(tk0,   l0);   tk0  +=4;
                                vst1q_f32(tk1,   r0);   tk1  +=4;
                                vst1q_f32(tk2,   l1);   tk2  +=4;
                                vst1q_f32(tk3,   r1);   tk3  +=4;
                                vst1q_f32(tk4,   l2);   tk4  +=4;
                                vst1q_f32(tk5,   r2);   tk5  +=4;
                                vst1q_f32(tk6,   l3);   tk6  +=4;
                                vst1q_f32(tk7,   r3);   tk7  +=4;
                                vst1q_f32(tk8,   l4);   tk8  +=4;
                                vst1q_f32(tk9,   r4);   tk9  +=4;
                                vst1q_f32(tk10,  l5);   tk10 +=4;
                                vst1q_f32(tk11,  r5);   tk11 +=4;
                                vst1q_f32(tk12,  l6);   tk12 +=4;
                                vst1q_f32(tk13,  r6);   tk13 +=4;
                                vst1q_f32(tk14,  l7);   tk14 +=4;
                                vst1q_f32(tk15,  r7);   tk15 +=4;
                            }
                        }
                    }
                    // }
                    kernelTran.accumBench();
                    
                } else {
                    // to-do 
                    // if (ic_cblock_idx != 0)
                    //     kernelBuf += 64 * oc_block * ic_block;
                    // printMatrix_v1(kernelBuf, 16*outputChannels / ocRegBlock, 4 * icBlock * ocRegBlock);
                }
                // printf("oc_cblock_idx = %d\n", oc_cblock_idx);
                // printMatrix_v1(transform_kernel, 16*outputChannels / ocRegBlock, 4 * icBlock * ocRegBlock);
                // printf("kernel transform over.\n");
            }

            if (is_input_transform) {
                // for (int tile_cblock_idx = tile_cblock_begin; tile_cblock_idx < tile_cblock_end; tile_cblock_idx += tileBlock) {
                inputTran.startBench();
                // memset(inputBuf, 0, (64 * ic_block * align_ceil(tileBlock, tileRegBlock) + 128) * sizeof(float));
                int tile_rblock_remain_start = tile_cblock_step - tile_cblock_step % tileRegBlock;
                float* inputBuf_copy = inputBuf + 64 * ic_block * tile_cblock_idx;
                int step = 4 * ic_block * tile_cblock_step_align;
                #pragma omp parallel for collapse(2) private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
                for (int tile_rblock_idx = 0; tile_rblock_idx < tile_rblock_remain_start; tile_rblock_idx += tileRegBlock) {
                    // float *inputBuf_copy = inputBuf + 4 * ic_block * tile_rblock_idx;
                    for (int ic = 0; ic < ic_cblock_step; ic++) {
                        // float *ti0  = inputBuf_copy + 4 * tileRegBlock * ic;
                        float *ti0  = inputBuf_copy + 4 * ic_block * tile_rblock_idx + 4 * tileRegBlock * ic;
                        float *ti1  = ti0  + step;
                        float *ti2  = ti1  + step;
                        float *ti3  = ti2  + step;
                        float *ti4  = ti3  + step;
                        float *ti5  = ti4  + step;
                        float *ti6  = ti5  + step;
                        float *ti7  = ti6  + step;
                        float *ti8  = ti7  + step;
                        float *ti9  = ti8  + step;
                        float *ti10 = ti9  + step;
                        float *ti11 = ti10 + step;
                        float *ti12 = ti11 + step;
                        float *ti13 = ti12 + step;
                        float *ti14 = ti13 + step;
                        float *ti15 = ti14 + step;

                        for (int tb = 0; tb < tileRegBlock; tb++) {
                            int tile = tile_cblock_idx + tile_rblock_idx + tb;
                            int tile_row = (tile/tileW)*6 - padHeight;
                            int tile_col = (tile%tileW)*6 - padWidth;
                            int in_pos   = tile_row * inputWidth + tile_col;

                            float *in_ptr0 = testInput + (ic_cblock_idx + ic) * inputWidth * inputHeight + in_pos;
                            float *in_ptr1 = in_ptr0 + inputWidth;
                            float *in_ptr2 = in_ptr1 + inputWidth;
                            float *in_ptr3 = in_ptr2 + inputWidth;
                            float *in_ptr4 = in_ptr3 + inputWidth;
                            float *in_ptr5 = in_ptr4 + inputWidth;
                            float *in_ptr6 = in_ptr5 + inputWidth;
                            float *in_ptr7 = in_ptr6 + inputWidth;

                            
                            if(tile_row >= 0 && tile_row+8 <= inputHeight && tile_col >= 0 && tile_col+8 <= inputWidth) {
                                l0 = vld1q_f32(in_ptr0);
                                r0 = vld1q_f32(in_ptr0 + 4);
                                l1 = vld1q_f32(in_ptr1);
                                r1 = vld1q_f32(in_ptr1 + 4);
                                l2 = vld1q_f32(in_ptr2);
                                r2 = vld1q_f32(in_ptr2 + 4);
                                l3 = vld1q_f32(in_ptr3);
                                r3 = vld1q_f32(in_ptr3 + 4);
                                l4 = vld1q_f32(in_ptr4);
                                r4 = vld1q_f32(in_ptr4 + 4);
                                l5 = vld1q_f32(in_ptr5);
                                r5 = vld1q_f32(in_ptr5 + 4);
                                l6 = vld1q_f32(in_ptr6);
                                r6 = vld1q_f32(in_ptr6 + 4);
                                l7 = vld1q_f32(in_ptr7);
                                r7 = vld1q_f32(in_ptr7 + 4);
                            } else {
                                float tmp[64];
                                memset(tmp, 0, 256);
                                int XStart = tile_row < 0 ? 0 : tile_row; 
                                int YStart = tile_col < 0 ? 0 : tile_col;
                                XStart     = XStart < inputHeight ? XStart : inputHeight;
                                YStart     = YStart < inputWidth  ? YStart : inputWidth;

                                int XEnd   = tile_row+8 < inputHeight ? tile_row+8 : inputHeight;
                                int YEnd   = tile_col+8 < inputWidth  ? tile_col+8 : inputWidth;
                                XEnd       = XEnd < 0 ? 0 : XEnd;
                                YEnd       = YEnd < 0 ? 0 : YEnd;
                                int YStep  = YEnd - YStart;
                                int XStep  = XEnd - XStart;

                                int len    = YStep * sizeof(float);
                                float *pLocal = testInput + (ic_cblock_idx + ic) * inputHeight * inputWidth + XStart * inputWidth + YStart;
                                float *ptmp   = tmp + (XStart - tile_row) * 8 + YStart - tile_col;
                                for(int ux = 0; ux < XStep; ux++, ptmp += 8, pLocal += inputWidth) {
                                    memcpy(ptmp, pLocal, len);
                                }

                                l0 = vld1q_f32(tmp);
                                r0 = vld1q_f32(tmp + 4);
                                l1 = vld1q_f32(tmp + 8);
                                r1 = vld1q_f32(tmp + 12);
                                l2 = vld1q_f32(tmp + 16);
                                r2 = vld1q_f32(tmp + 20);
                                l3 = vld1q_f32(tmp + 24);
                                r3 = vld1q_f32(tmp + 28);
                                l4 = vld1q_f32(tmp + 32);
                                r4 = vld1q_f32(tmp + 36);
                                l5 = vld1q_f32(tmp + 40);
                                r5 = vld1q_f32(tmp + 44);
                                l6 = vld1q_f32(tmp + 48);
                                r6 = vld1q_f32(tmp + 52);
                                l7 = vld1q_f32(tmp + 56);
                                r7 = vld1q_f32(tmp + 60);
                            }

                            input_transform_v1(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            neon_transpose4x4_inplace_f32_cpp_v1(l0, l1, l2, l3);
                            neon_transpose4x4_inplace_f32_cpp_v1(l4, l5, l6, l7);
                            input_transform_v1(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            neon_transpose4x4_inplace_f32_cpp_v1(r0, r1, r2, r3);
                            neon_transpose4x4_inplace_f32_cpp_v1(r4, r5, r6, r7);

                            input_transform_v1(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            input_transform_v1(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

                            vst1q_f32(ti0,  l0);    ti0  += 4;
                            vst1q_f32(ti1,  l4);    ti1  += 4;
                            vst1q_f32(ti2,  l1);    ti2  += 4;
                            vst1q_f32(ti3,  l5);    ti3  += 4;
                            vst1q_f32(ti4,  l2);    ti4  += 4;
                            vst1q_f32(ti5,  l6);    ti5  += 4;
                            vst1q_f32(ti6,  l3);    ti6  += 4;
                            vst1q_f32(ti7,  l7);    ti7  += 4;
                            vst1q_f32(ti8,  r0);    ti8  += 4;
                            vst1q_f32(ti9,  r4);    ti9  += 4;
                            vst1q_f32(ti10, r1);    ti10 += 4;
                            vst1q_f32(ti11, r5);    ti11 += 4;
                            vst1q_f32(ti12, r2);    ti12 += 4;
                            vst1q_f32(ti13, r6);    ti13 += 4;
                            vst1q_f32(ti14, r3);    ti14 += 4;
                            vst1q_f32(ti15, r7);    ti15 += 4;
                            
                        }
                    }
                }

                if (tile_rblock_remain_start < tile_cblock_step) {
                    // printf("tile_rblock_idx_dddd\n");
                    int tile_rblock_idx = tile_rblock_remain_start;
                    // float *inputBuf_copy = inputBuf + 64 * ic_block * tile_cblock_idx ;
                    int remain = tile_cblock_step - tile_rblock_remain_start;
                    int step = 4 * ic_block * tile_cblock_step_align;

                    for (int ic = 0; ic < ic_cblock_step; ic++) {
                        float *ti0  = inputBuf_copy + 4 * ic_block * tile_rblock_idx + 4 * tileRegBlock * ic;
                        float *ti1  = ti0  + step;
                        float *ti2  = ti1  + step;
                        float *ti3  = ti2  + step;
                        float *ti4  = ti3  + step;
                        float *ti5  = ti4  + step;
                        float *ti6  = ti5  + step;
                        float *ti7  = ti6  + step;
                        float *ti8  = ti7  + step;
                        float *ti9  = ti8  + step;
                        float *ti10 = ti9  + step;
                        float *ti11 = ti10 + step;
                        float *ti12 = ti11 + step;
                        float *ti13 = ti12 + step;
                        float *ti14 = ti13 + step;
                        float *ti15 = ti14 + step; 

                        for (int tb = 0; tb < remain; tb++) {
                            int tile = tile_cblock_idx + tile_rblock_idx + tb;
                            int tile_row = (tile/tileW)*6 - padHeight;
                            int tile_col = (tile%tileW)*6 - padWidth;
                            int in_pos   = tile_row * inputWidth + tile_col;

                            float *in_ptr0 = testInput + (ic_cblock_idx + ic) * inputWidth * inputHeight + in_pos;
                            float *in_ptr1 = in_ptr0 + inputWidth;
                            float *in_ptr2 = in_ptr1 + inputWidth;
                            float *in_ptr3 = in_ptr2 + inputWidth;
                            float *in_ptr4 = in_ptr3 + inputWidth;
                            float *in_ptr5 = in_ptr4 + inputWidth;
                            float *in_ptr6 = in_ptr5 + inputWidth;
                            float *in_ptr7 = in_ptr6 + inputWidth;

                            
                            if(tile_row >= 0 && tile_row+8 <= inputHeight && tile_col >= 0 && tile_col+8 <= inputWidth) {
                                l0 = vld1q_f32(in_ptr0);
                                r0 = vld1q_f32(in_ptr0 + 4);
                                l1 = vld1q_f32(in_ptr1);
                                r1 = vld1q_f32(in_ptr1 + 4);
                                l2 = vld1q_f32(in_ptr2);
                                r2 = vld1q_f32(in_ptr2 + 4);
                                l3 = vld1q_f32(in_ptr3);
                                r3 = vld1q_f32(in_ptr3 + 4);
                                l4 = vld1q_f32(in_ptr4);
                                r4 = vld1q_f32(in_ptr4 + 4);
                                l5 = vld1q_f32(in_ptr5);
                                r5 = vld1q_f32(in_ptr5 + 4);
                                l6 = vld1q_f32(in_ptr6);
                                r6 = vld1q_f32(in_ptr6 + 4);
                                l7 = vld1q_f32(in_ptr7);
                                r7 = vld1q_f32(in_ptr7 + 4);
                            } else {
                                float tmp[64];
                                memset(tmp, 0, 256);
                                int XStart = tile_row < 0 ? 0 : tile_row; 
                                int YStart = tile_col < 0 ? 0 : tile_col;
                                XStart     = XStart < inputHeight ? XStart : inputHeight;
                                YStart     = YStart < inputWidth  ? YStart : inputWidth;

                                int XEnd   = tile_row+8 < inputHeight ? tile_row+8 : inputHeight;
                                int YEnd   = tile_col+8 < inputWidth  ? tile_col+8 : inputWidth;
                                XEnd       = XEnd < 0 ? 0 : XEnd;
                                YEnd       = YEnd < 0 ? 0 : YEnd;
                                int YStep  = YEnd - YStart;
                                int XStep  = XEnd - XStart;

                                int len    = YStep * sizeof(float);
                                float *pLocal = testInput + (ic_cblock_idx + ic) * inputHeight * inputWidth + XStart * inputWidth + YStart;
                                float *ptmp   = tmp + (XStart - tile_row) * 8 + YStart - tile_col;
                                for(int ux = 0; ux < XStep; ux++, ptmp += 8, pLocal += inputWidth) {
                                    memcpy(ptmp, pLocal, len);
                                }

                                l0 = vld1q_f32(tmp);
                                r0 = vld1q_f32(tmp + 4);
                                l1 = vld1q_f32(tmp + 8);
                                r1 = vld1q_f32(tmp + 12);
                                l2 = vld1q_f32(tmp + 16);
                                r2 = vld1q_f32(tmp + 20);
                                l3 = vld1q_f32(tmp + 24);
                                r3 = vld1q_f32(tmp + 28);
                                l4 = vld1q_f32(tmp + 32);
                                r4 = vld1q_f32(tmp + 36);
                                l5 = vld1q_f32(tmp + 40);
                                r5 = vld1q_f32(tmp + 44);
                                l6 = vld1q_f32(tmp + 48);
                                r6 = vld1q_f32(tmp + 52);
                                l7 = vld1q_f32(tmp + 56);
                                r7 = vld1q_f32(tmp + 60);
                            }

                            input_transform_v1(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            neon_transpose4x4_inplace_f32_cpp_v1(l0, l1, l2, l3);
                            neon_transpose4x4_inplace_f32_cpp_v1(l4, l5, l6, l7);
                            input_transform_v1(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            neon_transpose4x4_inplace_f32_cpp_v1(r0, r1, r2, r3);
                            neon_transpose4x4_inplace_f32_cpp_v1(r4, r5, r6, r7);

                            input_transform_v1(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                            input_transform_v1(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                                            t1, t2, s1, s2, m1, m2, //Auxiliary
                                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

                            vst1q_f32(ti0,  l0);    ti0  += 4;
                            vst1q_f32(ti1,  l4);    ti1  += 4;
                            vst1q_f32(ti2,  l1);    ti2  += 4;
                            vst1q_f32(ti3,  l5);    ti3  += 4;
                            vst1q_f32(ti4,  l2);    ti4  += 4;
                            vst1q_f32(ti5,  l6);    ti5  += 4;
                            vst1q_f32(ti6,  l3);    ti6  += 4;
                            vst1q_f32(ti7,  l7);    ti7  += 4;
                            vst1q_f32(ti8,  r0);    ti8  += 4;
                            vst1q_f32(ti9,  r4);    ti9  += 4;
                            vst1q_f32(ti10, r1);    ti10 += 4;
                            vst1q_f32(ti11, r5);    ti11 += 4;
                            vst1q_f32(ti12, r2);    ti12 += 4;
                            vst1q_f32(ti13, r6);    ti13 += 4;
                            vst1q_f32(ti14, r3);    ti14 += 4;
                            vst1q_f32(ti15, r7);    ti15 += 4;
                            
                        }

                        for (int tb = remain; tb < tileRegBlock; tb++) {
                            vst1q_f32(ti0,  vZero);    ti0  += 4;
                            vst1q_f32(ti1,  vZero);    ti1  += 4;
                            vst1q_f32(ti2,  vZero);    ti2  += 4;
                            vst1q_f32(ti3,  vZero);    ti3  += 4;
                            vst1q_f32(ti4,  vZero);    ti4  += 4;
                            vst1q_f32(ti5,  vZero);    ti5  += 4;
                            vst1q_f32(ti6,  vZero);    ti6  += 4;
                            vst1q_f32(ti7,  vZero);    ti7  += 4;
                            vst1q_f32(ti8,  vZero);    ti8  += 4;
                            vst1q_f32(ti9,  vZero);    ti9  += 4;
                            vst1q_f32(ti10, vZero);    ti10 += 4;
                            vst1q_f32(ti11, vZero);    ti11 += 4;
                            vst1q_f32(ti12, vZero);    ti12 += 4;
                            vst1q_f32(ti13, vZero);    ti13 += 4;
                            vst1q_f32(ti14, vZero);    ti14 += 4;
                            vst1q_f32(ti15, vZero);    ti15 += 4;
                        }
                    
                    }
                }
                // }
                inputTran.accumBench();
                // printf("tile_cblock_idx = %d\n", tile_cblock_idx);
                // printf("input transform over.\n");
            }
            
            
            // printMatrix_v1(transform_input, 16*tileBlock / tileRegBlock, 4 * icBlock * tileRegBlock);

            GEMM.startBench();
            int oc_rblock_remain_start = oc_cblock_step - oc_cblock_step % ocRegBlock;
            #pragma omp parallel for collapse(2)
            for (int depth = 0; depth < 16; depth++) {
                // // for (int oc_cblock_idx = 0; oc_cblock_idx < outputChannels; oc_cblock_idx += oc_block) {
                    // // int oc_cblock_step = min(oc_block, oc_cblock_end - oc_cblock_begin);
                    // int oc_rblock_remain_start = oc_cblock_step - oc_cblock_step % ocRegBlock;
                    // float *kernel_copy = kernelBuf + 64 * ic_block * oc_cblock_idx   + depth * 4 * oc_cblock_step         * ic_block;
                    // float *input_copy  = inputBuf  + 64 * ic_block * tile_cblock_idx + depth * 4 * tile_cblock_step_align * ic_block;
                    // float *output_copy = gemmBuf   + (depth * outputChannels + oc_cblock_idx) * 4 * tileN + 4 * tile_cblock_idx;
                    // #pragma omp parallel for collapse(2)
                    for (int oc_rblock_idx = 0; oc_rblock_idx < oc_rblock_remain_start; oc_rblock_idx += ocRegBlock) {
                        
                        // float *kernel_ptr = kernel_copy + 4 * oc_rblock_idx * ic_block;
                        // float *input_ptr  = input_copy;
                        // float *output_ptr = output_copy + 4 * oc_rblock_idx * tileN;
                        for (int tile_rblock_idx = 0; tile_rblock_idx < tile_cblock_step_align; tile_rblock_idx += tileRegBlock) {
                            float *kernel_copy = kernelBuf + 64 * ic_block * oc_cblock_idx   + depth * 4 * oc_cblock_step         * ic_block;
                            float *input_copy  = inputBuf  + 64 * ic_block * tile_cblock_idx + depth * 4 * tile_cblock_step_align * ic_block;
                            float *output_copy = gemmBuf   + (depth * outputChannels + oc_cblock_idx) * 4 * tileN + 4 * tile_cblock_idx;
                            float *kernel_ptr = kernel_copy + 4 * oc_rblock_idx * ic_block;
                            float *input_ptr  = input_copy  + 4 * tile_rblock_idx * ic_block;
                            float *output_ptr = output_copy + 4 * oc_rblock_idx * tileN + 4 * tile_rblock_idx;
                            // printf("oc_rblock_idx = %d, tile_rblock_idx = %d, tid = %d\n", oc_rblock_idx, tile_rblock_idx, omp_get_thread_num());
                            // TensorGEMMInnerKernel4x4x4_v2(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, 16, tileN);
                            // TensorGEMMInnerKernel4x4x4_v1(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, 16, tileN);
                            tensorGEMM(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, tileN);
                            // input_ptr  += 4 * tileRegBlock * ic_block;        
                            // output_ptr += 4 * tileRegBlock;
                        }
                        // printf("-----\n");
                    }
                // }
            }
            
            #pragma omp parallel for
            for (int depth = 0; depth < 16; depth++) {
                if (oc_rblock_remain_start < oc_cblock_step) {
                    float *kernel_copy = kernelBuf + 64 * ic_block * oc_cblock_idx   + depth * 4 * oc_cblock_step         * ic_block;
                    float *input_copy  = inputBuf  + 64 * ic_block * tile_cblock_idx + depth * 4 * tile_cblock_step_align * ic_block;
                    float *output_copy = gemmBuf   + (depth * outputChannels + oc_cblock_idx) * 4 * tileN + 4 * tile_cblock_idx;
                    // printf("oc_rblock_remian_statrr\n");
                    // printf("oc_rblock_remain_start = %d, oc_block = %d\n", oc_rblock_remain_start, oc_cblock_step);
                    // int remain = oc_cblock_step - oc_rblock_remain_start;
                    float *kernel_ptr = kernel_copy + 4 * oc_rblock_remain_start * ic_block;
                    float *input_ptr  = input_copy;
                    float *output_ptr = output_copy + 4 * oc_rblock_remain_start * tileN;
                    
                    for (int tile_rblock_idx = 0; tile_rblock_idx < tile_cblock_step_align; tile_rblock_idx += tileRegBlock) {
                        // TensorGEMMInnerKernel4x4x4_v2(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, 16, tileN);
                        tensorGEMM_for_corner_case(output_ptr, kernel_ptr, input_ptr, ic_cblock_step, tileN);
                        input_ptr  += 4 * tileRegBlock * ic_block;
                        output_ptr += 4 * tileRegBlock;
                    }
                    
                }
            }
            
            GEMM.accumBench();

            // printf("tensorGEMM over.\n");
        // }
        // printf("ic_cblock_idx = %d.\n", ic_cblock_idx);
            
        }
    }

    outputTran.startBench();
    int step = 4 * outputChannels * tileN;
    int output_size = outputHeight * outputWidth;
    #pragma omp parallel for private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
    for (int oc = 0; oc < outputChannels; oc++) {
        float *gemmBuf_p0  = gemmBuf + 4 * oc * tileN;
        float *gemmBuf_p1  = gemmBuf_p0  + step;
        float *gemmBuf_p2  = gemmBuf_p1  + step;
        float *gemmBuf_p3  = gemmBuf_p2  + step;
        float *gemmBuf_p4  = gemmBuf_p3  + step;
        float *gemmBuf_p5  = gemmBuf_p4  + step;
        float *gemmBuf_p6  = gemmBuf_p5  + step;
        float *gemmBuf_p7  = gemmBuf_p6  + step;
        float *gemmBuf_p8  = gemmBuf_p7  + step;
        float *gemmBuf_p9  = gemmBuf_p8  + step;
        float *gemmBuf_p10 = gemmBuf_p9  + step;
        float *gemmBuf_p11 = gemmBuf_p10 + step;
        float *gemmBuf_p12 = gemmBuf_p11 + step;
        float *gemmBuf_p13 = gemmBuf_p12 + step;
        float *gemmBuf_p14 = gemmBuf_p13 + step;
        float *gemmBuf_p15 = gemmBuf_p14 + step;

        // #pragma omp parallel for private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2) \
        //         firstprivate(gemmBuf_p0, gemmBuf_p1, gemmBuf_p2, gemmBuf_p3, gemmBuf_p4, gemmBuf_p5, gemmBuf_p6, gemmBuf_p7, gemmBuf_p8, gemmBuf_p9, gemmBuf_p10, gemmBuf_p11, gemmBuf_p12, gemmBuf_p13, gemmBuf_p14, gemmBuf_p15)
        for (int tile = 0; tile < tileN; tile++) {
            l0 = vld1q_f32(gemmBuf_p0); 	gemmBuf_p0+=4;
            r0 = vld1q_f32(gemmBuf_p1);	    gemmBuf_p1+=4;
            l1 = vld1q_f32(gemmBuf_p2);	    gemmBuf_p2+=4;
            r1 = vld1q_f32(gemmBuf_p3);	    gemmBuf_p3+=4;
            l2 = vld1q_f32(gemmBuf_p4);	    gemmBuf_p4+=4;
            r2 = vld1q_f32(gemmBuf_p5);	    gemmBuf_p5+=4;
            l3 = vld1q_f32(gemmBuf_p6);	    gemmBuf_p6+=4;
            r3 = vld1q_f32(gemmBuf_p7);	    gemmBuf_p7+=4;
            l4 = vld1q_f32(gemmBuf_p8);	    gemmBuf_p8+=4;
            r4 = vld1q_f32(gemmBuf_p9);	    gemmBuf_p9+=4;
            l5 = vld1q_f32(gemmBuf_p10);	gemmBuf_p10+=4;
            r5 = vld1q_f32(gemmBuf_p11);	gemmBuf_p11+=4;
            l6 = vld1q_f32(gemmBuf_p12);	gemmBuf_p12+=4;
            r6 = vld1q_f32(gemmBuf_p13);	gemmBuf_p13+=4;
            l7 = vld1q_f32(gemmBuf_p14);	gemmBuf_p14+=4;
            r7 = vld1q_f32(gemmBuf_p15);	gemmBuf_p15+=4;

            winograd_f6k3_output_transform_inplace_v1(l0, l1, l2, l3, l4, l5, l6, l7);
            winograd_f6k3_output_transform_inplace_v1(r0, r1, r2, r3, r4, r5, r6, r7);
            neon_transpose4x4_inplace_f32_cpp_v1(l0, l1, l2, l3);
            neon_transpose4x4_inplace_f32_cpp_v1(r0, r1, r2, r3);
            neon_transpose4x4_inplace_f32_cpp_v1(l4, l5, l6, l7);
            neon_transpose4x4_inplace_f32_cpp_v1(r4, r5, r6, r7);
            winograd_f6k3_output_transform_inplace_v1(l0, l1, l2, l3, r0, r1, r2, r3);
            winograd_f6k3_output_transform_inplace_v1(l4, l5, l6, l7, r4, r5, r6, r7);

            int tileRow = (tile) / tileW * 6;
            int tileCol = (tile) % tileW * 6;

            float* output_p0 = baseResult + oc*output_size + tileRow*outputWidth + tileCol;
            int deltaX = outputHeight - tileRow;
            int deltaY = outputWidth - tileCol;
            // deltaX = deltaX<6?deltaX:6;
            // deltaY = deltaY<6?deltaY:6;
            if (deltaX >= 6 && deltaY >= 6) {
                float* output_p1 = output_p0 + outputWidth;
                float* output_p2 = output_p1 + outputWidth;
                float* output_p3 = output_p2 + outputWidth;
                float* output_p4 = output_p3 + outputWidth;
                float* output_p5 = output_p4 + outputWidth;

                vst1q_f32(output_p0, l0);
                vst1_f32(output_p0 + 4, vget_low_f32(l4));
                vst1q_f32(output_p1,  l1);
                vst1_f32(output_p1 + 4, vget_low_f32(l5));
                vst1q_f32(output_p2, l2);
                vst1_f32(output_p2 + 4, vget_low_f32(l6));
                vst1q_f32(output_p3, l3);
                vst1_f32(output_p3 + 4, vget_low_f32(l7));
                vst1q_f32(output_p4, r0);
                vst1_f32(output_p4 + 4, vget_low_f32(r4));
                vst1q_f32(output_p5, r1);
                vst1_f32(output_p5 + 4, vget_low_f32(r5));
            } else {
                // printf("------------------not entouasd\n");
                deltaX = min(6, deltaX);
                deltaY = min(6, deltaY);
                float tmp[36];
                memset(tmp, 0, 36*sizeof(float));
                vst1q_f32(tmp, l0);
                vst1_f32(tmp   + 4, vget_low_f32(l4));
                vst1q_f32(tmp  + 6, l1);
                vst1_f32(tmp  + 10, vget_low_f32(l5));
                vst1q_f32(tmp + 12, l2);
                vst1_f32(tmp  + 16, vget_low_f32(l6));
                vst1q_f32(tmp + 18, l3);
                vst1_f32(tmp  + 22, vget_low_f32(l7));
                vst1q_f32(tmp + 24, r0);
                vst1_f32(tmp  + 28, vget_low_f32(r4));
                vst1q_f32(tmp + 30, r1);
                vst1_f32(tmp  + 34, vget_low_f32(r5));

                for(int u = 0; u < deltaX; u++) 
                    memcpy(output_p0 + u*outputWidth, tmp + u*6, deltaY*sizeof(float));

            }
            
        }
    }
    outputTran.accumBench();
    
    // printf("outputTransform over.\n");
    // kernelTran.printBench("kernel", 1);
    // inputTran.printBench("inputTran", 1);
    // GEMM.printBench("TensorGEMM", 1);
    // outputTran.printBench("outputTran", 1);
    // _mm_free(transform_kernel);
    // _mm_free(transform_input);
    // _mm_free(gemm_buf);
    // printf("---------------------1 round over---------------.\n");

    return 1;
}

int winoF63_v1_multi_v1(float *baseResult, float *testInput, float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int padWidth, int padHeight, int strideWidth, int strideHeight, int tileBlock, float *gemmBuf, int ocBlock, float *kernelBuf, int icBlock, float *inputBuf, int tileRegBlock, int ocRegBlock, int scheduling, int num_threads,
                Timer& inputTran, Timer& kernelTran, Timer& GEMM, Timer& outputTran)
{
    /*
        outputBuf: tileBlock*ocBlock*48
        kernelBuf: icBlock*ocBlock*64
        inputBuf:  tileBlock*icBlock*64
    */
    // printf("multi_v1.\n");
    int enableOffKernel = scheduling%2;
    int loopOrdering    = scheduling/2;
    int outputHeight = inputHeight+ padHeight*2 -2;
    int outputWidth  = inputWidth + padWidth*2  -2;
    int tileH = (inputHeight+ padHeight*2 -2 + 5)/6;
    int tileW = (inputWidth + padWidth*2  -2 + 5)/6;
    int tileN = tileH*tileW;


    // assert(inputChannels%icBlock  == 0);
    assert(tileBlock%tileRegBlock == 0);
    assert(ocBlock%ocRegBlock     == 0);


    // Timer inputTran, kernelTran, GEMM, outputTran, storeBack;

    const float32x4_t f5    = vdupq_n_f32(5.0f);
    const float32x4_t f4    = vdupq_n_f32(4.0f);
    const float32x4_t f2    = vdupq_n_f32(2.0f);
    const float32x4_t f2_5  = vdupq_n_f32(2.5f);
    const float32x4_t f5_25 = vdupq_n_f32(5.25f);
    const float32x4_t f4_25 = vdupq_n_f32(4.25f);
    const float32x4_t f1_25 = vdupq_n_f32(1.25f);
    const float32x4_t f0_5  = vdupq_n_f32(0.5f);
    const float32x4_t f0_25 = vdupq_n_f32(0.25f);
    const float32x4_t vZero = vdupq_n_f32(0.0f);

    float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    float32x4_t m1, m2, s1, s2, t1, t2;//Auxiliary registers

    int ic_block = icBlock;
    int oc_block = ocBlock;
    int ti_block = tileBlock;

    // _mm_free(inputBuf);
    // _mm_free(kernelBuf);
    // _mm_free(gemmBuf);

    

    void (*tensorGEMM)(float *, const float *, float *, int, int); 
    void (*tensorGEMM_for_corner_case)(float *, const float *, float *, int, int);
    if (ocRegBlock == 4) {
        if (tileRegBlock == 4) {
            tensorGEMM = TensorGEMMInnerKernel4x4x4_v3;
            // tensorGEMM = TensorGEMMInnerKernel4x4x4_v1;
            int remain = outputChannels % ocRegBlock;
            if (remain == 3) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x3x4_v1;
            } else if (remain == 2) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x2x4_v1;
            } else if (remain == 1) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x1x4_v1;
            }
        } else if (tileRegBlock == 5) {
            tensorGEMM = TensorGEMMInnerKernel4x4x5_v1;
            // tensorGEMM = TensorGEMMInnerKernel4x4x5_v2;
            int remain = outputChannels % ocRegBlock;
            if (remain == 3) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x3x5_v1;
            } else if (remain == 2) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x2x5_v1;
            } else if (remain == 1) {
                tensorGEMM_for_corner_case = TensorGEMMInnerKernel4x1x5_v1;
            }
        }
    }

    for (int ib = 0; ib < inputChannels; ib += ic_block) {
        int ib_step = min(ic_block, inputChannels - ib);
        for (int ob = 0; ob < outputChannels; ob += oc_block) {
            int ob_step = min(oc_block, outputChannels - ob);
            if (!enableOffKernel) {
                // printf("kernel transform start.\n");
                kernelTran.startBench();
                // memset(kernelBuf, sizeof(float) * (64 * ocBlock * icBlock + 128), 0);
                int tk_offset = 4 * ocRegBlock * ic_block;
                #pragma omp parallel for private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
                for (int orb = 0; orb < ob_step; orb += ocRegBlock) {
                    int orb_step = min(ocRegBlock, ob_step - orb);
                    float *kernelBuf_copy = kernelBuf  + 64 * orb * ic_block;
                    float *kernel_copy    = testKernel + 9  * (ob + orb) * inputChannels;
                    float *tk0 = kernelBuf_copy;
                    float* tk1  = tk0  + tk_offset;
                    float* tk2  = tk1  + tk_offset;
                    float* tk3  = tk2  + tk_offset;
                    float* tk4  = tk3  + tk_offset;
                    float* tk5  = tk4  + tk_offset;
                    float* tk6  = tk5  + tk_offset;
                    float* tk7  = tk6  + tk_offset;
                    float* tk8  = tk7  + tk_offset;
                    float* tk9  = tk8  + tk_offset;
                    float* tk10 = tk9  + tk_offset;
                    float* tk11 = tk10 + tk_offset;
                    float* tk12 = tk11 + tk_offset;
                    float* tk13 = tk12 + tk_offset;
                    float* tk14 = tk13 + tk_offset;
                    float* tk15 = tk14 + tk_offset;

                    for (int ic_index = 0; ic_index < ib_step; ic_index++) {
                        for (int oc_index = 0; oc_index < orb_step; oc_index++) {
                            float *kernel_p = kernel_copy + 9 * oc_index * inputChannels + 9 * (ib + ic_index);
                            l0 = vld1q_f32(kernel_p);
                            l1 = vld1q_f32(kernel_p + 3);
                            l2 = vld1q_f32(kernel_p + 6);

                            // kernel transform
                            winograd_f6k3_kernel_transform_v1(l0, l1, l2, m1, m2, s1, l7, t1, t2, s2, r7, f4, f2);
                            neon_transpose4x4_inplace_f32_cpp_v1(m1, m2, s1, l7);
                            neon_transpose4x4_inplace_f32_cpp_v1(t1, t2, s2, r7);
                            winograd_f6k3_kernel_transform_v1(m1, m2, s1,  l0, l1, l2, l3, l4, l5, l6, l7, f4, f2);
                            winograd_f6k3_kernel_transform_v1(t1, t2, s2,  r0, r1, r2, r3, r4, r5, r6, r7, f4, f2);

                            vst1q_f32(tk0,   l0);   tk0  +=4;
                            vst1q_f32(tk1,   r0);   tk1  +=4;
                            vst1q_f32(tk2,   l1);   tk2  +=4;
                            vst1q_f32(tk3,   r1);   tk3  +=4;
                            vst1q_f32(tk4,   l2);   tk4  +=4;
                            vst1q_f32(tk5,   r2);   tk5  +=4;
                            vst1q_f32(tk6,   l3);   tk6  +=4;
                            vst1q_f32(tk7,   r3);   tk7  +=4;
                            vst1q_f32(tk8,   l4);   tk8  +=4;
                            vst1q_f32(tk9,   r4);   tk9  +=4;
                            vst1q_f32(tk10,  l5);   tk10 +=4;
                            vst1q_f32(tk11,  r5);   tk11 +=4;
                            vst1q_f32(tk12,  l6);   tk12 +=4;
                            vst1q_f32(tk13,  r6);   tk13 +=4;
                            vst1q_f32(tk14,  l7);   tk14 +=4;
                            vst1q_f32(tk15,  r7);   tk15 +=4;
                        }
                    }
                }
                kernelTran.accumBench();
            } else {
                // offline_kernel_transform();
            }

            // printf("kernel_transform over.\n");
        
            for (int tb = 0; tb < tileN; tb += tileBlock) {
                int tb_step = min(tileBlock, tileN - tb);
                int tb_step_align = align_ceil(tb_step, tileRegBlock);
                int trb_remain_start = tb_step - tb_step % tileRegBlock;
                if (ob == 0) {
                    // printf("input transform start.\n");
                    inputTran.startBench();
                    int ti_offset = 4 * tileRegBlock * ic_block;
                    float *inputBuf_copy = inputBuf + 64 * tb * ic_block;
                    #pragma omp parallel for collapse(2) private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
                    for (int trb = 0; trb < trb_remain_start; trb += tileRegBlock) {
                        for (int ic_index = 0; ic_index < ib_step; ic_index++) {
                            float *ti0  = inputBuf_copy + 64 * trb * ic_block + 4 * tileRegBlock * ic_index;
                            float *ti1  = ti0  + ti_offset;
                            float *ti2  = ti1  + ti_offset;
                            float *ti3  = ti2  + ti_offset;
                            float *ti4  = ti3  + ti_offset;
                            float *ti5  = ti4  + ti_offset;
                            float *ti6  = ti5  + ti_offset;
                            float *ti7  = ti6  + ti_offset;
                            float *ti8  = ti7  + ti_offset;
                            float *ti9  = ti8  + ti_offset;
                            float *ti10 = ti9  + ti_offset;
                            float *ti11 = ti10 + ti_offset;
                            float *ti12 = ti11 + ti_offset;
                            float *ti13 = ti12 + ti_offset;
                            float *ti14 = ti13 + ti_offset;
                            float *ti15 = ti14 + ti_offset;

                            for (int tile_index = 0; tile_index < tileRegBlock; tile_index++) {
                                int tile = tb + trb + tile_index;
                                int tile_row = (tile / tileW) * 6 - padHeight;
                                int tile_col = (tile % tileW) * 6 - padWidth;
                                int in_pos   = tile_row * inputWidth + tile_col;

                                float *in_p0 = testInput + (ib + ic_index) * inputWidth * inputHeight + in_pos;
                                float *in_p1 = in_p0 + inputWidth;
                                float *in_p2 = in_p1 + inputWidth;
                                float *in_p3 = in_p2 + inputWidth;
                                float *in_p4 = in_p3 + inputWidth;
                                float *in_p5 = in_p4 + inputWidth;
                                float *in_p6 = in_p5 + inputWidth;
                                float *in_p7 = in_p6 + inputWidth;

                                if(tile_row >= 0 && tile_row+8 <= inputHeight && tile_col >= 0 && tile_col+8 <= inputWidth) {
                                    l0 = vld1q_f32(in_p0);
                                    r0 = vld1q_f32(in_p0 + 4);
                                    l1 = vld1q_f32(in_p1);
                                    r1 = vld1q_f32(in_p1 + 4);
                                    l2 = vld1q_f32(in_p2);
                                    r2 = vld1q_f32(in_p2 + 4);
                                    l3 = vld1q_f32(in_p3);
                                    r3 = vld1q_f32(in_p3 + 4);
                                    l4 = vld1q_f32(in_p4);
                                    r4 = vld1q_f32(in_p4 + 4);
                                    l5 = vld1q_f32(in_p5);
                                    r5 = vld1q_f32(in_p5 + 4);
                                    l6 = vld1q_f32(in_p6);
                                    r6 = vld1q_f32(in_p6 + 4);
                                    l7 = vld1q_f32(in_p7);
                                    r7 = vld1q_f32(in_p7 + 4);
                                } else {
                                    float tmp[64];
                                    memset(tmp, 0, 256);
                                    int XStart = tile_row < 0 ? 0 : tile_row; 
                                    int YStart = tile_col < 0 ? 0 : tile_col;
                                    XStart     = XStart < inputHeight ? XStart : inputHeight;
                                    YStart     = YStart < inputWidth  ? YStart : inputWidth;

                                    int XEnd   = tile_row+8 < inputHeight ? tile_row+8 : inputHeight;
                                    int YEnd   = tile_col+8 < inputWidth  ? tile_col+8 : inputWidth;
                                    XEnd       = XEnd < 0 ? 0 : XEnd;
                                    YEnd       = YEnd < 0 ? 0 : YEnd;
                                    int YStep  = YEnd - YStart;
                                    int XStep  = XEnd - XStart;

                                    int len    = YStep * sizeof(float);
                                    float *pLocal = testInput + (ib + ic_index) * inputHeight * inputWidth + XStart * inputWidth + YStart;
                                    float *ptmp   = tmp + (XStart - tile_row) * 8 + YStart - tile_col;
                                    for(int ux = 0; ux < XStep; ux++, ptmp += 8, pLocal += inputWidth) {
                                        memcpy(ptmp, pLocal, len);
                                    }

                                    l0 = vld1q_f32(tmp);
                                    r0 = vld1q_f32(tmp + 4);
                                    l1 = vld1q_f32(tmp + 8);
                                    r1 = vld1q_f32(tmp + 12);
                                    l2 = vld1q_f32(tmp + 16);
                                    r2 = vld1q_f32(tmp + 20);
                                    l3 = vld1q_f32(tmp + 24);
                                    r3 = vld1q_f32(tmp + 28);
                                    l4 = vld1q_f32(tmp + 32);
                                    r4 = vld1q_f32(tmp + 36);
                                    l5 = vld1q_f32(tmp + 40);
                                    r5 = vld1q_f32(tmp + 44);
                                    l6 = vld1q_f32(tmp + 48);
                                    r6 = vld1q_f32(tmp + 52);
                                    l7 = vld1q_f32(tmp + 56);
                                    r7 = vld1q_f32(tmp + 60);
                                }

                                input_transform_v1(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                                neon_transpose4x4_inplace_f32_cpp_v1(l0, l1, l2, l3);
                                neon_transpose4x4_inplace_f32_cpp_v1(l4, l5, l6, l7);
                                input_transform_v1(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                                neon_transpose4x4_inplace_f32_cpp_v1(r0, r1, r2, r3);
                                neon_transpose4x4_inplace_f32_cpp_v1(r4, r5, r6, r7);

                                input_transform_v1(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                                input_transform_v1(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

                                vst1q_f32(ti0,  l0);    ti0  += 4;
                                vst1q_f32(ti1,  l4);    ti1  += 4;
                                vst1q_f32(ti2,  l1);    ti2  += 4;
                                vst1q_f32(ti3,  l5);    ti3  += 4;
                                vst1q_f32(ti4,  l2);    ti4  += 4;
                                vst1q_f32(ti5,  l6);    ti5  += 4;
                                vst1q_f32(ti6,  l3);    ti6  += 4;
                                vst1q_f32(ti7,  l7);    ti7  += 4;
                                vst1q_f32(ti8,  r0);    ti8  += 4;
                                vst1q_f32(ti9,  r4);    ti9  += 4;
                                vst1q_f32(ti10, r1);    ti10 += 4;
                                vst1q_f32(ti11, r5);    ti11 += 4;
                                vst1q_f32(ti12, r2);    ti12 += 4;
                                vst1q_f32(ti13, r6);    ti13 += 4;
                                vst1q_f32(ti14, r3);    ti14 += 4;
                                vst1q_f32(ti15, r7);    ti15 += 4;
                            }
                        }
                    }

                    if (trb_remain_start < tb_step) {
                        int trb = trb_remain_start;
                        float *inputBuf_copy_copy = inputBuf_copy + 64 * trb * ic_block;
                        int trb_remain = tb_step - trb_remain_start;
                        int ti_offset = 4 * tileRegBlock * ic_block; 

                        for (int ic_index = 0; ic_index < ib_step; ic_index++) {
                            float *ti0  = inputBuf_copy_copy + 4 * tileRegBlock * ic_index;
                            float *ti1  = ti0  + ti_offset;
                            float *ti2  = ti1  + ti_offset;
                            float *ti3  = ti2  + ti_offset;
                            float *ti4  = ti3  + ti_offset;
                            float *ti5  = ti4  + ti_offset;
                            float *ti6  = ti5  + ti_offset;
                            float *ti7  = ti6  + ti_offset;
                            float *ti8  = ti7  + ti_offset;
                            float *ti9  = ti8  + ti_offset;
                            float *ti10 = ti9  + ti_offset;
                            float *ti11 = ti10 + ti_offset;
                            float *ti12 = ti11 + ti_offset;
                            float *ti13 = ti12 + ti_offset;
                            float *ti14 = ti13 + ti_offset;
                            float *ti15 = ti14 + ti_offset; 

                            for (int tile_index = 0; tile_index < trb_remain; tile_index++) {
                                int tile = tb + trb + tile_index;
                                int tile_row = (tile / tileW) * 6 - padHeight;
                                int tile_col = (tile % tileW) * 6 - padWidth;
                                int in_pos   = tile_row * inputWidth + tile_col;

                                float *in_p0 = testInput + (ib + ic_index) * inputWidth * inputHeight + in_pos;
                                float *in_p1 = in_p0 + inputWidth;
                                float *in_p2 = in_p1 + inputWidth;
                                float *in_p3 = in_p2 + inputWidth;
                                float *in_p4 = in_p3 + inputWidth;
                                float *in_p5 = in_p4 + inputWidth;
                                float *in_p6 = in_p5 + inputWidth;
                                float *in_p7 = in_p6 + inputWidth;

                                
                                if(tile_row >= 0 && tile_row+8 <= inputHeight && tile_col >= 0 && tile_col+8 <= inputWidth) {
                                    l0 = vld1q_f32(in_p0);
                                    r0 = vld1q_f32(in_p0 + 4);
                                    l1 = vld1q_f32(in_p1);
                                    r1 = vld1q_f32(in_p1 + 4);
                                    l2 = vld1q_f32(in_p2);
                                    r2 = vld1q_f32(in_p2 + 4);
                                    l3 = vld1q_f32(in_p3);
                                    r3 = vld1q_f32(in_p3 + 4);
                                    l4 = vld1q_f32(in_p4);
                                    r4 = vld1q_f32(in_p4 + 4);
                                    l5 = vld1q_f32(in_p5);
                                    r5 = vld1q_f32(in_p5 + 4);
                                    l6 = vld1q_f32(in_p6);
                                    r6 = vld1q_f32(in_p6 + 4);
                                    l7 = vld1q_f32(in_p7);
                                    r7 = vld1q_f32(in_p7 + 4);
                                } else {
                                    float tmp[64];
                                    memset(tmp, 0, 256);
                                    int XStart = tile_row < 0 ? 0 : tile_row; 
                                    int YStart = tile_col < 0 ? 0 : tile_col;
                                    XStart     = XStart < inputHeight ? XStart : inputHeight;
                                    YStart     = YStart < inputWidth  ? YStart : inputWidth;

                                    int XEnd   = tile_row+8 < inputHeight ? tile_row+8 : inputHeight;
                                    int YEnd   = tile_col+8 < inputWidth  ? tile_col+8 : inputWidth;
                                    XEnd       = XEnd < 0 ? 0 : XEnd;
                                    YEnd       = YEnd < 0 ? 0 : YEnd;
                                    int YStep  = YEnd - YStart;
                                    int XStep  = XEnd - XStart;

                                    int len    = YStep * sizeof(float);
                                    float *pLocal = testInput + (ib + ic_index) * inputHeight * inputWidth + XStart * inputWidth + YStart;
                                    float *ptmp   = tmp + (XStart - tile_row) * 8 + YStart - tile_col;
                                    for(int ux = 0; ux < XStep; ux++, ptmp += 8, pLocal += inputWidth) {
                                        memcpy(ptmp, pLocal, len);
                                    }

                                    l0 = vld1q_f32(tmp);
                                    r0 = vld1q_f32(tmp + 4);
                                    l1 = vld1q_f32(tmp + 8);
                                    r1 = vld1q_f32(tmp + 12);
                                    l2 = vld1q_f32(tmp + 16);
                                    r2 = vld1q_f32(tmp + 20);
                                    l3 = vld1q_f32(tmp + 24);
                                    r3 = vld1q_f32(tmp + 28);
                                    l4 = vld1q_f32(tmp + 32);
                                    r4 = vld1q_f32(tmp + 36);
                                    l5 = vld1q_f32(tmp + 40);
                                    r5 = vld1q_f32(tmp + 44);
                                    l6 = vld1q_f32(tmp + 48);
                                    r6 = vld1q_f32(tmp + 52);
                                    l7 = vld1q_f32(tmp + 56);
                                    r7 = vld1q_f32(tmp + 60);
                                }

                                input_transform_v1(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                                neon_transpose4x4_inplace_f32_cpp_v1(l0, l1, l2, l3);
                                neon_transpose4x4_inplace_f32_cpp_v1(l4, l5, l6, l7);
                                input_transform_v1(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                                neon_transpose4x4_inplace_f32_cpp_v1(r0, r1, r2, r3);
                                neon_transpose4x4_inplace_f32_cpp_v1(r4, r5, r6, r7);

                                input_transform_v1(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                                input_transform_v1(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                                                t1, t2, s1, s2, m1, m2, //Auxiliary
                                                f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

                                vst1q_f32(ti0,  l0);    ti0  += 4;
                                vst1q_f32(ti1,  l4);    ti1  += 4;
                                vst1q_f32(ti2,  l1);    ti2  += 4;
                                vst1q_f32(ti3,  l5);    ti3  += 4;
                                vst1q_f32(ti4,  l2);    ti4  += 4;
                                vst1q_f32(ti5,  l6);    ti5  += 4;
                                vst1q_f32(ti6,  l3);    ti6  += 4;
                                vst1q_f32(ti7,  l7);    ti7  += 4;
                                vst1q_f32(ti8,  r0);    ti8  += 4;
                                vst1q_f32(ti9,  r4);    ti9  += 4;
                                vst1q_f32(ti10, r1);    ti10 += 4;
                                vst1q_f32(ti11, r5);    ti11 += 4;
                                vst1q_f32(ti12, r2);    ti12 += 4;
                                vst1q_f32(ti13, r6);    ti13 += 4;
                                vst1q_f32(ti14, r3);    ti14 += 4;
                                vst1q_f32(ti15, r7);    ti15 += 4;
                            }

                            for (int tile_index = trb_remain; tile_index < tileRegBlock; tile_index++) {
                                vst1q_f32(ti0,  vZero);    ti0  += 4;
                                vst1q_f32(ti1,  vZero);    ti1  += 4;
                                vst1q_f32(ti2,  vZero);    ti2  += 4;
                                vst1q_f32(ti3,  vZero);    ti3  += 4;
                                vst1q_f32(ti4,  vZero);    ti4  += 4;
                                vst1q_f32(ti5,  vZero);    ti5  += 4;
                                vst1q_f32(ti6,  vZero);    ti6  += 4;
                                vst1q_f32(ti7,  vZero);    ti7  += 4;
                                vst1q_f32(ti8,  vZero);    ti8  += 4;
                                vst1q_f32(ti9,  vZero);    ti9  += 4;
                                vst1q_f32(ti10, vZero);    ti10 += 4;
                                vst1q_f32(ti11, vZero);    ti11 += 4;
                                vst1q_f32(ti12, vZero);    ti12 += 4;
                                vst1q_f32(ti13, vZero);    ti13 += 4;
                                vst1q_f32(ti14, vZero);    ti14 += 4;
                                vst1q_f32(ti15, vZero);    ti15 += 4;
                            }
                        }
                    }

                    inputTran.accumBench();

                    // printf("input_transform over.\n");
                }
                

                // printf("gemm start.\n");
                GEMM.startBench();
                int orb_remain_start = ob_step - ob_step % ocRegBlock;
                int block_size = 64 * ocBlock * tileBlock;
                int rblock_size = 64 * ocRegBlock * tileRegBlock;
                int output_block_col  = (tileN + tileBlock - 1) / tileBlock;
                int output_rblock_col = tileBlock / tileRegBlock;
                int input_offset  = 4 * tileRegBlock * ic_block;
                int kernel_offset = 4 * ocRegBlock   * ic_block;
                int output_offset = 4 * tileRegBlock * ocRegBlock;
                float *input_copy  = inputBuf + 64 * tb * ic_block;
                float *output_copy = gemmBuf  + (ob / oc_block * output_block_col + tb / tileBlock) * block_size;
                #pragma omp parallel for collapse(2) 
                for (int orb = 0; orb < orb_remain_start; orb += ocRegBlock) {
                    for (int trb = 0; trb < tb_step_align; trb += tileRegBlock) {
                        float *input_p  = input_copy + 64 * trb * ic_block;
                        float *kernel_p = kernelBuf  + 64 * orb * ic_block;
                        float *output_p = output_copy + (orb / ocRegBlock * output_rblock_col + trb / tileRegBlock) * rblock_size;
                        for (int depth = 0; depth < 16; depth++) {
                            tensorGEMM(output_p, kernel_p, input_p, ib_step, tileRegBlock);
                            input_p  += input_offset;
                            kernel_p += kernel_offset;
                            output_p += output_offset;
                        }
                    }
                }

                if (orb_remain_start < ob_step) {
                    for (int trb = 0; trb < tb_step_align; trb += tileRegBlock) {
                        float *input_p  = input_copy + 64 * trb * ic_block;
                        float *kernel_p = kernelBuf  + 64 * orb_remain_start * ic_block;
                        float *output_p = output_copy + (orb_remain_start / ocRegBlock * output_rblock_col + trb / tileRegBlock) * rblock_size;
                        for (int depth = 0; depth < 16; depth++) {
                            tensorGEMM_for_corner_case(output_p, kernel_p, input_p, ib_step, tileRegBlock);
                            input_p  += input_offset;
                            kernel_p += kernel_offset;
                            output_p += output_offset;
                        }
                    }
                }
                GEMM.accumBench();

                // printf("gemm over.\n");

            }
        }
    }

    // printf("output transform start.\n");
    outputTran.startBench();
    int output_size = outputHeight * outputWidth;
    int block_size = 64 * ocBlock * tileBlock;
    int rblock_size = 64 * ocRegBlock * tileRegBlock;
    int output_block_col  = align_ceil(tileN, tileBlock) / tileBlock;
    int output_rblock_col = tileBlock / tileRegBlock;
    int output_offset = 4 * tileRegBlock * ocRegBlock;
    
    #pragma omp parallel for collapse(2) private(l0, l1, l2, l3, l4, l5, l6, l7, r0, r1, r2, r3, r4, r5, r6, r7, m1, m2, s1, s2, t1, t2)
    for (int ob = 0; ob < outputChannels; ob += oc_block) {
        for (int tb = 0; tb < tileN; tb += tileBlock) {
            int ob_step = min(ocBlock, outputChannels - ob);
            int tb_step = min(tileBlock, tileN - tb);
            float *gemmBuf_copy = gemmBuf + (ob / oc_block * output_block_col + tb / tileBlock) * block_size;
            for (int orb = 0; orb < ob_step; orb += ocRegBlock) {
                for (int trb = 0; trb < tb_step; trb += tileRegBlock) {
                    int orb_step = min(ocRegBlock, ob_step - orb);
                    int trb_step = min(tileRegBlock, tb_step - trb);
                    float *gemmBuf_copy_copy = gemmBuf_copy + (orb / ocRegBlock * output_rblock_col + trb / tileRegBlock) * rblock_size;
                    for (int orb_index = 0; orb_index < orb_step; orb_index++) {
                        float *gemmBuf_p0  = gemmBuf_copy_copy + 4 * orb_index * tileRegBlock;
                        float *gemmBuf_p1  = gemmBuf_p0  + output_offset;
                        float *gemmBuf_p2  = gemmBuf_p1  + output_offset;
                        float *gemmBuf_p3  = gemmBuf_p2  + output_offset;
                        float *gemmBuf_p4  = gemmBuf_p3  + output_offset;
                        float *gemmBuf_p5  = gemmBuf_p4  + output_offset;
                        float *gemmBuf_p6  = gemmBuf_p5  + output_offset;
                        float *gemmBuf_p7  = gemmBuf_p6  + output_offset;
                        float *gemmBuf_p8  = gemmBuf_p7  + output_offset;
                        float *gemmBuf_p9  = gemmBuf_p8  + output_offset;
                        float *gemmBuf_p10 = gemmBuf_p9  + output_offset;
                        float *gemmBuf_p11 = gemmBuf_p10 + output_offset;
                        float *gemmBuf_p12 = gemmBuf_p11 + output_offset;
                        float *gemmBuf_p13 = gemmBuf_p12 + output_offset;
                        float *gemmBuf_p14 = gemmBuf_p13 + output_offset;
                        float *gemmBuf_p15 = gemmBuf_p14 + output_offset;
                        for (int trb_index = 0; trb_index < trb_step; trb_index++) {
                            l0 = vld1q_f32(gemmBuf_p0); 	gemmBuf_p0+=4;
                            r0 = vld1q_f32(gemmBuf_p1);	    gemmBuf_p1+=4;
                            l1 = vld1q_f32(gemmBuf_p2);	    gemmBuf_p2+=4;
                            r1 = vld1q_f32(gemmBuf_p3);	    gemmBuf_p3+=4;
                            l2 = vld1q_f32(gemmBuf_p4);	    gemmBuf_p4+=4;
                            r2 = vld1q_f32(gemmBuf_p5);	    gemmBuf_p5+=4;
                            l3 = vld1q_f32(gemmBuf_p6);	    gemmBuf_p6+=4;
                            r3 = vld1q_f32(gemmBuf_p7);	    gemmBuf_p7+=4;
                            l4 = vld1q_f32(gemmBuf_p8);	    gemmBuf_p8+=4;
                            r4 = vld1q_f32(gemmBuf_p9);	    gemmBuf_p9+=4;
                            l5 = vld1q_f32(gemmBuf_p10);	gemmBuf_p10+=4;
                            r5 = vld1q_f32(gemmBuf_p11);	gemmBuf_p11+=4;
                            l6 = vld1q_f32(gemmBuf_p12);	gemmBuf_p12+=4;
                            r6 = vld1q_f32(gemmBuf_p13);	gemmBuf_p13+=4;
                            l7 = vld1q_f32(gemmBuf_p14);	gemmBuf_p14+=4;
                            r7 = vld1q_f32(gemmBuf_p15);	gemmBuf_p15+=4;

                            winograd_f6k3_output_transform_inplace_v1(l0, l1, l2, l3, l4, l5, l6, l7);
                            winograd_f6k3_output_transform_inplace_v1(r0, r1, r2, r3, r4, r5, r6, r7);
                            neon_transpose4x4_inplace_f32_cpp_v1(l0, l1, l2, l3);
                            neon_transpose4x4_inplace_f32_cpp_v1(r0, r1, r2, r3);
                            neon_transpose4x4_inplace_f32_cpp_v1(l4, l5, l6, l7);
                            neon_transpose4x4_inplace_f32_cpp_v1(r4, r5, r6, r7);
                            winograd_f6k3_output_transform_inplace_v1(l0, l1, l2, l3, r0, r1, r2, r3);
                            winograd_f6k3_output_transform_inplace_v1(l4, l5, l6, l7, r4, r5, r6, r7);

                            int tile = tb + trb + trb_index;
                            int tileRow = (tile) / tileW * 6;
                            int tileCol = (tile) % tileW * 6;
                            float* output_p0 = baseResult + (ob + orb + orb_index)*output_size + tileRow*outputWidth + tileCol;
                            int deltaX = outputHeight - tileRow;
                            int deltaY = outputWidth - tileCol;

                            if (deltaX >= 6 && deltaY >= 6) {
                                float* output_p1 = output_p0 + outputWidth;
                                float* output_p2 = output_p1 + outputWidth;
                                float* output_p3 = output_p2 + outputWidth;
                                float* output_p4 = output_p3 + outputWidth;
                                float* output_p5 = output_p4 + outputWidth;

                                vst1q_f32(output_p0, l0);
                                vst1_f32(output_p0 + 4, vget_low_f32(l4));
                                vst1q_f32(output_p1,  l1);
                                vst1_f32(output_p1 + 4, vget_low_f32(l5));
                                vst1q_f32(output_p2, l2);
                                vst1_f32(output_p2 + 4, vget_low_f32(l6));
                                vst1q_f32(output_p3, l3);
                                vst1_f32(output_p3 + 4, vget_low_f32(l7));
                                vst1q_f32(output_p4, r0);
                                vst1_f32(output_p4 + 4, vget_low_f32(r4));
                                vst1q_f32(output_p5, r1);
                                vst1_f32(output_p5 + 4, vget_low_f32(r5));
                            } else {
                                // printf("------------------not entouasd\n");
                                deltaX = min(6, deltaX);
                                deltaY = min(6, deltaY);
                                float tmp[36];
                                memset(tmp, 0, 36*sizeof(float));
                                vst1q_f32(tmp, l0);
                                vst1_f32(tmp   + 4, vget_low_f32(l4));
                                vst1q_f32(tmp  + 6, l1);
                                vst1_f32(tmp  + 10, vget_low_f32(l5));
                                vst1q_f32(tmp + 12, l2);
                                vst1_f32(tmp  + 16, vget_low_f32(l6));
                                vst1q_f32(tmp + 18, l3);
                                vst1_f32(tmp  + 22, vget_low_f32(l7));
                                vst1q_f32(tmp + 24, r0);
                                vst1_f32(tmp  + 28, vget_low_f32(r4));
                                vst1q_f32(tmp + 30, r1);
                                vst1_f32(tmp  + 34, vget_low_f32(r5));

                                for(int u = 0; u < deltaX; u++) 
                                    memcpy(output_p0 + u*outputWidth, tmp + u*6, deltaY*sizeof(float));

                            }

                        }
                    }
                }
            }
        }
    }
    
    outputTran.accumBench();

    // printf("output transform over.\n");

    // _mm_free(inputBuf);
    // _mm_free(kernelBuf);
    // _mm_free(gemmBuf);

    return 1;
}