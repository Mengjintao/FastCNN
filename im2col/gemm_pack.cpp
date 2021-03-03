#include <stdlib.h>
#include <arm_neon.h>
#include <assert.h>
#include <stdio.h>
#include <omp.h>
#include "../utility/common.h"
#include "gemm_pack.h"

#define min(i, j) ((i) < (j) ? (i): (j))
#define __ARM_ASM

// 在计算C大分块的内部4x4小分块时，对A分块中的每4行进行重排，相当于对 4 x k block of A进行重排
// a b c          a 1 e 5
// 1 2 3   --->   b 2 f 6
// e f g   --->   c 3 g 7
// 5 6 7          
// 用于GEMM_v1的packing A
void pack_a_v1(int M, int K, float* A, int lda, float* NewA)
{
    assert(K != 0 && M != 0 && K % 4 == 0 && M % 4 == 0);
    float *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    float *new_a_offset;
    float  ctemp1,  ctemp2,  ctemp3,  ctemp4;
    float  ctemp5,  ctemp6,  ctemp7,  ctemp8;
    float  ctemp9, ctemp10, ctemp11, ctemp12;
    float ctemp13, ctemp14, ctemp15, ctemp16;

    a_offset = A;
    new_a_offset = NewA;

    for (int j = 0; j < M; j += 4)
    {
        a_offset1 = a_offset;
        a_offset2 = a_offset1 + lda;
        a_offset3 = a_offset2 + lda;
        a_offset4 = a_offset3 + lda;
        a_offset += 4 * lda;
        for (int i = 0; i < K; i += 4)
        {
            ctemp1 = *(a_offset1 + 0);
            ctemp2 = *(a_offset1 + 1);
            ctemp3 = *(a_offset1 + 2);
            ctemp4 = *(a_offset1 + 3);

            ctemp5 = *(a_offset2 + 0);
            ctemp6 = *(a_offset2 + 1);
            ctemp7 = *(a_offset2 + 2);
            ctemp8 = *(a_offset2 + 3);
        
            ctemp9  = *(a_offset3 + 0);
            ctemp10 = *(a_offset3 + 1);
            ctemp11 = *(a_offset3 + 2);
            ctemp12 = *(a_offset3 + 3);
        
            ctemp13 = *(a_offset4 + 0);
            ctemp14 = *(a_offset4 + 1);
            ctemp15 = *(a_offset4 + 2);
            ctemp16 = *(a_offset4 + 3);

            *(new_a_offset + 0) = ctemp1;
            *(new_a_offset + 1) = ctemp5;
            *(new_a_offset + 2) = ctemp9;
            *(new_a_offset + 3) = ctemp13;

            *(new_a_offset + 4) = ctemp2;
            *(new_a_offset + 5) = ctemp6;
            *(new_a_offset + 6) = ctemp10;
            *(new_a_offset + 7) = ctemp14;

            *(new_a_offset + 8 ) = ctemp3;
            *(new_a_offset + 9 ) = ctemp7;
            *(new_a_offset + 10) = ctemp11;
            *(new_a_offset + 11) = ctemp15;

            *(new_a_offset + 12) = ctemp4;
            *(new_a_offset + 13) = ctemp8;
            *(new_a_offset + 14) = ctemp12;
            *(new_a_offset + 15) = ctemp16;

            a_offset1 += 4;
            a_offset2 += 4;
            a_offset3 += 4;
            a_offset4 += 4;

            new_a_offset += 16;
        }
    }
}

// 在计算C大分块的内部4x4小分块时，对B分块中的每4列进行重排，使每次进行计算时B分块中的4列处于地址连续的状态
// 原本B分块中的每4列不是地址连续的
// 相当于对 k x 4 blocks of B 进行重排
/*
suppose that k and n is mutiple of 4
pack B means
Input:
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
0 1 2 3  4 5 6 7
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
8 9 a b  c d e f
Pack it zigzag, not like pack A
Output:
0 1 2 3 0 1 2 3 0 1 2 3 0 1 2 3
8 9 a b 8 9 a b 8 9 a b 8 9 a b
4 5 6 7 4 5 6 7 4 5 6 7 4 5 6 7
c d e f c d e f c d e f c d e f
*/
// 用于GEMM_v1的packing B
void pack_b_v1(int K, int N, float *B, int ldb, float *NewB)
{
    assert(K != 0 && N != 0 && K % 4 == 0 && N % 4 == 0);
    float *a_offset, *a_offset1, *a_offset2, *a_offset3, *a_offset4;
    float *b_offset, *b_offset1;
    float  ctemp1,  ctemp2,  ctemp3,  ctemp4;
    float  ctemp5,  ctemp6,  ctemp7,  ctemp8;
    float  ctemp9, ctemp10, ctemp11, ctemp12;
    float ctemp13, ctemp14, ctemp15, ctemp16;

    a_offset = B;
    b_offset = NewB;

    for (int j = 0; j < K; j += 4)
    {
        a_offset1 = a_offset;
        a_offset2 = a_offset1 + ldb;
        a_offset3 = a_offset2 + ldb;
        a_offset4 = a_offset3 + ldb;
        a_offset += 4 * ldb;

        b_offset1 = b_offset;
        b_offset += 16;
            
        for (int i = 0; i < N; i += 4)
        {
            ctemp1 = *(a_offset1 + 0);
            ctemp2 = *(a_offset1 + 1);
            ctemp3 = *(a_offset1 + 2);
            ctemp4 = *(a_offset1 + 3);

            ctemp5  = *(a_offset2 + 0);
            ctemp6  = *(a_offset2 + 1);
            ctemp7  = *(a_offset2 + 2);
            ctemp8  = *(a_offset2 + 3);

            ctemp9  = *(a_offset3 + 0);
            ctemp10 = *(a_offset3 + 1);
            ctemp11 = *(a_offset3 + 2);
            ctemp12 = *(a_offset3 + 3);

            ctemp13 = *(a_offset4 + 0);
            ctemp14 = *(a_offset4 + 1);
            ctemp15 = *(a_offset4 + 2);
            ctemp16 = *(a_offset4 + 3);

            a_offset1 += 4;
            a_offset2 += 4;
            a_offset3 += 4;
            a_offset4 += 4;

            *(b_offset1 +  0) = ctemp1;
            *(b_offset1 +  1) = ctemp2;
            *(b_offset1 +  2) = ctemp3;
            *(b_offset1 +  3) = ctemp4;

            *(b_offset1 +  4) = ctemp5;
            *(b_offset1 +  5) = ctemp6;
            *(b_offset1 +  6) = ctemp7;
            *(b_offset1 +  7) = ctemp8;

            *(b_offset1 +  8) = ctemp9;
            *(b_offset1 +  9) = ctemp10;
            *(b_offset1 + 10) = ctemp11;
            *(b_offset1 + 11) = ctemp12;

            *(b_offset1 + 12) = ctemp13;
            *(b_offset1 + 13) = ctemp14;
            *(b_offset1 + 14) = ctemp15;
            *(b_offset1 + 15) = ctemp16;

            b_offset1 += K * 4;

        }
    }
    
}

// 对 M x kc 做8行packing
void pack_a_v2(int M, int kc_adjust, float* A, int lda, float* packA, int mc, int kc, int nc, const int ROW_BATCH, const int COL_BATCH) 
{
    // const int ROW_BATCH = 8;
    float* aPtr = A, *packAPtr = packA;
    
    // 每次处理一个panel
    for (int i = 0; i < M; i += mc)
    {
        // 确定一个Penel的高度
        int m_len = min(mc, M - i);
        int remainRowStart = m_len - m_len % ROW_BATCH;
        // 对panel内部进行packing，每次对panel内部的8行进行处理
        int m = 0;
        for (; m < remainRowStart; m += ROW_BATCH)
        {
            aPtr = A + (i + m) * lda;
            // 每次处理panel内部的8行1列
            for (int k = 0; k < kc_adjust; k++)
            {
                int aPtrIdx = 0;
                for (int j = 0; j < ROW_BATCH; j++)
                {
                    packAPtr[j] = aPtr[aPtrIdx];
                    aPtrIdx += lda;
                }
                packAPtr += ROW_BATCH;
                aPtr++;
            }
        }

        aPtr = A + (i + remainRowStart) * lda;
        // 如果剩余不足8行，直接处理
        int remainRow = m_len - remainRowStart;
        // 每次处理panel内部的1列
        for (int k = 0; k < kc_adjust; k++)
        {
            int aPtrIdx = 0;
            for (int j = 0; j < remainRow; j++)
            {
                packAPtr[j] = aPtr[aPtrIdx];
                aPtrIdx += lda;
            }
            packAPtr += remainRow;
            aPtr++;
        } 
    }
}

// 对 kc x nc 做8列的packing
void pack_b_v2_8x8(int kc_adjust, int nc_adjust, float *B, int ldb, float *packB, const int ROW_BATCH, const int COL_BATCH)
{
    float *BPtr, *packBPtr;
    int remainColStart = nc_adjust - nc_adjust % COL_BATCH;

    // 每次处理B的一行
    for (int k = 0; k < kc_adjust; k++) {
        BPtr = B + ldb * k;
        packBPtr = packB + COL_BATCH * k;
        int step = COL_BATCH * kc_adjust;

        // 在B的一行中一次拷贝8个元素
        int j = 0;
        for (; j < remainColStart; j += COL_BATCH) {

#ifdef __ARM_ASM
            asm volatile(
                // "prfm   pldl1keep, [%0, #256]   \n"
                "ld1    {v0.4s, v1.4s}, [%0]    \n"
                "st1    {v0.4s, v1.4s}, [%1]    \n"
                : "=r"(BPtr),  // %0
                "=r"(packBPtr) // %1
                : "0"(BPtr),
                "1"(packBPtr)
                : "cc", "memory", "v0", "v1");

#else
            float32x4_t a0, a1;
            a0 = vld1q_f32(BPtr    );
            a1 = vld1q_f32(BPtr + 4);

            vst1q_f32(packBPtr    , a0);
            vst1q_f32(packBPtr + 4, a1);

#endif
            BPtr += COL_BATCH;
            packBPtr += step;
        }

        // 如果B的一行中还剩余几个（不足8个），直接拷贝，其他值直接设为0（相当于不足8个填充为8个）
        if (j < nc_adjust) {
            int remain = nc_adjust - j;
            packBPtr = packB + j * kc_adjust + COL_BATCH * k;
            for (int i = 0; i < remain; i++) {
                packBPtr[i] = BPtr[i];
                // printf("_b_%f\n", BPtr[i]);
                // *packBPtr = *BPtr;
                // packBPtr++;
                // BPtr++;
            }

            for (int i = remain; i < COL_BATCH; i++) {
                packBPtr[i] = 0.0f;
                // *packBPtr = 0.0f;
                // packBPtr++;
            }
        }
    }
}

void pack_b_v2_8x12(int kc_adjust, int nc_adjust, float *B, int ldb, float *packB, const int ROW_BATCH, const int COL_BATCH)
{
    float *BPtr, *packBPtr;
    int remainColStart = nc_adjust - nc_adjust % COL_BATCH;

    // 每次处理B的一行
    for (int k = 0; k < kc_adjust; k++) {
        BPtr = B + ldb * k;
        packBPtr = packB + COL_BATCH * k;
        int step = COL_BATCH * kc_adjust;

        // 在B的一行中一次拷贝8个元素
        int j = 0;
        for (; j < remainColStart; j += COL_BATCH) {

#ifdef __ARM_ASM
            // 用于kernel 8 x 12
            asm volatile(
                // "prfm   pldl1keep, [%0, #128]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s}, [%0]    \n"
                "st1    {v0.4s, v1.4s, v2.4s}, [%1]    \n"
                : "=r"(BPtr),  // %0
                "=r"(packBPtr) // %1
                : "0"(BPtr),
                "1"(packBPtr)
                : "cc", "memory", "v0", "v1", "v2");

            
#else
            // kernel 8 x 12
            float32x4_t a0, a1, a2;
            a0 = vld1q_f32(BPtr    );
            a1 = vld1q_f32(BPtr + 4);
            a2 = vld1q_f32(BPtr + 8);

            vst1q_f32(packBPtr    , a0);
            vst1q_f32(packBPtr + 4, a1);
            vst1q_f32(packBPtr + 8, a2);

#endif
            BPtr += COL_BATCH;
            packBPtr += step;
        }

        // 如果B的一行中还剩余几个（不足8个），直接拷贝，其他值直接设为0（相当于不足8个填充为8个）
        if (j < nc_adjust) {
            int remain = nc_adjust - j;
            packBPtr = packB + j * kc_adjust + COL_BATCH * k;
            for (int i = 0; i < remain; i++) {
                packBPtr[i] = BPtr[i];
                // printf("_b_%f\n", BPtr[i]);
                // *packBPtr = *BPtr;
                // packBPtr++;
                // BPtr++;
            }

            for (int i = remain; i < COL_BATCH; i++) {
                packBPtr[i] = 0.0f;
                // *packBPtr = 0.0f;
                // packBPtr++;
            }
        }
    }
}

void pack_b_v2_4x16(int kc_adjust, int nc_adjust, float *B, int ldb, float *packB, const int ROW_BATCH, const int COL_BATCH)
{
    float *BPtr, *packBPtr;
    int remainColStart = nc_adjust - nc_adjust % COL_BATCH;

    // 每次处理B的一行
    for (int k = 0; k < kc_adjust; k++) {
        BPtr = B + ldb * k;
        packBPtr = packB + COL_BATCH * k;
        int step = COL_BATCH * kc_adjust;

        // 在B的一行中一次拷贝8个元素
        int j = 0;
        for (; j < remainColStart; j += COL_BATCH) {

#ifdef __ARM_ASM
            // 用于kernel 4 x 16
            asm volatile(
                // "prfm   pldl1keep, [%0, #128]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0]    \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1]    \n"
                : "=r"(BPtr),  // %0
                "=r"(packBPtr) // %1
                : "0"(BPtr),
                "1"(packBPtr)
                : "cc", "memory", "v0", "v1", "v2", "v3");

#else
            // kernel 4 x 16
            float32x4_t a0, a1, a2, a3;
            a0 = vld1q_f32(BPtr    );
            a1 = vld1q_f32(BPtr + 4);
            a2 = vld1q_f32(BPtr + 8);
            a3 = vld1q_f32(BPtr + 12);

            vst1q_f32(packBPtr    , a0);
            vst1q_f32(packBPtr + 4, a1);
            vst1q_f32(packBPtr + 8, a2);
            vst1q_f32(packBPtr + 12, a3);

#endif
            BPtr += COL_BATCH;
            packBPtr += step;
        }

        // 如果B的一行中还剩余几个（不足8个），直接拷贝，其他值直接设为0（相当于不足8个填充为8个）
        if (j < nc_adjust) {
            int remain = nc_adjust - j;
            packBPtr = packB + j * kc_adjust + COL_BATCH * k;
            for (int i = 0; i < remain; i++) {
                packBPtr[i] = BPtr[i];
                // printf("_b_%f\n", BPtr[i]);
                // *packBPtr = *BPtr;
                // packBPtr++;
                // BPtr++;
            }

            for (int i = remain; i < COL_BATCH; i++) {
                packBPtr[i] = 0.0f;
                // *packBPtr = 0.0f;
                // packBPtr++;
            }
        }
    }
}

// 对 kc x nc 做8列的packing（多线程版本）
// 这个版本在v2的基础上支持多线程并行packing
void pack_b_multithread_kernel_8x8_v2(int kc_adjust, int k_interval, int nc_adjust, float *B, int ldb, float *packB, const int ROW_BATCH, const int COL_BATCH)
{
    float *BPtr = B, *packBPtr;
    int remainColStart = nc_adjust - nc_adjust % COL_BATCH;

    // 每次处理B的一行
    for (int k = 0; k < k_interval; k++) {
        BPtr = B + ldb * k;
        packBPtr = packB + COL_BATCH * k;
        int step = COL_BATCH * kc_adjust;

        // 在B的一行中一次拷贝8个元素
        int j = 0;
        for (; j < remainColStart; j += COL_BATCH) {

#ifdef __ARM_ASM
            asm volatile(
                // "prfm   pldl1keep, [%0, #128]   \n"
                "ld1    {v0.4s, v1.4s}, [%0]    \n"
                "st1    {v0.4s, v1.4s}, [%1]    \n"
                : "=r"(BPtr),  // %0
                "=r"(packBPtr) // %1
                : "0"(BPtr),
                "1"(packBPtr)
                : "cc", "memory", "v0", "v1");
#else
            vst1q_f32(packBPtr    , vld1q_f32(BPtr    ));
            vst1q_f32(packBPtr + 4, vld1q_f32(BPtr + 4));

#endif
            BPtr += COL_BATCH;
            packBPtr += step;
        }

        // 如果B的一行中还剩余几个（不足8个），直接拷贝，其他值直接设为0（相当于不足8个填充为8个）
        if (j < nc_adjust) {
            int remain = nc_adjust - j;
            packBPtr = packB + j * kc_adjust + COL_BATCH * k;
            for (int i = 0; i < remain; i++)
            {
                packBPtr[i] = BPtr[i];
            }

            for (int i = remain; i < COL_BATCH; i++)
                packBPtr[i] = 0.0f;

        }
    }
}

void pack_b_multithread_kernel_8x12_v2(int kc_adjust, int k_interval, int nc_adjust, float *B, int ldb, float *packB, const int ROW_BATCH, const int COL_BATCH)
{
    float *BPtr = B, *packBPtr;

    int remainColStart = nc_adjust - nc_adjust % COL_BATCH;

    // 每次处理B的一行
    for (int k = 0; k < k_interval; k++) {
        BPtr = B + ldb * k;
        packBPtr = packB + COL_BATCH * k;
        int step = COL_BATCH * kc_adjust;

        // 在B的一行中一次拷贝8个元素
        int j = 0;
        for (; j < remainColStart; j += COL_BATCH) {
#ifdef __ARM_ASM
            // 用于kernel 8 x 12
            asm volatile(
                // "prfm   pldl1keep, [%0, #128]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s}, [%0]    \n"
                "st1    {v0.4s, v1.4s, v2.4s}, [%1]    \n"
                : "=r"(BPtr),  // %0
                "=r"(packBPtr) // %1
                : "0"(BPtr),
                "1"(packBPtr)
                : "cc", "memory", "v0", "v1", "v2");
#else
            // kernel 8 x 12
            vst1q_f32(packBPtr    , vld1q_f32(BPtr    ));
            vst1q_f32(packBPtr + 4, vld1q_f32(BPtr + 4));
            vst1q_f32(packBPtr + 8, vld1q_f32(BPtr + 8));
#endif
            BPtr += COL_BATCH;
            packBPtr += step;
        }

        // 如果B的一行中还剩余几个（不足8个），直接拷贝，其他值直接设为0（相当于不足8个填充为8个）
        if (j < nc_adjust) {
            int remain = nc_adjust - j;
            packBPtr = packB + j * kc_adjust + COL_BATCH * k;
            for (int i = 0; i < remain; i++)
            {
                packBPtr[i] = BPtr[i];
            }

            for (int i = remain; i < COL_BATCH; i++)
                packBPtr[i] = 0.0f;

        }
    }
}

void pack_b_multithread_kernel_4x16_v2(int kc_adjust, int k_interval, int nc_adjust, float *B, int ldb, float *packB, const int ROW_BATCH, const int COL_BATCH)
{
    float *BPtr = B, *packBPtr;
    int remainColStart = nc_adjust - nc_adjust % COL_BATCH;

    // 每次处理B的一行
    for (int k = 0; k < k_interval; k++) {
        BPtr = B + ldb * k;
        packBPtr = packB + COL_BATCH * k;
        int step = COL_BATCH * kc_adjust;

        // 在B的一行中一次拷贝8个元素
        int j = 0;
        for (; j < remainColStart; j += COL_BATCH) {
#ifdef __ARM_ASM
            // 用于kernel 4 x 16
            asm volatile(
                // "prfm   pldl1keep, [%0, #128]   \n"
                "ld1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%0]    \n"
                "st1    {v0.4s, v1.4s, v2.4s, v3.4s}, [%1]    \n"
                : "=r"(BPtr),  // %0
                "=r"(packBPtr) // %1
                : "0"(BPtr),
                "1"(packBPtr)
                : "cc", "memory", "v0", "v1", "v2", "v3");
#else
            // kernel 4 x 16
            vst1q_f32(packBPtr    , vld1q_f32(BPtr    ));
            vst1q_f32(packBPtr + 4, vld1q_f32(BPtr + 4));
            vst1q_f32(packBPtr + 8, vld1q_f32(BPtr + 8));
            vst1q_f32(packBPtr + 12, vld1q_f32(BPtr + 12));
#endif
            BPtr += COL_BATCH;
            packBPtr += step;
        }

        // 如果B的一行中还剩余几个（不足8个），直接拷贝，其他值直接设为0（相当于不足8个填充为8个）
        if (j < nc_adjust) {
            int remain = nc_adjust - j;
            packBPtr = packB + j * kc_adjust + COL_BATCH * k;
            for (int i = 0; i < remain; i++)
            {
                packBPtr[i] = BPtr[i];
            }

            for (int i = remain; i < COL_BATCH; i++)
                packBPtr[i] = 0.0f;

        }
    }
}

// 对 kc x nc 做8列的packing（多线程版本）
// 这个版本在v3的基础上，变换了packing顺序。让packB顺序写，而让B跳着读
void pack_b_multithread_kernel_8x8_v3(int kc_adjust, int n_interval, int n_ceil, int nc_adjust, float *B, int ldb, float *packB, const int ROW_BATCH, const int COL_BATCH)
{
    float *BPtr, *packBPtr;
    int group = n_ceil / COL_BATCH;
    int tid = omp_get_thread_num();
    int remain_col = n_ceil - n_interval;

    if (remain_col == 0) {
        int g = 0;
        for (; g < group; g++) {
            BPtr = B + g * COL_BATCH;
            packBPtr = packB + g * COL_BATCH * kc_adjust;

            for (int k = 0; k < kc_adjust; k++) {
                vst1q_f32(packBPtr    , vld1q_f32(BPtr    ));
                vst1q_f32(packBPtr + 4, vld1q_f32(BPtr + 4));

                packBPtr += COL_BATCH;
                BPtr += ldb;
            }
        }
    }
    else {
        int g = 0;
        for (; g < group - 1; g++) {
            BPtr = B + g * COL_BATCH;
            packBPtr = packB + g * COL_BATCH * kc_adjust;

            for (int k = 0; k < kc_adjust; k++) {
                vst1q_f32(packBPtr    , vld1q_f32(BPtr    ));
                vst1q_f32(packBPtr + 4, vld1q_f32(BPtr + 4));

                packBPtr += COL_BATCH;
                BPtr += ldb;
            }
        }

        BPtr = B + (group - 1) * COL_BATCH;
        packBPtr = packB + (group - 1) * COL_BATCH * kc_adjust;
        int remain_col_start = COL_BATCH - remain_col;
        for (int k = 0; k < kc_adjust; k++) {
            int i = 0;
            for (; i < remain_col_start; i++) 
                packBPtr[i] = BPtr[i];

            for (; i < COL_BATCH; i++)
                packBPtr[i] = 0;

            packBPtr += COL_BATCH;
            BPtr += ldb;
        }
    }
}

void pack_b_multithread_kernel_8x12_v3(int kc_adjust, int n_interval, int n_ceil, int nc_adjust, float *B, int ldb, float *packB, const int ROW_BATCH, const int COL_BATCH)
{
    float *BPtr, *packBPtr;
    int group = n_ceil / COL_BATCH;
    int tid = omp_get_thread_num();
    int remain_col = n_ceil - n_interval;

    if (remain_col == 0) {
        int g = 0;
        for (; g < group; g++) {
            BPtr = B + g * COL_BATCH;
            packBPtr = packB + g * COL_BATCH * kc_adjust;

            for (int k = 0; k < kc_adjust; k++) {
                vst1q_f32(packBPtr    , vld1q_f32(BPtr    ));
                vst1q_f32(packBPtr + 4, vld1q_f32(BPtr + 4));
                vst1q_f32(packBPtr + 8, vld1q_f32(BPtr + 8));

                packBPtr += COL_BATCH;
                BPtr += ldb;
            }
        }
    }
    else {
        int g = 0;
        for (; g < group - 1; g++) {
            BPtr = B + g * COL_BATCH;
            packBPtr = packB + g * COL_BATCH * kc_adjust;

            for (int k = 0; k < kc_adjust; k++) {
                vst1q_f32(packBPtr    , vld1q_f32(BPtr    ));
                vst1q_f32(packBPtr + 4, vld1q_f32(BPtr + 4));
                vst1q_f32(packBPtr + 8, vld1q_f32(BPtr + 8));

                packBPtr += COL_BATCH;
                BPtr += ldb;
            }
        }

        BPtr = B + (group - 1) * COL_BATCH;
        packBPtr = packB + (group - 1) * COL_BATCH * kc_adjust;
        int remain_col_start = COL_BATCH - remain_col;
        for (int k = 0; k < kc_adjust; k++) {
            int i = 0;
            for (; i < remain_col_start; i++) 
                packBPtr[i] = BPtr[i];

            for (; i < COL_BATCH; i++)
                packBPtr[i] = 0;

            packBPtr += COL_BATCH;
            BPtr += ldb;
        }
    }
}

void pack_b_multithread_kernel_4x16_v3(int kc_adjust, int n_interval, int n_ceil, int nc_adjust, float *B, int ldb, float *packB, const int ROW_BATCH, const int COL_BATCH)
{
    float *BPtr, *packBPtr;
    int group = n_ceil / COL_BATCH;
    int tid = omp_get_thread_num();
    int remain_col = n_ceil - n_interval;

    if (remain_col == 0) {
        int g = 0;
        for (; g < group; g++) {
            BPtr = B + g * COL_BATCH;
            packBPtr = packB + g * COL_BATCH * kc_adjust;

            for (int k = 0; k < kc_adjust; k++) {
                vst1q_f32(packBPtr    , vld1q_f32(BPtr    ));
                vst1q_f32(packBPtr + 4, vld1q_f32(BPtr + 4));
                vst1q_f32(packBPtr + 8, vld1q_f32(BPtr + 8));
                vst1q_f32(packBPtr + 12, vld1q_f32(BPtr + 12));

                packBPtr += COL_BATCH;
                BPtr += ldb;
            }
        }
    }
    else {
        int g = 0;
        for (; g < group - 1; g++) {
            BPtr = B + g * COL_BATCH;
            packBPtr = packB + g * COL_BATCH * kc_adjust;

            for (int k = 0; k < kc_adjust; k++) {
                vst1q_f32(packBPtr    , vld1q_f32(BPtr    ));
                vst1q_f32(packBPtr + 4, vld1q_f32(BPtr + 4));
                vst1q_f32(packBPtr + 8, vld1q_f32(BPtr + 8));
                vst1q_f32(packBPtr + 12, vld1q_f32(BPtr + 12));

                packBPtr += COL_BATCH;
                BPtr += ldb;
            }
        }

        BPtr = B + (group - 1) * COL_BATCH;
        packBPtr = packB + (group - 1) * COL_BATCH * kc_adjust;
        int remain_col_start = COL_BATCH - remain_col;
        for (int k = 0; k < kc_adjust; k++) {
            int i = 0;
            for (; i < remain_col_start; i++) 
                packBPtr[i] = BPtr[i];

            for (; i < COL_BATCH; i++)
                packBPtr[i] = 0;

            packBPtr += COL_BATCH;
            BPtr += ldb;
        }
    }
}

void pack_b_multithread_8x8_v1(int kc_adjust, int nc_adjust, int thread_num, int tid, float *B, int ldb, float *packB, int kt, int nt, const int ROW_BATCH, const int COL_BATCH) {
    // 版本1
    // 主线程对B做一次packing，其他线程直接利用缓存一致性获取packing结果
    // 需要加入同步机制
    #pragma omp barrier
    // 让主线程来packing B
    #pragma omp master 
    {
        pack_b_v2_8x8(kc_adjust, nc_adjust, B + kt*ldb + nt, ldb, packB, ROW_BATCH, COL_BATCH);
    }
    #pragma omp barrier
}

void pack_b_multithread_8x8_v2(int kc_adjust, int nc_adjust, int thread_num, int tid, float *B, int ldb, float *packB, int kt, int nt, const int ROW_BATCH, const int COL_BATCH) {
    // 版本2
    // TO-DO 考虑让每个线程负责packing B的一部分，然后共享
    // 在kc维度上做切分，切分成8份，每个线程负责对自己的那部分做packing
    #pragma omp barrier
    int k_interval = (kc_adjust + (thread_num - 1)) / thread_num;
    if (tid * k_interval < kc_adjust) {
        if ((tid + 1) * k_interval > kc_adjust) {
            pack_b_multithread_kernel_8x8_v2(kc_adjust, kc_adjust - tid * k_interval, nc_adjust,
                        B + (kt + tid * k_interval) * ldb + nt, 
                        ldb, packB + COL_BATCH * tid * k_interval, ROW_BATCH, COL_BATCH);
        }
        else {
            pack_b_multithread_kernel_8x8_v2(kc_adjust, k_interval, nc_adjust,
                        B + (kt + tid * k_interval) * ldb + nt,
                            ldb, packB + COL_BATCH * tid * k_interval, ROW_BATCH, COL_BATCH);
        }
    }
    #pragma omp barrier
}

void pack_b_multithread_8x8_v3(int kc_adjust, int nc_adjust, int thread_num, int tid, float *B, int ldb, float *packB, int kt, int nt, const int ROW_BATCH, const int COL_BATCH) {
    #pragma omp barrier
    int n_block_num = (nc_adjust + (COL_BATCH - 1)) / COL_BATCH;
    int n_block_num_per_thread = n_block_num / thread_num;
    int n_remain_block_num = n_block_num % thread_num;
    int n_interval = n_block_num_per_thread * COL_BATCH;
    int n_from;
    int n_to;
    if (tid < n_remain_block_num) {
        n_from = tid * (n_block_num_per_thread + 1) * COL_BATCH;
        n_to = n_from + (n_block_num_per_thread + 1) * COL_BATCH;
    }
    else{
        n_from = (tid * n_block_num_per_thread + n_remain_block_num) * COL_BATCH;
        n_to = n_from + n_block_num_per_thread * COL_BATCH;
    }
    if (n_to > nc_adjust) 
        n_to = nc_adjust;
    
    if (n_to - n_from > 0) {
        n_interval = n_to - n_from;
        int n_ceil = align_ceil(n_interval, COL_BATCH);
        pack_b_multithread_kernel_8x8_v3(kc_adjust, n_interval, n_ceil, nc_adjust, B + kt*ldb + nt + n_from, ldb, packB + n_from * kc_adjust, ROW_BATCH, COL_BATCH);
    }
    #pragma omp barrier
}

void pack_b_multithread_8x12_v1(int kc_adjust, int nc_adjust, int thread_num, int tid, float *B, int ldb, float *packB, int kt, int nt, const int ROW_BATCH, const int COL_BATCH) {
    // 版本1
    // 主线程对B做一次packing，其他线程直接利用缓存一致性获取packing结果
    // 需要加入同步机制
    #pragma omp barrier
    // 让主线程来packing B
    #pragma omp master 
    {
        pack_b_v2_8x12(kc_adjust, nc_adjust, B + kt*ldb + nt, ldb, packB, ROW_BATCH, COL_BATCH);
    }
    #pragma omp barrier
}

void pack_b_multithread_8x12_v2(int kc_adjust, int nc_adjust, int thread_num, int tid, float *B, int ldb, float *packB, int kt, int nt, const int ROW_BATCH, const int COL_BATCH) {
    #pragma omp barrier
    int k_interval = (kc_adjust + (thread_num - 1)) / thread_num;
    if (tid * k_interval < kc_adjust) {
        if ((tid + 1) * k_interval > kc_adjust) {
            pack_b_multithread_kernel_8x12_v2(kc_adjust, kc_adjust - tid * k_interval, nc_adjust,
                        B + (kt + tid * k_interval) * ldb + nt, 
                        ldb, packB + COL_BATCH * tid * k_interval, ROW_BATCH, COL_BATCH);
        }
        else {
            pack_b_multithread_kernel_8x12_v2(kc_adjust, k_interval, nc_adjust,
                        B + (kt + tid * k_interval) * ldb + nt,
                            ldb, packB + COL_BATCH * tid * k_interval, ROW_BATCH, COL_BATCH);
        }
    }
    #pragma omp barrier
}

void pack_b_multithread_8x12_v3(int kc_adjust, int nc_adjust, int thread_num, int tid, float *B, int ldb, float *packB, int kt, int nt, const int ROW_BATCH, const int COL_BATCH) {
    #pragma omp barrier
    int n_block_num = (nc_adjust + (COL_BATCH - 1)) / COL_BATCH;
    int n_block_num_per_thread = n_block_num / thread_num;
    int n_remain_block_num = n_block_num % thread_num;
    int n_interval = n_block_num_per_thread * COL_BATCH;
    int n_from;
    int n_to;
    if (tid < n_remain_block_num) {
        n_from = tid * (n_block_num_per_thread + 1) * COL_BATCH;
        n_to = n_from + (n_block_num_per_thread + 1) * COL_BATCH;
    }
    else{
        n_from = (tid * n_block_num_per_thread + n_remain_block_num) * COL_BATCH;
        n_to = n_from + n_block_num_per_thread * COL_BATCH;
    }
    if (n_to > nc_adjust) 
        n_to = nc_adjust;
    
    if (n_to - n_from > 0) {
        n_interval = n_to - n_from;
        int n_ceil = align_ceil(n_interval, COL_BATCH);
        pack_b_multithread_kernel_8x12_v3(kc_adjust, n_interval, n_ceil, nc_adjust, B + kt*ldb + nt + n_from, ldb, packB + n_from * kc_adjust, ROW_BATCH, COL_BATCH);
    }
    #pragma omp barrier
}

void pack_b_multithread_4x16_v1(int kc_adjust, int nc_adjust, int thread_num, int tid, float *B, int ldb, float *packB, int kt, int nt, const int ROW_BATCH, const int COL_BATCH) {
    // 版本1
    // 主线程对B做一次packing，其他线程直接利用缓存一致性获取packing结果
    // 需要加入同步机制
    #pragma omp barrier
    // 让主线程来packing B
    #pragma omp master 
    {
        pack_b_v2_4x16(kc_adjust, nc_adjust, B + kt*ldb + nt, ldb, packB, ROW_BATCH, COL_BATCH);
    }
    #pragma omp barrier
}

void pack_b_multithread_4x16_v2(int kc_adjust, int nc_adjust, int thread_num, int tid, float *B, int ldb, float *packB, int kt, int nt, const int ROW_BATCH, const int COL_BATCH) {
    #pragma omp barrier
    int k_interval = (kc_adjust + (thread_num - 1)) / thread_num;
    if (tid * k_interval < kc_adjust) {
        if ((tid + 1) * k_interval > kc_adjust) {
            pack_b_multithread_kernel_4x16_v2(kc_adjust, kc_adjust - tid * k_interval, nc_adjust,
                        B + (kt + tid * k_interval) * ldb + nt, 
                        ldb, packB + COL_BATCH * tid * k_interval, ROW_BATCH, COL_BATCH);
        }
        else {
            pack_b_multithread_kernel_4x16_v2(kc_adjust, k_interval, nc_adjust,
                        B + (kt + tid * k_interval) * ldb + nt,
                            ldb, packB + COL_BATCH * tid * k_interval, ROW_BATCH, COL_BATCH);
        }
    }
    #pragma omp barrier
}

void pack_b_multithread_4x16_v3(int kc_adjust, int nc_adjust, int thread_num, int tid, float *B, int ldb, float *packB, int kt, int nt, const int ROW_BATCH, const int COL_BATCH) {
    #pragma omp barrier
    int n_block_num = (nc_adjust + (COL_BATCH - 1)) / COL_BATCH;
    int n_block_num_per_thread = n_block_num / thread_num;
    int n_remain_block_num = n_block_num % thread_num;
    int n_interval = n_block_num_per_thread * COL_BATCH;
    int n_from;
    int n_to;
    if (tid < n_remain_block_num) {
        n_from = tid * (n_block_num_per_thread + 1) * COL_BATCH;
        n_to = n_from + (n_block_num_per_thread + 1) * COL_BATCH;
    }
    else{
        n_from = (tid * n_block_num_per_thread + n_remain_block_num) * COL_BATCH;
        n_to = n_from + n_block_num_per_thread * COL_BATCH;
    }
    if (n_to > nc_adjust) 
        n_to = nc_adjust;
    
    if (n_to - n_from > 0) {
        n_interval = n_to - n_from;
        int n_ceil = align_ceil(n_interval, COL_BATCH);
        pack_b_multithread_kernel_4x16_v3(kc_adjust, n_interval, n_ceil, nc_adjust, B + kt*ldb + nt + n_from, ldb, packB + n_from * kc_adjust, ROW_BATCH, COL_BATCH);
    }
    #pragma omp barrier
}

void load_c_v2_8x8(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    for (int m = 0; m < mc_adjust; m++) {
        float *CPtr = C + m * ldc;
        float *packCPtr = packC + m * nc_ceil;
        // 行中每 COL_BATCH 个进行移动，不足 COL_BATCH 个的填充0
        int remain_col_start = nc_adjust - nc_adjust % COL_BATCH;
        int n = 0;
        for (; n < remain_col_start; n += COL_BATCH)
        {
            vst1q_f32(packCPtr + n, vld1q_f32(CPtr + n));
            vst1q_f32(packCPtr + n + 4, vld1q_f32(CPtr + n + 4));
        }
        for (; n < nc_adjust; n++) 
            packCPtr[n] = CPtr[n];

        for (; n < nc_ceil; n++) 
            packCPtr[n] = 0.0f;

    }
}

void load_c_v2_8x12(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    for (int m = 0; m < mc_adjust; m++) {
        float *CPtr = C + m * ldc;
        float *packCPtr = packC + m * nc_ceil;
        // 行中每 COL_BATCH 个进行移动，不足 COL_BATCH 个的填充0
        int remain_col_start = nc_adjust - nc_adjust % COL_BATCH;
        int n = 0;
        for (; n < remain_col_start; n += COL_BATCH)
        {
            vst1q_f32(packCPtr + n, vld1q_f32(CPtr + n));
            vst1q_f32(packCPtr + n + 4, vld1q_f32(CPtr + n + 4));
            vst1q_f32(packCPtr + n + 8, vld1q_f32(CPtr + n + 8));
        }
        for (; n < nc_adjust; n++) 
            packCPtr[n] = CPtr[n];

        for (; n < nc_ceil; n++) 
            packCPtr[n] = 0.0f;

    }
}

void load_c_v2_4x16(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    for (int m = 0; m < mc_adjust; m++) {
        float *CPtr = C + m * ldc;
        float *packCPtr = packC + m * nc_ceil;
        // 行中每 COL_BATCH 个进行移动，不足 COL_BATCH 个的填充0
        int remain_col_start = nc_adjust - nc_adjust % COL_BATCH;
        int n = 0;
        for (; n < remain_col_start; n += COL_BATCH)
        {
            vst1q_f32(packCPtr + n, vld1q_f32(CPtr + n));
            vst1q_f32(packCPtr + n + 4, vld1q_f32(CPtr + n + 4));
            vst1q_f32(packCPtr + n + 8, vld1q_f32(CPtr + n + 8));
            vst1q_f32(packCPtr + n + 12, vld1q_f32(CPtr + n + 12));
        }
        for (; n < nc_adjust; n++) 
            packCPtr[n] = CPtr[n];

        for (; n < nc_ceil; n++) 
            packCPtr[n] = 0.0f;

    }
}

void load_c_v2_8x8_pack(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    pack_b_v2_8x8(mc_adjust, nc_adjust, C, ldc, packC, ROW_BATCH, COL_BATCH);
}

void load_c_v2_8x12_pack(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    pack_b_v2_8x12(mc_adjust, nc_adjust, C, ldc, packC, ROW_BATCH, COL_BATCH);
}

void load_c_v2_4x16_pack(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    pack_b_v2_4x16(mc_adjust, nc_adjust, C, ldc, packC, ROW_BATCH, COL_BATCH);
}

void write_c_v2_8x8(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    for (int m = 0; m < mc_adjust; m++) {
        float *CPtr = C + m * ldc;
        float *packCPtr = packC + m * nc_ceil;
        int remain_col_start = nc_adjust - nc_adjust % COL_BATCH;
        // 一次写 COL_BATCH 个
        int n = 0;
        for (; n < remain_col_start; n += COL_BATCH)
        {
            vst1q_f32(CPtr, vld1q_f32(packCPtr));
            vst1q_f32(CPtr + 4, vld1q_f32(packCPtr + 4));
            CPtr += COL_BATCH;
            packCPtr += COL_BATCH;
        }

        // 不足8个，单独处理
        for (; n < nc_adjust; n++)
        {
            *CPtr = *packCPtr;
            ++CPtr;
            ++packCPtr;
        }
    }
}

void write_c_v2_8x12(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    for (int m = 0; m < mc_adjust; m++) {
        float *CPtr = C + m * ldc;
        float *packCPtr = packC + m * nc_ceil;
        int remain_col_start = nc_adjust - nc_adjust % COL_BATCH;
        // 一次写 COL_BATCH 个
        int n = 0;
        for (; n < remain_col_start; n += COL_BATCH)
        {
            vst1q_f32(CPtr, vld1q_f32(packCPtr));
            vst1q_f32(CPtr + 4, vld1q_f32(packCPtr + 4));
            vst1q_f32(CPtr + 8, vld1q_f32(packCPtr + 8));
            CPtr += COL_BATCH;
            packCPtr += COL_BATCH;
        }

        // 不足8个，单独处理
        for (; n < nc_adjust; n++)
        {
            *CPtr = *packCPtr;
            ++CPtr;
            ++packCPtr;
        }
    }
}

void write_c_v2_4x16(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    for (int m = 0; m < mc_adjust; m++) {
        float *CPtr = C + m * ldc;
        float *packCPtr = packC + m * nc_ceil;
        int remain_col_start = nc_adjust - nc_adjust % COL_BATCH;
        // 一次写 COL_BATCH 个
        int n = 0;
        for (; n < remain_col_start; n += COL_BATCH)
        {
            vst1q_f32(CPtr, vld1q_f32(packCPtr));
            vst1q_f32(CPtr + 4, vld1q_f32(packCPtr + 4));
            vst1q_f32(CPtr + 8, vld1q_f32(packCPtr + 8));
            vst1q_f32(CPtr + 12, vld1q_f32(packCPtr + 12));
            CPtr += COL_BATCH;
            packCPtr += COL_BATCH;
        }

        // 不足8个，单独处理
        for (; n < nc_adjust; n++)
        {
            *CPtr = *packCPtr;
            ++CPtr;
            ++packCPtr;
        }
    }
}

void write_c_v2_8x8_unpack(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    for (int m = 0; m < mc_adjust; m++) {
        float* CPtr = C + m * ldc;
        float* packCPtr = packC + m * COL_BATCH;
        int remainColStart = nc_adjust - nc_adjust % COL_BATCH;
        for (int n = 0; n < remainColStart; n += COL_BATCH) {
            vst1q_f32(CPtr, vld1q_f32(packCPtr));
            vst1q_f32(CPtr + 4, vld1q_f32(packCPtr + 4));
            CPtr += COL_BATCH;
            packCPtr += COL_BATCH * mc_adjust;
        }

        for (int n = remainColStart; n < nc_adjust; n++) {
            *CPtr = *packCPtr;
            ++CPtr;
            ++packCPtr;
        }
    }
}

void write_c_v2_8x12_unpack(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    for (int m = 0; m < mc_adjust; m++) {
        float* CPtr = C + m * ldc;
        float* packCPtr = packC + m * COL_BATCH;
        int remainColStart = nc_adjust - nc_adjust % COL_BATCH;
        for (int n = 0; n < remainColStart; n += COL_BATCH) {
            vst1q_f32(CPtr, vld1q_f32(packCPtr));
            vst1q_f32(CPtr + 4, vld1q_f32(packCPtr + 4));
            vst1q_f32(CPtr + 8, vld1q_f32(packCPtr + 8));
            CPtr += COL_BATCH;
            packCPtr += COL_BATCH * mc_adjust;
        }

        for (int n = remainColStart; n < nc_adjust; n++) {
            *CPtr = *packCPtr;
            ++CPtr;
            ++packCPtr;
        }
    }
}

void write_c_v2_4x16_unpack(int mc_adjust, int nc_adjust, int nc_ceil, float* C, float* packC, int ldc, const int ROW_BATCH, const int COL_BATCH) {
    for (int m = 0; m < mc_adjust; m++) {
        float* CPtr = C + m * ldc;
        float* packCPtr = packC + m * COL_BATCH;
        int remainColStart = nc_adjust - nc_adjust % COL_BATCH;
        for (int n = 0; n < remainColStart; n += COL_BATCH) {
            vst1q_f32(CPtr, vld1q_f32(packCPtr));
            vst1q_f32(CPtr + 4, vld1q_f32(packCPtr + 4));
            vst1q_f32(CPtr + 8, vld1q_f32(packCPtr + 8));
            vst1q_f32(CPtr + 12, vld1q_f32(packCPtr + 12));
            CPtr += COL_BATCH;
            packCPtr += COL_BATCH * mc_adjust;
        }

        for (int n = remainColStart; n < nc_adjust; n++) {
            *CPtr = *packCPtr;
            ++CPtr;
            ++packCPtr;
        }
    }
}

// 对 M x kc 做8行packing（多线程版本）
// omp并行版本
void PackMatrix_a_multithread(int M, int kc_adjust, float* A, int lda, float* packA, int mc, int kc, int nc, const int ROW_BATCH, const int COL_BATCH) 
{
    // 每次处理一个panel
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i += mc)
    {
        float *aPtr = A + i * lda, *packAPtr = packA + i * kc_adjust;
        int tid = omp_get_thread_num();
        // 确定一个Penel的高度
        int m_len = min(mc, M - i);
        int remainRowStart = m_len - m_len % ROW_BATCH;
        // 对panel内部进行packing，每次对panel内部的8行进行处理
        int m = 0;
        for (; m < remainRowStart; m += ROW_BATCH)
        {
            aPtr = A + (i + m) * lda;
            // 每次处理panel内部的8行1列
            for (int k = 0; k < kc_adjust; k++)
            {
                int aPtrIdx = 0;
                for (int j = 0; j < ROW_BATCH; j++)
                {
                    packAPtr[j] = aPtr[aPtrIdx];
                    aPtrIdx += lda;
                }
                packAPtr += ROW_BATCH;
                aPtr++;
            }
        }

        aPtr = A + (i + remainRowStart) * lda;
        // 如果剩余不足8行，直接处理
        int remainRow = m_len - remainRowStart;
        // 每次处理panel内部的1列
        for (int k = 0; k < kc_adjust; k++)
        {
            int aPtrIdx = 0;
            for (int j = 0; j < remainRow; j++)
            {
                packAPtr[j] = aPtr[aPtrIdx];
                aPtrIdx += lda;
            }
            packAPtr += remainRow;
            aPtr++;
        } 
    }
}

