#include <stdlib.h>
#include <arm_neon.h>
#include <assert.h>
#include <stdio.h>
#include "../utility/helper.h"

#define __ARM_ASM

typedef void (*InnerKernel)(int, float *, float *, float *, int);

// 用于GEMM_v1的register kernel
void kernel_4x4(int M, int N, int K, float *A, float* B, float *C, int ldc) 
{
    assert(M > 0 && N > 0 && K > 0);
    assert(M % 4 == 0 && N % 4 == 0 && K % 4 == 0);
    float* B_ptr = B;
    float* C_ptr = C;

    int i, j;
    // 每次计算C中4x4的分块
    for (i = 0; i < M; i += 4)
    {
        for (j = 0; j < N; j += 4)
        {
            // 预取数据
            // __builtin_prefetch(B, 0, 3);
            // __builtin_prefetch(A, 0, 3);

            // 保存C中4x4的结果
            float32x4_t c0 = {0};
            float32x4_t c1 = {0};
            float32x4_t c2 = {0};
            float32x4_t c3 = {0};

            for (int p = 0; p < K; p += 4) {
                // 读取B中的4个
                float32x4_t b0 = vld1q_f32(B);
                float32x4_t a0 = vld1q_f32(A);

                // 将B中的4个值分别与A中的每个值做乘加
                c0 = vmlaq_laneq_f32(c0, b0, a0, 0);
                c1 = vmlaq_laneq_f32(c1, b0, a0, 1);
                c2 = vmlaq_laneq_f32(c2, b0, a0, 2);
                c3 = vmlaq_laneq_f32(c3, b0, a0, 3);

                float32x4_t b1 = vld1q_f32(B+4);
                float32x4_t a1 = vld1q_f32(A+4);

                c0 = vmlaq_laneq_f32(c0, b1, a1, 0);
                c1 = vmlaq_laneq_f32(c1, b1, a1, 1);
                c2 = vmlaq_laneq_f32(c2, b1, a1, 2);
                c3 = vmlaq_laneq_f32(c3, b1, a1, 3);

                float32x4_t b2 = vld1q_f32(B+8);
                float32x4_t a2 = vld1q_f32(A+8);

                c0 = vmlaq_laneq_f32(c0, b2, a2, 0);
                c1 = vmlaq_laneq_f32(c1, b2, a2, 1);
                c2 = vmlaq_laneq_f32(c2, b2, a2, 2);
                c3 = vmlaq_laneq_f32(c3, b2, a2, 3);

                float32x4_t b3 = vld1q_f32(B+12);
                float32x4_t a3 = vld1q_f32(A+12);

                c0 = vmlaq_laneq_f32(c0, b3, a3, 0);
                c1 = vmlaq_laneq_f32(c1, b3, a3, 1);
                c2 = vmlaq_laneq_f32(c2, b3, a3, 2);
                c3 = vmlaq_laneq_f32(c3, b3, a3, 3);
                
                // __builtin_prefetch(B+16, 0, 3);
                // __builtin_prefetch(A+16, 0, 3);

                B += 16;
                A += 16;

            }

            // 将C的4x4结果写回
            c0 = vaddq_f32(vld1q_f32(C), c0);
            c1 = vaddq_f32(vld1q_f32(C + ldc), c1);
            c2 = vaddq_f32(vld1q_f32(C + 2*ldc), c2);
            c3 = vaddq_f32(vld1q_f32(C + 3*ldc), c3);

            vst1q_f32(C, c0);
            vst1q_f32(C + ldc, c1);
            vst1q_f32(C + 2*ldc, c2);
            vst1q_f32(C + 3*ldc, c3);

            C += 4;
            A -= 4 * K;

        }

        A += 4 * K;
        B = B_ptr;
        C_ptr += 4 * ldc;
        C = C_ptr;
    }
}

// 计算C中的一个8x8分块
void kernel_8x8(int kc_adjust, float *packA, float* packB, float *packC, int ldc, const int PREFETCH_A, const int PREFETCH_B, const int PREFETCH_C) 
{
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;
    
    float* packCPtr0 = packC;
    float* packCPtr1 = packC + 1 * ldc;
    float* packCPtr2 = packC + 2 * ldc;
    float* packCPtr3 = packC + 3 * ldc;
    float* packCPtr4 = packC + 4 * ldc;
    float* packCPtr5 = packC + 5 * ldc;
    float* packCPtr6 = packC + 6 * ldc;
    float* packCPtr7 = packC + 7 * ldc;

#ifdef __ARM_ASM
    asm volatile(
        "mov    x4, %20                   \n"
        "mov    x5, %22                   \n"
        "mov    x6, %23                   \n"

        "ld1    {v4.4s, v5.4s}, [%2]      \n"
        "ld1    {v6.4s, v7.4s}, [%3]      \n"
        "ld1    {v8.4s, v9.4s}, [%4]      \n"
        "ld1    {v10.4s, v11.4s}, [%5]    \n"
        "ld1    {v12.4s, v13.4s}, [%6]    \n"
        "ld1    {v14.4s, v15.4s}, [%7]    \n"
        "ld1    {v16.4s, v17.4s}, [%8]    \n"
        "ld1    {v18.4s, v19.4s}, [%9]    \n"

        "cmp    w4, #0                    \n"
        "beq    3f                         \n"

        "ld1    {v0.4s, v1.4s}, [%0], #32 \n"   // load packA, packA+4
        "ld1    {v2.4s, v3.4s}, [%1], #32 \n"   // load packB, packB+4

        "lsr   x9, x4, 0x2               \n"   // unroll by 4, get main loop counter
        "cmp   w9, #0                    \n"
        "beq   1f                         \n"
        
    "0:                               \n"

        "subs   w9, w9, 0x1               \n"

        "prfm   pldl1keep, [%0, x5]     \n"
        "ld1    {v20.4s, v21.4s}, [%0], #32 \n"   // load packA, packA+4

        "fmla   v4.4s, v2.4s, v0.s[0]    \n"
        "fmla   v5.4s, v3.4s, v0.s[0]    \n"

        "fmla   v6.4s, v2.4s, v0.s[1]    \n"
        "fmla   v7.4s, v3.4s, v0.s[1]    \n"
//        "prfm   pldl1keep, [%0, x5]     \n"

        "fmla   v8.4s, v2.4s, v0.s[2]    \n"
        "fmla   v9.4s, v3.4s, v0.s[2]    \n"

        "fmla   v10.4s, v2.4s, v0.s[3]    \n"
        "fmla   v11.4s, v3.4s, v0.s[3]    \n"
        
        "prfm   pldl1keep, [%1, x6]     \n"
        "ld1    {v22.4s, v23.4s}, [%1], #32 \n"   // load packB, packB+4

        "fmla   v12.4s, v2.4s, v1.s[0]    \n"
        "fmla   v13.4s, v3.4s, v1.s[0]    \n"
        
        "fmla   v14.4s, v2.4s, v1.s[1]    \n"
        "fmla   v15.4s, v3.4s, v1.s[1]    \n"
//        "prfm   pldl1keep, [%1, x6]     \n"

        "fmla   v16.4s, v2.4s, v1.s[2]    \n"
        "fmla   v17.4s, v3.4s, v1.s[2]    \n"

        "fmla   v18.4s, v2.4s, v1.s[3]    \n"
        "fmla   v19.4s, v3.4s, v1.s[3]    \n"
        
        "ld1    {v24.4s, v25.4s}, [%0], #32 \n"   // load packA, packA+4
        
        "fmla   v4.4s, v22.4s, v20.s[0]    \n"
        "fmla   v5.4s, v23.4s, v20.s[0]    \n"
        
        "fmla   v6.4s, v22.4s, v20.s[1]    \n"
        "fmla   v7.4s, v23.4s, v20.s[1]    \n"

        "fmla   v8.4s, v22.4s, v20.s[2]    \n"
        "fmla   v9.4s, v23.4s, v20.s[2]    \n"

        "fmla   v10.4s, v22.4s, v20.s[3]    \n"
        "fmla   v11.4s, v23.4s, v20.s[3]    \n"
        
        "ld1    {v26.4s, v27.4s}, [%1], #32 \n"   // load packB, packB+4
        
        "fmla   v12.4s, v22.4s, v21.s[0]    \n"
        "fmla   v13.4s, v23.4s, v21.s[0]    \n"
        
        "fmla   v14.4s, v22.4s, v21.s[1]    \n"
        "fmla   v15.4s, v23.4s, v21.s[1]    \n"

        "fmla   v16.4s, v22.4s, v21.s[2]    \n"
        "fmla   v17.4s, v23.4s, v21.s[2]    \n"

        "fmla   v18.4s, v22.4s, v21.s[3]    \n"
        "fmla   v19.4s, v23.4s, v21.s[3]    \n"

        "prfm   pldl1keep, [%0, x5]     \n"
        "ld1    {v28.4s, v29.4s}, [%0], #32 \n"   // load packA, packA+4
        
        "fmla   v4.4s, v26.4s, v24.s[0]    \n"
        "fmla   v5.4s, v27.4s, v24.s[0]    \n"
        
        "fmla   v6.4s, v26.4s, v24.s[1]    \n"
        "fmla   v7.4s, v27.4s, v24.s[1]    \n"

        "fmla   v8.4s, v26.4s, v24.s[2]    \n"
        "fmla   v9.4s, v27.4s, v24.s[2]    \n"

        "fmla   v10.4s, v26.4s, v24.s[3]    \n"
        "fmla   v11.4s, v27.4s, v24.s[3]    \n"
        
        "prfm   pldl1keep, [%1, x6]     \n"
        "ld1    {v30.4s, v31.4s}, [%1], #32 \n"   // load packB, packB+4
        
        "fmla   v12.4s, v26.4s, v25.s[0]    \n"
        "fmla   v13.4s, v27.4s, v25.s[0]    \n"
        
        "fmla   v14.4s, v26.4s, v25.s[1]    \n"
        "fmla   v15.4s, v27.4s, v25.s[1]    \n"
//        "prfm   pldl1keep, [%1, x6]     \n"

        "fmla   v16.4s, v26.4s, v25.s[2]    \n"
        "fmla   v17.4s, v27.4s, v25.s[2]    \n"

        "fmla   v18.4s, v26.4s, v25.s[3]    \n"
        "fmla   v19.4s, v27.4s, v25.s[3]    \n"

        "ld1    {v0.4s, v1.4s}, [%0], #32 \n"   // load packA, packA+4
        
        "fmla   v4.4s, v30.4s, v28.s[0]    \n"
        "fmla   v5.4s, v31.4s, v28.s[0]    \n"
        
        "fmla   v6.4s, v30.4s, v28.s[1]    \n"
        "fmla   v7.4s, v31.4s, v28.s[1]    \n"

        "fmla   v8.4s, v30.4s, v28.s[2]    \n"
        "fmla   v9.4s, v31.4s, v28.s[2]    \n"

        "fmla   v10.4s, v30.4s, v28.s[3]    \n"
        "fmla   v11.4s, v31.4s, v28.s[3]    \n"
        
        "ld1    {v2.4s, v3.4s}, [%1], #32 \n"   // load packB, packB+4
        
        "fmla   v12.4s, v30.4s, v29.s[0]    \n"
        "fmla   v13.4s, v31.4s, v29.s[0]    \n"
        
        "fmla   v14.4s, v30.4s, v29.s[1]    \n"
        "fmla   v15.4s, v31.4s, v29.s[1]    \n"

        "fmla   v16.4s, v30.4s, v29.s[2]    \n"
        "fmla   v17.4s, v31.4s, v29.s[2]    \n"

        "fmla   v18.4s, v30.4s, v29.s[3]    \n"
        "fmla   v19.4s, v31.4s, v29.s[3]    \n"

        "bne    0b                           \n"

    "1:                           \n"
        "ands   x10, x4, 0x3              \n"   // unroll by 4, get remain loop counter
        "beq    3f                         \n"

    "2:                               \n"
        // "ld1    {v0.4s, v1.4s}, [%0], #32 \n"   // load packA, packA+4
        // "ld1    {v2.4s, v3.4s}, [%1], #32 \n"   // load packB, packB+4

        "subs   w10, w10, 0x1             \n"

        "prfm   pldl1keep, [%0, x5]     \n"

        "fmla   v4.4s, v2.4s, v0.s[0]    \n"
        "fmla   v5.4s, v3.4s, v0.s[0]    \n"

        "fmla   v6.4s, v2.4s, v0.s[1]    \n"
        "fmla   v7.4s, v3.4s, v0.s[1]    \n"

        "fmla   v8.4s, v2.4s, v0.s[2]    \n"
        "fmla   v9.4s, v3.4s, v0.s[2]    \n"

        "fmla   v10.4s, v2.4s, v0.s[3]    \n"
        "fmla   v11.4s, v3.4s, v0.s[3]    \n"
        
        "prfm   pldl1keep, [%1, x6]     \n"

        "fmla   v12.4s, v2.4s, v1.s[0]    \n"
        "fmla   v13.4s, v3.4s, v1.s[0]    \n"

        "fmla   v14.4s, v2.4s, v1.s[1]    \n"
        "fmla   v15.4s, v3.4s, v1.s[1]    \n"

        "fmla   v16.4s, v2.4s, v1.s[2]    \n"
        "fmla   v17.4s, v3.4s, v1.s[2]    \n"

        "fmla   v18.4s, v2.4s, v1.s[3]    \n"
        "fmla   v19.4s, v3.4s, v1.s[3]    \n"
        
        "ld1    {v0.4s, v1.4s}, [%0], #32 \n"   // load packA, packA+4
        "ld1    {v2.4s, v3.4s}, [%1], #32 \n"   // load packB, packB+4
        
        "bne    2b                        \n"

    "3:                                   \n"

        "st1    {v4.4s, v5.4s}, [%2]      \n"
        "st1    {v6.4s, v7.4s}, [%3]      \n"
        "st1    {v8.4s, v9.4s}, [%4]      \n"
        "st1    {v10.4s, v11.4s}, [%5]      \n"
        "st1    {v12.4s, v13.4s}, [%6]      \n"
        "st1    {v14.4s, v15.4s}, [%7]      \n"
        "st1    {v16.4s, v17.4s}, [%8]      \n"
        "st1    {v18.4s, v19.4s}, [%9]      \n"

        : "=r"(packAPtr), // %0
        "=r"(packBPtr),   // %1
        "=r"(packCPtr0),    // %2
        "=r"(packCPtr1),    // %3
        "=r"(packCPtr2),    // %4
        "=r"(packCPtr3),    // %5
        "=r"(packCPtr4),    // %6
        "=r"(packCPtr5),    // %7
        "=r"(packCPtr6),    // %8
        "=r"(packCPtr7)     // %9
        : "0"(packAPtr),
        "1"(packBPtr),
        "2"(packCPtr0),
        "3"(packCPtr1),
        "4"(packCPtr2),
        "5"(packCPtr3),
        "6"(packCPtr4),
        "7"(packCPtr5),
        "8"(packCPtr6),
        "9"(packCPtr7),
        "r"(kc_adjust), // %20
        "r"(ldc),       // %21
        "r"(PREFETCH_A),// %22
        "r"(PREFETCH_B),// %23
        "r"(PREFETCH_C) // %24
        : "cc", "memory", "x4", "x5", "x6", "x9", "x10", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", 
        "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

#else

    float32x4_t a0, a1;
    float32x4_t b0, b1;
    float32x4_t c00, c01, c10, c11, c20, c21, c30, c31;
    float32x4_t c40, c41, c50, c51, c60, c61, c70, c71;

    // c00  = vld1q_f32(packCPtr);
    // c01  = vld1q_f32(packCPtr + 4);
    // packCPtr += ldc;

    // c10  = vld1q_f32(packCPtr);
    // c11  = vld1q_f32(packCPtr + 4);
    // packCPtr += ldc;

    // c20  = vld1q_f32(packCPtr);
    // c21  = vld1q_f32(packCPtr + 4);
    // packCPtr += ldc;

    // c30  = vld1q_f32(packCPtr);
    // c31  = vld1q_f32(packCPtr + 4);
    // packCPtr += ldc;

    // c40  = vld1q_f32(packCPtr);
    // c41  = vld1q_f32(packCPtr + 4);
    // packCPtr += ldc;

    // c50  = vld1q_f32(packCPtr);
    // c51  = vld1q_f32(packCPtr + 4);
    // packCPtr += ldc;

    // c60  = vld1q_f32(packCPtr);
    // c61  = vld1q_f32(packCPtr + 4);
    // packCPtr += ldc;

    // c70  = vld1q_f32(packCPtr);
    // c71  = vld1q_f32(packCPtr + 4);
    // packCPtr += ldc;

    c00  = vld1q_f32(packCPtr0);
    c01  = vld1q_f32(packCPtr0 + 4);

    c10  = vld1q_f32(packCPtr1);
    c11  = vld1q_f32(packCPtr1 + 4);

    c20  = vld1q_f32(packCPtr2);
    c21  = vld1q_f32(packCPtr2 + 4);

    c30  = vld1q_f32(packCPtr3);
    c31  = vld1q_f32(packCPtr3 + 4);

    c40  = vld1q_f32(packCPtr4);
    c41  = vld1q_f32(packCPtr4 + 4);

    c50  = vld1q_f32(packCPtr5);
    c51  = vld1q_f32(packCPtr5 + 4);

    c60  = vld1q_f32(packCPtr6);
    c61  = vld1q_f32(packCPtr6 + 4);

    c70  = vld1q_f32(packCPtr7);
    c71  = vld1q_f32(packCPtr7 + 4);

    // A中packing之后的8行与B中packing之后的8列进行计算
    // 计算C的8x8分块
    for (int k = 0; k < kc_adjust; k++)
    {
        a0 = vld1q_f32(packAPtr);
        a1 = vld1q_f32(packAPtr + 4);
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);

        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        
        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);

        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);

        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);

        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);

        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);

        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);

        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);

        packAPtr += 8;
        packBPtr += 8;
    }

    // 将计算结果写回到C的8x8分块中
    packCPtr = packC;

    // vst1q_f32(packCPtr    , c00);
    // vst1q_f32(packCPtr + 4, c01);
    // packCPtr += ldc;

    // vst1q_f32(packCPtr    , c10);
    // vst1q_f32(packCPtr + 4, c11);
    // packCPtr += ldc;

    // vst1q_f32(packCPtr    , c20);
    // vst1q_f32(packCPtr + 4, c21);
    // packCPtr += ldc;

    // vst1q_f32(packCPtr    , c30);
    // vst1q_f32(packCPtr + 4, c31);
    // packCPtr += ldc;

    // vst1q_f32(packCPtr    , c40);
    // vst1q_f32(packCPtr + 4, c41);
    // packCPtr += ldc;

    // vst1q_f32(packCPtr    , c50);
    // vst1q_f32(packCPtr + 4, c51);
    // packCPtr += ldc;

    // vst1q_f32(packCPtr    , c60);
    // vst1q_f32(packCPtr + 4, c61);
    // packCPtr += ldc;

    // vst1q_f32(packCPtr    , c70);
    // vst1q_f32(packCPtr + 4, c71);
    // packCPtr += ldc;

    vst1q_f32(packCPtr0    , c00);
    vst1q_f32(packCPtr0 + 4, c01);

    vst1q_f32(packCPtr1    , c10);
    vst1q_f32(packCPtr1 + 4, c11);

    vst1q_f32(packCPtr2    , c20);
    vst1q_f32(packCPtr2 + 4, c21);

    vst1q_f32(packCPtr3    , c30);
    vst1q_f32(packCPtr3 + 4, c31);

    vst1q_f32(packCPtr4    , c40);
    vst1q_f32(packCPtr4 + 4, c41);

    vst1q_f32(packCPtr5    , c50);
    vst1q_f32(packCPtr5 + 4, c51);

    vst1q_f32(packCPtr6    , c60);
    vst1q_f32(packCPtr6 + 4, c61);

    vst1q_f32(packCPtr7    , c70);
    vst1q_f32(packCPtr7 + 4, c71);
#endif  // __ARM_ASM
}

void kernel_4x16(int kc_adjust, float *packA, float* packB, float *packC, int ldc, const int PREFETCH_A, const int PREFETCH_B, const int PREFETCH_C) {
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;

    float* packCPtr0 = packC;
    float* packCPtr1 = packC + 1 * ldc;
    float* packCPtr2 = packC + 2 * ldc;
    float* packCPtr3 = packC + 3 * ldc;

// #ifdef __ARM_ASM
//     asm volatile(
//         "mov    x4, %12                   \n"
//         "mov    x5, %14                   \n"
//         "mov    x6, %15                   \n"

//         "mov    x8,  %2                   \n"
//         "mov    x9,  %3                   \n"
//         "mov    x10, %4                   \n"
//         "mov    x11, %5                   \n"
        
//         "mov    x12, %2                   \n"
//         "mov    x13, %3                   \n"
//         "mov    x14, %4                   \n"
//         "mov    x15, %5                   \n"

//         // // load这里有bug，顺序需要调整一下
//         // "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%2]      \n"
//         // "ld1    {v20.4s, v21.4s, v22.4s, v23.4s}, [%3]      \n"
//         // "ld1    {v24.4s, v25.4s, v26.4s, v27.4s}, [%4]      \n"
//         // "ld1    {v28.4s, v29.4s, v30.4s, v31.4s}, [%5]      \n"
        
//         "ld4    {v16.s, v17.s, v18.s, v19.s}[0], [x8],  #16 \n"
//         "ld4    {v16.s, v17.s, v18.s, v19.s}[1], [x9],  #16 \n"
//         "ld4    {v16.s, v17.s, v18.s, v19.s}[2], [x10], #16 \n"
//         "ld4    {v16.s, v17.s, v18.s, v19.s}[3], [x11], #16 \n"

//         "ld4    {v20.s, v21.s, v22.s, v23.s}[0], [x8],  #16 \n"
//         "ld4    {v20.s, v21.s, v22.s, v23.s}[1], [x9],  #16 \n"
//         "ld4    {v20.s, v21.s, v22.s, v23.s}[2], [x10], #16 \n"
//         "ld4    {v20.s, v21.s, v22.s, v23.s}[3], [x11], #16 \n"

//         "ld4    {v24.s, v25.s, v26.s, v27.s}[0], [x8],  #16 \n"
//         "ld4    {v24.s, v25.s, v26.s, v27.s}[1], [x9],  #16 \n"
//         "ld4    {v24.s, v25.s, v26.s, v27.s}[2], [x10], #16 \n"
//         "ld4    {v24.s, v25.s, v26.s, v27.s}[3], [x11], #16 \n"

//         "ld4    {v28.s, v29.s, v30.s, v31.s}[0], [x8]       \n"
//         "ld4    {v28.s, v29.s, v30.s, v31.s}[1], [x9]       \n"
//         "ld4    {v28.s, v29.s, v30.s, v31.s}[2], [x10]      \n"
//         "ld4    {v28.s, v29.s, v30.s, v31.s}[3], [x11]      \n"

//         "cmp    w4, #0                    \n"
//         "beq    2f                        \n"

//         "ld1    {v0.4s}, [%0],        #16 \n"   // load packA, packA
//         "ld1    {v4.4s, v5.4s}, [%1], #32 \n"   // load packB, packB+12

//         "and    x10, x4, 0x3              \n"
//         "lsr    x9,  x4, 0x2              \n"

//     "0:                                   \n"   // loop4
//         "fmla   v16.4s, v0.4s, v4.s[0]    \n"
//         "ld1    {v6.4s, v7.4s}, [%1], #32 \n"
//         "subs   x9, x9, 0x1               \n"
//         "fmla   v17.4s, v0.4s, v4.s[1]    \n"
//         "fmla   v18.4s, v0.4s, v4.s[2]    \n"
//         "fmla   v19.4s, v0.4s, v4.s[3]    \n"
//         "ld1    {v1.4s}, [%0],        #16 \n"   // load packA, packA
//         "fmla   v20.4s, v0.4s, v5.s[0]    \n"
//         "fmla   v21.4s, v0.4s, v5.s[1]    \n"
//         "fmla   v22.4s, v0.4s, v5.s[2]    \n"
//         "fmla   v23.4s, v0.4s, v5.s[3]    \n"
//         "ld1    {v4.4s, v5.4s}, [%1], #32 \n"   // load packB, packB+12
//         "fmla   v24.4s, v0.4s, v6.s[0]    \n"
//         "fmla   v25.4s, v0.4s, v6.s[1]    \n"
//         "fmla   v26.4s, v0.4s, v6.s[2]    \n"
//         "fmla   v27.4s, v0.4s, v6.s[3]    \n"
//         "prfm   pldl1keep, [%0, #128]     \n"
//         "fmla   v28.4s, v0.4s, v7.s[0]    \n"
//         "fmla   v29.4s, v0.4s, v7.s[1]    \n"
//         "prfm   pldl1keep, [%1, #128]    \n"
//         "fmla   v30.4s, v0.4s, v7.s[2]    \n"
//         "fmla   v31.4s, v0.4s, v7.s[3]    \n"

//         "ld1    {v6.4s, v7.4s}, [%1], #32 \n"
//         "fmla   v16.4s, v1.4s, v4.s[0]    \n"
//         "fmla   v17.4s, v1.4s, v4.s[1]    \n"
//         "fmla   v18.4s, v1.4s, v4.s[2]    \n"
//         "fmla   v19.4s, v1.4s, v4.s[3]    \n"
//         "ld1    {v0.4s}, [%0],        #16 \n"   // load packA, packA
//         "fmla   v20.4s, v1.4s, v5.s[0]    \n"
//         "fmla   v21.4s, v1.4s, v5.s[1]    \n"
//         "fmla   v22.4s, v1.4s, v5.s[2]    \n"
//         "fmla   v23.4s, v1.4s, v5.s[3]    \n"
//         "ld1    {v4.4s, v5.4s}, [%1], #32 \n"   // load packB, packB+12
//         "fmla   v24.4s, v1.4s, v6.s[0]    \n"
//         "fmla   v25.4s, v1.4s, v6.s[1]    \n"
//         "fmla   v26.4s, v1.4s, v6.s[2]    \n"
//         "fmla   v27.4s, v1.4s, v6.s[3]    \n"
//         "prfm   pldl1keep, [%1, 0x180]    \n"
//         "fmla   v28.4s, v1.4s, v7.s[0]    \n"
//         "fmla   v29.4s, v1.4s, v7.s[1]    \n"
//         "prfm   pldl1keep, [%1, 0x140]    \n"
//         "fmla   v30.4s, v1.4s, v7.s[2]    \n"
//         "fmla   v31.4s, v1.4s, v7.s[3]    \n"

//         "ld1    {v6.4s, v7.4s}, [%1], #32 \n"
//         "fmla   v16.4s, v0.4s, v4.s[0]    \n"
//         "fmla   v17.4s, v0.4s, v4.s[1]    \n"
//         "fmla   v18.4s, v0.4s, v4.s[2]    \n"
//         "fmla   v19.4s, v0.4s, v4.s[3]    \n"
//         "ld1    {v0.4s}, [%0],        #16 \n"   // load packA, packA
//         "fmla   v20.4s, v0.4s, v5.s[0]    \n"
//         "fmla   v21.4s, v0.4s, v5.s[1]    \n"
//         "fmla   v22.4s, v0.4s, v5.s[2]    \n"
//         "fmla   v23.4s, v0.4s, v5.s[3]    \n"
//         "ld1    {v4.4s, v5.4s}, [%1], #32 \n"   // load packB, packB+12
//         "fmla   v24.4s, v0.4s, v6.s[0]    \n"
//         "fmla   v25.4s, v0.4s, v6.s[1]    \n"
//         "fmla   v26.4s, v0.4s, v6.s[2]    \n"
//         "fmla   v27.4s, v0.4s, v6.s[3]    \n"
//         "prfm   pldl1keep, [%1, 0x180]    \n"
//         "fmla   v28.4s, v0.4s, v7.s[0]    \n"
//         "fmla   v29.4s, v0.4s, v7.s[1]    \n"
//         "fmla   v30.4s, v0.4s, v7.s[2]    \n"
//         "fmla   v31.4s, v0.4s, v7.s[3]    \n"

//         "ld1    {v6.4s, v7.4s}, [%1], #32 \n"
//         "fmla   v16.4s, v1.4s, v4.s[0]    \n"
//         "fmla   v17.4s, v1.4s, v4.s[1]    \n"
//         "fmla   v18.4s, v1.4s, v4.s[2]    \n"
//         "fmla   v19.4s, v1.4s, v4.s[3]    \n"
//         "ld1    {v0.4s}, [%0],        #16 \n"   // load packA, packA
//         "fmla   v20.4s, v1.4s, v5.s[0]    \n"
//         "fmla   v21.4s, v1.4s, v5.s[1]    \n"
//         "fmla   v22.4s, v1.4s, v5.s[2]    \n"
//         "fmla   v23.4s, v1.4s, v5.s[3]    \n"
//         "ld1    {v4.4s, v5.4s}, [%1], #32 \n"   // load packB, packB+12
//         "fmla   v24.4s, v1.4s, v6.s[0]    \n"
//         "fmla   v25.4s, v1.4s, v6.s[1]    \n"
//         "fmla   v26.4s, v1.4s, v6.s[2]    \n"
//         "fmla   v27.4s, v1.4s, v6.s[3]    \n"
//         "fmla   v28.4s, v1.4s, v7.s[0]    \n"
//         "fmla   v29.4s, v1.4s, v7.s[1]    \n"
//         "fmla   v30.4s, v1.4s, v7.s[2]    \n"
//         "fmla   v31.4s, v1.4s, v7.s[3]    \n"

//         "bne    0b                        \n"

//         "2:                               \n"   // save_result
//         "st4    {v16.s, v17.s, v18.s, v19.s}[0], [x12], #16 \n"
//         "st4    {v16.s, v17.s, v18.s, v19.s}[1], [x13], #16 \n"
//         "st4    {v16.s, v17.s, v18.s, v19.s}[2], [x14], #16 \n"
//         "st4    {v16.s, v17.s, v18.s, v19.s}[3], [x15], #16 \n"

//         "st4    {v20.s, v21.s, v22.s, v23.s}[0], [x12], #16 \n"
//         "st4    {v20.s, v21.s, v22.s, v23.s}[1], [x13], #16 \n"
//         "st4    {v20.s, v21.s, v22.s, v23.s}[2], [x14], #16 \n"
//         "st4    {v20.s, v21.s, v22.s, v23.s}[3], [x15], #16 \n"

//         "st4    {v24.s, v25.s, v26.s, v27.s}[0], [x12], #16 \n"
//         "st4    {v24.s, v25.s, v26.s, v27.s}[1], [x13], #16 \n"
//         "st4    {v24.s, v25.s, v26.s, v27.s}[2], [x14], #16 \n"
//         "st4    {v24.s, v25.s, v26.s, v27.s}[3], [x15], #16 \n"

//         "st4    {v28.s, v29.s, v30.s, v31.s}[0], [x12]      \n"
//         "st4    {v28.s, v29.s, v30.s, v31.s}[1], [x13]      \n"
//         "st4    {v28.s, v29.s, v30.s, v31.s}[2], [x14]      \n"
//         "st4    {v28.s, v29.s, v30.s, v31.s}[3], [x15]      \n"

//         : "=r"(packAPtr), // %0
//         "=r"(packBPtr),   // %1
//         "=r"(packCPtr0),    // %2
//         "=r"(packCPtr1),    // %3
//         "=r"(packCPtr2),    // %4
//         "=r"(packCPtr3)    // %5
//         : "0"(packAPtr),
//         "1"(packBPtr),
//         "2"(packCPtr0),
//         "3"(packCPtr1),
//         "4"(packCPtr2),
//         "5"(packCPtr3),
//         "r"(kc_adjust), // %12
//         "r"(ldc),       // %13
//         "r"(PREFETCH_A),// %14
//         "r"(PREFETCH_B),// %15
//         "r"(PREFETCH_C) // %16
//         : "cc", "memory", "x4", "x5", "x6", "x9", "x10", 
//                           "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
//                           "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
//                           "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
//                           "v30", "v31");

#ifdef __ARM_ASM
        asm volatile(
        "mov    x4, %12                   \n"
        "mov    x5, %14                   \n"
        "mov    x6, %15                   \n"

        "ld1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%2]      \n"
        "ld1    {v14.4s, v15.4s, v16.4s, v17.4s}, [%3]      \n"
        "ld1    {v18.4s, v19.4s, v20.4s, v21.4s}, [%4]      \n"
        "ld1    {v22.4s, v23.4s, v24.4s, v25.4s}, [%5]      \n"

        "cmp    w4, #0                    \n"
        "beq    8f                        \n"

        "ld1    {v0.4s}, [%0],        #16 \n"   // load packA, packA
        "ld1    {v2.4s, v3.4s, v4.4s, v5.4s}, [%1], #64 \n"   // load packB, packB+12
        
        "lsr    x9, x4, 0x1               \n"
        "cmp    w9, #0                    \n"
        "beq    5f                        \n"

    "4:                          \n"

        "subs   w9, w9, 0x1               \n"

        "prfm   pldl1keep, [%0, x5]       \n"
        "ld1    {v1.4s}, [%0],        #16 \n"   // load packA, packA
        "fmla   v10.4s, v2.4s, v0.s[0]    \n"
        "fmla   v11.4s, v3.4s, v0.s[0]    \n"
        "fmla   v12.4s, v4.4s, v0.s[0]    \n"
        "fmla   v13.4s, v5.4s, v0.s[0]    \n"
        
        "fmla   v14.4s, v2.4s, v0.s[1]    \n"
        "fmla   v15.4s, v3.4s, v0.s[1]    \n"
        "fmla   v16.4s, v4.4s, v0.s[1]    \n"
        "fmla   v17.4s, v5.4s, v0.s[1]    \n"

        "prfm   pldl1keep, [%1, x6]       \n"
        "ld1    {v6.4s, v7.4s, v8.4s, v9.4s}, [%1], #64 \n"   // load packB, packB+12
        "fmla   v18.4s, v2.4s, v0.s[2]    \n"
        "fmla   v19.4s, v3.4s, v0.s[2]    \n"
        "fmla   v20.4s, v4.4s, v0.s[2]    \n"
        "fmla   v21.4s, v5.4s, v0.s[2]    \n"

        "fmla   v22.4s, v2.4s, v0.s[3]    \n"
        "fmla   v23.4s, v3.4s, v0.s[3]    \n"
        "fmla   v24.4s, v4.4s, v0.s[3]    \n"
        "fmla   v25.4s, v5.4s, v0.s[3]    \n"

        "ld1    {v0.4s}, [%0],        #16 \n"   // load packA, packA
        "fmla   v10.4s, v6.4s, v1.s[0]    \n"
        "fmla   v11.4s, v7.4s, v1.s[0]    \n"
        "fmla   v12.4s, v8.4s, v1.s[0]    \n"
        "fmla   v13.4s, v9.4s, v1.s[0]    \n"
        
        "fmla   v14.4s, v6.4s, v1.s[1]    \n"
        "fmla   v15.4s, v7.4s, v1.s[1]    \n"
        "fmla   v16.4s, v8.4s, v1.s[1]    \n"
        "fmla   v17.4s, v9.4s, v1.s[1]    \n"

        "prfm   pldl1keep, [%1, x6]       \n"
        "ld1    {v2.4s, v3.4s, v4.4s, v5.4s}, [%1], #64 \n"   // load packB, packB+12
        "fmla   v18.4s, v6.4s, v1.s[2]    \n"
        "fmla   v19.4s, v7.4s, v1.s[2]    \n"
        "fmla   v20.4s, v8.4s, v1.s[2]    \n"
        "fmla   v21.4s, v9.4s, v1.s[2]    \n"

        "fmla   v22.4s, v6.4s, v1.s[3]    \n"
        "fmla   v23.4s, v7.4s, v1.s[3]    \n"
        "fmla   v24.4s, v8.4s, v1.s[3]    \n"
        "fmla   v25.4s, v9.4s, v1.s[3]    \n"
        
        "bne    4b                        \n"

    "5:                                   \n"

        "ands   x10, x4, 0x1              \n"
        "beq    7f                        \n"

    "6:                                   \n"

        "fmla   v10.4s, v2.4s, v0.s[0]    \n"
        "fmla   v11.4s, v3.4s, v0.s[0]    \n"
        "fmla   v12.4s, v4.4s, v0.s[0]    \n"
        "fmla   v13.4s, v5.4s, v0.s[0]    \n"
        
        "fmla   v14.4s, v2.4s, v0.s[1]    \n"
        "fmla   v15.4s, v3.4s, v0.s[1]    \n"
        "fmla   v16.4s, v4.4s, v0.s[1]    \n"
        "fmla   v17.4s, v5.4s, v0.s[1]    \n"

        "fmla   v18.4s, v2.4s, v0.s[2]    \n"
        "fmla   v19.4s, v3.4s, v0.s[2]    \n"
        "fmla   v20.4s, v4.4s, v0.s[2]    \n"
        "fmla   v21.4s, v5.4s, v0.s[2]    \n"

        "fmla   v22.4s, v2.4s, v0.s[3]    \n"
        "fmla   v23.4s, v3.4s, v0.s[3]    \n"
        "fmla   v24.4s, v4.4s, v0.s[3]    \n"
        "fmla   v25.4s, v5.4s, v0.s[3]    \n"

    "7:                                   \n"

        "st1    {v10.4s, v11.4s, v12.4s, v13.4s}, [%2]      \n"
        "st1    {v14.4s, v15.4s, v16.4s, v17.4s}, [%3]      \n"
        "st1    {v18.4s, v19.4s, v20.4s, v21.4s}, [%4]      \n"
        "st1    {v22.4s, v23.4s, v24.4s, v25.4s}, [%5]      \n"
    
    "8:                                   \n"

        : "=r"(packAPtr), // %0
        "=r"(packBPtr),   // %1
        "=r"(packCPtr0),    // %2
        "=r"(packCPtr1),    // %3
        "=r"(packCPtr2),    // %4
        "=r"(packCPtr3)    // %5
        : "0"(packAPtr),
        "1"(packBPtr),
        "2"(packCPtr0),
        "3"(packCPtr1),
        "4"(packCPtr2),
        "5"(packCPtr3),
        "r"(kc_adjust), // %12
        "r"(ldc),       // %13
        "r"(PREFETCH_A),// %14
        "r"(PREFETCH_B),// %15
        "r"(PREFETCH_C) // %16
        : "cc", "memory", "x4", "x5", "x6", "x9", "x10", 
                          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
                          "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19",
                          "v20", "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29",
                          "v30", "v31");

// #ifdef __ARM_ASM
//         asm volatile(
//         "mov    x4, %12                   \n"
//         "mov    x5, %14                   \n"
//         "mov    x6, %15                   \n"

//         "ld1    {v5.4s,  v6.4s,  v7.4s,  v8.4s},  [%2]      \n"
//         "ld1    {v9.4s,  v10.4s, v11.4s, v12.4s}, [%3]      \n"
//         "ld1    {v13.4s, v14.4s, v15.4s, v16.4s}, [%4]      \n"
//         "ld1    {v17.4s, v18.4s, v19.4s, v20.4s}, [%5]      \n"

//         "cmp    w4, #0                    \n"
//         "beq    1f                        \n"

//         "0:                               \n"

//         "ld1    {v0.4s}, [%0],        #16 \n"   // load packA, packA
//         "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%1], #64 \n"   // load packB, packB+12

//         "prfm   pldl1keep, [%0, x5]      \n"
//         "fmla   v5.4s, v1.4s, v0.s[0]    \n"
//         "fmla   v6.4s, v2.4s, v0.s[0]    \n"
//         "fmla   v7.4s, v3.4s, v0.s[0]    \n"
//         "fmla   v8.4s, v4.4s, v0.s[0]    \n"
        
//         "fmla   v9.4s,  v1.4s, v0.s[1]    \n"
//         "fmla   v10.4s, v2.4s, v0.s[1]    \n"
//         "fmla   v11.4s, v3.4s, v0.s[1]    \n"
//         "fmla   v12.4s, v4.4s, v0.s[1]    \n"

//         "prfm   pldl1keep, [%1, x6]       \n"
//         "fmla   v13.4s, v1.4s, v0.s[2]    \n"
//         "fmla   v14.4s, v2.4s, v0.s[2]    \n"
//         "fmla   v15.4s, v3.4s, v0.s[2]    \n"
//         "fmla   v16.4s, v4.4s, v0.s[2]    \n"

//         "fmla   v17.4s, v1.4s, v0.s[3]    \n"
//         "fmla   v18.4s, v2.4s, v0.s[3]    \n"
//         "fmla   v19.4s, v3.4s, v0.s[3]    \n"
//         "fmla   v20.4s, v4.4s, v0.s[3]    \n"
        
//         "subs   w4, w4, #1                \n"
//         "bne    0b                        \n"

//         "1:                               \n"

//         "st1    {v5.4s,  v6.4s,  v7.4s,  v8.4s},  [%2]      \n"
//         "st1    {v9.4s,  v10.4s, v11.4s, v12.4s}, [%3]      \n"
//         "st1    {v13.4s, v14.4s, v15.4s, v16.4s}, [%4]      \n"
//         "st1    {v17.4s, v18.4s, v19.4s, v20.4s}, [%5]      \n"

//         : "=r"(packAPtr), // %0
//         "=r"(packBPtr),   // %1
//         "=r"(packCPtr0),    // %2
//         "=r"(packCPtr1),    // %3
//         "=r"(packCPtr2),    // %4
//         "=r"(packCPtr3)    // %5
//         : "0"(packAPtr),
//         "1"(packBPtr),
//         "2"(packCPtr0),
//         "3"(packCPtr1),
//         "4"(packCPtr2),
//         "5"(packCPtr3),
//         "r"(kc_adjust), // %12
//         "r"(ldc),       // %13
//         "r"(PREFETCH_A),// %14
//         "r"(PREFETCH_B),// %15
//         "r"(PREFETCH_C) // %16
//         : "cc", "memory", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20");

#else
    float32x4_t a0;
    float32x4_t b0, b1, b2, b3;
    float32x4_t c00, c01, c02, c03, c10, c11, c12, c13;
    float32x4_t c20, c21, c22, c23, c30, c31, c32, c33;

    // 加载C的 4x16 block
    c00 = vld1q_f32(packCPtr0);
    c01 = vld1q_f32(packCPtr0 + 4);
    c02 = vld1q_f32(packCPtr0 + 8);
    c03 = vld1q_f32(packCPtr0 + 12);

    c10 = vld1q_f32(packCPtr1);
    c11 = vld1q_f32(packCPtr1 + 4);
    c12 = vld1q_f32(packCPtr1 + 8);
    c13 = vld1q_f32(packCPtr1 + 12);

    c20 = vld1q_f32(packCPtr2);
    c21 = vld1q_f32(packCPtr2 + 4);
    c22 = vld1q_f32(packCPtr2 + 8);
    c23 = vld1q_f32(packCPtr2 + 12);

    c30 = vld1q_f32(packCPtr3);
    c31 = vld1q_f32(packCPtr3 + 4);
    c32 = vld1q_f32(packCPtr3 + 8);
    c33 = vld1q_f32(packCPtr3 + 12);

    // 计算
    for (int k = 0; k < kc_adjust; k++) {
        a0 = vld1q_f32(packAPtr);
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        b2 = vld1q_f32(packBPtr + 8);
        b3 = vld1q_f32(packBPtr + 12);

        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);
        c03 = vfmaq_laneq_f32(c03, b3, a0, 0);

        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);
        c13 = vfmaq_laneq_f32(c13, b3, a0, 1);

        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);
        c23 = vfmaq_laneq_f32(c23, b3, a0, 2);

        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
        c33 = vfmaq_laneq_f32(c33, b3, a0, 3);

        packAPtr += 4;
        packBPtr += 16;
    }

    // 将结果写回
    vst1q_f32(packCPtr0     , c00);
    vst1q_f32(packCPtr0 + 4 , c01);
    vst1q_f32(packCPtr0 + 8 , c02);
    vst1q_f32(packCPtr0 + 12, c03);

    vst1q_f32(packCPtr1     , c10);
    vst1q_f32(packCPtr1 + 4 , c11);
    vst1q_f32(packCPtr1 + 8 , c12);
    vst1q_f32(packCPtr1 + 12, c13);

    vst1q_f32(packCPtr2     , c20);
    vst1q_f32(packCPtr2 + 4 , c21);
    vst1q_f32(packCPtr2 + 8 , c22);
    vst1q_f32(packCPtr2 + 12, c23);

    vst1q_f32(packCPtr3     , c30);
    vst1q_f32(packCPtr3 + 4 , c31);
    vst1q_f32(packCPtr3 + 8 , c32);
    vst1q_f32(packCPtr3 + 12, c33);
#endif
}

void kernel_8x12(int kc_adjust, float *packA, float* packB, float *packC, int ldc, const int PREFETCH_A, const int PREFETCH_B, const int PREFETCH_C) {
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;

    float* packCPtr0 = packC;
    float* packCPtr1 = packC + 1 * ldc;
    float* packCPtr2 = packC + 2 * ldc;
    float* packCPtr3 = packC + 3 * ldc;
    float* packCPtr4 = packC + 4 * ldc;
    float* packCPtr5 = packC + 5 * ldc;
    float* packCPtr6 = packC + 6 * ldc;
    float* packCPtr7 = packC + 7 * ldc;

#ifdef __ARM_ASM
    asm volatile(
        "mov    x4, %20                   \n"
        "mov    x5, %22                   \n"
        "mov    x6, %23                   \n"
        
        "cmp    w4, #0                    \n"
        "beq    4f                  \n"

        // "ld1    {v5.4s,  v6.4s,  v7.4s},  [%2]    \n"
        "ld1    {v8.4s,  v9.4s,  v10.4s}, [%2]    \n"
        "ld1    {v11.4s, v12.4s, v13.4s}, [%3]    \n"
        "ld1    {v14.4s, v15.4s, v16.4s}, [%4]    \n"
        "ld1    {v17.4s, v18.4s, v19.4s}, [%5]    \n"
        "ld1    {v20.4s, v21.4s, v22.4s}, [%6]    \n"
        "ld1    {v23.4s, v24.4s, v25.4s}, [%7]    \n"
        "ld1    {v26.4s, v27.4s, v28.4s}, [%8]    \n"
        "ld1    {v29.4s, v30.4s, v31.4s}, [%9]    \n"

        "ld1    {v0.4s}, [%0], #16 \n"   // load packA
        "ld1    {v2.4s, v3.4s, v4.4s}, [%1], #48 \n"   // load packB, packB+4
        
        "lsr    x9, x4, 0x1               \n"
        "cmp    w9, #0                    \n"
        "beq    1f            \n"

    "0:                          \n"
        "subs   w9, w9, #1                \n"

        "prfm   pldl1keep, [%0, x5]     \n"
        
        "fmla   v8.4s,  v2.4s, v0.s[0]    \n"
        "fmla   v9.4s,  v3.4s, v0.s[0]    \n"
        "fmla   v10.4s, v4.4s, v0.s[0]    \n"
        
        "fmla   v11.4s, v2.4s, v0.s[1]    \n"
        "fmla   v12.4s, v3.4s, v0.s[1]    \n"
        "fmla   v13.4s, v4.4s, v0.s[1]    \n"
        "ld1    {v1.4s}, [%0], #16 \n"   // load packA
        
        "fmla   v14.4s, v2.4s, v0.s[2]    \n"
        "fmla   v15.4s, v3.4s, v0.s[2]    \n"
        "fmla   v16.4s, v4.4s, v0.s[2]    \n"
        
        "prfm   pldl1keep, [%1, x6]     \n"
        "fmla   v17.4s, v2.4s, v0.s[3]    \n"
        "fmla   v18.4s, v3.4s, v0.s[3]    \n"
        "fmla   v19.4s, v4.4s, v0.s[3]    \n"
        "ld1    {v5.4s, v6.4s, v7.4s}, [%1], #48 \n"   // load packB, packB+4
        
        "fmla   v20.4s, v2.4s, v1.s[0]    \n"
        "fmla   v21.4s, v3.4s, v1.s[0]    \n"
        "fmla   v22.4s, v4.4s, v1.s[0]    \n"

        "fmla   v23.4s, v2.4s, v1.s[1]    \n"
        "fmla   v24.4s, v3.4s, v1.s[1]    \n"
        "fmla   v25.4s, v4.4s, v1.s[1]    \n"
        "ld1    {v0.4s}, [%0], #16 \n"   // load packA
        
        "fmla   v26.4s, v2.4s, v1.s[2]    \n"
        "fmla   v27.4s, v3.4s, v1.s[2]    \n"
        "fmla   v28.4s, v4.4s, v1.s[2]    \n"

        "fmla   v29.4s, v2.4s, v1.s[3]    \n"
        "fmla   v30.4s, v3.4s, v1.s[3]    \n"
        "fmla   v31.4s, v4.4s, v1.s[3]    \n"
        "prfm   pldl1keep, [%0, x5]     \n"

        "fmla   v8.4s,  v5.4s, v0.s[0]    \n"
        "fmla   v9.4s,  v6.4s, v0.s[0]    \n"
        "fmla   v10.4s, v7.4s, v0.s[0]    \n"
        
        "fmla   v11.4s, v5.4s, v0.s[1]    \n"
        "fmla   v12.4s, v6.4s, v0.s[1]    \n"
        "fmla   v13.4s, v7.4s, v0.s[1]    \n"
        "ld1    {v1.4s}, [%0], #16 \n"   // load packA
        
        "fmla   v14.4s, v5.4s, v0.s[2]    \n"
        "fmla   v15.4s, v6.4s, v0.s[2]    \n"
        "fmla   v16.4s, v7.4s, v0.s[2]    \n"
        
        "prfm   pldl1keep, [%1, x6]     \n"
        "fmla   v17.4s, v5.4s, v0.s[3]    \n"
        "fmla   v18.4s, v6.4s, v0.s[3]    \n"
        "fmla   v19.4s, v7.4s, v0.s[3]    \n"
        "ld1    {v2.4s, v3.4s, v4.4s}, [%1], #48 \n"   // load packB, packB+4
        
        "fmla   v20.4s, v5.4s, v1.s[0]    \n"
        "fmla   v21.4s, v6.4s, v1.s[0]    \n"
        "fmla   v22.4s, v7.4s, v1.s[0]    \n"

        "fmla   v23.4s, v5.4s, v1.s[1]    \n"
        "fmla   v24.4s, v6.4s, v1.s[1]    \n"
        "fmla   v25.4s, v7.4s, v1.s[1]    \n"
        
        "ld1    {v0.4s}, [%0], #16 \n"   // load packA
        "fmla   v26.4s, v5.4s, v1.s[2]    \n"
        "fmla   v27.4s, v6.4s, v1.s[2]    \n"
        "fmla   v28.4s, v7.4s, v1.s[2]    \n"

        "fmla   v29.4s, v5.4s, v1.s[3]    \n"
        "fmla   v30.4s, v6.4s, v1.s[3]    \n"
        "fmla   v31.4s, v7.4s, v1.s[3]    \n"

        "bne    0b                \n"

    "1:                      \n"
        "ands   x10, x4, 0x1              \n"
        "beq    3f          \n"

    "2:                          \n"
        "prfm   pldl1keep, [%0, x5]     \n"
        "ld1    {v1.4s}, [%0], #16 \n"   // load packA
        
        "fmla   v8.4s,  v2.4s, v0.s[0]    \n"
        "fmla   v9.4s,  v3.4s, v0.s[0]    \n"
        "fmla   v10.4s, v4.4s, v0.s[0]    \n"
        
        "fmla   v11.4s, v2.4s, v0.s[1]    \n"
        "fmla   v12.4s, v3.4s, v0.s[1]    \n"
        "fmla   v13.4s, v4.4s, v0.s[1]    \n"
        
        "fmla   v14.4s, v2.4s, v0.s[2]    \n"
        "fmla   v15.4s, v3.4s, v0.s[2]    \n"
        "fmla   v16.4s, v4.4s, v0.s[2]    \n"
        
        "fmla   v17.4s, v2.4s, v0.s[3]    \n"
        "fmla   v18.4s, v3.4s, v0.s[3]    \n"
        "fmla   v19.4s, v4.4s, v0.s[3]    \n"
        
        "fmla   v20.4s, v2.4s, v1.s[0]    \n"
        "fmla   v21.4s, v3.4s, v1.s[0]    \n"
        "fmla   v22.4s, v4.4s, v1.s[0]    \n"

        "fmla   v23.4s, v2.4s, v1.s[1]    \n"
        "fmla   v24.4s, v3.4s, v1.s[1]    \n"
        "fmla   v25.4s, v4.4s, v1.s[1]    \n"
        
        "fmla   v26.4s, v2.4s, v1.s[2]    \n"
        "fmla   v27.4s, v3.4s, v1.s[2]    \n"
        "fmla   v28.4s, v4.4s, v1.s[2]    \n"

        "fmla   v29.4s, v2.4s, v1.s[3]    \n"
        "fmla   v30.4s, v3.4s, v1.s[3]    \n"
        "fmla   v31.4s, v4.4s, v1.s[3]    \n"

    "3:                    \n"

        // "st1    {v5.4s,  v6.4s,  v7.4s},  [%2]    \n"
        "st1    {v8.4s,  v9.4s,  v10.4s}, [%2]    \n"
        "st1    {v11.4s, v12.4s, v13.4s}, [%3]    \n"
        "st1    {v14.4s, v15.4s, v16.4s}, [%4]    \n"
        "st1    {v17.4s, v18.4s, v19.4s}, [%5]    \n"
        "st1    {v20.4s, v21.4s, v22.4s}, [%6]    \n"
        "st1    {v23.4s, v24.4s, v25.4s}, [%7]    \n"
        "st1    {v26.4s, v27.4s, v28.4s}, [%8]    \n"
        "st1    {v29.4s, v30.4s, v31.4s}, [%9]    \n"

    "4:                                    \n"

        : "=r"(packAPtr), // %0
        "=r"(packBPtr),   // %1
        "=r"(packCPtr0),    // %2
        "=r"(packCPtr1),    // %3
        "=r"(packCPtr2),    // %4
        "=r"(packCPtr3),    // %5
        "=r"(packCPtr4),    // %6
        "=r"(packCPtr5),    // %7
        "=r"(packCPtr6),    // %8
        "=r"(packCPtr7)     // %9
        : "0"(packAPtr),
        "1"(packBPtr),
        "2"(packCPtr0),
        "3"(packCPtr1),
        "4"(packCPtr2),
        "5"(packCPtr3),
        "6"(packCPtr4),
        "7"(packCPtr5),
        "8"(packCPtr6),
        "9"(packCPtr7),
        "r"(kc_adjust), // %20
        "r"(ldc),       // %21
        "r"(PREFETCH_A),// %22
        "r"(PREFETCH_B),// %23
        "r"(PREFETCH_C) // %24
        : "cc", "memory", "x4", "x5", "x6", "x9", "x10", 
                          "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", 
                          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                          "v29", "v30", "v31");

// #ifdef __ARM_ASM
//     asm volatile(
//         "mov    x4, %20                   \n"
//         "mov    x5, %22                   \n"
//         "mov    x6, %23                   \n"

//         "ld1    {v5.4s,  v6.4s,  v7.4s},  [%2]    \n"
//         "ld1    {v8.4s,  v9.4s,  v10.4s}, [%3]    \n"
//         "ld1    {v11.4s, v12.4s, v13.4s}, [%4]    \n"
//         "ld1    {v14.4s, v15.4s, v16.4s}, [%5]    \n"
//         "ld1    {v17.4s, v18.4s, v19.4s}, [%6]    \n"
//         "ld1    {v20.4s, v21.4s, v22.4s}, [%7]    \n"
//         "ld1    {v23.4s, v24.4s, v25.4s}, [%8]    \n"
//         "ld1    {v26.4s, v27.4s, v28.4s}, [%9]    \n"

//         "cmp    w4, #0                    \n"
//         "beq    1f                        \n"

//         "0:                               \n"

//         "ld1    {v0.4s, v1.4s}, [%0], #32 \n"   // load packA, packA+4
//         "ld1    {v2.4s, v3.4s, v4.4s}, [%1], #48 \n"   // load packB, packB+4

//         "fmla   v5.4s, v2.4s, v0.s[0]    \n"
//         "fmla   v6.4s, v3.4s, v0.s[0]    \n"
//         "fmla   v7.4s, v4.4s, v0.s[0]    \n"
        
//         "fmla   v8.4s,  v2.4s, v0.s[1]    \n"
//         "fmla   v9.4s,  v3.4s, v0.s[1]    \n"
//         "fmla   v10.4s, v4.4s, v0.s[1]    \n"
//         "prfm   pldl1keep, [%0, x5]     \n"
        
//         "fmla   v11.4s, v2.4s, v0.s[2]    \n"
//         "fmla   v12.4s, v3.4s, v0.s[2]    \n"
//         "fmla   v13.4s, v4.4s, v0.s[2]    \n"
        
//         "fmla   v14.4s, v2.4s, v0.s[3]    \n"
//         "fmla   v15.4s, v3.4s, v0.s[3]    \n"
//         "fmla   v16.4s, v4.4s, v0.s[3]    \n"
        
//         "fmla   v17.4s, v2.4s, v1.s[0]    \n"
//         "fmla   v18.4s, v3.4s, v1.s[0]    \n"
//         "prfm   pldl1keep, [%1, x6]     \n"
//         "fmla   v19.4s, v4.4s, v1.s[0]    \n"
        
//         "fmla   v20.4s, v2.4s, v1.s[1]    \n"
//         "fmla   v21.4s, v3.4s, v1.s[1]    \n"
//         "fmla   v22.4s, v4.4s, v1.s[1]    \n"
        
//         "fmla   v23.4s, v2.4s, v1.s[2]    \n"
//         "fmla   v24.4s, v3.4s, v1.s[2]    \n"
//         "fmla   v25.4s, v4.4s, v1.s[2]    \n"

//         "fmla   v26.4s, v2.4s, v1.s[3]    \n"
//         "fmla   v27.4s, v3.4s, v1.s[3]    \n"
//         "fmla   v28.4s, v4.4s, v1.s[3]    \n"

//         "subs   w4, w4, #1                \n"
//         "bne    0b                        \n"

//         "1:                               \n"

//         "st1    {v5.4s,  v6.4s,  v7.4s},  [%2]    \n"
//         "st1    {v8.4s,  v9.4s,  v10.4s}, [%3]    \n"
//         "st1    {v11.4s, v12.4s, v13.4s}, [%4]    \n"
//         "st1    {v14.4s, v15.4s, v16.4s}, [%5]    \n"
//         "st1    {v17.4s, v18.4s, v19.4s}, [%6]    \n"
//         "st1    {v20.4s, v21.4s, v22.4s}, [%7]    \n"
//         "st1    {v23.4s, v24.4s, v25.4s}, [%8]    \n"
//         "st1    {v26.4s, v27.4s, v28.4s}, [%9]    \n"

//         : "=r"(packAPtr), // %0
//         "=r"(packBPtr),   // %1
//         "=r"(packCPtr0),    // %2
//         "=r"(packCPtr1),    // %3
//         "=r"(packCPtr2),    // %4
//         "=r"(packCPtr3),    // %5
//         "=r"(packCPtr4),    // %6
//         "=r"(packCPtr5),    // %7
//         "=r"(packCPtr6),    // %8
//         "=r"(packCPtr7)     // %9
//         : "0"(packAPtr),
//         "1"(packBPtr),
//         "2"(packCPtr0),
//         "3"(packCPtr1),
//         "4"(packCPtr2),
//         "5"(packCPtr3),
//         "6"(packCPtr4),
//         "7"(packCPtr5),
//         "8"(packCPtr6),
//         "9"(packCPtr7),
//         "r"(kc_adjust), // %20
//         "r"(ldc),       // %21
//         "r"(PREFETCH_A),// %22
//         "r"(PREFETCH_B),// %23
//         "r"(PREFETCH_C) // %24
//         : "cc", "memory", "x4", "x5", "x6", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
//                           "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", 
//                           "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28");

#else
    
    float32x4_t a0, a1;
    float32x4_t b0, b1, b2;
    float32x4_t c00, c01, c02, c10, c11, c12;
    float32x4_t c20, c21, c22, c30, c31, c32;
    float32x4_t c40, c41, c42, c50, c51, c52;
    float32x4_t c60, c61, c62, c70, c71, c72;

    // 加载c的 8 x 12 block
    c00 = vld1q_f32(packCPtr0);
    c01 = vld1q_f32(packCPtr0 + 4);
    c02 = vld1q_f32(packCPtr0 + 8);

    c10 = vld1q_f32(packCPtr1);
    c11 = vld1q_f32(packCPtr1 + 4);
    c12 = vld1q_f32(packCPtr1 + 8);

    c20 = vld1q_f32(packCPtr2);
    c21 = vld1q_f32(packCPtr2 + 4);
    c22 = vld1q_f32(packCPtr2 + 8);

    c30 = vld1q_f32(packCPtr3);
    c31 = vld1q_f32(packCPtr3 + 4);
    c32 = vld1q_f32(packCPtr3 + 8);

    c40 = vld1q_f32(packCPtr4);
    c41 = vld1q_f32(packCPtr4 + 4);
    c42 = vld1q_f32(packCPtr4 + 8);

    c50 = vld1q_f32(packCPtr5);
    c51 = vld1q_f32(packCPtr5 + 4);
    c52 = vld1q_f32(packCPtr5 + 8);

    c60 = vld1q_f32(packCPtr6);
    c61 = vld1q_f32(packCPtr6 + 4);
    c62 = vld1q_f32(packCPtr6 + 8);

    c70 = vld1q_f32(packCPtr7);
    c71 = vld1q_f32(packCPtr7 + 4);
    c72 = vld1q_f32(packCPtr7 + 8);

    // 计算
    for (int k = 0; k < kc_adjust; k++) {
        a0 = vld1q_f32(packAPtr);
        a1 = vld1q_f32(packAPtr + 4);
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        b2 = vld1q_f32(packBPtr + 8);

        c00 = vfmaq_laneq_f32(c00, b0, a0, 0);
        c01 = vfmaq_laneq_f32(c01, b1, a0, 0);
        c02 = vfmaq_laneq_f32(c02, b2, a0, 0);

        c10 = vfmaq_laneq_f32(c10, b0, a0, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a0, 1);
        c12 = vfmaq_laneq_f32(c12, b2, a0, 1);

        c20 = vfmaq_laneq_f32(c20, b0, a0, 2);
        c21 = vfmaq_laneq_f32(c21, b1, a0, 2);
        c22 = vfmaq_laneq_f32(c22, b2, a0, 2);

        c30 = vfmaq_laneq_f32(c30, b0, a0, 3);
        c31 = vfmaq_laneq_f32(c31, b1, a0, 3);
        c32 = vfmaq_laneq_f32(c32, b2, a0, 3);
 
        c40 = vfmaq_laneq_f32(c40, b0, a1, 0);
        c41 = vfmaq_laneq_f32(c41, b1, a1, 0);
        c42 = vfmaq_laneq_f32(c42, b2, a1, 0);

        c50 = vfmaq_laneq_f32(c50, b0, a1, 1);
        c51 = vfmaq_laneq_f32(c51, b1, a1, 1);
        c52 = vfmaq_laneq_f32(c52, b2, a1, 1);

        c60 = vfmaq_laneq_f32(c60, b0, a1, 2);
        c61 = vfmaq_laneq_f32(c61, b1, a1, 2);
        c62 = vfmaq_laneq_f32(c62, b2, a1, 2);

        c70 = vfmaq_laneq_f32(c70, b0, a1, 3);
        c71 = vfmaq_laneq_f32(c71, b1, a1, 3);
        c72 = vfmaq_laneq_f32(c72, b2, a1, 3);

        packAPtr += 8;
        packBPtr += 12;
    }

    // 将c写回内存
    vst1q_f32(packCPtr0    , c00);
    vst1q_f32(packCPtr0 + 4, c01);
    vst1q_f32(packCPtr0 + 8, c02);

    vst1q_f32(packCPtr1    , c10);
    vst1q_f32(packCPtr1 + 4, c11);
    vst1q_f32(packCPtr1 + 8, c12);

    vst1q_f32(packCPtr2    , c20);
    vst1q_f32(packCPtr2 + 4, c21);
    vst1q_f32(packCPtr2 + 8, c22);

    vst1q_f32(packCPtr3    , c30);
    vst1q_f32(packCPtr3 + 4, c31);
    vst1q_f32(packCPtr3 + 8, c32);

    vst1q_f32(packCPtr4    , c40);
    vst1q_f32(packCPtr4 + 4, c41);
    vst1q_f32(packCPtr4 + 8, c42);

    vst1q_f32(packCPtr5    , c50);
    vst1q_f32(packCPtr5 + 4, c51);
    vst1q_f32(packCPtr5 + 8, c52);

    vst1q_f32(packCPtr6    , c60);
    vst1q_f32(packCPtr6 + 4, c61);
    vst1q_f32(packCPtr6 + 8, c62);

    vst1q_f32(packCPtr7    , c70);
    vst1q_f32(packCPtr7 + 4, c71);
    vst1q_f32(packCPtr7 + 8, c72);
#endif
}

void kernel_8x8_packC(int kc_adjust, float *packA, float* packB, float *packC, int ldc, const int PREFETCH_A, const int PREFETCH_B, const int PREFETCH_C) 
{
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;
    float* packCPtr0 = packC;

#ifdef __ARM_ASM
    asm volatile(
        "mov    x4, %8                     \n"
        "mov    x5, %10                    \n"
        "mov    x6, %11                    \n"
        "mov    x7, %12                    \n"

        "ld1    {v4.4s, v5.4s, v6.4s, v7.4s},     [%2], #64      \n"
        "ld1    {v8.4s, v9.4s, v10.4s, v11.4s},   [%2], #64      \n"
        "ld1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%2], #64      \n"
        "ld1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%2], #64      \n"

        // "prfm   pldl1keep, [%2, #128]     \n"
        
        "cmp    w4, #0                    \n"
        "beq    1f                        \n"

        "0:                               \n"

        "ld1    {v0.4s, v1.4s}, [%0], #32 \n"   // load packA, packA+4
        "ld1    {v2.4s, v3.4s}, [%1], #32 \n"   // load packB, packB+4

        "prfm   pldl1keep, [%0, x5]     \n"
        "fmla   v4.4s, v2.4s, v0.s[0]    \n"
        "fmla   v5.4s, v3.4s, v0.s[0]    \n"

        "fmla   v6.4s, v2.4s, v0.s[1]    \n"
        "fmla   v7.4s, v3.4s, v0.s[1]    \n"

        "fmla   v8.4s, v2.4s, v0.s[2]    \n"
        "fmla   v9.4s, v3.4s, v0.s[2]    \n"
        "prfm   pldl1keep, [%1, x6]     \n"

        "fmla   v10.4s, v2.4s, v0.s[3]    \n"
        "fmla   v11.4s, v3.4s, v0.s[3]    \n"

        "fmla   v12.4s, v2.4s, v1.s[0]    \n"
        "fmla   v13.4s, v3.4s, v1.s[0]    \n"
        "prfm   pldl1keep, [%2, x7]     \n"

        "fmla   v14.4s, v2.4s, v1.s[1]    \n"
        "fmla   v15.4s, v3.4s, v1.s[1]    \n"

        "fmla   v16.4s, v2.4s, v1.s[2]    \n"
        "fmla   v17.4s, v3.4s, v1.s[2]    \n"

        "fmla   v18.4s, v2.4s, v1.s[3]    \n"
        "fmla   v19.4s, v3.4s, v1.s[3]    \n"
        
        
        "subs   w4, w4, #1                \n"
        "bne    0b                        \n"

        "1:                               \n"

        "st1    {v4.4s, v5.4s, v6.4s, v7.4s},     [%3], #64      \n"
        "st1    {v8.4s, v9.4s, v10.4s, v11.4s},   [%3], #64      \n"
        "st1    {v12.4s, v13.4s, v14.4s, v15.4s}, [%3], #64     \n"
        "st1    {v16.4s, v17.4s, v18.4s, v19.4s}, [%3], #64      \n"

        : "=r"(packAPtr), // %0
        "=r"(packBPtr),   // %1
        "=r"(packCPtr),   // %2
        "=r"(packCPtr0)   // %3 
        : "0"(packAPtr),
        "1"(packBPtr),
        "2"(packCPtr),
        "3"(packCPtr0),
        "r"(kc_adjust), // %8
        "r"(ldc),       // %9
        "r"(PREFETCH_A),// %10
        "r"(PREFETCH_B),// %11
        "r"(PREFETCH_C) // %12
        : "cc", "memory", "x4", "x5", "x6", "x7", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20");

#else

    float32x4_t a0, a1;
    float32x4_t b0, b1;
    float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;
    float32x4_t c8, c9, c10, c11, c12, c13, c14, c15;

    c0   = vld1q_f32(packC);
    c1   = vld1q_f32(packC + 4);
    c2   = vld1q_f32(packC + 8);
    c3   = vld1q_f32(packC + 12);
    c4   = vld1q_f32(packC + 16);
    c5   = vld1q_f32(packC + 20);
    c6   = vld1q_f32(packC + 24);
    c7   = vld1q_f32(packC + 28);
    c8   = vld1q_f32(packC + 32);
    c9   = vld1q_f32(packC + 36);
    c10  = vld1q_f32(packC + 40);
    c11  = vld1q_f32(packC + 44);
    c12  = vld1q_f32(packC + 48);
    c13  = vld1q_f32(packC + 52);
    c14  = vld1q_f32(packC + 56);
    c15  = vld1q_f32(packC + 60);

    // A中packing之后的8行与B中packing之后的8列进行计算
    // 计算C的8x8分块
    for (int k = 0; k < kc_adjust; k++)
    {
        a0 = vld1q_f32(packAPtr);
        a1 = vld1q_f32(packAPtr + 4);
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);

        c0 = vfmaq_laneq_f32(c0, b0, a0, 0);
        c1 = vfmaq_laneq_f32(c1, b1, a0, 0);
        
        c2 = vfmaq_laneq_f32(c2, b0, a0, 1);
        c3 = vfmaq_laneq_f32(c3, b1, a0, 1);

        c4 = vfmaq_laneq_f32(c4, b0, a0, 2);
        c5 = vfmaq_laneq_f32(c5, b1, a0, 2);

        c6 = vfmaq_laneq_f32(c6, b0, a0, 3);
        c7 = vfmaq_laneq_f32(c7, b1, a0, 3);

        c8 = vfmaq_laneq_f32(c8, b0, a1, 0);
        c9 = vfmaq_laneq_f32(c9, b1, a1, 0);

        c10 = vfmaq_laneq_f32(c10, b0, a1, 1);
        c11 = vfmaq_laneq_f32(c11, b1, a1, 1);

        c12 = vfmaq_laneq_f32(c12, b0, a1, 2);
        c13 = vfmaq_laneq_f32(c13, b1, a1, 2);

        c14 = vfmaq_laneq_f32(c14, b0, a1, 3);
        c15 = vfmaq_laneq_f32(c15, b1, a1, 3);

        packAPtr += 8;
        packBPtr += 8;
    }

    // 将计算结果写回到C的8x8分块中
    // packCPtr = packC;

    vst1q_f32(packC     , c0);
    vst1q_f32(packC +  4, c1);
    vst1q_f32(packC +  8, c2);
    vst1q_f32(packC + 12, c3);
    vst1q_f32(packC + 16, c4);
    vst1q_f32(packC + 20, c5);
    vst1q_f32(packC + 24, c6);
    vst1q_f32(packC + 28, c7);
    vst1q_f32(packC + 32, c8);
    vst1q_f32(packC + 36, c9);
    vst1q_f32(packC + 40, c10);
    vst1q_f32(packC + 44, c11);
    vst1q_f32(packC + 48, c12);
    vst1q_f32(packC + 52, c13);
    vst1q_f32(packC + 56, c14);
    vst1q_f32(packC + 60, c15);

#endif  // __ARM_ASM
}

void kernel_4x16_packC(int kc_adjust, float *packA, float* packB, float *packC, int ldc, const int PREFETCH_A, const int PREFETCH_B, const int PREFETCH_C) {
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;
    float* packCPtr0 = packC;

#ifdef __ARM_ASM
    asm volatile(
        "mov    x4, %8                     \n"
        "mov    x5, %10                     \n"
        "mov    x6, %11                     \n"
        "mov    x7, %12                     \n"

        "ld1    {v5.4s,  v6.4s,  v7.4s,  v8.4s},  [%2], #64      \n"
        "ld1    {v9.4s,  v10.4s, v11.4s, v12.4s}, [%2], #64      \n"
        "ld1    {v13.4s, v14.4s, v15.4s, v16.4s}, [%2], #64     \n"
        "ld1    {v17.4s, v18.4s, v19.4s, v20.4s}, [%2], #64      \n"

        // "prfm   pldl1keep, [%2, #128]     \n"
        
        "cmp    w4, #0                    \n"
        "beq    1f                        \n"

        "0:                               \n"

        "ld1    {v0.4s}, [%0], #16 \n"   // load packA, packA+4
        "ld1    {v1.4s, v2.4s, v3.4s, v4.4s}, [%1], #64 \n"   // load packB, packB+4

        // "prfm   pldl1keep, [%0, #128]     \n"
        "prfm   pldl1keep, [%0, x5]     \n"
        "fmla   v5.4s, v1.4s, v0.s[0]    \n"
        "fmla   v6.4s, v2.4s, v0.s[0]    \n"
        "fmla   v7.4s, v3.4s, v0.s[0]    \n"
        "fmla   v8.4s, v4.4s, v0.s[0]    \n"
        
        "fmla   v9.4s,  v1.4s, v0.s[1]    \n"
        "fmla   v10.4s, v2.4s, v0.s[1]    \n"
        "prfm   pldl1keep, [%1, x6]       \n"
        "fmla   v11.4s, v3.4s, v0.s[1]    \n"
        "fmla   v12.4s, v4.4s, v0.s[1]    \n"
        
        "fmla   v13.4s, v1.4s, v0.s[2]    \n"
        "fmla   v14.4s, v2.4s, v0.s[2]    \n"
        "prfm   pldl1keep, [%2, x7]       \n"
        "fmla   v15.4s, v3.4s, v0.s[2]    \n"
        "fmla   v16.4s, v4.4s, v0.s[2]    \n"

        "fmla   v17.4s, v1.4s, v0.s[3]    \n"
        "fmla   v18.4s, v2.4s, v0.s[3]    \n"
        "fmla   v19.4s, v3.4s, v0.s[3]    \n"
        "fmla   v20.4s, v4.4s, v0.s[3]    \n"
        
        "subs   w4, w4, #1                \n"
        "bne    0b                        \n"

        "1:                               \n"

        "st1    {v5.4s,  v6.4s,  v7.4s,  v8.4s},  [%3], #64      \n"
        "st1    {v9.4s,  v10.4s, v11.4s, v12.4s}, [%3], #64      \n"
        "st1    {v13.4s, v14.4s, v15.4s, v16.4s}, [%3], #64     \n"
        "st1    {v17.4s, v18.4s, v19.4s, v20.4s}, [%3], #64      \n"

        : "=r"(packAPtr), // %0
        "=r"(packBPtr),   // %1
        "=r"(packCPtr),   // %2
        "=r"(packCPtr0)   // %3 
        : "0"(packAPtr),
        "1"(packBPtr),
        "2"(packCPtr),
        "3"(packCPtr0),
        "r"(kc_adjust), // %8
        "r"(ldc),       // %9
        "r"(PREFETCH_A),// %10
        "r"(PREFETCH_B),// %11
        "r"(PREFETCH_C) // %12
        : "cc", "memory", "x4", "x5", "x6", "x7", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10",
                          "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", 
                          "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28");

#else
    float32x4_t a0;
    float32x4_t b0, b1, b2, b3;
    float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;
    float32x4_t c8, c9, c10, c11, c12, c13, c14, c15;

    c0   = vld1q_f32(packC);
    c1   = vld1q_f32(packC + 4);
    c2   = vld1q_f32(packC + 8);
    c3   = vld1q_f32(packC + 12);
    c4   = vld1q_f32(packC + 16);
    c5   = vld1q_f32(packC + 20);
    c6   = vld1q_f32(packC + 24);
    c7   = vld1q_f32(packC + 28);
    c8   = vld1q_f32(packC + 32);
    c9   = vld1q_f32(packC + 36);
    c10  = vld1q_f32(packC + 40);
    c11  = vld1q_f32(packC + 44);
    c12  = vld1q_f32(packC + 48);
    c13  = vld1q_f32(packC + 52);
    c14  = vld1q_f32(packC + 56);
    c15  = vld1q_f32(packC + 60);

    // 计算
    for (int k = 0; k < kc_adjust; k++) {
        a0 = vld1q_f32(packAPtr);
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        b2 = vld1q_f32(packBPtr + 8);
        b3 = vld1q_f32(packBPtr + 12);

        c0  = vfmaq_laneq_f32(c0, b0, a0, 0);
        c1  = vfmaq_laneq_f32(c1, b1, a0, 0);
        c2  = vfmaq_laneq_f32(c2, b2, a0, 0);
        c3  = vfmaq_laneq_f32(c3, b3, a0, 0);

        c4  = vfmaq_laneq_f32(c4, b0, a0, 1);
        c5  = vfmaq_laneq_f32(c5, b1, a0, 1);
        c6  = vfmaq_laneq_f32(c6, b2, a0, 1);
        c7  = vfmaq_laneq_f32(c7, b3, a0, 1);

        c8  = vfmaq_laneq_f32(c8, b0, a0, 2);
        c9  = vfmaq_laneq_f32(c9, b1, a0, 2);
        c10 = vfmaq_laneq_f32(c10, b2, a0, 2);
        c11 = vfmaq_laneq_f32(c11, b3, a0, 2);

        c12 = vfmaq_laneq_f32(c12, b0, a0, 3);
        c13 = vfmaq_laneq_f32(c13, b1, a0, 3);
        c14 = vfmaq_laneq_f32(c14, b2, a0, 3);
        c15 = vfmaq_laneq_f32(c15, b3, a0, 3);

        packAPtr += 4;
        packBPtr += 16;
    }

    // 将结果写回
    vst1q_f32(packC     , c0);
    vst1q_f32(packC +  4, c1);
    vst1q_f32(packC +  8, c2);
    vst1q_f32(packC + 12, c3);
    vst1q_f32(packC + 16, c4);
    vst1q_f32(packC + 20, c5);
    vst1q_f32(packC + 24, c6);
    vst1q_f32(packC + 28, c7);
    vst1q_f32(packC + 32, c8);
    vst1q_f32(packC + 36, c9);
    vst1q_f32(packC + 40, c10);
    vst1q_f32(packC + 44, c11);
    vst1q_f32(packC + 48, c12);
    vst1q_f32(packC + 52, c13);
    vst1q_f32(packC + 56, c14);
    vst1q_f32(packC + 60, c15);
#endif
}

void kernel_8x12_packC(int kc_adjust, float *packA, float* packB, float *packC, int ldc, const int PREFETCH_A, const int PREFETCH_B, const int PREFETCH_C) {
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;
    float* packCPtr0 = packC;

#ifdef __ARM_ASM
    asm volatile(
        "mov    x4, %8                     \n"
        "mov    x5, %10                    \n"
        "mov    x6, %11                    \n"
        "mov    x7, %12                    \n"

        "ld1    {v5.4s,  v6.4s,  v7.4s,  v8.4s},  [%2], #64      \n"
        "ld1    {v9.4s,  v10.4s, v11.4s, v12.4s}, [%2], #64      \n"
        "ld1    {v13.4s, v14.4s, v15.4s, v16.4s}, [%2], #64      \n"
        "ld1    {v17.4s, v18.4s, v19.4s, v20.4s}, [%2], #64      \n"
        "ld1    {v21.4s, v22.4s, v23.4s, v24.4s}, [%2], #64      \n"
        "ld1    {v25.4s, v26.4s, v27.4s, v28.4s}, [%2], #64      \n"

        // "prfm   pldl1keep, [%2, #128]     \n"
        
        "cmp    w4, #0                    \n"
        "beq    1f                        \n"

        "0:                               \n"

        "ld1    {v0.4s, v1.4s}, [%0], #32 \n"   // load packA, packA+4
        "ld1    {v2.4s, v3.4s, v4.4s}, [%1], #48 \n"   // load packB, packB+4

        "prfm   pldl1keep, [%0, x5]     \n"
        // "prfm   pldl1keep, [%0, #128]     \n"

        "fmla   v5.4s, v2.4s, v0.s[0]    \n"
        "fmla   v6.4s, v3.4s, v0.s[0]    \n"
        "fmla   v7.4s, v4.4s, v0.s[0]    \n"
        
        "fmla   v8.4s,  v2.4s, v0.s[1]    \n"
        "fmla   v9.4s,  v3.4s, v0.s[1]    \n"
        "fmla   v10.4s, v4.4s, v0.s[1]    \n"
        
        "prfm   pldl1keep, [%1, x6]     \n"
        "fmla   v11.4s, v2.4s, v0.s[2]    \n"
        "fmla   v12.4s, v3.4s, v0.s[2]    \n"
        "fmla   v13.4s, v4.4s, v0.s[2]    \n"
        
        "fmla   v14.4s, v2.4s, v0.s[3]    \n"
        "fmla   v15.4s, v3.4s, v0.s[3]    \n"
        "fmla   v16.4s, v4.4s, v0.s[3]    \n"
        
        "fmla   v17.4s, v2.4s, v1.s[0]    \n"
        "fmla   v18.4s, v3.4s, v1.s[0]    \n"
        "prfm   pldl1keep, [%2, x7]     \n"
        "fmla   v19.4s, v4.4s, v1.s[0]    \n"
        
        "fmla   v20.4s, v2.4s, v1.s[1]    \n"
        "fmla   v21.4s, v3.4s, v1.s[1]    \n"
        "fmla   v22.4s, v4.4s, v1.s[1]    \n"
        
        "fmla   v23.4s, v2.4s, v1.s[2]    \n"
        "fmla   v24.4s, v3.4s, v1.s[2]    \n"
        "fmla   v25.4s, v4.4s, v1.s[2]    \n"

        "fmla   v26.4s, v2.4s, v1.s[3]    \n"
        "fmla   v27.4s, v3.4s, v1.s[3]    \n"
        "fmla   v28.4s, v4.4s, v1.s[3]    \n"
        
        "subs   w4, w4, #1                \n"
        "bne    0b                        \n"

        "1:                               \n"

        "st1    {v5.4s,  v6.4s,  v7.4s,  v8.4s},  [%3], #64      \n"
        "st1    {v9.4s,  v10.4s, v11.4s, v12.4s}, [%3], #64      \n"
        "st1    {v13.4s, v14.4s, v15.4s, v16.4s}, [%3], #64      \n"
        "st1    {v17.4s, v18.4s, v19.4s, v20.4s}, [%3], #64      \n"
        "st1    {v21.4s, v22.4s, v23.4s, v24.4s}, [%3], #64      \n"
        "st1    {v25.4s, v26.4s, v27.4s, v28.4s}, [%3], #64      \n"

        : "=r"(packAPtr), // %0
        "=r"(packBPtr),   // %1
        "=r"(packCPtr),   // %2
        "=r"(packCPtr0)   // %3 
        : "0"(packAPtr),
        "1"(packBPtr),
        "2"(packCPtr),
        "3"(packCPtr0),
        "r"(kc_adjust), // %8
        "r"(ldc),       // %9
        "r"(PREFETCH_A),// %10
        "r"(PREFETCH_B),// %11
        "r"(PREFETCH_C) // %12
        : "cc", "memory", "x4", "x5", "x6", "x7", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20");

#else
    float32x4_t a0, a1;
    float32x4_t b0, b1, b2;
    float32x4_t c0, c1, c2, c3, c4, c5, c6, c7;
    float32x4_t c8, c9, c10, c11, c12, c13, c14, c15;
    float32x4_t c16, c17, c18, c19, c20, c21, c22, c23;

    // 加载c的 8 x 12 block
    c0   = vld1q_f32(packC);
    c1   = vld1q_f32(packC + 4);
    c2   = vld1q_f32(packC + 8);
    c3   = vld1q_f32(packC + 12);
    c4   = vld1q_f32(packC + 16);
    c5   = vld1q_f32(packC + 20);
    c6   = vld1q_f32(packC + 24);
    c7   = vld1q_f32(packC + 28);
    c8   = vld1q_f32(packC + 32);
    c9   = vld1q_f32(packC + 36);
    c10  = vld1q_f32(packC + 40);
    c11  = vld1q_f32(packC + 44);
    c12  = vld1q_f32(packC + 48);
    c13  = vld1q_f32(packC + 52);
    c14  = vld1q_f32(packC + 56);
    c15  = vld1q_f32(packC + 60);
    c16  = vld1q_f32(packC + 64);
    c17  = vld1q_f32(packC + 68);
    c18  = vld1q_f32(packC + 72);
    c19  = vld1q_f32(packC + 76);
    c20  = vld1q_f32(packC + 80);
    c21  = vld1q_f32(packC + 84);
    c22  = vld1q_f32(packC + 88);
    c23  = vld1q_f32(packC + 92);

    // 计算
    for (int k = 0; k < kc_adjust; k++) {
        a0 = vld1q_f32(packAPtr);
        a1 = vld1q_f32(packAPtr + 4);
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        b2 = vld1q_f32(packBPtr + 8);

        c0 = vfmaq_laneq_f32(c0, b0, a0, 0);
        c1 = vfmaq_laneq_f32(c1, b1, a0, 0);
        c2 = vfmaq_laneq_f32(c2, b2, a0, 0);

        c3 = vfmaq_laneq_f32(c3, b0, a0, 1);
        c4 = vfmaq_laneq_f32(c4, b1, a0, 1);
        c5 = vfmaq_laneq_f32(c5, b2, a0, 1);

        c6 = vfmaq_laneq_f32(c6, b0, a0, 2);
        c7 = vfmaq_laneq_f32(c7, b1, a0, 2);
        c8 = vfmaq_laneq_f32(c8, b2, a0, 2);

        c9 = vfmaq_laneq_f32(c9, b0, a0, 3);
        c10 = vfmaq_laneq_f32(c10, b1, a0, 3);
        c11 = vfmaq_laneq_f32(c11, b2, a0, 3);
 
        c12 = vfmaq_laneq_f32(c12, b0, a1, 0);
        c13 = vfmaq_laneq_f32(c13, b1, a1, 0);
        c14 = vfmaq_laneq_f32(c14, b2, a1, 0);

        c15 = vfmaq_laneq_f32(c15, b0, a1, 1);
        c16 = vfmaq_laneq_f32(c16, b1, a1, 1);
        c17 = vfmaq_laneq_f32(c17, b2, a1, 1);

        c18 = vfmaq_laneq_f32(c18, b0, a1, 2);
        c19 = vfmaq_laneq_f32(c19, b1, a1, 2);
        c20 = vfmaq_laneq_f32(c20, b2, a1, 2);

        c21 = vfmaq_laneq_f32(c21, b0, a1, 3);
        c22 = vfmaq_laneq_f32(c22, b1, a1, 3);
        c23 = vfmaq_laneq_f32(c23, b2, a1, 3);

        packAPtr += 8;
        packBPtr += 12;
    }

    // 将c写回内存
    vst1q_f32(packC     , c0);
    vst1q_f32(packC +  4, c1);
    vst1q_f32(packC +  8, c2);
    vst1q_f32(packC + 12, c3);
    vst1q_f32(packC + 16, c4);
    vst1q_f32(packC + 20, c5);
    vst1q_f32(packC + 24, c6);
    vst1q_f32(packC + 28, c7);
    vst1q_f32(packC + 32, c8);
    vst1q_f32(packC + 36, c9);
    vst1q_f32(packC + 40, c10);
    vst1q_f32(packC + 44, c11);
    vst1q_f32(packC + 48, c12);
    vst1q_f32(packC + 52, c13);
    vst1q_f32(packC + 56, c14);
    vst1q_f32(packC + 60, c15);
    vst1q_f32(packC + 64, c16);
    vst1q_f32(packC + 68, c17);
    vst1q_f32(packC + 72, c18);
    vst1q_f32(packC + 76, c19);
    vst1q_f32(packC + 80, c20);
    vst1q_f32(packC + 84, c21);
    vst1q_f32(packC + 88, c22);
    vst1q_f32(packC + 92, c23);
#endif
}

template <int N>
void kernel_Nx8_for_8x8_template(int kc_adjust, float *packA, float *packB, float *packC, int ldc)
{
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* cPtr = packC;

    float32x4_t a0, a1, a2, a3, a4, a5, a6, a7;
    float32x4_t b0, b1;
    float32x4_t c00, c01, c10, c11, c20, c21, c30, c31;
    float32x4_t c40, c41, c50, c51, c60, c61, c70, c71;

    // 加载C的Nx8分块
    if (N > 0)
    {
        c00 = vld1q_f32(cPtr);
        c01 = vld1q_f32(cPtr + 4);
        cPtr += ldc;
    }
    if (N > 1)
    {
        c10 = vld1q_f32(cPtr);
        c11 = vld1q_f32(cPtr + 4);
        cPtr += ldc;
    }
    if (N > 2)
    {
        c20 = vld1q_f32(cPtr);
        c21 = vld1q_f32(cPtr + 4);
        cPtr += ldc; 
    }
    if (N > 3)
    {
        c30 = vld1q_f32(cPtr);
        c31 = vld1q_f32(cPtr + 4);
        cPtr += ldc; 
    }

    if (N > 4)
    {
        c40 = vld1q_f32(cPtr);
        c41 = vld1q_f32(cPtr + 4);
        cPtr += ldc; 
    }
    if (N > 5)
    {
        c50 = vld1q_f32(cPtr);
        c51 = vld1q_f32(cPtr + 4);
        cPtr += ldc; 
    }
    if (N > 6)
    {
        c60 = vld1q_f32(cPtr);
        c61 = vld1q_f32(cPtr + 4);
        cPtr += ldc; 
    }
    if (N > 7)
    {
        c70 = vld1q_f32(cPtr);
        c71 = vld1q_f32(cPtr + 4);
        cPtr += ldc; 
    }

    // 计算Nx8的分块
    for (int p = 0; p < kc_adjust; p++)
    {
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        if (N > 0)
        {
            a0 = vdupq_n_f32(packAPtr[0]);
            c00 = vfmaq_f32(c00, a0, b0);
            c01 = vfmaq_f32(c01, a0, b1);
        }
        if (N > 1)
        {
            a1 = vdupq_n_f32(packAPtr[1]);
            c10 = vfmaq_f32(c10, a1, b0);
            c11 = vfmaq_f32(c11, a1, b1);
        }
        if (N > 2)
        {
            a2 = vdupq_n_f32(packAPtr[2]);
            c20 = vfmaq_f32(c20, a2, b0);
            c21 = vfmaq_f32(c21, a2, b1);
        }
        if (N > 3)
        {
            a3 = vdupq_n_f32(packAPtr[3]);
            c30 = vfmaq_f32(c30, a3, b0);
            c31 = vfmaq_f32(c31, a3, b1);
        }
        if (N > 4)
        {
            a4 = vdupq_n_f32(packAPtr[4]);
            c40 = vfmaq_f32(c40, a4, b0);
            c41 = vfmaq_f32(c41, a4, b1);
        }
        if (N > 5)
        {
            a5 = vdupq_n_f32(packAPtr[5]);
            c50 = vfmaq_f32(c50, a5, b0);
            c51 = vfmaq_f32(c51, a5, b1);
        }
        if (N > 6)
        {
            a6 = vdupq_n_f32(packAPtr[6]);
            c60 = vfmaq_f32(c60, a6, b0);
            c61 = vfmaq_f32(c61, a6, b1);
        }
        if (N > 7)
        {
            a7 = vdupq_n_f32(packAPtr[7]);
            c70 = vfmaq_f32(c70, a7, b0);
            c71 = vfmaq_f32(c71, a7, b1);
        }

        packBPtr += 8;
        packAPtr += N;
    }

    // 将结果写回
    cPtr = packC;
    if (N > 0)
    {
        vst1q_f32(cPtr    , c00);
        vst1q_f32(cPtr + 4, c01);
        cPtr += ldc;
    }
    if (N > 1)
    {
        vst1q_f32(cPtr    , c10);
        vst1q_f32(cPtr + 4, c11);
        cPtr += ldc;
    }
    if (N > 2)
    {
        vst1q_f32(cPtr    , c20);
        vst1q_f32(cPtr + 4, c21);
        cPtr += ldc;
    }
    if (N > 3)
    {
        vst1q_f32(cPtr    , c30);
        vst1q_f32(cPtr + 4, c31);
        cPtr += ldc;
    }
    if (N > 4)
    {
        vst1q_f32(cPtr    , c40);
        vst1q_f32(cPtr + 4, c41);
        cPtr += ldc;
    }
    if (N > 5)
    {
        vst1q_f32(cPtr    , c50);
        vst1q_f32(cPtr + 4, c51);
        cPtr += ldc;
    }
    if (N > 6)
    {
        vst1q_f32(cPtr    , c60);
        vst1q_f32(cPtr + 4, c61);
        cPtr += ldc;
    }
    if (N > 7)
    {
        vst1q_f32(cPtr    , c70);
        vst1q_f32(cPtr + 4, c71);
        cPtr += ldc;
    }
}

InnerKernel get_kernel_Nx8(int k) {
    if (k == 1)
        return kernel_Nx8_for_8x8_template<1>;
    else if (k == 2)
        return kernel_Nx8_for_8x8_template<2>;
    else if (k == 3)
        return kernel_Nx8_for_8x8_template<3>;
    else if (k == 4)
        return kernel_Nx8_for_8x8_template<4>;
    else if (k == 5)
        return kernel_Nx8_for_8x8_template<5>;
    else if (k == 6)
        return kernel_Nx8_for_8x8_template<6>;
    else if (k == 7)
        return kernel_Nx8_for_8x8_template<7>;
    return kernel_Nx8_for_8x8_template<8>;
}

template <int N>
void kernel_Nx8_for_8x8_packC_template(int kc_adjust, float *packA, float *packB, float *packC, int ldc) {
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;

    float32x4_t a0, a1, a2, a3, a4, a5, a6, a7;
    float32x4_t b0, b1;
    float32x4_t c00, c01, c10, c11, c20, c21, c30, c31;
    float32x4_t c40, c41, c50, c51, c60, c61, c70, c71;

    // 加载C的Nx8分块
    if (N > 0)
    {
        c00 = vld1q_f32(packCPtr);
        c01 = vld1q_f32(packCPtr + 4);
        packCPtr += 8;
    }
    if (N > 1)
    {
        c10 = vld1q_f32(packCPtr);
        c11 = vld1q_f32(packCPtr + 4);
        packCPtr += 8;
    }
    if (N > 2)
    {
        c20 = vld1q_f32(packCPtr);
        c21 = vld1q_f32(packCPtr + 4);
        packCPtr += 8; 
    }
    if (N > 3)
    {
        c30 = vld1q_f32(packCPtr);
        c31 = vld1q_f32(packCPtr + 4);
        packCPtr += 8; 
    }

    if (N > 4)
    {
        c40 = vld1q_f32(packCPtr);
        c41 = vld1q_f32(packCPtr + 4);
        packCPtr += 8; 
    }
    if (N > 5)
    {
        c50 = vld1q_f32(packCPtr);
        c51 = vld1q_f32(packCPtr + 4);
        packCPtr += 8; 
    }
    if (N > 6)
    {
        c60 = vld1q_f32(packCPtr);
        c61 = vld1q_f32(packCPtr + 4);
        packCPtr += 8; 
    }
    if (N > 7)
    {
        c70 = vld1q_f32(packCPtr);
        c71 = vld1q_f32(packCPtr + 4);
        packCPtr += 8; 
    }

    // 计算Nx8的分块
    for (int p = 0; p < kc_adjust; p++)
    {
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        if (N > 0)
        {
            a0 = vdupq_n_f32(packAPtr[0]);
            c00 = vfmaq_f32(c00, a0, b0);
            c01 = vfmaq_f32(c01, a0, b1);
        }
        if (N > 1)
        {
            a1 = vdupq_n_f32(packAPtr[1]);
            c10 = vfmaq_f32(c10, a1, b0);
            c11 = vfmaq_f32(c11, a1, b1);
        }
        if (N > 2)
        {
            a2 = vdupq_n_f32(packAPtr[2]);
            c20 = vfmaq_f32(c20, a2, b0);
            c21 = vfmaq_f32(c21, a2, b1);
        }
        if (N > 3)
        {
            a3 = vdupq_n_f32(packAPtr[3]);
            c30 = vfmaq_f32(c30, a3, b0);
            c31 = vfmaq_f32(c31, a3, b1);
        }
        if (N > 4)
        {
            a4 = vdupq_n_f32(packAPtr[4]);
            c40 = vfmaq_f32(c40, a4, b0);
            c41 = vfmaq_f32(c41, a4, b1);
        }
        if (N > 5)
        {
            a5 = vdupq_n_f32(packAPtr[5]);
            c50 = vfmaq_f32(c50, a5, b0);
            c51 = vfmaq_f32(c51, a5, b1);
        }
        if (N > 6)
        {
            a6 = vdupq_n_f32(packAPtr[6]);
            c60 = vfmaq_f32(c60, a6, b0);
            c61 = vfmaq_f32(c61, a6, b1);
        }
        if (N > 7)
        {
            a7 = vdupq_n_f32(packAPtr[7]);
            c70 = vfmaq_f32(c70, a7, b0);
            c71 = vfmaq_f32(c71, a7, b1);
        }

        packBPtr += 8;
        packAPtr += N;
    }

    // 将结果写回
    packCPtr = packC;
    if (N > 0)
    {
        vst1q_f32(packCPtr    , c00);
        vst1q_f32(packCPtr + 4, c01);
        packCPtr += 8;
    }
    if (N > 1)
    {
        vst1q_f32(packCPtr    , c10);
        vst1q_f32(packCPtr + 4, c11);
        packCPtr += 8;
    }
    if (N > 2)
    {
        vst1q_f32(packCPtr    , c20);
        vst1q_f32(packCPtr + 4, c21);
        packCPtr += 8;
    }
    if (N > 3)
    {
        vst1q_f32(packCPtr    , c30);
        vst1q_f32(packCPtr + 4, c31);
        packCPtr += 8;
    }
    if (N > 4)
    {
        vst1q_f32(packCPtr    , c40);
        vst1q_f32(packCPtr + 4, c41);
        packCPtr += 8;
    }
    if (N > 5)
    {
        vst1q_f32(packCPtr    , c50);
        vst1q_f32(packCPtr + 4, c51);
        packCPtr += 8;
    }
    if (N > 6)
    {
        vst1q_f32(packCPtr    , c60);
        vst1q_f32(packCPtr + 4, c61);
        packCPtr += 8;
    }
    if (N > 7)
    {
        vst1q_f32(packCPtr    , c70);
        vst1q_f32(packCPtr + 4, c71);
        packCPtr += 8;
    }

}

InnerKernel get_kernel_Nx8_packC(int k) {
    if (k == 1)
        return kernel_Nx8_for_8x8_packC_template<1>;
    else if (k == 2)
        return kernel_Nx8_for_8x8_packC_template<2>;
    else if (k == 3)
        return kernel_Nx8_for_8x8_packC_template<3>;
    else if (k == 4)
        return kernel_Nx8_for_8x8_packC_template<4>;
    else if (k == 5)
        return kernel_Nx8_for_8x8_packC_template<5>;
    else if (k == 6)
        return kernel_Nx8_for_8x8_packC_template<6>;
    else if (k == 7)
        return kernel_Nx8_for_8x8_packC_template<7>;
    return kernel_Nx8_for_8x8_packC_template<8>;
}

template <int N>
void kernel_Nx12_for_8x12_template(int kc_adjust, float *packA, float *packB, float *packC, int ldc)
{
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;

    float32x4_t a0, a1, a2, a3;
    float32x4_t b0, b1, b2;
    float32x4_t c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;
    float32x4_t c40, c41, c42, c50, c51, c52, c60, c61, c62, c70, c71, c72;

    // 加载C的Nx12分块
    if (N > 0)
    {
        c00 = vld1q_f32(packCPtr);
        c01 = vld1q_f32(packCPtr + 4);
        c02 = vld1q_f32(packCPtr + 8);
        packCPtr += ldc;
    }
    if (N > 1)
    {
        c10 = vld1q_f32(packCPtr);
        c11 = vld1q_f32(packCPtr + 4);
        c12 = vld1q_f32(packCPtr + 8);
        packCPtr += ldc;
    }
    if (N > 2)
    {
        c20 = vld1q_f32(packCPtr);
        c21 = vld1q_f32(packCPtr + 4);
        c22 = vld1q_f32(packCPtr + 8);
        packCPtr += ldc; 
    }
    if (N > 3)
    {
        c30 = vld1q_f32(packCPtr);
        c31 = vld1q_f32(packCPtr + 4);
        c32 = vld1q_f32(packCPtr + 8);
        packCPtr += ldc; 
    }

    if (N > 4)
    {
        c40 = vld1q_f32(packCPtr);
        c41 = vld1q_f32(packCPtr + 4);
        c42 = vld1q_f32(packCPtr + 8);
        packCPtr += ldc; 
    }
    if (N > 5)
    {
        c50 = vld1q_f32(packCPtr);
        c51 = vld1q_f32(packCPtr + 4);
        c52 = vld1q_f32(packCPtr + 8);
        packCPtr += ldc; 
    }
    if (N > 6)
    {
        c60 = vld1q_f32(packCPtr);
        c61 = vld1q_f32(packCPtr + 4);
        c62 = vld1q_f32(packCPtr + 8);
        packCPtr += ldc; 
    }
    if (N > 7)
    {
        c70 = vld1q_f32(packCPtr);
        c71 = vld1q_f32(packCPtr + 4);
        c72 = vld1q_f32(packCPtr + 8);
        packCPtr += ldc; 
    }

    // 计算Nx12的分块
    for (int p = 0; p < kc_adjust; p++)
    {
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        b2 = vld1q_f32(packBPtr + 8);
        if (N > 0)
        {
            a0 = vdupq_n_f32(packAPtr[0]);
            c00 = vfmaq_f32(c00, a0, b0);
            c01 = vfmaq_f32(c01, a0, b1);
            c02 = vfmaq_f32(c02, a0, b2);
        }
        if (N > 1)
        {
            a1 = vdupq_n_f32(packAPtr[1]);
            c10 = vfmaq_f32(c10, a1, b0);
            c11 = vfmaq_f32(c11, a1, b1);
            c12 = vfmaq_f32(c12, a1, b2);
        }
        if (N > 2)
        {
            a2 = vdupq_n_f32(packAPtr[2]);
            c20 = vfmaq_f32(c20, a2, b0);
            c21 = vfmaq_f32(c21, a2, b1);
            c22 = vfmaq_f32(c22, a2, b2);
        }
        if (N > 3)
        {
            a3 = vdupq_n_f32(packAPtr[3]);
            c30 = vfmaq_f32(c30, a3, b0);
            c31 = vfmaq_f32(c31, a3, b1);
            c32 = vfmaq_f32(c32, a3, b2);
        }
        if (N > 4)
        {
            a0 = vdupq_n_f32(packAPtr[4]);
            c40 = vfmaq_f32(c40, a0, b0);
            c41 = vfmaq_f32(c41, a0, b1);
            c42 = vfmaq_f32(c42, a0, b2);
        }
        if (N > 5)
        {
            a1 = vdupq_n_f32(packAPtr[5]);
            c50 = vfmaq_f32(c50, a1, b0);
            c51 = vfmaq_f32(c51, a1, b1);
            c52 = vfmaq_f32(c52, a1, b2);
        }
        if (N > 6)
        {
            a2 = vdupq_n_f32(packAPtr[6]);
            c60 = vfmaq_f32(c60, a2, b0);
            c61 = vfmaq_f32(c61, a2, b1);
            c62 = vfmaq_f32(c62, a2, b2);
        }
        if (N > 7)
        {
            a3 = vdupq_n_f32(packAPtr[7]);
            c70 = vfmaq_f32(c70, a3, b0);
            c71 = vfmaq_f32(c71, a3, b1);
            c72 = vfmaq_f32(c72, a3, b2);
        }

        packBPtr += 12;
        packAPtr += N;
    }

    // 将结果写回
    packCPtr = packC;
    if (N > 0)
    {
        vst1q_f32(packCPtr    , c00);
        vst1q_f32(packCPtr + 4, c01);
        vst1q_f32(packCPtr + 8, c02);
        packCPtr += ldc;
    }
    if (N > 1)
    {
        vst1q_f32(packCPtr    , c10);
        vst1q_f32(packCPtr + 4, c11);
        vst1q_f32(packCPtr + 8, c12);
        packCPtr += ldc;
    }
    if (N > 2)
    {
        vst1q_f32(packCPtr    , c20);
        vst1q_f32(packCPtr + 4, c21);
        vst1q_f32(packCPtr + 8, c22);
        packCPtr += ldc;
    }
    if (N > 3)
    {
        vst1q_f32(packCPtr    , c30);
        vst1q_f32(packCPtr + 4, c31);
        vst1q_f32(packCPtr + 8, c32);
        packCPtr += ldc;
    }
    if (N > 4)
    {
        vst1q_f32(packCPtr    , c40);
        vst1q_f32(packCPtr + 4, c41);
        vst1q_f32(packCPtr + 8, c42);
        packCPtr += ldc;
    }
    if (N > 5)
    {
        vst1q_f32(packCPtr    , c50);
        vst1q_f32(packCPtr + 4, c51);
        vst1q_f32(packCPtr + 8, c52);
        packCPtr += ldc;
    }
    if (N > 6)
    {
        vst1q_f32(packCPtr    , c60);
        vst1q_f32(packCPtr + 4, c61);
        vst1q_f32(packCPtr + 8, c62);
        packCPtr += ldc;
    }
    if (N > 7)
    {
        vst1q_f32(packCPtr    , c70);
        vst1q_f32(packCPtr + 4, c71);
        vst1q_f32(packCPtr + 8, c72);
        packCPtr += ldc;
    }
}

InnerKernel get_kernel_Nx12(int k) {
    if (k == 1)
        return kernel_Nx12_for_8x12_template<1>;
    else if (k == 2)
        return kernel_Nx12_for_8x12_template<2>;
    else if (k == 3)
        return kernel_Nx12_for_8x12_template<3>;
    else if (k == 4)
        return kernel_Nx12_for_8x12_template<4>;
    else if (k == 5)
        return kernel_Nx12_for_8x12_template<5>;
    else if (k == 6)
        return kernel_Nx12_for_8x12_template<6>;
    else if (k == 7)
        return kernel_Nx12_for_8x12_template<7>;
    return kernel_Nx12_for_8x12_template<8>;
}

template <int N>
void kernel_Nx12_for_8x12_packC_template(int kc_adjust, float *packA, float *packB, float *packC, int ldc)
{
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;

    float32x4_t a0, a1, a2, a3;
    float32x4_t b0, b1, b2;
    float32x4_t c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;
    float32x4_t c40, c41, c42, c50, c51, c52, c60, c61, c62, c70, c71, c72;

    // 加载C的Nx12分块
    if (N > 0)
    {
        c00 = vld1q_f32(packCPtr);
        c01 = vld1q_f32(packCPtr + 4);
        c02 = vld1q_f32(packCPtr + 8);
        packCPtr += 12;
    }
    if (N > 1)
    {
        c10 = vld1q_f32(packCPtr);
        c11 = vld1q_f32(packCPtr + 4);
        c12 = vld1q_f32(packCPtr + 8);
        packCPtr += 12;
    }
    if (N > 2)
    {
        c20 = vld1q_f32(packCPtr);
        c21 = vld1q_f32(packCPtr + 4);
        c22 = vld1q_f32(packCPtr + 8);
        packCPtr += 12; 
    }
    if (N > 3)
    {
        c30 = vld1q_f32(packCPtr);
        c31 = vld1q_f32(packCPtr + 4);
        c32 = vld1q_f32(packCPtr + 8);
        packCPtr += 12; 
    }

    if (N > 4)
    {
        c40 = vld1q_f32(packCPtr);
        c41 = vld1q_f32(packCPtr + 4);
        c42 = vld1q_f32(packCPtr + 8);
        packCPtr += 12; 
    }
    if (N > 5)
    {
        c50 = vld1q_f32(packCPtr);
        c51 = vld1q_f32(packCPtr + 4);
        c52 = vld1q_f32(packCPtr + 8);
        packCPtr += 12; 
    }
    if (N > 6)
    {
        c60 = vld1q_f32(packCPtr);
        c61 = vld1q_f32(packCPtr + 4);
        c62 = vld1q_f32(packCPtr + 8);
        packCPtr += 12; 
    }
    if (N > 7)
    {
        c70 = vld1q_f32(packCPtr);
        c71 = vld1q_f32(packCPtr + 4);
        c72 = vld1q_f32(packCPtr + 8);
        packCPtr += 12; 
    }

    // 计算Nx12的分块
    for (int p = 0; p < kc_adjust; p++)
    {
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        b2 = vld1q_f32(packBPtr + 8);
        if (N > 0)
        {
            a0 = vdupq_n_f32(packAPtr[0]);
            c00 = vfmaq_f32(c00, a0, b0);
            c01 = vfmaq_f32(c01, a0, b1);
            c02 = vfmaq_f32(c02, a0, b2);
        }
        if (N > 1)
        {
            a1 = vdupq_n_f32(packAPtr[1]);
            c10 = vfmaq_f32(c10, a1, b0);
            c11 = vfmaq_f32(c11, a1, b1);
            c12 = vfmaq_f32(c12, a1, b2);
        }
        if (N > 2)
        {
            a2 = vdupq_n_f32(packAPtr[2]);
            c20 = vfmaq_f32(c20, a2, b0);
            c21 = vfmaq_f32(c21, a2, b1);
            c22 = vfmaq_f32(c22, a2, b2);
        }
        if (N > 3)
        {
            a3 = vdupq_n_f32(packAPtr[3]);
            c30 = vfmaq_f32(c30, a3, b0);
            c31 = vfmaq_f32(c31, a3, b1);
            c32 = vfmaq_f32(c32, a3, b2);
        }
        if (N > 4)
        {
            a0 = vdupq_n_f32(packAPtr[4]);
            c40 = vfmaq_f32(c40, a0, b0);
            c41 = vfmaq_f32(c41, a0, b1);
            c42 = vfmaq_f32(c42, a0, b2);
        }
        if (N > 5)
        {
            a1 = vdupq_n_f32(packAPtr[5]);
            c50 = vfmaq_f32(c50, a1, b0);
            c51 = vfmaq_f32(c51, a1, b1);
            c52 = vfmaq_f32(c52, a1, b2);
        }
        if (N > 6)
        {
            a2 = vdupq_n_f32(packAPtr[6]);
            c60 = vfmaq_f32(c60, a2, b0);
            c61 = vfmaq_f32(c61, a2, b1);
            c62 = vfmaq_f32(c62, a2, b2);
        }
        if (N > 7)
        {
            a3 = vdupq_n_f32(packAPtr[7]);
            c70 = vfmaq_f32(c70, a3, b0);
            c71 = vfmaq_f32(c71, a3, b1);
            c72 = vfmaq_f32(c72, a3, b2);
        }

        packBPtr += 12;
        packAPtr += N;
    }

    // 将结果写回
    packCPtr = packC;
    if (N > 0)
    {
        vst1q_f32(packCPtr    , c00);
        vst1q_f32(packCPtr + 4, c01);
        vst1q_f32(packCPtr + 8, c02);
        packCPtr += 12;
    }
    if (N > 1)
    {
        vst1q_f32(packCPtr    , c10);
        vst1q_f32(packCPtr + 4, c11);
        vst1q_f32(packCPtr + 8, c12);
        packCPtr += 12;
    }
    if (N > 2)
    {
        vst1q_f32(packCPtr    , c20);
        vst1q_f32(packCPtr + 4, c21);
        vst1q_f32(packCPtr + 8, c22);
        packCPtr += 12;
    }
    if (N > 3)
    {
        vst1q_f32(packCPtr    , c30);
        vst1q_f32(packCPtr + 4, c31);
        vst1q_f32(packCPtr + 8, c32);
        packCPtr += 12;
    }
    if (N > 4)
    {
        vst1q_f32(packCPtr    , c40);
        vst1q_f32(packCPtr + 4, c41);
        vst1q_f32(packCPtr + 8, c42);
        packCPtr += 12;
    }
    if (N > 5)
    {
        vst1q_f32(packCPtr    , c50);
        vst1q_f32(packCPtr + 4, c51);
        vst1q_f32(packCPtr + 8, c52);
        packCPtr += 12;
    }
    if (N > 6)
    {
        vst1q_f32(packCPtr    , c60);
        vst1q_f32(packCPtr + 4, c61);
        vst1q_f32(packCPtr + 8, c62);
        packCPtr += 12;
    }
    if (N > 7)
    {
        vst1q_f32(packCPtr    , c70);
        vst1q_f32(packCPtr + 4, c71);
        vst1q_f32(packCPtr + 8, c72);
        packCPtr += 12;
    }
}

InnerKernel get_kernel_Nx12_packC(int k) {
        if (k == 1)
        return kernel_Nx12_for_8x12_packC_template<1>;
    else if (k == 2)
        return kernel_Nx12_for_8x12_packC_template<2>;
    else if (k == 3)
        return kernel_Nx12_for_8x12_packC_template<3>;
    else if (k == 4)
        return kernel_Nx12_for_8x12_packC_template<4>;
    else if (k == 5)
        return kernel_Nx12_for_8x12_packC_template<5>;
    else if (k == 6)
        return kernel_Nx12_for_8x12_packC_template<6>;
    else if (k == 7)
        return kernel_Nx12_for_8x12_packC_template<7>;
    return kernel_Nx12_for_8x12_packC_template<8>;
}

template <int N>
void kernel_Nx16_for_4x16_template(int kc_adjust, float *packA, float *packB, float *packC, int ldc)
{
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;

    float32x4_t a0, a1, a2, a3;
    float32x4_t b0, b1, b2, b3;
    float32x4_t c00, c01, c02, c03;
    float32x4_t c10, c11, c12, c13;
    float32x4_t c20, c21, c22, c23;
    float32x4_t c30, c31, c32, c33;

    // 加载C的Nx16分块
    if (N > 0)
    {
        c00 = vld1q_f32(packCPtr);
        c01 = vld1q_f32(packCPtr + 4);
        c02 = vld1q_f32(packCPtr + 8);
        c03 = vld1q_f32(packCPtr + 12);
        packCPtr += ldc;
    }
    if (N > 1)
    {
        c10 = vld1q_f32(packCPtr);
        c11 = vld1q_f32(packCPtr + 4);
        c12 = vld1q_f32(packCPtr + 8);
        c13 = vld1q_f32(packCPtr + 12);
        packCPtr += ldc;
    }
    if (N > 2)
    {
        c20 = vld1q_f32(packCPtr);
        c21 = vld1q_f32(packCPtr + 4);
        c22 = vld1q_f32(packCPtr + 8);
        c23 = vld1q_f32(packCPtr + 12);
        packCPtr += ldc; 
    }
    if (N > 3)
    {
        c30 = vld1q_f32(packCPtr);
        c31 = vld1q_f32(packCPtr + 4);
        c32 = vld1q_f32(packCPtr + 8);
        c33 = vld1q_f32(packCPtr + 12);
        packCPtr += ldc; 
    }

    // 计算Nx16的分块
    for (int p = 0; p < kc_adjust; p++)
    {
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        b2 = vld1q_f32(packBPtr + 8);
        b3 = vld1q_f32(packBPtr + 12);
        if (N > 0)
        {
            a0 = vdupq_n_f32(packAPtr[0]);
            c00 = vfmaq_f32(c00, a0, b0);
            c01 = vfmaq_f32(c01, a0, b1);
            c02 = vfmaq_f32(c02, a0, b2);
            c03 = vfmaq_f32(c03, a0, b3);
        }
        if (N > 1)
        {
            a1 = vdupq_n_f32(packAPtr[1]);
            c10 = vfmaq_f32(c10, a1, b0);
            c11 = vfmaq_f32(c11, a1, b1);
            c12 = vfmaq_f32(c12, a1, b2);
            c13 = vfmaq_f32(c13, a1, b3);
        }
        if (N > 2)
        {
            a2 = vdupq_n_f32(packAPtr[2]);
            c20 = vfmaq_f32(c20, a2, b0);
            c21 = vfmaq_f32(c21, a2, b1);
            c22 = vfmaq_f32(c22, a2, b2);
            c23 = vfmaq_f32(c23, a2, b3);
        }
        if (N > 3)
        {
            a3 = vdupq_n_f32(packAPtr[3]);
            c30 = vfmaq_f32(c30, a3, b0);
            c31 = vfmaq_f32(c31, a3, b1);
            c32 = vfmaq_f32(c32, a3, b2);
            c33 = vfmaq_f32(c33, a3, b3);
        }

        packBPtr += 16;
        packAPtr += N;
    }

    // 将结果写回
    packCPtr = packC;
    if (N > 0)
    {
        vst1q_f32(packCPtr    , c00);
        vst1q_f32(packCPtr + 4, c01);
        vst1q_f32(packCPtr + 8, c02);
        vst1q_f32(packCPtr + 12, c03);
        packCPtr += ldc;
    }
    if (N > 1)
    {
        vst1q_f32(packCPtr    , c10);
        vst1q_f32(packCPtr + 4, c11);
        vst1q_f32(packCPtr + 8, c12);
        vst1q_f32(packCPtr + 12, c13);
        packCPtr += ldc;
    }
    if (N > 2)
    {
        vst1q_f32(packCPtr    , c20);
        vst1q_f32(packCPtr + 4, c21);
        vst1q_f32(packCPtr + 8, c22);
        vst1q_f32(packCPtr + 12, c23);
        packCPtr += ldc;
    }
    if (N > 3)
    {
        vst1q_f32(packCPtr    , c30);
        vst1q_f32(packCPtr + 4, c31);
        vst1q_f32(packCPtr + 8, c32);
        vst1q_f32(packCPtr + 12, c33);
        packCPtr += ldc;
    }

}

InnerKernel get_kernel_Nx16(int k) {
    if (k == 1)
        return kernel_Nx16_for_4x16_template<1>;
    else if (k == 2)
        return kernel_Nx16_for_4x16_template<2>;
    else if (k == 3)
        return kernel_Nx16_for_4x16_template<3>;
    return kernel_Nx16_for_4x16_template<4>;
}

template <int N>
void kernel_Nx16_for_4x16_packC_template(int kc_adjust, float *packA, float *packB, float *packC, int ldc)
{
    float* packAPtr = packA;
    float* packBPtr = packB;
    float* packCPtr = packC;

    float32x4_t a0, a1, a2, a3;
    float32x4_t b0, b1, b2, b3;
    float32x4_t c00, c01, c02, c03;
    float32x4_t c10, c11, c12, c13;
    float32x4_t c20, c21, c22, c23;
    float32x4_t c30, c31, c32, c33;

    // 加载C的Nx16分块
    if (N > 0)
    {
        c00 = vld1q_f32(packCPtr);
        c01 = vld1q_f32(packCPtr + 4);
        c02 = vld1q_f32(packCPtr + 8);
        c03 = vld1q_f32(packCPtr + 12);
        packCPtr += 16;
    }
    if (N > 1)
    {
        c10 = vld1q_f32(packCPtr);
        c11 = vld1q_f32(packCPtr + 4);
        c12 = vld1q_f32(packCPtr + 8);
        c13 = vld1q_f32(packCPtr + 12);
        packCPtr += 16;
    }
    if (N > 2)
    {
        c20 = vld1q_f32(packCPtr);
        c21 = vld1q_f32(packCPtr + 4);
        c22 = vld1q_f32(packCPtr + 8);
        c23 = vld1q_f32(packCPtr + 12);
        packCPtr += 16; 
    }
    if (N > 3)
    {
        c30 = vld1q_f32(packCPtr);
        c31 = vld1q_f32(packCPtr + 4);
        c32 = vld1q_f32(packCPtr + 8);
        c33 = vld1q_f32(packCPtr + 12);
        packCPtr += 16; 
    }

    // 计算Nx16的分块
    for (int p = 0; p < kc_adjust; p++)
    {
        b0 = vld1q_f32(packBPtr);
        b1 = vld1q_f32(packBPtr + 4);
        b2 = vld1q_f32(packBPtr + 8);
        b3 = vld1q_f32(packBPtr + 12);
        if (N > 0)
        {
            a0 = vdupq_n_f32(packAPtr[0]);
            c00 = vfmaq_f32(c00, a0, b0);
            c01 = vfmaq_f32(c01, a0, b1);
            c02 = vfmaq_f32(c02, a0, b2);
            c03 = vfmaq_f32(c03, a0, b3);
        }
        if (N > 1)
        {
            a1 = vdupq_n_f32(packAPtr[1]);
            c10 = vfmaq_f32(c10, a1, b0);
            c11 = vfmaq_f32(c11, a1, b1);
            c12 = vfmaq_f32(c12, a1, b2);
            c13 = vfmaq_f32(c13, a1, b3);
        }
        if (N > 2)
        {
            a2 = vdupq_n_f32(packAPtr[2]);
            c20 = vfmaq_f32(c20, a2, b0);
            c21 = vfmaq_f32(c21, a2, b1);
            c22 = vfmaq_f32(c22, a2, b2);
            c23 = vfmaq_f32(c23, a2, b3);
        }
        if (N > 3)
        {
            a3 = vdupq_n_f32(packAPtr[3]);
            c30 = vfmaq_f32(c30, a3, b0);
            c31 = vfmaq_f32(c31, a3, b1);
            c32 = vfmaq_f32(c32, a3, b2);
            c33 = vfmaq_f32(c33, a3, b3);
        }

        packBPtr += 16;
        packAPtr += N;
    }

    // 将结果写回
    packCPtr = packC;
    if (N > 0)
    {
        vst1q_f32(packCPtr    , c00);
        vst1q_f32(packCPtr + 4, c01);
        vst1q_f32(packCPtr + 8, c02);
        vst1q_f32(packCPtr + 12, c03);
        packCPtr += 16;
    }
    if (N > 1)
    {
        vst1q_f32(packCPtr    , c10);
        vst1q_f32(packCPtr + 4, c11);
        vst1q_f32(packCPtr + 8, c12);
        vst1q_f32(packCPtr + 12, c13);
        packCPtr += 16;
    }
    if (N > 2)
    {
        vst1q_f32(packCPtr    , c20);
        vst1q_f32(packCPtr + 4, c21);
        vst1q_f32(packCPtr + 8, c22);
        vst1q_f32(packCPtr + 12, c23);
        packCPtr += 16;
    }
    if (N > 3)
    {
        vst1q_f32(packCPtr    , c30);
        vst1q_f32(packCPtr + 4, c31);
        vst1q_f32(packCPtr + 8, c32);
        vst1q_f32(packCPtr + 12, c33);
        packCPtr += 16;
    }

}

InnerKernel get_kernel_Nx16_packC(int k) {
    if (k == 1)
        return kernel_Nx16_for_4x16_packC_template<1>;
    else if (k == 2)
        return kernel_Nx16_for_4x16_packC_template<2>;
    else if (k == 3)
        return kernel_Nx16_for_4x16_packC_template<3>;
    return kernel_Nx16_for_4x16_packC_template<4>;
}
