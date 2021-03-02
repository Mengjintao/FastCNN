#include <stdlib.h>
#include <arm_neon.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include "../utility/helper.h"
#include "TensorGEMM.h"

void printTensor(const float* matrix, int row, int col)
{
    printf("--------------\n");
    for(int i=0; i<row; i++)
    {
        for(int j=0; j<col; j++)
            printf("%5.2f ", *(matrix+i*col+j));
        printf("\n");
    }
}





void retransformKernel2(float* kernel, int outputChannels, int inputChannels, int ocBlock, int icBlock, int ocRegBlock)
{
//    assert(outputChannels % ocRegBlock ==0);
    assert(ocBlock % ocRegBlock ==0);

    float *tmp = new float [ocBlock*inputChannels*9];
    /*
         printf("Begin\n");
         for(int i=0;i<outputChannels;i++)
         {
                 for(int j=0;j<inputChannels;j++)
         {
             for(int k=0;k<1;k++)        printf("%3.3f ", *(kernel+i*inputChannels*9 + j*9 + k));
         }
         printf("\n");
         }
     */
    for(int oc=0; oc<outputChannels; oc+=ocBlock)
    {
        int ocStart = oc;
        int ocEnd   = oc+ocBlock;
        if(ocEnd>outputChannels)        ocEnd = outputChannels;
        int ocDis   = ocEnd - ocStart;
        int pos = 0;
        for(int ic=0; ic<inputChannels; ic+=icBlock)
        {
            int icStart = ic;
            int icEnd   = ic + icBlock;
            if(icEnd>inputChannels)     icEnd = inputChannels;
            int icDis   = icEnd - icStart;
            int blockSize = ocDis * icDis * 9;
            for(int ocN=0; ocN<ocDis; ocN+=ocRegBlock)
            {
                int ocRStart = ocN;
                int ocREnd   = ocN + ocRegBlock;
                if(ocREnd>ocDis)        ocREnd = ocDis;
                int ocRDis   = ocREnd - ocRStart;
                for(int i=0; i<ocRDis; i++)
                    for(int icN=0; icN<icDis; icN++)
                    {
                        for(int k=0; k<9; k++)
                            tmp[pos + (ocN)*icDis*9+ icN*ocRDis*9 + i*9 +k] = kernel[(oc+ocN+i)*inputChannels*9 + (ic+icN)*9 + k];
                    }
            }
            pos += blockSize;
        }
//        memcpy(kernel + oc*inputChannels*9,  tmp,  ocBlock * inputChannels * 9 * sizeof(float));
        memcpy(kernel + oc*inputChannels*9, tmp, pos * sizeof(float));
    }
    /*
        printf("End\n");
        for(int i=0;i<outputChannels;i++)
        {
                for(int j=0;j<inputChannels;j++)
        {
                for(int k=0;k<1;k++)    printf("%3.3f ", *(kernel+i*inputChannels*9 + j*9 + k));
        }
        printf("\n");
        }
    */
    delete [] tmp;
    tmp = NULL;
}

void retransformKernel(float* kernel, int outputChannels, int inputChannels, int ocBlock)
{
    assert(outputChannels % ocBlock ==0);

    float *tmp = new float [ocBlock*inputChannels*9];

    for(int oc=0; oc<outputChannels; oc+=ocBlock)
    {
        for(int i=0; i<ocBlock; i++)
            for(int j=0; j<inputChannels; j++)
            {
                for(int k=0; k<9; k++)
                    tmp[j*ocBlock*9+i*9+k] = kernel[(oc+i)*inputChannels*9 + j*9 + k];
            }
        memcpy(kernel + oc*inputChannels*9,  tmp,  ocBlock * inputChannels * 9 * sizeof(float));
    }
}

/*
 * Input transform:
 *
 * First traverse all the output channels and then the input channels,
 * so that each input frame would be only transformed once.
 * The out buffer for UT x VT result would be heavily flushed.
 * We can do cache blocking against the output channels if necessary.
 */

/*
 * Reshape:
 * Dimensions: (nRowBlocks * nColBlocks * 16 / 16) x (Input channels * 16)
 * Illustrations:
 * INPUT CHANNEL IDS:   |   0   |   1   |   2   |   3   |...
 * 4x4 Data blocks:     |BLOCK 0|BLOCK 0|BLOCK 0|BLOCK 0|
 *                      |BLOCK 1|BLOCK 1|BLOCK 1|BLOCK 1|
 *                      |BLOCK 2|BLOCK 2|BLOCK 2|BLOCK 2|
 */

static inline void input_transform(
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

static inline void neon_transpose4x4_inplace_f32_cpp(
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



//testKernel[OC X IC X 9], testInput[IC X H X W], testResult[OC X (H-2) x (W-2)]
int winoF63_naive(float *baseResult, float *testInput, float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int padWidth, int padHeight, int num_threads)
{
    printf("winoF63 inputHW, padHW, channel: %d %d %d %d %d %d\n", inputHeight,inputWidth, padHeight, padWidth, inputChannels, outputChannels);
    int outputHeight = inputHeight+padHeight*2 -2; 
    int outputWidth  = inputWidth +padWidth *2 -2;
    int outputPage   = outputHeight * outputWidth;
    for(int oc=0; oc<outputChannels; oc++)
    {
        for(int tile=0; tile<outputHeight*outputWidth; tile++)
        {
            int startH = tile/(outputWidth);
            int startW = tile%(outputWidth);

            float sum = 0;
            for(int ic=0; ic<inputChannels; ic++)
            {
                //float *ip = testInput  + ic*(inputHeight*inputWidth) + startH * inputWidth + startW;
                float *ip = testInput  + ic*(inputHeight*inputWidth);
                float *kp = testKernel + oc*(inputChannels*9) + ic*9;

                float total = 0;
                for(int i=0; i<3; i++)	for(int j=0; j<3; j++)
                    {
			int x = startH+i - padHeight;
			int y = startW+j - padWidth;
			float value = 0;
			if(x>=0 && x<inputHeight && y>=0 && y<inputWidth)	value = *(ip + x*inputWidth + y);
			else value = 0;	
//                        float *tip = ip + i*(inputWidth) + j;
                        float *tkp = kp + i*3            + j;
//                        total += (*tip) * (*tkp);
                        total += value * (*tkp);
                    }
                sum += total;
            }
            *(baseResult + oc*outputPage + tile) = sum;
        }
    }

    return 1;
}

void extractMatrix_4tile_8x8(float *inputBlock, float *testInput, int ic, int tile, int inputHeight, int inputWidth)
{
    int tileH = (inputHeight-2)/6;
    int tileW = (inputWidth -2)/6;

    for(int i=0; i<4; i++)
    {
        int tileRow = (tile+i)/tileW;
        int tileCol = (tile+i)%tileW;
        float *pInput = testInput + ic*inputHeight*inputWidth + tileRow*inputWidth*6 + tileCol*6;
        for(int x=0; x<8; x++)
            memcpy(inputBlock+i*64+x*8, pInput+x*inputWidth, 8*sizeof(float));
    }
}

void inputTransform8x8(float *output, float *input)
{
    const float32x4_t f5    = vdupq_n_f32( 5.0f);
    const float32x4_t f4    = vdupq_n_f32( 4.0f);
    const float32x4_t f2    = vdupq_n_f32( 2.0f);
    const float32x4_t f2_5  = vdupq_n_f32( 2.5f);
    const float32x4_t f5_25 = vdupq_n_f32(5.25f);
    const float32x4_t f4_25 = vdupq_n_f32(4.25f);
    const float32x4_t f1_25 = vdupq_n_f32(1.25f);
    const float32x4_t f0_5  = vdupq_n_f32( 0.5f);
    const float32x4_t f0_25 = vdupq_n_f32( 0.25f);
    const float32x4_t vZero = vdupq_n_f32( 0.0f);

    float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    float32x4_t m1, m2, s1, s2, t1, t2;//Auxiliary registers
    float32x4_t d0, d1, d2, d3, d4, d5, d6, d7;


    float *ext = input;
    l0 = vld1q_f32(ext);
    r0 = vld1q_f32(ext + 4);
    l1 = vld1q_f32(ext + 8);
    r1 = vld1q_f32(ext + 12);
    l2 = vld1q_f32(ext + 16);
    r2 = vld1q_f32(ext + 20);
    l3 = vld1q_f32(ext + 24);
    r3 = vld1q_f32(ext + 28);
    l4 = vld1q_f32(ext + 32);
    r4 = vld1q_f32(ext + 36);
    l5 = vld1q_f32(ext + 40);
    r5 = vld1q_f32(ext + 44);
    l6 = vld1q_f32(ext + 48);
    r6 = vld1q_f32(ext + 52);
    l7 = vld1q_f32(ext + 56);
    r7 = vld1q_f32(ext + 60);

    input_transform(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                    t1, t2, s1, s2, m1, m2, //Auxiliary
                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
    /*
            input_transform(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                            t1, t2, s1, s2, m1, m2, //Auxiliary
                            f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

        float *outp = output;
            vst1q_f32(outp,       l0);
            vst1q_f32(outp + 4,   r0);
            vst1q_f32(outp + 8,   l1);
            vst1q_f32(outp + 12,  r1);
            vst1q_f32(outp + 16,  l2);
            vst1q_f32(outp + 20,  r2);
            vst1q_f32(outp + 24,  l3);
            vst1q_f32(outp + 28,  r3);

            vst1q_f32(outp + 32, l4);
            vst1q_f32(outp + 36, r4);
            vst1q_f32(outp + 40, l5);
            vst1q_f32(outp + 44, r5);
            vst1q_f32(outp + 48, l6);
            vst1q_f32(outp + 52, r6);
            vst1q_f32(outp + 56, l7);
            vst1q_f32(outp + 60, r7);
    */
    neon_transpose4x4_inplace_f32_cpp(l0, l1, l2, l3);
    neon_transpose4x4_inplace_f32_cpp(l4, l5, l6, l7);
    input_transform(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                    t1, t2, s1, s2, m1, m2, //Auxiliary
                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
    neon_transpose4x4_inplace_f32_cpp(r0, r1, r2, r3);
    neon_transpose4x4_inplace_f32_cpp(r4, r5, r6, r7);
    input_transform(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                    t1, t2, s1, s2, m1, m2, //Auxiliary
                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
    input_transform(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                    t1, t2, s1, s2, m1, m2, //Auxiliary
                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

    float *outp = output;
    vst1q_f32(outp,       l0);
    vst1q_f32(outp + 4,  l4);
    vst1q_f32(outp + 8,  l1);
    vst1q_f32(outp + 12,  l5);
    vst1q_f32(outp + 16,  l2);
    vst1q_f32(outp + 20,  l6);
    vst1q_f32(outp + 24,  l3);
    vst1q_f32(outp + 28, l7);

    vst1q_f32(outp + 32, r0);
    vst1q_f32(outp + 36, r4);
    vst1q_f32(outp + 40, r1);
    vst1q_f32(outp + 44, r5);
    vst1q_f32(outp + 48, r2);
    vst1q_f32(outp + 52, r6);
    vst1q_f32(outp + 56, r3);
    vst1q_f32(outp + 60, r7);
}


void inputTransform4x8x8(float *output, float* input)
{
    const float32x4_t f5    = vdupq_n_f32( 5.0f);
    const float32x4_t f4    = vdupq_n_f32( 4.0f);
    const float32x4_t f2    = vdupq_n_f32( 2.0f);
    const float32x4_t f2_5  = vdupq_n_f32( 2.5f);
    const float32x4_t f5_25 = vdupq_n_f32(5.25f);
    const float32x4_t f4_25 = vdupq_n_f32(4.25f);
    const float32x4_t f1_25 = vdupq_n_f32(1.25f);
    const float32x4_t f0_5  = vdupq_n_f32( 0.5f);
    const float32x4_t f0_25 = vdupq_n_f32( 0.25f);
    const float32x4_t vZero = vdupq_n_f32( 0.0f);

    float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    float32x4_t m1, m2, s1, s2, t1, t2;//Auxiliary registers

    for(int i=0; i<4; i++)
    {
        float *ext = input+i*64;
        l0 = vld1q_f32(ext    );
        r0 = vld1q_f32(ext + 4);
        l1 = vld1q_f32(ext + 8);
        r1 = vld1q_f32(ext + 12);
        l2 = vld1q_f32(ext + 16);
        r2 = vld1q_f32(ext + 20);
        l3 = vld1q_f32(ext + 24);
        r3 = vld1q_f32(ext + 28);
        l4 = vld1q_f32(ext + 32);
        r4 = vld1q_f32(ext + 36);
        l5 = vld1q_f32(ext + 40);
        r5 = vld1q_f32(ext + 44);
        l6 = vld1q_f32(ext + 48);
        r6 = vld1q_f32(ext + 52);
        l7 = vld1q_f32(ext + 56);
        r7 = vld1q_f32(ext + 60);

        input_transform(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                        t1, t2, s1, s2, m1, m2, //Auxiliary
                        f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
        neon_transpose4x4_inplace_f32_cpp(l0, l1, l2, l3);
        neon_transpose4x4_inplace_f32_cpp(l4, l5, l6, l7);
        input_transform(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                        t1, t2, s1, s2, m1, m2, //Auxiliary
                        f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
        neon_transpose4x4_inplace_f32_cpp(r0, r1, r2, r3);
        neon_transpose4x4_inplace_f32_cpp(r4, r5, r6, r7);
        input_transform(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                        t1, t2, s1, s2, m1, m2, //Auxiliary
                        f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
        input_transform(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                        t1, t2, s1, s2, m1, m2, //Auxiliary
                        f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

        float *outp = output + i*4;
        vst1q_f32(outp,       l0);
        vst1q_f32(outp + 16,  l4);
        vst1q_f32(outp + 32,  l1);
        vst1q_f32(outp + 48,  l5);

        vst1q_f32(outp + 64,  l2);
        vst1q_f32(outp + 80,  l6);
        vst1q_f32(outp + 96,  l3);
        vst1q_f32(outp + 112, l7);

        vst1q_f32(outp + 128, r0);
        vst1q_f32(outp + 144, r4);
        vst1q_f32(outp + 160, r1);
        vst1q_f32(outp + 176, r5);

        vst1q_f32(outp + 192, r2);
        vst1q_f32(outp + 208, r6);
        vst1q_f32(outp + 224, r3);
        vst1q_f32(outp + 240, r7);
    }
}

void naive_gemmtemp(int M, int N, int L, float *A, float *B, float *C)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] = 0.f;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int k = 0; k < L; k++)
            {
                C[i * N + j] += A[i * L + k] * B[k * N + j];
            }
        }
    }
}

void transposetemp(size_t m, size_t n, float *in, float *out) //  A[m][n] -> A[n][m]
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            out[j * m + i] = in[i * n + j];
}

void kernelTransform8x8(float *kernelBlock, float *kernel)
{
    float ktm[24] =
    {
        1.0f, 0.0f, 0.0f,
        -2.0f / 9, -2.0f / 9, -2.0f / 9,
        -2.0f / 9, 2.0f / 9, -2.0f / 9,
        1.0f / 90, 1.0f / 45, 2.0f / 45,
        1.0f / 90, -1.0f / 45, 2.0f / 45,
        1.0f / 45, 1.0f / 90, 1.0f / 180,
        1.0f / 45, -1.0f / 90, 1.0f / 180,
        0.0f, 0.0f, 1.0f
    };

    float midBlock[24];
    float outBlock[24];
    float bigBlock[64];
    float32x4_t w0, w1, w2, w3;

    //print_floats(kernel, 3, 3);
    naive_gemmtemp(8, 3, 3, ktm, kernel, midBlock);
    transposetemp(8, 3, midBlock, outBlock);
    naive_gemmtemp(8, 8, 3, ktm, outBlock, kernelBlock);

    /*    for (int i = 0; i < 16; ++i)
        {
            float32x4_t reg;
            reg = vld1q_f32(bigBlock + i * 4);
            vst1q_f32(kernelBlock + i * 16, reg);
        }
    */
}

void kernelTransform4x8x8(float *kernelBlock, float *kernel, int ldk)
{
    float ktm[24] =
    {
        1.0f, 0.0f, 0.0f,
        -2.0f / 9, -2.0f / 9, -2.0f / 9,
        -2.0f / 9, 2.0f / 9, -2.0f / 9,
        1.0f / 90, 1.0f / 45, 2.0f / 45,
        1.0f / 90, -1.0f / 45, 2.0f / 45,
        1.0f / 45, 1.0f / 90, 1.0f / 180,
        1.0f / 45, -1.0f / 90, 1.0f / 180,
        0.0f, 0.0f, 1.0f
    };

    float midBlock[24];
    float outBlock[24];
    float bigBlock[64];
    float32x4_t w0, w1, w2, w3;

    for(int i=0; i<4; i++)
    {
        //print_floats(kernel, 3, 3);
        naive_gemmtemp(8, 3, 3, ktm, kernel+i*ldk, midBlock);
        transposetemp( 8, 3, midBlock, outBlock);
        naive_gemmtemp(8, 8, 3, ktm, outBlock, bigBlock);


        /*        float *kp = kernelBlock+i*4;
                for(int j=0;j<16;j++)
                {
                    for(int k=0;k<4;k++)    *(kp+j*16+k) = bigBlock[j*4+k];
                }
        */
    }
}

static inline void winograd_f6k3_kernel_transform(
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

static inline void winograd_f6k3_output_transform_inplace(
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


void outputTransform4x8x8(float *out, float *output)
{
    float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    for(int i=0; i<4; i++)
    {
        float *wp = output+i*4;
        l0 = vld1q_f32(wp);
        r0 = vld1q_f32(wp + 16);
        l1 = vld1q_f32(wp + 32);
        r1 = vld1q_f32(wp + 48);
        l2 = vld1q_f32(wp + 64);
        r2 = vld1q_f32(wp + 80);
        l3 = vld1q_f32(wp + 96);
        r3 = vld1q_f32(wp + 112);
        l4 = vld1q_f32(wp + 128);
        r4 = vld1q_f32(wp + 144);
        l5 = vld1q_f32(wp + 160);
        r5 = vld1q_f32(wp + 176);
        l6 = vld1q_f32(wp + 192);
        r6 = vld1q_f32(wp + 208);
        l7 = vld1q_f32(wp + 224);
        r7 = vld1q_f32(wp + 240);

        winograd_f6k3_output_transform_inplace(l0, l1, l2, l3, l4, l5, l6, l7);
        winograd_f6k3_output_transform_inplace(r0, r1, r2, r3, r4, r5, r6, r7);
        neon_transpose4x4_inplace_f32_cpp(l0, l1, l2, l3);
        neon_transpose4x4_inplace_f32_cpp(l4, l5, l6, l7);
        neon_transpose4x4_inplace_f32_cpp(r0, r1, r2, r3);
        neon_transpose4x4_inplace_f32_cpp(r4, r5, r6, r7);
        winograd_f6k3_output_transform_inplace(l0, l1, l2, l3, r0, r1, r2, r3);
        winograd_f6k3_output_transform_inplace(l4, l5, l6, l7, r4, r5, r6, r7);

        float *ext = out + i*48;

        vst1q_f32(ext, l0);
        vst1q_f32(ext + 4, l4);
        vst1q_f32(ext + 8, l1);
        vst1q_f32(ext + 12, l5);
        vst1q_f32(ext + 16, l2);
        vst1q_f32(ext + 20, l6);
        vst1q_f32(ext + 24, l3);
        vst1q_f32(ext + 28, l7);
        vst1q_f32(ext + 32, r0);
        vst1q_f32(ext + 36, r4);
        vst1q_f32(ext + 40, r1);
        vst1q_f32(ext + 44, r5);
    }
}

void printRegisters(float32x4_t l0, float32x4_t r0, float32x4_t l1, float32x4_t r1, float32x4_t l2, float32x4_t r2, float32x4_t l3, float32x4_t r3,
                    float32x4_t l4, float32x4_t r4, float32x4_t l5, float32x4_t r5, float32x4_t l6, float32x4_t r6, float32x4_t l7, float32x4_t r7)
{
    float pBuf[64];
    vst1q_f32(pBuf,     l0);
    vst1q_f32(pBuf+4,   r0);
    vst1q_f32(pBuf+8,   l1);
    vst1q_f32(pBuf+12,  r1);
    vst1q_f32(pBuf+16,  l2);
    vst1q_f32(pBuf+20,  r2);
    vst1q_f32(pBuf+24,  l3);
    vst1q_f32(pBuf+28,  r3);
    vst1q_f32(pBuf+32,  l4);
    vst1q_f32(pBuf+36,  r4);
    vst1q_f32(pBuf+40,  l5);
    vst1q_f32(pBuf+44,  r5);
    vst1q_f32(pBuf+48,  l6);
    vst1q_f32(pBuf+52,  r6);
    vst1q_f32(pBuf+56,  l7);
    vst1q_f32(pBuf+60,  r7);

    printf("Register:\n");
    for(int i=0; i<8; i++)
    {
        for(int j=0; j<8; j++)    printf("%3.3f\t", pBuf[i*8 + j]);
        printf("\n");
    }
}
/*
void offlineKernelTransform(float *offlineKernel, float* testKernel, int outputChannels, int inputChannels, int ocBlock, int ocRegBlock)
{
    float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    float32x4_t m1, m2, s1, s2, t1, t2;//Auxiliary registers

    const float32x4_t f4    = vdupq_n_f32(4.0f);
    const float32x4_t f2    = vdupq_n_f32(2.0f);
    float *kernel = testKernel;
    for(int oc=0; oc<outputChannels; oc+=ocBlock)
    {
            int ocEnd    = oc+ocBlock<outputChannels?oc+ocBlock:outputChannels;
            int ocStep   = ocEnd - oc;

            int   lineWidth = 4*ocRegBlock*inputChannels;
            for(int kN=0; kN<ocBlock; kN+=ocRegBlock)
            {
                //error is find here. there should be no oc inside the loops
                float *kernel= testKernel    + (oc+kN)*inputChannels*9;
                float *pk0   = offlineKernel + (oc+kN)*inputChannels*64;
                float *pk1   = pk0  + lineWidth;
                float *pk2   = pk1  + lineWidth;
                float *pk3   = pk2  + lineWidth;
                float *pk4   = pk3  + lineWidth;
                float *pk5   = pk4  + lineWidth;
                float *pk6   = pk5  + lineWidth;
                float *pk7   = pk6  + lineWidth;
                float *pk8   = pk7  + lineWidth;
                float *pk9   = pk8  + lineWidth;
                float *pk10  = pk9  + lineWidth;
                float *pk11  = pk10 + lineWidth;
                float *pk12  = pk11 + lineWidth;
                float *pk13  = pk12 + lineWidth;
                float *pk14  = pk13 + lineWidth;
                float *pk15  = pk14 + lineWidth;

                for(int ic=0; ic<inputChannels; ic++)
                    for(int kRN=0; kRN<ocRegBlock; kRN++)
                    {
                        //printTensor(kernel, 3, 3);
                        l0 = vld1q_f32(kernel);
                        l1 = vld1q_f32(kernel+3);
                        l2 = vld1q_f32(kernel+6);
                        kernel += 9;

                        winograd_f6k3_kernel_transform(l0, l1, l2, m1, m2, s1, l7, t1, t2, s2, r7, f4, f2);
                        neon_transpose4x4_inplace_f32_cpp(m1, m2, s1, l7);
                        neon_transpose4x4_inplace_f32_cpp(t1, t2, s2, r7);
                        winograd_f6k3_kernel_transform(m1, m2, s1,  l0, l1, l2, l3, l4, l5, l6, l7, f4, f2);
                        winograd_f6k3_kernel_transform(t1, t2, s2,  r0, r1, r2, r3, r4, r5, r6, r7, f4, f2);

                        vst1q_f32(pk0,   l0);
                        pk0  +=4;
                        vst1q_f32(pk1,   r0);
                        pk1  +=4;
                        vst1q_f32(pk2,   l1);
                        pk2  +=4;
                        vst1q_f32(pk3,   r1);
                        pk3  +=4;
                        vst1q_f32(pk4,   l2);
                        pk4  +=4;
                        vst1q_f32(pk5,   r2);
                        pk5  +=4;
                        vst1q_f32(pk6,   l3);
                        pk6  +=4;
                        vst1q_f32(pk7,   r3);
                        pk7  +=4;
                        vst1q_f32(pk8,   l4);
                        pk8  +=4;
                        vst1q_f32(pk9,   r4);
                        pk9  +=4;
                        vst1q_f32(pk10,  l5);
                        pk10 +=4;
                        vst1q_f32(pk11,  r5);
                        pk11 +=4;
                        vst1q_f32(pk12,  l6);
                        pk12 +=4;
                        vst1q_f32(pk13,  r6);
                        pk13 +=4;
                        vst1q_f32(pk14,  l7);
                        pk14 +=4;
                        vst1q_f32(pk15,  r7);
                        pk15 +=4;
                    }
            }
    }
}

*/

void offlineKernelTransform(float *offlineKernel, float* testKernel, int outputChannels, int inputChannels, int ocBlock, int ocRegBlock)
{
    float32x4_t l0, l1, l2, l3, l4, l5, l6, l7;
    float32x4_t r0, r1, r2, r3, r4, r5, r6, r7;
    float32x4_t m1, m2, s1, s2, t1, t2;//Auxiliary registers

    const float32x4_t f4 = vdupq_n_f32(4.0f);
    const float32x4_t f2 = vdupq_n_f32(2.0f);
    for(int oc=0; oc<outputChannels; oc+=ocBlock)
    {
            int ocEnd  = oc+ocBlock<outputChannels?oc+ocBlock:outputChannels;
            int ocStep = ocEnd - oc;

            int lineWidth = 4*ocRegBlock*inputChannels;
            for(int kN=0; kN<ocBlock; kN+=ocRegBlock)
            {
                //error is find here. there should be no oc inside the loops
                float *kernel= testKernel    + (oc+kN)*inputChannels*9;
                float *pk0   = offlineKernel + (oc+kN)*inputChannels*64;
                float *pk1   = pk0  + lineWidth;
                float *pk2   = pk1  + lineWidth;
                float *pk3   = pk2  + lineWidth;
                float *pk4   = pk3  + lineWidth;
                float *pk5   = pk4  + lineWidth;
                float *pk6   = pk5  + lineWidth;
                float *pk7   = pk6  + lineWidth;
                float *pk8   = pk7  + lineWidth;
                float *pk9   = pk8  + lineWidth;
                float *pk10  = pk9  + lineWidth;
                float *pk11  = pk10 + lineWidth;
                float *pk12  = pk11 + lineWidth;
                float *pk13  = pk12 + lineWidth;
                float *pk14  = pk13 + lineWidth;
                float *pk15  = pk14 + lineWidth;

                for(int ic=0; ic<inputChannels; ic++)
                    for(int kRN=0; kRN<ocRegBlock; kRN++)
                    {
                        //printTensor(kernel, 3, 3);
                        l0 = vld1q_f32(kernel);
                        l1 = vld1q_f32(kernel+3);
                        l2 = vld1q_f32(kernel+6);
                        kernel += 9;

                        winograd_f6k3_kernel_transform(l0, l1, l2, m1, m2, s1, l7, t1, t2, s2, r7, f4, f2);
                        neon_transpose4x4_inplace_f32_cpp(m1, m2, s1, l7);
                        neon_transpose4x4_inplace_f32_cpp(t1, t2, s2, r7);
                        winograd_f6k3_kernel_transform(m1, m2, s1,  l0, l1, l2, l3, l4, l5, l6, l7, f4, f2);
                        winograd_f6k3_kernel_transform(t1, t2, s2,  r0, r1, r2, r3, r4, r5, r6, r7, f4, f2);

                        vst1q_f32(pk0,   l0);
                        pk0  +=4;
                        vst1q_f32(pk1,   r0);
                        pk1  +=4;
                        vst1q_f32(pk2,   l1);
                        pk2  +=4;
                        vst1q_f32(pk3,   r1);
                        pk3  +=4;
                        vst1q_f32(pk4,   l2);
                        pk4  +=4;
                        vst1q_f32(pk5,   r2);
                        pk5  +=4;
                        vst1q_f32(pk6,   l3);
                        pk6  +=4;
                        vst1q_f32(pk7,   r3);
                        pk7  +=4;
                        vst1q_f32(pk8,   l4);
                        pk8  +=4;
                        vst1q_f32(pk9,   r4);
                        pk9  +=4;
                        vst1q_f32(pk10,  l5);
                        pk10 +=4;
                        vst1q_f32(pk11,  r5);
                        pk11 +=4;
                        vst1q_f32(pk12,  l6);
                        pk12 +=4;
                        vst1q_f32(pk13,  r6);
                        pk13 +=4;
                        vst1q_f32(pk14,  l7);
                        pk14 +=4;
                        vst1q_f32(pk15,  r7);
                        pk15 +=4;
                    }
            }
    }
}

int winoF63(float *baseResult, float *testInput, const float *testKernel, int inputChannels, int outputChannels, int inputHeight, int inputWidth, int padWidth, int padHeight, int strideWidth, int strideHeight, int tileBlock, float *buf, int ocBlock, float *kernelBuf, int icBlock, float *inputBuf, int tileRegBlock, int ocRegBlock, bool enableOffKernel, int num_threads)
{
    /*
        outputBuf: tileBlock*ocBlock*48
        kernelBuf: icBlock*ocBlock*64
        inputBuf:  tileBlock*icBlock*64
    */
    
    int outputHeight = inputHeight+ padHeight*2 -2;
    int outputWidth  = inputWidth + padWidth*2  -2;
    int tileH = (inputHeight+ padHeight*2 -2 + 5)/6;
    int tileW = (inputWidth + padWidth*2  -2 + 5)/6;
    int tileN = tileH*tileW;

    printf("%d %d\n", inputChannels, icBlock);
    assert(inputChannels%icBlock  == 0);
    assert(outputChannels%ocBlock == 0);

    assert(ocBlock%ocRegBlock     == 0);
    assert(tileBlock%tileRegBlock == 0);

    void (*tensorGEMM)(float *, const float *, float *, int, int);
    void (*funGEMM)(float *, const float *, float *, int, int);
    if (ocRegBlock == 4)
    {
        if (tileRegBlock == 5)
            tensorGEMM = TensorGEMMInnerKernel4x5x4;
        else if (tileRegBlock == 4)
            tensorGEMM = TensorGEMMInnerKernel4x4x4;
        else if (tileRegBlock == 3)
            tensorGEMM = TensorGEMMInnerKernel4x3x4;
        else if (tileRegBlock == 2)
            tensorGEMM = TensorGEMMInnerKernel4x2x4;
        else if (tileRegBlock == 1)
            tensorGEMM = TensorGEMMInnerKernel4x1x4;

        if (tileN % tileRegBlock == 4)
            funGEMM = TensorGEMMInnerKernel4x4x4;
        else if (tileN % tileRegBlock == 3)
            funGEMM = TensorGEMMInnerKernel4x3x4;
        else if (tileN % tileRegBlock == 2)
            funGEMM = TensorGEMMInnerKernel4x2x4;
        else if (tileN % tileRegBlock == 1)
            funGEMM = TensorGEMMInnerKernel4x1x4;
    }
    else if (ocRegBlock == 5)
    {
        if (tileRegBlock == 5)
	    assert(0);
        if (tileRegBlock == 4)
            tensorGEMM = TensorGEMMInnerKernel5x4x4;
        else if (tileRegBlock == 3)
            tensorGEMM = TensorGEMMInnerKernel5x3x4;
        else if (tileRegBlock == 2)
            tensorGEMM = TensorGEMMInnerKernel5x2x4;
        else if (tileRegBlock == 1)
            tensorGEMM = TensorGEMMInnerKernel5x1x4;

        if (tileN % tileRegBlock == 4)
            funGEMM = TensorGEMMInnerKernel5x4x4;
        else if (tileN % tileRegBlock == 3)
            funGEMM = TensorGEMMInnerKernel5x3x4;
        else if (tileN % tileRegBlock == 2)
            funGEMM = TensorGEMMInnerKernel5x2x4;
        else if (tileN % tileRegBlock == 1)
            funGEMM = TensorGEMMInnerKernel5x1x4;
    }
    else if (ocRegBlock == 6)
    {
        if (tileRegBlock == 5)
            tensorGEMM = TensorGEMMInnerKernel6x5x4;
        else if (tileRegBlock == 4)
            tensorGEMM = TensorGEMMInnerKernel6x4x4;
        else if (tileRegBlock == 3)
            tensorGEMM = TensorGEMMInnerKernel6x3x4;
        else if (tileRegBlock == 2)
            tensorGEMM = TensorGEMMInnerKernel6x2x4;
        else if (tileRegBlock == 1)
            tensorGEMM = TensorGEMMInnerKernel6x1x4;

        if (tileN % tileRegBlock == 5)
            funGEMM = TensorGEMMInnerKernel6x5x4;
        else if (tileN % tileRegBlock == 4)
            funGEMM = TensorGEMMInnerKernel6x4x4;
        else if (tileN % tileRegBlock == 3)
            funGEMM = TensorGEMMInnerKernel6x3x4;
        else if (tileN % tileRegBlock == 2)
            funGEMM = TensorGEMMInnerKernel6x2x4;
        else if (tileN % tileRegBlock == 1)
            funGEMM = TensorGEMMInnerKernel6x1x4;
    }
    else if (ocRegBlock == 7)
    {
        if (tileRegBlock == 6)
            tensorGEMM = TensorGEMMInnerKernel7x6x4;
        else if (tileRegBlock == 5)
            tensorGEMM = TensorGEMMInnerKernel7x5x4;
        else if (tileRegBlock == 4)
            tensorGEMM = TensorGEMMInnerKernel7x4x4;
        else if (tileRegBlock == 3)
            tensorGEMM = TensorGEMMInnerKernel7x3x4;
        else if (tileRegBlock == 2)
            tensorGEMM = TensorGEMMInnerKernel7x2x4;
        else if (tileRegBlock == 1)
            tensorGEMM = TensorGEMMInnerKernel7x1x4;

        if (tileN % tileRegBlock == 6)
            funGEMM = TensorGEMMInnerKernel7x6x4;
        else if (tileN % tileRegBlock == 5)
            funGEMM = TensorGEMMInnerKernel7x5x4;
        else if (tileN % tileRegBlock == 4)
            funGEMM = TensorGEMMInnerKernel7x4x4;
        else if (tileN % tileRegBlock == 3)
            funGEMM = TensorGEMMInnerKernel7x3x4;
        else if (tileN % tileRegBlock == 2)
            funGEMM = TensorGEMMInnerKernel7x2x4;
        else if (tileN % tileRegBlock == 1)
            funGEMM = TensorGEMMInnerKernel7x1x4;
    }
    else if (ocRegBlock == 3)
    {
        if (tileRegBlock == 7)
            tensorGEMM = TensorGEMMInnerKernel3x7x4;
        else if (tileRegBlock == 6)
            tensorGEMM = TensorGEMMInnerKernel3x6x4;
        else if (tileRegBlock == 5)
            tensorGEMM = TensorGEMMInnerKernel3x5x4;
        else if (tileRegBlock == 4)
            tensorGEMM = TensorGEMMInnerKernel3x4x4;
        else if (tileRegBlock == 3)
            tensorGEMM = TensorGEMMInnerKernel3x3x4;
        else if (tileRegBlock == 2)
            tensorGEMM = TensorGEMMInnerKernel3x2x4;
        else if (tileRegBlock == 1)
            tensorGEMM = TensorGEMMInnerKernel3x1x4;

        if (tileN % tileRegBlock == 7)
            funGEMM = TensorGEMMInnerKernel3x7x4;
        else if (tileN % tileRegBlock == 6)
            funGEMM = TensorGEMMInnerKernel3x6x4;
        else if (tileN % tileRegBlock == 5)
            funGEMM = TensorGEMMInnerKernel3x5x4;
        else if (tileN % tileRegBlock == 4)
            funGEMM = TensorGEMMInnerKernel3x4x4;
        else if (tileN % tileRegBlock == 3)
            funGEMM = TensorGEMMInnerKernel3x3x4;
        else if (tileN % tileRegBlock == 2)
            funGEMM = TensorGEMMInnerKernel3x2x4;
        else if (tileN % tileRegBlock == 1)
            funGEMM = TensorGEMMInnerKernel3x1x4;
    }
    else if (ocRegBlock == 2)
    {
        if (tileRegBlock == 10)
            tensorGEMM = TensorGEMMInnerKernel2x10x4;
        else if (tileRegBlock == 1)
            tensorGEMM = TensorGEMMInnerKernel2x1x4;
    }


/*
    void (*tensorGEMM)(float *, const float *, float *, int, int);
    if(tileRegBlock==4      && ocRegBlock==4)	tensorGEMM = TensorGEMMInnerKernel4x4x4;
    else if(tileRegBlock==5 && ocRegBlock==4)	tensorGEMM = TensorGEMMInnerKernel4x5x4;

    void (*funGEMM)(float *, const float *, float *, int, int);
    if(tileN%tileRegBlock==4)		funGEMM = TensorGEMMInnerKernel4x4x4;
    else if(tileN%tileRegBlock==3)	funGEMM = TensorGEMMInnerKernel4x3x4;
    else if(tileN%tileRegBlock==2)	funGEMM = TensorGEMMInnerKernel4x2x4;
    else if(tileN%tileRegBlock==1)	funGEMM = TensorGEMMInnerKernel4x1x4;
*/
    Timer inputTran, kernelTran, GEMM, outputTran, storeBack;

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

#ifdef _OPENMP
#ifdef __aarch64__
//#pragma omp parallel for num_threads(num_threads) collapse(2) schedule(static)
#endif
#endif

    for(int oc=0; oc<outputChannels; oc+=ocBlock)
    {
        for(int tile=0; tile<tileN; tile+=tileBlock)
        {
            int ocEnd    = oc+ocBlock<outputChannels?oc+ocBlock:outputChannels;
            int ocStep   = ocEnd - oc;
            int tileEnd  = tile+tileBlock<tileN ? tile+tileBlock : tileN;
            int tileStep = tileEnd - tile;

	    if(!enableOffKernel)
            { 
            kernelTran.startBench();
	    int   lineWidth = 4*ocRegBlock*inputChannels;
            for(int kN=0; kN<ocBlock; kN+=ocRegBlock)
            {
                //error is find here. there should be no oc inside the loops
                const float *kernel= testKernel + (oc+kN)*inputChannels*9;
                float *pk0   = kernelBuf  + kN*inputChannels*64;
                float *pk1   = pk0  + lineWidth;
                float *pk2   = pk1  + lineWidth;
                float *pk3   = pk2  + lineWidth;
                float *pk4   = pk3  + lineWidth;
                float *pk5   = pk4  + lineWidth;
                float *pk6   = pk5  + lineWidth;
                float *pk7   = pk6  + lineWidth;
                float *pk8   = pk7  + lineWidth;
                float *pk9   = pk8  + lineWidth;
                float *pk10  = pk9  + lineWidth;
                float *pk11  = pk10 + lineWidth;
                float *pk12  = pk11 + lineWidth;
                float *pk13  = pk12 + lineWidth;
                float *pk14  = pk13 + lineWidth;
                float *pk15  = pk14 + lineWidth;

                for(int ic=0; ic<inputChannels; ic++)
                    for(int kRN=0; kRN<ocRegBlock; kRN++)
                    {
                        //printTensor(kernel, 3, 3);
                        l0 = vld1q_f32(kernel);
                        l1 = vld1q_f32(kernel+3);
                        l2 = vld1q_f32(kernel+6);
                        kernel += 9;

                        winograd_f6k3_kernel_transform(l0, l1, l2, m1, m2, s1, l7, t1, t2, s2, r7, f4, f2);
                        neon_transpose4x4_inplace_f32_cpp(m1, m2, s1, l7);
                        neon_transpose4x4_inplace_f32_cpp(t1, t2, s2, r7);
                        winograd_f6k3_kernel_transform(m1, m2, s1,  l0, l1, l2, l3, l4, l5, l6, l7, f4, f2);
                        winograd_f6k3_kernel_transform(t1, t2, s2,  r0, r1, r2, r3, r4, r5, r6, r7, f4, f2);

                        vst1q_f32(pk0,   l0);
                        pk0  +=4;
                        vst1q_f32(pk1,   r0);
                        pk1  +=4;
                        vst1q_f32(pk2,   l1);
                        pk2  +=4;
                        vst1q_f32(pk3,   r1);
                        pk3  +=4;
                        vst1q_f32(pk4,   l2);
                        pk4  +=4;
                        vst1q_f32(pk5,   r2);
                        pk5  +=4;
                        vst1q_f32(pk6,   l3);
                        pk6  +=4;
                        vst1q_f32(pk7,   r3);
                        pk7  +=4;
                        vst1q_f32(pk8,   l4);
                        pk8  +=4;
                        vst1q_f32(pk9,   r4);
                        pk9  +=4;
                        vst1q_f32(pk10,  l5);
                        pk10 +=4;
                        vst1q_f32(pk11,  r5);
                        pk11 +=4;
                        vst1q_f32(pk12,  l6);
                        pk12 +=4;
                        vst1q_f32(pk13,  r6);
                        pk13 +=4;
                        vst1q_f32(pk14,  l7);
                        pk14 +=4;
                        vst1q_f32(pk15,  r7);
                        pk15 +=4;
                    }
            }
            kernelTran.accumBench();
	    }

//            printf("kernelTran\n");
//            printTensor(kernelBuf, 32, 16);

            int lineWidth = 4 * tileRegBlock * inputChannels;
            int stepSize = tileRegBlock*inputChannels*64 - tileRegBlock*4;
            inputTran.startBench();

            for(int ic=0; ic<inputChannels; ic++)
            {
                int tileRow = (tile/tileW)*6 - padHeight;
                int tileCol = (tile%tileW)*6 - padWidth;
                int inPos   = tileRow*inputWidth + tileCol;

                float *pIn0 = testInput + ic*inputHeight*inputWidth + inPos;
                float *pIn1 = pIn0 + inputWidth;
                float *pIn2 = pIn1 + inputWidth;
                float *pIn3 = pIn2 + inputWidth;
                float *pIn4 = pIn3 + inputWidth;
                float *pIn5 = pIn4 + inputWidth;
                float *pIn6 = pIn5 + inputWidth;
                float *pIn7 = pIn6 + inputWidth;

                //if(tileRow>=0 && tileRow+8 <= inputWidth && tileCol>=0 && tileCol+8 <= inputWidth)
                if(tileRow>=0 && tileRow+8 <= inputHeight && tileCol>=0 && tileCol+8 <= inputWidth)
                {
                    __builtin_prefetch(pIn0, 0, 1);
                    __builtin_prefetch(pIn1, 0, 1);
                    __builtin_prefetch(pIn2, 0, 1);
                    __builtin_prefetch(pIn3, 0, 1);
                    __builtin_prefetch(pIn4, 0, 1);
                    __builtin_prefetch(pIn5, 0, 1);
                    __builtin_prefetch(pIn6, 0, 1);
                    __builtin_prefetch(pIn7, 0, 1);
                }

                float *p0  = inputBuf + ic*4*tileRegBlock;
                float *p1  = p0  + lineWidth;
                float *p2  = p1  + lineWidth;
                float *p3  = p2  + lineWidth;
                float *p4  = p3  + lineWidth;
                float *p5  = p4  + lineWidth;
                float *p6  = p5  + lineWidth;
                float *p7  = p6  + lineWidth;
                float *p8  = p7  + lineWidth;
                float *p9  = p8  + lineWidth;
                float *p10 = p9  + lineWidth;
                float *p11 = p10 + lineWidth;
                float *p12 = p11 + lineWidth;
                float *p13 = p12 + lineWidth;
                float *p14 = p13 + lineWidth;
                float *p15 = p14 + lineWidth;
                __builtin_prefetch(p0,  1, 1);
                __builtin_prefetch(p1,  1, 1);
                __builtin_prefetch(p2,  1, 1);
                __builtin_prefetch(p3,  1, 1);
                __builtin_prefetch(p4,  1, 1);
                __builtin_prefetch(p5,  1, 1);
                __builtin_prefetch(p6,  1, 1);
                __builtin_prefetch(p7,  1, 1);
                __builtin_prefetch(p8,  1, 1);
                __builtin_prefetch(p9,  1, 1);
                __builtin_prefetch(p10, 1, 1);
                __builtin_prefetch(p11, 1, 1);
                __builtin_prefetch(p12, 1, 1);
                __builtin_prefetch(p13, 1, 1);
                __builtin_prefetch(p14, 1, 1);
                __builtin_prefetch(p15, 1, 1);

                int tNRmd = tileStep%tileRegBlock;
                int tNEnd = tileStep - tNRmd;

                for(int tN=0; tN<tileStep; tN++)
                {
                    int curRow = (tile+tN)/tileW*6 - padHeight;
                    int curCol = (tile+tN)%tileW*6 - padWidth;
		    int rest = pIn0 - testInput + ic*inputHeight*inputWidth;
                    //if(curRow>=0 && curRow+8 <= inputWidth && curCol>=0 && curCol+8 <= inputWidth)
                    if(curRow>=0 && curRow+8<=inputHeight && curCol>=0 && curCol+8<=inputWidth)
                    {
                        l0 = vld1q_f32(pIn0);
                        r0 = vld1q_f32(pIn0 + 4);
                        l1 = vld1q_f32(pIn1);
                        r1 = vld1q_f32(pIn1 + 4);
                        l2 = vld1q_f32(pIn2);
                        r2 = vld1q_f32(pIn2 + 4);
                        l3 = vld1q_f32(pIn3);
                        r3 = vld1q_f32(pIn3 + 4);
                        l4 = vld1q_f32(pIn4);
                        r4 = vld1q_f32(pIn4 + 4);
                        l5 = vld1q_f32(pIn5);
                        r5 = vld1q_f32(pIn5 + 4);
                        l6 = vld1q_f32(pIn6);
                        r6 = vld1q_f32(pIn6 + 4);
                        l7 = vld1q_f32(pIn7);
                        r7 = vld1q_f32(pIn7 + 4);
                    }
                    else
                    {
                        float tmp[64];
			memset(tmp, 0, 256);
			int XStart = curRow<0?0:curRow; 
			int YStart = curCol<0?0:curCol;
			XStart     = XStart<inputHeight?XStart:inputHeight;
			YStart     = YStart<inputWidth?YStart:inputWidth;

			int XEnd   = curRow+8<inputHeight?curRow+8:inputHeight;
			int YEnd   = curCol+8<inputWidth?curCol+8:inputWidth;
			XEnd       = XEnd<0?0:XEnd;
			YEnd       = YEnd<0?0:YEnd;
			int YStep  = YEnd-YStart;
		        int XStep  = XEnd-XStart;	

			int len    = YStep*sizeof(float);
			float *pLocal = testInput + ic*inputHeight*inputWidth + XStart*inputWidth + YStart;
			float *ptmp   = tmp + (XStart-curRow)*8 + YStart-curCol;
			for(int ux=0; ux<XStep; ux++, ptmp+=8, pLocal+=inputWidth)
			{
//			    	printf("(%d %d)<-(%d %d) Step=%d\n", ux-curRow, YStart-curCol, ux, YStart, YStep);
				memcpy(ptmp, pLocal, len);
			}

/*			float *pLocal = testInput + ic*inputHeight*inputWidth;
                        for(int ux=XStart; ux<XEnd; ux++)
			{
//			    printf("(%d %d)<-(%d %d) Step=%d\n", ux-curRow, YStart-curCol, ux, YStart, YStep);
			    memcpy(tmp+(ux-curRow)*8+(YStart-curCol), pLocal+ ux*inputWidth+YStart, YStep*sizeof(float)); 	
			}
//			printTensor(tmp, 8, 8);
*/                      
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
                    pIn0 += 6;
                    pIn1 += 6;
                    pIn2 += 6;
                    pIn3 += 6;
                    pIn4 += 6;
                    pIn5 += 6;
                    pIn6 += 6;
                    pIn7 += 6;

                    int row = (tile+tN+1)/tileW*6 - padHeight;
                    int col = (tile+tN+1)%tileW*6 - padWidth;
                    //Reflash the value if tileRow goes to the next line;
                    if((tN != tileStep-1) && curRow != row)
                    {
                        pIn0 = testInput + ic*inputHeight*inputWidth +  row*inputWidth + col;
                        pIn1 = pIn0 + inputWidth;
                        pIn2 = pIn1 + inputWidth;
                        pIn3 = pIn2 + inputWidth;
                        pIn4 = pIn3 + inputWidth;
                        pIn5 = pIn4 + inputWidth;
                        pIn6 = pIn5 + inputWidth;
                        pIn7 = pIn6 + inputWidth;

                        //if(row>=0 && row+8 < inputWidth && col>=0 && col+8 < inputWidth)
                        if(row>=0 && row+8 < inputHeight && col>=0 && col+8 < inputWidth)
                        {
                            __builtin_prefetch(pIn0, 0, 1);
                            __builtin_prefetch(pIn1, 0, 1);
                            __builtin_prefetch(pIn2, 0, 1);
                            __builtin_prefetch(pIn3, 0, 1);
                            __builtin_prefetch(pIn4, 0, 1);
                            __builtin_prefetch(pIn5, 0, 1);
                            __builtin_prefetch(pIn6, 0, 1);
                            __builtin_prefetch(pIn7, 0, 1);
                        }
                    }
                    if(!(tN%tileRegBlock) && tN)
                    {
                        p0 += stepSize;
                        p1 += stepSize;
                        p2 += stepSize;
                        p3 += stepSize;
                        p4 += stepSize;
                        p5 += stepSize;
                        p6 += stepSize;
                        p7 += stepSize;
                        p8 += stepSize;
                        p9 += stepSize;
                        p10+= stepSize;
                        p11+= stepSize;
                        p12+= stepSize;
                        p13+= stepSize;
                        p14+= stepSize;
                        p15+= stepSize;

                        __builtin_prefetch(p0, 1, 1);
                        __builtin_prefetch(p1, 1, 1);
                        __builtin_prefetch(p2, 1, 1);
                        __builtin_prefetch(p3, 1, 1);
                        __builtin_prefetch(p4, 1, 1);
                        __builtin_prefetch(p5, 1, 1);
                        __builtin_prefetch(p6, 1, 1);
                        __builtin_prefetch(p7, 1, 1);
                        __builtin_prefetch(p8, 1, 1);
                        __builtin_prefetch(p9, 1, 1);
                        __builtin_prefetch(p10, 1, 1);
                        __builtin_prefetch(p11, 1, 1);
                        __builtin_prefetch(p12, 1, 1);
                        __builtin_prefetch(p13, 1, 1);
                        __builtin_prefetch(p14, 1, 1);
                        __builtin_prefetch(p15, 1, 1);
                    }

//                    printRegisters(l0, r0, l1, r1, l2, r2, l3, r3, l4, r4, l5, r5, l6, r6, l7, r7);
                    input_transform(l0, l1, l2, l3, l4, l5, l6, l7, //Target
                                    t1, t2, s1, s2, m1, m2, //Auxiliary
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                    neon_transpose4x4_inplace_f32_cpp(l0, l1, l2, l3);
                    neon_transpose4x4_inplace_f32_cpp(l4, l5, l6, l7);
                    input_transform(r0, r1, r2, r3, r4, r5, r6, r7, //Target
                                    t1, t2, s1, s2, m1, m2, //Auxiliary
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                    neon_transpose4x4_inplace_f32_cpp(r0, r1, r2, r3);
                    neon_transpose4x4_inplace_f32_cpp(r4, r5, r6, r7);

                    input_transform(l0, l1, l2, l3, r0, r1, r2, r3, //Target
                                    t1, t2, s1, s2, m1, m2, //Auxiliary
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants
                    input_transform(l4, l5, l6, l7, r4, r5, r6, r7, //Target
                                    t1, t2, s1, s2, m1, m2, //Auxiliary
                                    f5_25, f4_25, f4, f2_5, f2, f1_25, f0_5, f0_25); //Constants

                    vst1q_f32(p0,  l0);
                    p0+=4;
                    vst1q_f32(p1,  l4);
                    p1+=4;
                    vst1q_f32(p2,  l1);
                    p2+=4;
                    vst1q_f32(p3,  l5);
                    p3+=4;
                    vst1q_f32(p4,  l2);
                    p4+=4;
                    vst1q_f32(p5,  l6);
                    p5+=4;
                    vst1q_f32(p6,  l3);
                    p6+=4;
                    vst1q_f32(p7,  l7);
                    p7+=4;
                    vst1q_f32(p8,  r0);
                    p8+=4;
                    vst1q_f32(p9,  r4);
                    p9+=4;
                    vst1q_f32(p10, r1);
                    p10+=4;
                    vst1q_f32(p11, r5);
                    p11+=4;
                    vst1q_f32(p12, r2);
                    p12+=4;
                    vst1q_f32(p13, r6);
                    p13+=4;
                    vst1q_f32(p14, r3);
                    p14+=4;
                    vst1q_f32(p15, r7);
                    p15+=4;
                }
            }
            inputTran.accumBench();
//            printf("inputTran\n");
//            printTensor(inputBuf, 16, 16);


            GEMM.startBench();
            float *gemmBuf = buf + tileRegBlock*ocRegBlock*36;
	    int cellSize   = tileRegBlock*ocRegBlock*64;
            int colStep    = tileBlock / tileRegBlock;
	    int extStep    = tileRegBlock  * ocRegBlock   * 4;
	    int inputStep  = inputChannels * tileRegBlock * 4;
	    int kernelStep = inputChannels * ocRegBlock   * 4;
	    int tensorStep = tileRegBlock*4;
            for(int kN=0; kN<ocBlock; kN+=ocRegBlock)
            {
                int tileStepRmd = tileStep % tileRegBlock;
                int tileStepEnd = tileStep - tileStepRmd;

                float *pInputBuf  = inputBuf;
                float *ext        = gemmBuf   + (kN/ocRegBlock*colStep)* cellSize;
		const float *pKernel;
		pKernel = kernelBuf + ((enableOffKernel?oc:0) + kN) * inputChannels * 64;

                for(int tN=0; tN<tileStepEnd; tN+=tileRegBlock)
                {
                    const float *pKernelBuf = pKernel;
                    for(int depth=0; depth<16; depth++)
                    {
                        tensorGEMM(ext, pKernelBuf,  pInputBuf, inputChannels, tensorStep);
                        ext        += extStep;
                        pInputBuf  += inputStep;
                        pKernelBuf += kernelStep;
                    }

                }

                if(tileStepRmd)
                {
                    const float *pKernelBuf = pKernel;
                    for(int depth=0; depth<16; depth++)
                    {
                        funGEMM(ext, pKernelBuf,  pInputBuf, inputChannels, tensorStep);
                        ext        += extStep;
                        pInputBuf  += inputStep;
                        pKernelBuf += kernelStep;
                    }
                }
            }
            GEMM.accumBench();

	    outputTran.startBench();
            memset(buf, 0, sizeof(float)*36*tileRegBlock*ocRegBlock);
            int step = tileRegBlock*ocRegBlock*4;

	    for(int kN=0; kN<ocBlock; kN+=ocRegBlock)
                for(int tN=0; tN<tileStep; tN+=tileRegBlock)
                {
                    int tNEnd  = tN+tileRegBlock<tileStep?tN+tileRegBlock:tileStep;
                    int tNStep = tNEnd - tN;
                    float *pOutputBuf = gemmBuf + (kN/ocRegBlock*colStep + tN/tileRegBlock)*cellSize;

                    __builtin_prefetch(pOutputBuf, 	    0, 1);
                    __builtin_prefetch(pOutputBuf+cellSize, 0, 1);
		    float *pOut0 = pOutputBuf;
		    float *pOut1 = pOutputBuf + step*1;
		    float *pOut2 = pOutputBuf + step*2;
		    float *pOut3 = pOutputBuf + step*3;
		    float *pOut4 = pOutputBuf + step*4;
		    float *pOut5 = pOutputBuf + step*5;
		    float *pOut6 = pOutputBuf + step*6;
		    float *pOut7 = pOutputBuf + step*7;
		    float *pOut8 = pOutputBuf + step*8;
		    float *pOut9 = pOutputBuf + step*9;
		    float *pOut10= pOutputBuf + step*10;
		    float *pOut11= pOutputBuf + step*11;
		    float *pOut12= pOutputBuf + step*12;
		    float *pOut13= pOutputBuf + step*13;
		    float *pOut14= pOutputBuf + step*14;
		    float *pOut15= pOutputBuf + step*15;

	    	    int outputPage   = outputHeight * outputWidth;
/*		    float *pWrite = baseResult + (oc+kN)*outputPage + (tile+tN)/tileW*6*outputHeight + (tile+tN)%tileW*6;
		    for(int kRN=0;kRN<ocRegBlock;kRN++)
		    {
                    	__builtin_prefetch(pWrite, 	    		    1, 1);
                    	__builtin_prefetch(pWrite+outputHeight, 	    1, 1);
                    	__builtin_prefetch(pWrite+outputHeight*2, 	    1, 1);
                    	__builtin_prefetch(pWrite+outputHeight*3, 	    1, 1);
                    	__builtin_prefetch(pWrite+outputHeight*4, 	    1, 1);
                    	__builtin_prefetch(pWrite+outputHeight*5, 	    1, 1);
			pWrite += outputPage;
	            }
*/
                    for(int kRN=0; kRN<ocRegBlock; kRN++)
		    {
//                        float *ext = buf + ((kN+kRN)*tileBlock + (tN))*36;
//                        float *ext = buf + kRN*tileRegBlock*36;
                        float *ext = buf + kRN*tNStep*36;
     		        for(int tRN=0; tRN<tNStep; tRN++)
                        {
                            l0 = vld1q_f32(pOut0); 	pOut0+=4;
                            r0 = vld1q_f32(pOut1);	pOut1+=4;
                            l1 = vld1q_f32(pOut2);	pOut2+=4;
                            r1 = vld1q_f32(pOut3);	pOut3+=4;
                            l2 = vld1q_f32(pOut4);	pOut4+=4;
                            r2 = vld1q_f32(pOut5);	pOut5+=4;
                            l3 = vld1q_f32(pOut6);	pOut6+=4;
                            r3 = vld1q_f32(pOut7);	pOut7+=4;
                            l4 = vld1q_f32(pOut8);	pOut8+=4;
                            r4 = vld1q_f32(pOut9);	pOut9+=4;
                            l5 = vld1q_f32(pOut10);	pOut10+=4;
                            r5 = vld1q_f32(pOut11);	pOut11+=4;
                            l6 = vld1q_f32(pOut12);	pOut12+=4;
                            r6 = vld1q_f32(pOut13);	pOut13+=4;
                            l7 = vld1q_f32(pOut14);	pOut14+=4;
                            r7 = vld1q_f32(pOut15);	pOut15+=4;

/*			
                          float *pOut = pOutputBuf + kRN*tNStep*4 + tRN*4;
                            l0 = vld1q_f32(pOut);
                            pOut += step;
                            r0 = vld1q_f32(pOut);
                            pOut += step;
                            l1 = vld1q_f32(pOut);
                            pOut += step;
                            r1 = vld1q_f32(pOut);
                            pOut+=step;
                            l2 = vld1q_f32(pOut);
                            pOut+=step;
                            r2 = vld1q_f32(pOut);
                            pOut+=step;
                            l3 = vld1q_f32(pOut);
                            pOut+=step;
                            r3 = vld1q_f32(pOut);
                            pOut+=step;
                            l4 = vld1q_f32(pOut);
                            pOut+=step;
                            r4 = vld1q_f32(pOut);
                            pOut+=step;
                            l5 = vld1q_f32(pOut);
                            pOut+=step;
                            r5 = vld1q_f32(pOut);
                            pOut+=step;
                            l6 = vld1q_f32(pOut);
                            pOut+=step;
                            r6 = vld1q_f32(pOut);
                            pOut+=step;
                            l7 = vld1q_f32(pOut);
                            pOut+=step;
                            r7 = vld1q_f32(pOut);
                            pOut+=step;
*/

//                            printRegisters(l0, r0, l1, r1, l2, r2, l3, r3, l4, r4, l5, r5, l6, r6, l7, r7);
                            winograd_f6k3_output_transform_inplace(l0, l1, l2, l3, l4, l5, l6, l7);
                            winograd_f6k3_output_transform_inplace(r0, r1, r2, r3, r4, r5, r6, r7);
                            neon_transpose4x4_inplace_f32_cpp(l0, l1, l2, l3);
                            neon_transpose4x4_inplace_f32_cpp(r0, r1, r2, r3);
                            neon_transpose4x4_inplace_f32_cpp(l4, l5, l6, l7);
                            neon_transpose4x4_inplace_f32_cpp(r4, r5, r6, r7);
                            winograd_f6k3_output_transform_inplace(l0, l1, l2, l3, r0, r1, r2, r3);
                            winograd_f6k3_output_transform_inplace(l4, l5, l6, l7, r4, r5, r6, r7);
//			    printRegisters(l0, l4, l1, l5, l2, l6, l3, l7, r0, r4, r1, r5, r2, r6, r3, r7);


			    vst1q_f32(ext,      l0);
                            vst1_f32(ext + 4,  vget_low_f32(l4));
                            vst1q_f32(ext + 6,  l1);
                            vst1_f32(ext + 10, vget_low_f32(l5));
                            vst1q_f32(ext + 12, l2);
                            vst1_f32(ext + 16, vget_low_f32(l6));
                            vst1q_f32(ext + 18, l3);
                            vst1_f32(ext + 22, vget_low_f32(l7));
                            vst1q_f32(ext + 24, r0);
                            vst1_f32(ext + 28, vget_low_f32(r4));
                            vst1q_f32(ext + 30, r1);
                            vst1_f32(ext + 34, vget_low_f32(r5));

/*
                            vst1q_f32(ext,      l0);
                            vst1q_f32(ext + 4,  l4);
                            vst1q_f32(ext + 8,  l1);
                            vst1q_f32(ext + 12, l5);
                            vst1q_f32(ext + 16, l2);
                            vst1q_f32(ext + 20, l6);
                            vst1q_f32(ext + 24, l3);
                            vst1q_f32(ext + 28, l7);
                            vst1q_f32(ext + 32, r0);
                            vst1q_f32(ext + 36, r4);
                            vst1q_f32(ext + 40, r1);
                            vst1q_f32(ext + 44, r5);
*/			    ext += 36;
                        }
		    }


		    for(int kRN=0;kRN<ocRegBlock;kRN++)
			for(int tRN=0;tRN<tNStep;tRN++)
			{
	                    int tileRow = (tile+tN+tRN)/tileW*6;
        	            int tileCol = (tile+tN+tRN)%tileW*6;
				
			    //float *output = baseResult + (oc+kN+kRN)*outputPage + tileRow*outputHeight + tileCol;
                            float *output = baseResult + (oc+kN+kRN)*outputPage + tileRow*outputWidth + tileCol;

                    	    int deltaX = outputHeight - tileRow;
                    	    //int deltaY = outputHeight - tileCol;
                            int deltaY = outputWidth - tileCol;
                    	    deltaX = deltaX<6?deltaX:6;
                    	    deltaY = deltaY<6?deltaY:6;
//                    	    for(int u=0;u<deltaX;u++)   memcpy(output+u*outputHeight, buf+(kN+kRN)*tileBlock*36 + (tN+tRN)*36  + u*6, deltaY*4); 
//                    	    for(int u=0;u<deltaX;u++)   memcpy(output+u*outputHeight, buf+(kRN)*tileRegBlock*36 + (tRN)*36  + u*6, deltaY*4); 
                    	    for(int u=0;u<deltaX;u++)   memcpy(output+u*outputWidth, buf+(kRN)*tNStep*36 + (tRN)*36 + u*6, deltaY*4);//memcpy(output+u*outputHeight, buf+(kRN)*tNStep*36 + (tRN)*36  + u*6, deltaY*4); 
											//bug may be here.
			}
		}

	    
	    
	    /*
	    //Wait to fuse outputTransform into oneStep!!!!!
            for(int kN=0; kN<ocBlock; kN++)
            {
		for(int i=0; i<tileStep; i++)
                {
                    int tileRow = (tile+i)/tileW*6;
                    int tileCol = (tile+i)%tileW*6;

                    float *output = baseResult + (oc+kN)*outputPage + tileRow*outputHeight + tileCol;

		    int deltaX = outputHeight - tileRow;
		    int deltaY = outputHeight - tileCol; 
		    deltaX = deltaX<6?deltaX:6;
		    deltaY = deltaY<6?deltaY:6;
		    for(int u=0;u<deltaX;u++)	memcpy(output+u*outputHeight, buf+kN*tileBlock*36 + i*36  + u*6, deltaY*4);

                    //store a 6x6 matrix back to output
//                    	for(int u=0; u<6; u++)  for(int v=0; v<6; v++)
//                        {
//                            if(tileRow+u<inputHeight-2 && tileCol+v<inputHeight-2)
////                                *(output+u*(inputWidth-2)+v) = buf[kN*tileBlock*48 + i*48  + u*8 + v];
//                                *(output+u*(inputWidth-2)+v) = buf[kN*tileBlock*36 + i*36  + u*6 + v];
//                        }

                }
	    }
*/
            outputTran.accumBench();
        }

    }

    kernelTran.printBench("kernel", 1);
    inputTran.printBench("inputTran", 1);
    GEMM.printBench("TensorGEMM", 1);
    outputTran.printBench("outputTran", 1);
    return 1;
}
