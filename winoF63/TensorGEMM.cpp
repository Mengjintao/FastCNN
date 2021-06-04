#include <stdlib.h>
#include <arm_neon.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>


void print_floats(float* arr, const int len)
{
    for (int i = 0; i < len; ++i)
    {
        printf("%5.2f ", arr[i]);
    }
    printf("\n\n");
}


void TensorGEMMInnerKernel4x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00;
    float32x4_t vc10;
    float32x4_t vc20;
    float32x4_t vc30;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0;
    vc00 = vdupq_n_f32(0.f);
    vc10 = vdupq_n_f32(0.f);
    vc20 = vdupq_n_f32(0.f);
    vc30 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up); 
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
        v0 = vld1q_f32(vp); 
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc10 = vfmaq_f32(vc10, u1, v0);
        vc20 = vfmaq_f32(vc20, u2, v0);
        vc30 = vfmaq_f32(vc30, u3, v0);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc10);
    vst1q_f32(wp + 8,  vc20);
    vst1q_f32(wp + 12, vc30);
}


void TensorGEMMInnerKernel4x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01;
    float32x4_t vc10, vc11;
    float32x4_t vc20, vc21;
    float32x4_t vc30, vc31;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
        v0 = vld1q_f32(vp); 
        v1 = vld1q_f32(vp + 4);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);

    vst1q_f32(wp + 8, vc10);
    vst1q_f32(wp + 12, vc11);

    vst1q_f32(wp + 16, vc20);
    vst1q_f32(wp + 20, vc21);

    vst1q_f32(wp + 24, vc30);
    vst1q_f32(wp + 28, vc31);
}

void TensorGEMMInnerKernel4x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02;
    float32x4_t vc10, vc11, vc12;
    float32x4_t vc20, vc21, vc22;
    float32x4_t vc30, vc31, vc32;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);

    vst1q_f32(wp + 12, vc10);
    vst1q_f32(wp + 16, vc11);
    vst1q_f32(wp + 20, vc12);

    vst1q_f32(wp + 24, vc20);
    vst1q_f32(wp + 28, vc21);
    vst1q_f32(wp + 32, vc22);

    vst1q_f32(wp + 36, vc30);
    vst1q_f32(wp + 40, vc31);
    vst1q_f32(wp + 44, vc32);
}


void TensorGEMMInnerKernel4x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t vc30, vc31, vc32, vc33;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);

	vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        vc33 = vfmaq_f32(vc33, u3, v3);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc10);
    vst1q_f32(wp + 20, vc11);
    vst1q_f32(wp + 24, vc12);
    vst1q_f32(wp + 28, vc13);
    vst1q_f32(wp + 32, vc20);
    vst1q_f32(wp + 36, vc21);
    vst1q_f32(wp + 40, vc22);
    vst1q_f32(wp + 44, vc23);
    vst1q_f32(wp + 48, vc30);
    vst1q_f32(wp + 52, vc31);
    vst1q_f32(wp + 56, vc32);
    vst1q_f32(wp + 60, vc33);
}

void TensorGEMMInnerKernel2x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t vc30, vc31, vc32, vc33;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        up += 8;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);

	vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);

        // vc20 = vfmaq_f32(vc20, u2, v0);
        // vc21 = vfmaq_f32(vc21, u2, v1);
        // vc22 = vfmaq_f32(vc22, u2, v2);
        // vc23 = vfmaq_f32(vc23, u2, v3);

        // vc30 = vfmaq_f32(vc30, u3, v0);
        // vc31 = vfmaq_f32(vc31, u3, v1);
        // vc32 = vfmaq_f32(vc32, u3, v2);
        // vc33 = vfmaq_f32(vc33, u3, v3);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc10);
    vst1q_f32(wp + 20, vc11);
    vst1q_f32(wp + 24, vc12);
    vst1q_f32(wp + 28, vc13);
    // vst1q_f32(wp + 32, vc20);
    // vst1q_f32(wp + 36, vc21);
    // vst1q_f32(wp + 40, vc22);
    // vst1q_f32(wp + 44, vc23);
    // vst1q_f32(wp + 48, vc30);
    // vst1q_f32(wp + 52, vc31);
    // vst1q_f32(wp + 56, vc32);
    // vst1q_f32(wp + 60, vc33);
}

void TensorGEMMInnerKernel1x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t vc30, vc31, vc32, vc33;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        up += 4;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);

	// vc10 = vfmaq_f32(vc10, u1, v0);
    //     vc11 = vfmaq_f32(vc11, u1, v1);
    //     vc12 = vfmaq_f32(vc12, u1, v2);
    //     vc13 = vfmaq_f32(vc13, u1, v3);

    //     vc20 = vfmaq_f32(vc20, u2, v0);
    //     vc21 = vfmaq_f32(vc21, u2, v1);
    //     vc22 = vfmaq_f32(vc22, u2, v2);
    //     vc23 = vfmaq_f32(vc23, u2, v3);

    //     vc30 = vfmaq_f32(vc30, u3, v0);
    //     vc31 = vfmaq_f32(vc31, u3, v1);
    //     vc32 = vfmaq_f32(vc32, u3, v2);
    //     vc33 = vfmaq_f32(vc33, u3, v3);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    // vst1q_f32(wp + 16, vc10);
    // vst1q_f32(wp + 20, vc11);
    // vst1q_f32(wp + 24, vc12);
    // vst1q_f32(wp + 28, vc13);
    // vst1q_f32(wp + 32, vc20);
    // vst1q_f32(wp + 36, vc21);
    // vst1q_f32(wp + 40, vc22);
    // vst1q_f32(wp + 44, vc23);
    // vst1q_f32(wp + 48, vc30);
    // vst1q_f32(wp + 52, vc31);
    // vst1q_f32(wp + 56, vc32);
    // vst1q_f32(wp + 60, vc33);
}

void TensorGEMMInnerKernel4x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03, vc04;
    float32x4_t vc10, vc11, vc12, vc13, vc14;
    float32x4_t vc20, vc21, vc22, vc23, vc24;
    float32x4_t vc30, vc31, vc32, vc33, vc34;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3, v4;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    vc34 = vdupq_n_f32(0.f);

    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        //if(oc == 0){
        //print_floats(vp, 16);
        //print_floats(up, 16);
        //}
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);
        vc24 = vfmaq_f32(vc24, u2, v4);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        vc33 = vfmaq_f32(vc33, u3, v3);
        vc34 = vfmaq_f32(vc34, u3, v4);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc04);

    vst1q_f32(wp + 20, vc10);
    vst1q_f32(wp + 24, vc11);
    vst1q_f32(wp + 28, vc12);
    vst1q_f32(wp + 32, vc13);
    vst1q_f32(wp + 36, vc14);

    vst1q_f32(wp + 40, vc20);
    vst1q_f32(wp + 44, vc21);
    vst1q_f32(wp + 48, vc22);
    vst1q_f32(wp + 52, vc23);
    vst1q_f32(wp + 56, vc24);

    vst1q_f32(wp + 60, vc30);
    vst1q_f32(wp + 64, vc31);
    vst1q_f32(wp + 68, vc32);
    vst1q_f32(wp + 72, vc33);
    vst1q_f32(wp + 76, vc34);
}

void TensorGEMMInnerKernel1x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03, vc04;
    float32x4_t vc10, vc11, vc12, vc13, vc14;
    float32x4_t vc20, vc21, vc22, vc23, vc24;
    float32x4_t vc30, vc31, vc32, vc33, vc34;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3, v4;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    vc34 = vdupq_n_f32(0.f);

    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        //if(oc == 0){
        //print_floats(vp, 16);
        //print_floats(up, 16);
        //}
        u0 = vld1q_f32(up);
        up += 4;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc04);
}

void TensorGEMMInnerKernel2x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03, vc04;
    float32x4_t vc10, vc11, vc12, vc13, vc14;
    float32x4_t vc20, vc21, vc22, vc23, vc24;
    float32x4_t vc30, vc31, vc32, vc33, vc34;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3, v4;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    vc34 = vdupq_n_f32(0.f);

    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        //if(oc == 0){
        //print_floats(vp, 16);
        //print_floats(up, 16);
        //}
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        up += 8;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc04);

    vst1q_f32(wp + 20, vc10);
    vst1q_f32(wp + 24, vc11);
    vst1q_f32(wp + 28, vc12);
    vst1q_f32(wp + 32, vc13);
    vst1q_f32(wp + 36, vc14);
}



void TensorGEMMInnerKernel5x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t vc30, vc31, vc32, vc33;
    float32x4_t vc40, vc41, vc42, vc43;

    float32x4_t u0, u1, u2, u3, u4;
    float32x4_t v0, v1, v2, v3;
    
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);
    vc43 = vdupq_n_f32(0.f);

    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        //if(oc == 0){
        //print_floats(vp, 16);
        //print_floats(up, 16);
        //}
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        up += 20;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        vc33 = vfmaq_f32(vc33, u3, v3);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);
        vc43 = vfmaq_f32(vc43, u4, v3);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);

    vst1q_f32(wp + 16, vc10);
    vst1q_f32(wp + 20, vc11);
    vst1q_f32(wp + 24, vc12);
    vst1q_f32(wp + 28, vc13);

    vst1q_f32(wp + 32, vc20);
    vst1q_f32(wp + 36, vc21);
    vst1q_f32(wp + 40, vc22);
    vst1q_f32(wp + 44, vc23);

    vst1q_f32(wp + 48, vc30);
    vst1q_f32(wp + 52, vc31);
    vst1q_f32(wp + 56, vc32);
    vst1q_f32(wp + 60, vc33);
    
    vst1q_f32(wp + 64, vc40);
    vst1q_f32(wp + 68, vc41);
    vst1q_f32(wp + 72, vc42);
    vst1q_f32(wp + 76, vc43);
}

// Test
void TensorGEMMInnerKernel5x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01, vc02;
    float32x4_t vc10, vc11, vc12;
    float32x4_t vc20, vc21, vc22;
    float32x4_t vc30, vc31, vc32;
    float32x4_t vc40, vc41, vc42;

    float32x4_t u0, u1, u2, u3, u4;
    float32x4_t v0, v1, v2;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        up += 20;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);

    }
    float* wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);

    vst1q_f32(wp + 12, vc10);
    vst1q_f32(wp + 16, vc11);
    vst1q_f32(wp + 20, vc12);

    vst1q_f32(wp + 24, vc20);
    vst1q_f32(wp + 28, vc21);
    vst1q_f32(wp + 32, vc22);

    vst1q_f32(wp + 36, vc30);
    vst1q_f32(wp + 40, vc31);
    vst1q_f32(wp + 44, vc32);

    vst1q_f32(wp + 48, vc40);
    vst1q_f32(wp + 52, vc41);
    vst1q_f32(wp + 56, vc42);
}


void TensorGEMMInnerKernel5x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01;
    float32x4_t vc10, vc11;
    float32x4_t vc20, vc21;
    float32x4_t vc30, vc31;
    float32x4_t vc40, vc41;

    float32x4_t u0, u1, u2, u3, u4;
    float32x4_t v0, v1;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        up += 20;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);

    }
    float* wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);

    vst1q_f32(wp + 8, vc10);
    vst1q_f32(wp + 12, vc11);

    vst1q_f32(wp + 16, vc20);
    vst1q_f32(wp + 20, vc21);

    vst1q_f32(wp + 24, vc30);
    vst1q_f32(wp + 28, vc31);

    vst1q_f32(wp + 32, vc40);
    vst1q_f32(wp + 36, vc41);
}

void TensorGEMMInnerKernel5x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00;
    float32x4_t vc10;
    float32x4_t vc20;
    float32x4_t vc30;
    float32x4_t vc40;

    float32x4_t u0, u1, u2, u3, u4;
    float32x4_t v0;

    vc00 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        up += 20;
        v0 = vld1q_f32(vp);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);

        vc10 = vfmaq_f32(vc10, u1, v0);

        vc20 = vfmaq_f32(vc20, u2, v0);

        vc30 = vfmaq_f32(vc30, u3, v0);

        vc40 = vfmaq_f32(vc40, u4, v0);

    }
    float* wp = WTp;
    vst1q_f32(wp,      vc00);

    vst1q_f32(wp + 4,  vc10);

    vst1q_f32(wp + 8,  vc20);

    vst1q_f32(wp + 12, vc30);

    vst1q_f32(wp + 16, vc40);
}

/*
void TensorGEMMInnerKernel7x3x4(float* WTp, float* UTp, float* vp, int inChannels)
{
    float32x4_t vc00, vc01, vc02;
    float32x4_t vc10, vc11, vc12;
    float32x4_t vc20, vc21, vc22;
    float32x4_t vc30, vc31, vc32;
    float32x4_t vc40, vc41, vc42;
    float32x4_t vc50, vc51, vc52;
    float32x4_t vc60, vc61, vc62;
    
    float32x4_t u0, u1, u2, u3, u4, u5, u6;
    float32x4_t v0, v1, v2;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);
    
    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);
    vc52 = vdupq_n_f32(0.f);
    
    vc60 = vdupq_n_f32(0.f);
    vc61 = vdupq_n_f32(0.f);
    vc62 = vdupq_n_f32(0.f);
    // printf("WTp offset %d\n", WTp - WT);
    //if(oc == 0)
    //      printf("vp offset %d i %d block offset %d depth offset %d\n", vp - (float*)pack_arr, i, (i - start_block_id) * inChannels * depth,d * depth * inChannels);
    // printf("up offset %d\n", up - UT);
    float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        u6 = vld1q_f32(up + 24);
        up += 28;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        vp += 12;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        
	vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);
        
	vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);
        vc52 = vfmaq_f32(vc52, u5, v2);
        
	vc60 = vfmaq_f32(vc60, u6, v0);
        vc61 = vfmaq_f32(vc61, u6, v1);
        vc62 = vfmaq_f32(vc62, u6, v2);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    
    vst1q_f32(wp + 12, vc10);
    vst1q_f32(wp + 16, vc11);
    vst1q_f32(wp + 20, vc12);
    
    vst1q_f32(wp + 24, vc20);
    vst1q_f32(wp + 28, vc21);
    vst1q_f32(wp + 32, vc22);
   
    vst1q_f32(wp + 36, vc30);
    vst1q_f32(wp + 40, vc31);
    vst1q_f32(wp + 44, vc32);

    vst1q_f32(wp + 48, vc40);
    vst1q_f32(wp + 52, vc41);
    vst1q_f32(wp + 56, vc42);

    vst1q_f32(wp + 60, vc50);
    vst1q_f32(wp + 64, vc51);
    vst1q_f32(wp + 68, vc52);
    
    vst1q_f32(wp + 72, vc60);
    vst1q_f32(wp + 76, vc61);
    vst1q_f32(wp + 80, vc62);
}
 */
/* 
void TensorGEMMInnerKernel3x7x4(float* WTp, float* UTp, float* vp, int inChannels)
{
    float32x4_t vc00, vc01, vc02, vc03, vc04, vc05, vc06;
    float32x4_t vc10, vc11, vc12, vc13, vc14, vc15, vc16;
    float32x4_t vc20, vc21, vc22, vc23, vc24, vc25, vc26;
    float32x4_t u0, u1, u2;
    float32x4_t v0, v1, v2, v3, v4, v5, v6;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);
    vc05 = vdupq_n_f32(0.f);
    vc06 = vdupq_n_f32(0.f);
	
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);
    vc15 = vdupq_n_f32(0.f);
    vc16 = vdupq_n_f32(0.f);
    
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);
    vc25 = vdupq_n_f32(0.f);
    vc26 = vdupq_n_f32(0.f);
    float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        up += 12;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        v5 = vld1q_f32(vp + 20);
        v6 = vld1q_f32(vp + 24);
        vp += 28;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);
        vc05 = vfmaq_f32(vc05, u0, v5);
        vc06 = vfmaq_f32(vc06, u0, v6);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);
        vc15 = vfmaq_f32(vc15, u1, v5);
        vc16 = vfmaq_f32(vc16, u1, v6);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);
        vc24 = vfmaq_f32(vc24, u2, v4);
        vc25 = vfmaq_f32(vc25, u2, v5);
        vc26 = vfmaq_f32(vc26, u2, v6);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc04);
    vst1q_f32(wp + 20, vc05);
    vst1q_f32(wp + 24, vc06);

    vst1q_f32(wp + 28, vc10);
    vst1q_f32(wp + 32, vc11);
    vst1q_f32(wp + 36, vc12);
    vst1q_f32(wp + 40, vc13);
    vst1q_f32(wp + 44, vc14);
    vst1q_f32(wp + 48, vc15);
    vst1q_f32(wp + 52, vc16);

    vst1q_f32(wp + 56, vc20);
    vst1q_f32(wp + 60, vc21);
    vst1q_f32(wp + 64, vc22);
    vst1q_f32(wp + 68, vc23);
    vst1q_f32(wp + 72, vc24);
    vst1q_f32(wp + 76, vc25);
    vst1q_f32(wp + 80, vc26);
}
 */


void TensorGEMMInnerKernel2x10x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03, vc04, vc05, vc06, vc07, vc08, vc09;
    float32x4_t vc10, vc11, vc12, vc13, vc14, vc15, vc16, vc17, vc18, vc19;
    float32x4_t u0, u1;
    float32x4_t v0, v1, v2, v3, v4, v5, v6, v7, v8, v9;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);
    vc05 = vdupq_n_f32(0.f);
    vc06 = vdupq_n_f32(0.f);
    vc07 = vdupq_n_f32(0.f);
    vc08 = vdupq_n_f32(0.f);
    vc09 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);
    vc15 = vdupq_n_f32(0.f);
    vc16 = vdupq_n_f32(0.f);
    vc17 = vdupq_n_f32(0.f);
    vc18 = vdupq_n_f32(0.f);
    vc19 = vdupq_n_f32(0.f);

    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        up += 8;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        v5 = vld1q_f32(vp + 20);
        v6 = vld1q_f32(vp + 24);
        v7 = vld1q_f32(vp + 28);
        v8 = vld1q_f32(vp + 32);
        v9 = vld1q_f32(vp + 36);
        vp += 40;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);
        vc05 = vfmaq_f32(vc05, u0, v5);
        vc06 = vfmaq_f32(vc06, u0, v6);
        vc07 = vfmaq_f32(vc07, u0, v7);
        vc08 = vfmaq_f32(vc08, u0, v8);
        vc09 = vfmaq_f32(vc09, u0, v9);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);
        vc15 = vfmaq_f32(vc15, u1, v5);
        vc16 = vfmaq_f32(vc16, u1, v6);
        vc17 = vfmaq_f32(vc17, u1, v7);
        vc18 = vfmaq_f32(vc18, u1, v8);
        vc19 = vfmaq_f32(vc19, u1, v9);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc04);
    vst1q_f32(wp + 20, vc05);
    vst1q_f32(wp + 24, vc06);
    vst1q_f32(wp + 28, vc07);
    vst1q_f32(wp + 32, vc08);
    vst1q_f32(wp + 36, vc09);

    vst1q_f32(wp + 40, vc10);
    vst1q_f32(wp + 44, vc11);
    vst1q_f32(wp + 48, vc12);
    vst1q_f32(wp + 52, vc13);
    vst1q_f32(wp + 56, vc14);
    vst1q_f32(wp + 60, vc15);
    vst1q_f32(wp + 64, vc16);
    vst1q_f32(wp + 68, vc17);
    vst1q_f32(wp + 72, vc18);
    vst1q_f32(wp + 76, vc19);
}


void TensorGEMMInnerKernel7x6x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01, vc02, vc03, vc04, vc05;
    float32x4_t vc10, vc11, vc12, vc13, vc14, vc15;
    float32x4_t vc20, vc21, vc22, vc23, vc24, vc25;
    float32x4_t vc30, vc31, vc32, vc33, vc34, vc35;
    float32x4_t vc40, vc41, vc42, vc43, vc44, vc45;
    float32x4_t vc50, vc51, vc52, vc53, vc54, vc55;
    float32x4_t vc60, vc61, vc62, vc63, vc64, vc65;

    float32x4_t u0, u1, u2, u3, u4, u5, u6;
    float32x4_t v0, v1, v2, v3, v4, v5;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);
    vc05 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);
    vc15 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);
    vc25 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    vc34 = vdupq_n_f32(0.f);
    vc35 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);
    vc43 = vdupq_n_f32(0.f);
    vc44 = vdupq_n_f32(0.f);
    vc45 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);
    vc52 = vdupq_n_f32(0.f);
    vc53 = vdupq_n_f32(0.f);
    vc54 = vdupq_n_f32(0.f);
    vc55 = vdupq_n_f32(0.f);

    vc60 = vdupq_n_f32(0.f);
    vc61 = vdupq_n_f32(0.f);
    vc62 = vdupq_n_f32(0.f);
    vc63 = vdupq_n_f32(0.f);
    vc64 = vdupq_n_f32(0.f);
    vc65 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        u6 = vld1q_f32(up + 24);
        up += 28;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        v5 = vld1q_f32(vp + 20);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);
        vc05 = vfmaq_f32(vc05, u0, v5);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);
        vc15 = vfmaq_f32(vc15, u1, v5);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);
        vc24 = vfmaq_f32(vc24, u2, v4);
        vc25 = vfmaq_f32(vc25, u2, v5);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        vc33 = vfmaq_f32(vc33, u3, v3);
        vc34 = vfmaq_f32(vc34, u3, v4);
        vc35 = vfmaq_f32(vc35, u3, v5);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);
        vc43 = vfmaq_f32(vc43, u4, v3);
        vc44 = vfmaq_f32(vc44, u4, v4);
        vc45 = vfmaq_f32(vc45, u4, v5);

        vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);
        vc52 = vfmaq_f32(vc52, u5, v2);
        vc53 = vfmaq_f32(vc53, u5, v3);
        vc54 = vfmaq_f32(vc54, u5, v4);
        vc55 = vfmaq_f32(vc55, u5, v5);

        vc60 = vfmaq_f32(vc60, u6, v0);
        vc61 = vfmaq_f32(vc61, u6, v1);
        vc62 = vfmaq_f32(vc62, u6, v2);
        vc63 = vfmaq_f32(vc63, u6, v3);
        vc64 = vfmaq_f32(vc64, u6, v4);
        vc65 = vfmaq_f32(vc65, u6, v5);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);
    vst1q_f32(wp + 8,   vc02);
    vst1q_f32(wp + 12,  vc03);
    vst1q_f32(wp + 16,  vc04);
    vst1q_f32(wp + 20,  vc05);

    vst1q_f32(wp + 24,  vc10);
    vst1q_f32(wp + 28,  vc11);
    vst1q_f32(wp + 32,  vc12);
    vst1q_f32(wp + 36,  vc13);
    vst1q_f32(wp + 40,  vc14);
    vst1q_f32(wp + 44,  vc15);

    vst1q_f32(wp + 48,  vc20);
    vst1q_f32(wp + 52,  vc21);
    vst1q_f32(wp + 56,  vc22);
    vst1q_f32(wp + 60,  vc23);
    vst1q_f32(wp + 64,  vc24);
    vst1q_f32(wp + 68,  vc25);

    vst1q_f32(wp + 72,  vc30);
    vst1q_f32(wp + 76,  vc31);
    vst1q_f32(wp + 80,  vc32);
    vst1q_f32(wp + 84,  vc33);
    vst1q_f32(wp + 88,  vc34);
    vst1q_f32(wp + 92,  vc35);

    vst1q_f32(wp + 96,  vc40);
    vst1q_f32(wp + 100, vc41);
    vst1q_f32(wp + 104, vc42);
    vst1q_f32(wp + 108, vc43);
    vst1q_f32(wp + 112, vc44);
    vst1q_f32(wp + 116, vc45);

    vst1q_f32(wp + 120, vc50);
    vst1q_f32(wp + 124, vc51);
    vst1q_f32(wp + 128, vc52);
    vst1q_f32(wp + 132, vc53);
    vst1q_f32(wp + 136, vc54);
    vst1q_f32(wp + 140, vc55);

    vst1q_f32(wp + 144, vc60);
    vst1q_f32(wp + 148, vc61);
    vst1q_f32(wp + 152, vc62);
    vst1q_f32(wp + 156, vc63);
    vst1q_f32(wp + 160, vc64);
    vst1q_f32(wp + 164, vc65);
}

void TensorGEMMInnerKernel7x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01, vc02, vc03, vc04;
    float32x4_t vc10, vc11, vc12, vc13, vc14;
    float32x4_t vc20, vc21, vc22, vc23, vc24;
    float32x4_t vc30, vc31, vc32, vc33, vc34;
    float32x4_t vc40, vc41, vc42, vc43, vc44;
    float32x4_t vc50, vc51, vc52, vc53, vc54;
    float32x4_t vc60, vc61, vc62, vc63, vc64;

    float32x4_t u0, u1, u2, u3, u4, u5, u6;
    float32x4_t v0, v1, v2, v3, v4;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    vc34 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);
    vc43 = vdupq_n_f32(0.f);
    vc44 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);
    vc52 = vdupq_n_f32(0.f);
    vc53 = vdupq_n_f32(0.f);
    vc54 = vdupq_n_f32(0.f);

    vc60 = vdupq_n_f32(0.f);
    vc61 = vdupq_n_f32(0.f);
    vc62 = vdupq_n_f32(0.f);
    vc63 = vdupq_n_f32(0.f);
    vc64 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        u6 = vld1q_f32(up + 24);
        up += 28;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);
        vc24 = vfmaq_f32(vc24, u2, v4);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        vc33 = vfmaq_f32(vc33, u3, v3);
        vc34 = vfmaq_f32(vc34, u3, v4);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);
        vc43 = vfmaq_f32(vc43, u4, v3);
        vc44 = vfmaq_f32(vc44, u4, v4);

        vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);
        vc52 = vfmaq_f32(vc52, u5, v2);
        vc53 = vfmaq_f32(vc53, u5, v3);
        vc54 = vfmaq_f32(vc54, u5, v4);

        vc60 = vfmaq_f32(vc60, u6, v0);
        vc61 = vfmaq_f32(vc61, u6, v1);
        vc62 = vfmaq_f32(vc62, u6, v2);
        vc63 = vfmaq_f32(vc63, u6, v3);
        vc64 = vfmaq_f32(vc64, u6, v4);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);
    vst1q_f32(wp + 8,   vc02);
    vst1q_f32(wp + 12,  vc03);
    vst1q_f32(wp + 16,  vc04);

    vst1q_f32(wp + 20,  vc10);
    vst1q_f32(wp + 24,  vc11);
    vst1q_f32(wp + 28,  vc12);
    vst1q_f32(wp + 32,  vc13);
    vst1q_f32(wp + 36,  vc14);

    vst1q_f32(wp + 40,  vc20);
    vst1q_f32(wp + 44,  vc21);
    vst1q_f32(wp + 48,  vc22);
    vst1q_f32(wp + 52,  vc23);
    vst1q_f32(wp + 56,  vc24);

    vst1q_f32(wp + 60,  vc30);
    vst1q_f32(wp + 64,  vc31);
    vst1q_f32(wp + 68,  vc32);
    vst1q_f32(wp + 72,  vc33);
    vst1q_f32(wp + 76,  vc34);

    vst1q_f32(wp + 80,  vc40);
    vst1q_f32(wp + 84,  vc41);
    vst1q_f32(wp + 88,  vc42);
    vst1q_f32(wp + 92,  vc43);
    vst1q_f32(wp + 96,  vc44);

    vst1q_f32(wp + 100, vc50);
    vst1q_f32(wp + 104, vc51);
    vst1q_f32(wp + 108, vc52);
    vst1q_f32(wp + 112, vc53);
    vst1q_f32(wp + 116, vc54);

    vst1q_f32(wp + 120, vc60);
    vst1q_f32(wp + 124, vc61);
    vst1q_f32(wp + 128, vc62);
    vst1q_f32(wp + 132, vc63);
    vst1q_f32(wp + 136, vc64);
}

void TensorGEMMInnerKernel7x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t vc30, vc31, vc32, vc33;
    float32x4_t vc40, vc41, vc42, vc43;
    float32x4_t vc50, vc51, vc52, vc53;
    float32x4_t vc60, vc61, vc62, vc63;

    float32x4_t u0, u1, u2, u3, u4, u5, u6;
    float32x4_t v0, v1, v2, v3;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);
    vc43 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);
    vc52 = vdupq_n_f32(0.f);
    vc53 = vdupq_n_f32(0.f);

    vc60 = vdupq_n_f32(0.f);
    vc61 = vdupq_n_f32(0.f);
    vc62 = vdupq_n_f32(0.f);
    vc63 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        u6 = vld1q_f32(up + 24);
        up += 28;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        vc33 = vfmaq_f32(vc33, u3, v3);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);
        vc43 = vfmaq_f32(vc43, u4, v3);

        vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);
        vc52 = vfmaq_f32(vc52, u5, v2);
        vc53 = vfmaq_f32(vc53, u5, v3);

        vc60 = vfmaq_f32(vc60, u6, v0);
        vc61 = vfmaq_f32(vc61, u6, v1);
        vc62 = vfmaq_f32(vc62, u6, v2);
        vc63 = vfmaq_f32(vc63, u6, v3);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);
    vst1q_f32(wp + 8,   vc02);
    vst1q_f32(wp + 12,  vc03);

    vst1q_f32(wp + 16,  vc10);
    vst1q_f32(wp + 20,  vc11);
    vst1q_f32(wp + 24,  vc12);
    vst1q_f32(wp + 28,  vc13);

    vst1q_f32(wp + 32,  vc20);
    vst1q_f32(wp + 36,  vc21);
    vst1q_f32(wp + 40,  vc22);
    vst1q_f32(wp + 44,  vc23);

    vst1q_f32(wp + 48,  vc30);
    vst1q_f32(wp + 52,  vc31);
    vst1q_f32(wp + 56,  vc32);
    vst1q_f32(wp + 60,  vc33);

    vst1q_f32(wp + 64,  vc40);
    vst1q_f32(wp + 68,  vc41);
    vst1q_f32(wp + 72,  vc42);
    vst1q_f32(wp + 76,  vc43);

    vst1q_f32(wp + 80,  vc50);
    vst1q_f32(wp + 84,  vc51);
    vst1q_f32(wp + 88,  vc52);
    vst1q_f32(wp + 92,  vc53);

    vst1q_f32(wp + 96,  vc60);
    vst1q_f32(wp + 100, vc61);
    vst1q_f32(wp + 104, vc62);
    vst1q_f32(wp + 108, vc63);
}

void TensorGEMMInnerKernel7x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01, vc02;
    float32x4_t vc10, vc11, vc12;
    float32x4_t vc20, vc21, vc22;
    float32x4_t vc30, vc31, vc32;
    float32x4_t vc40, vc41, vc42;
    float32x4_t vc50, vc51, vc52;
    float32x4_t vc60, vc61, vc62;

    float32x4_t u0, u1, u2, u3, u4, u5, u6;
    float32x4_t v0, v1, v2;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);
    vc52 = vdupq_n_f32(0.f);

    vc60 = vdupq_n_f32(0.f);
    vc61 = vdupq_n_f32(0.f);
    vc62 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        u6 = vld1q_f32(up + 24);
        up += 28;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);

        vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);
        vc52 = vfmaq_f32(vc52, u5, v2);

        vc60 = vfmaq_f32(vc60, u6, v0);
        vc61 = vfmaq_f32(vc61, u6, v1);
        vc62 = vfmaq_f32(vc62, u6, v2);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);
    vst1q_f32(wp + 8,   vc02);

    vst1q_f32(wp + 12,  vc10);
    vst1q_f32(wp + 16,  vc11);
    vst1q_f32(wp + 20,  vc12);

    vst1q_f32(wp + 24,  vc20);
    vst1q_f32(wp + 28,  vc21);
    vst1q_f32(wp + 32,  vc22);

    vst1q_f32(wp + 36,  vc30);
    vst1q_f32(wp + 40,  vc31);
    vst1q_f32(wp + 44,  vc32);

    vst1q_f32(wp + 48,  vc40);
    vst1q_f32(wp + 52, vc41);
    vst1q_f32(wp + 56, vc42);

    vst1q_f32(wp + 60, vc50);
    vst1q_f32(wp + 64, vc51);
    vst1q_f32(wp + 68, vc52);

    vst1q_f32(wp + 72, vc60);
    vst1q_f32(wp + 76, vc61);
    vst1q_f32(wp + 80, vc62);

}

void TensorGEMMInnerKernel7x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01;
    float32x4_t vc10, vc11;
    float32x4_t vc20, vc21;
    float32x4_t vc30, vc31;
    float32x4_t vc40, vc41;
    float32x4_t vc50, vc51;
    float32x4_t vc60, vc61;

    float32x4_t u0, u1, u2, u3, u4, u5, u6;
    float32x4_t v0, v1;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);

    vc60 = vdupq_n_f32(0.f);
    vc61 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        u6 = vld1q_f32(up + 24);
        up += 28;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);

        vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);

        vc60 = vfmaq_f32(vc60, u6, v0);
        vc61 = vfmaq_f32(vc61, u6, v1);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);

    vst1q_f32(wp + 8,   vc10);
    vst1q_f32(wp + 12,  vc11);

    vst1q_f32(wp + 16,  vc20);
    vst1q_f32(wp + 20,  vc21);

    vst1q_f32(wp + 24,  vc30);
    vst1q_f32(wp + 28,  vc31);

    vst1q_f32(wp + 32,  vc40);
    vst1q_f32(wp + 36,  vc41);

    vst1q_f32(wp + 40,  vc50);
    vst1q_f32(wp + 44,  vc51);

    vst1q_f32(wp + 48,  vc60);
    vst1q_f32(wp + 52,  vc61);

}

void TensorGEMMInnerKernel7x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00;
    float32x4_t vc10;
    float32x4_t vc20;
    float32x4_t vc30;
    float32x4_t vc40;
    float32x4_t vc50;
    float32x4_t vc60;

    float32x4_t u0, u1, u2, u3, u4, u5, u6;
    float32x4_t v0;

    vc00 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);

    vc60 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        u6 = vld1q_f32(up + 24);
        up += 28;
        v0 = vld1q_f32(vp);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);

        vc10 = vfmaq_f32(vc10, u1, v0);

        vc20 = vfmaq_f32(vc20, u2, v0);

        vc30 = vfmaq_f32(vc30, u3, v0);

        vc40 = vfmaq_f32(vc40, u4, v0);

        vc50 = vfmaq_f32(vc50, u5, v0);

        vc60 = vfmaq_f32(vc60, u6, v0);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);

    vst1q_f32(wp + 4,   vc10);

    vst1q_f32(wp + 8,   vc20);

    vst1q_f32(wp + 12,  vc30);

    vst1q_f32(wp + 16,  vc40);

    vst1q_f32(wp + 20,  vc50);

    vst1q_f32(wp + 24,  vc60);

}

void TensorGEMMInnerKernel6x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01, vc02, vc03, vc04;
    float32x4_t vc10, vc11, vc12, vc13, vc14;
    float32x4_t vc20, vc21, vc22, vc23, vc24;
    float32x4_t vc30, vc31, vc32, vc33, vc34;
    float32x4_t vc40, vc41, vc42, vc43, vc44;
    float32x4_t vc50, vc51, vc52, vc53, vc54;

    float32x4_t u0, u1, u2, u3, u4, u5;
    float32x4_t v0, v1, v2, v3, v4;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);
    vc34 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);
    vc43 = vdupq_n_f32(0.f);
    vc44 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);
    vc52 = vdupq_n_f32(0.f);
    vc53 = vdupq_n_f32(0.f);
    vc54 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        up += 24;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);
        vc24 = vfmaq_f32(vc24, u2, v4);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        vc33 = vfmaq_f32(vc33, u3, v3);
        vc34 = vfmaq_f32(vc34, u3, v4);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);
        vc43 = vfmaq_f32(vc43, u4, v3);
        vc44 = vfmaq_f32(vc44, u4, v4);

        vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);
        vc52 = vfmaq_f32(vc52, u5, v2);
        vc53 = vfmaq_f32(vc53, u5, v3);
        vc54 = vfmaq_f32(vc54, u5, v4);
    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);
    vst1q_f32(wp + 8,   vc02);
    vst1q_f32(wp + 12,  vc03);
    vst1q_f32(wp + 16,  vc04);

    vst1q_f32(wp + 20,  vc10);
    vst1q_f32(wp + 24,  vc11);
    vst1q_f32(wp + 28,  vc12);
    vst1q_f32(wp + 32,  vc13);
    vst1q_f32(wp + 36,  vc14);

    vst1q_f32(wp + 40,  vc20);
    vst1q_f32(wp + 44,  vc21);
    vst1q_f32(wp + 48,  vc22);
    vst1q_f32(wp + 52,  vc23);
    vst1q_f32(wp + 56,  vc24);

    vst1q_f32(wp + 60,  vc30);
    vst1q_f32(wp + 64,  vc31);
    vst1q_f32(wp + 68,  vc32);
    vst1q_f32(wp + 72,  vc33);
    vst1q_f32(wp + 76,  vc34);

    vst1q_f32(wp + 80,  vc40);
    vst1q_f32(wp + 84,  vc41);
    vst1q_f32(wp + 88,  vc42);
    vst1q_f32(wp + 92,  vc43);
    vst1q_f32(wp + 96,  vc44);

    vst1q_f32(wp + 100, vc50);
    vst1q_f32(wp + 104, vc51);
    vst1q_f32(wp + 108, vc52);
    vst1q_f32(wp + 112, vc53);
    vst1q_f32(wp + 116, vc54);
}

void TensorGEMMInnerKernel6x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t vc30, vc31, vc32, vc33;
    float32x4_t vc40, vc41, vc42, vc43;
    float32x4_t vc50, vc51, vc52, vc53;

    float32x4_t u0, u1, u2, u3, u4, u5;
    float32x4_t v0, v1, v2, v3;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);
    vc33 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);
    vc43 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);
    vc52 = vdupq_n_f32(0.f);
    vc53 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        up += 24;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);
        vc33 = vfmaq_f32(vc33, u3, v3);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);
        vc43 = vfmaq_f32(vc43, u4, v3);

        vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);
        vc52 = vfmaq_f32(vc52, u5, v2);
        vc53 = vfmaq_f32(vc53, u5, v3);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);
    vst1q_f32(wp + 8,   vc02);
    vst1q_f32(wp + 12,  vc03);

    vst1q_f32(wp + 16,  vc10);
    vst1q_f32(wp + 20,  vc11);
    vst1q_f32(wp + 24,  vc12);
    vst1q_f32(wp + 28,  vc13);

    vst1q_f32(wp + 32,  vc20);
    vst1q_f32(wp + 36,  vc21);
    vst1q_f32(wp + 40,  vc22);
    vst1q_f32(wp + 44,  vc23);

    vst1q_f32(wp + 48,  vc30);
    vst1q_f32(wp + 52,  vc31);
    vst1q_f32(wp + 56,  vc32);
    vst1q_f32(wp + 60,  vc33);

    vst1q_f32(wp + 64,  vc40);
    vst1q_f32(wp + 68,  vc41);
    vst1q_f32(wp + 72,  vc42);
    vst1q_f32(wp + 76,  vc43);

    vst1q_f32(wp + 80,  vc50);
    vst1q_f32(wp + 84,  vc51);
    vst1q_f32(wp + 88,  vc52);
    vst1q_f32(wp + 92,  vc53);
}

void TensorGEMMInnerKernel6x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01, vc02;
    float32x4_t vc10, vc11, vc12;
    float32x4_t vc20, vc21, vc22;
    float32x4_t vc30, vc31, vc32;
    float32x4_t vc40, vc41, vc42;
    float32x4_t vc50, vc51, vc52;

    float32x4_t u0, u1, u2, u3, u4, u5;
    float32x4_t v0, v1, v2;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);
    vc32 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);
    vc42 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);
    vc52 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        up += 24;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);
        vc32 = vfmaq_f32(vc32, u3, v2);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);
        vc42 = vfmaq_f32(vc42, u4, v2);

        vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);
        vc52 = vfmaq_f32(vc52, u5, v2);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);
    vst1q_f32(wp + 8,   vc02);

    vst1q_f32(wp + 12,  vc10);
    vst1q_f32(wp + 16,  vc11);
    vst1q_f32(wp + 20,  vc12);

    vst1q_f32(wp + 24,  vc20);
    vst1q_f32(wp + 28,  vc21);
    vst1q_f32(wp + 32,  vc22);

    vst1q_f32(wp + 36,  vc30);
    vst1q_f32(wp + 40,  vc31);
    vst1q_f32(wp + 44,  vc32);

    vst1q_f32(wp + 48,  vc40);
    vst1q_f32(wp + 52, vc41);
    vst1q_f32(wp + 56, vc42);

    vst1q_f32(wp + 60, vc50);
    vst1q_f32(wp + 64, vc51);
    vst1q_f32(wp + 68, vc52);

}

void TensorGEMMInnerKernel6x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01;
    float32x4_t vc10, vc11;
    float32x4_t vc20, vc21;
    float32x4_t vc30, vc31;
    float32x4_t vc40, vc41;
    float32x4_t vc50, vc51;

    float32x4_t u0, u1, u2, u3, u4, u5;
    float32x4_t v0, v1;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);
    vc31 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);
    vc41 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);
    vc51 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        up += 24;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);

        vc30 = vfmaq_f32(vc30, u3, v0);
        vc31 = vfmaq_f32(vc31, u3, v1);

        vc40 = vfmaq_f32(vc40, u4, v0);
        vc41 = vfmaq_f32(vc41, u4, v1);

        vc50 = vfmaq_f32(vc50, u5, v0);
        vc51 = vfmaq_f32(vc51, u5, v1);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);

    vst1q_f32(wp + 8,   vc10);
    vst1q_f32(wp + 12,  vc11);

    vst1q_f32(wp + 16,  vc20);
    vst1q_f32(wp + 20,  vc21);

    vst1q_f32(wp + 24,  vc30);
    vst1q_f32(wp + 28,  vc31);

    vst1q_f32(wp + 32,  vc40);
    vst1q_f32(wp + 36,  vc41);

    vst1q_f32(wp + 40,  vc50);
    vst1q_f32(wp + 44,  vc51);

}

void TensorGEMMInnerKernel6x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00;
    float32x4_t vc10;
    float32x4_t vc20;
    float32x4_t vc30;
    float32x4_t vc40;
    float32x4_t vc50;

    float32x4_t u0, u1, u2, u3, u4, u5, u6;
    float32x4_t v0;

    vc00 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);

    vc30 = vdupq_n_f32(0.f);

    vc40 = vdupq_n_f32(0.f);

    vc50 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        u4 = vld1q_f32(up + 16);
        u5 = vld1q_f32(up + 20);
        up += 24;
        v0 = vld1q_f32(vp);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);

        vc10 = vfmaq_f32(vc10, u1, v0);

        vc20 = vfmaq_f32(vc20, u2, v0);

        vc30 = vfmaq_f32(vc30, u3, v0);

        vc40 = vfmaq_f32(vc40, u4, v0);

        vc50 = vfmaq_f32(vc50, u5, v0);
    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);

    vst1q_f32(wp + 4,   vc10);

    vst1q_f32(wp + 8,   vc20);

    vst1q_f32(wp + 12,  vc30);

    vst1q_f32(wp + 16,  vc40);

    vst1q_f32(wp + 20,  vc50);

}

void TensorGEMMInnerKernel3x7x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03, vc04, vc05, vc06;
    float32x4_t vc10, vc11, vc12, vc13, vc14, vc15, vc16;
    float32x4_t vc20, vc21, vc22, vc23, vc24, vc25, vc26;
    float32x4_t u0, u1, u2;
    float32x4_t v0, v1, v2, v3, v4, v5, v6;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);
    vc05 = vdupq_n_f32(0.f);
    vc06 = vdupq_n_f32(0.f);
	
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);
    vc15 = vdupq_n_f32(0.f);
    vc16 = vdupq_n_f32(0.f);
    
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);
    vc25 = vdupq_n_f32(0.f);
    vc26 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        up += 12;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        v5 = vld1q_f32(vp + 20);
        v6 = vld1q_f32(vp + 24);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);
        vc05 = vfmaq_f32(vc05, u0, v5);
        vc06 = vfmaq_f32(vc06, u0, v6);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);
        vc15 = vfmaq_f32(vc15, u1, v5);
        vc16 = vfmaq_f32(vc16, u1, v6);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);
        vc24 = vfmaq_f32(vc24, u2, v4);
        vc25 = vfmaq_f32(vc25, u2, v5);
        vc26 = vfmaq_f32(vc26, u2, v6);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc04);
    vst1q_f32(wp + 20, vc05);
    vst1q_f32(wp + 24, vc06);

    vst1q_f32(wp + 28, vc10);
    vst1q_f32(wp + 32, vc11);
    vst1q_f32(wp + 36, vc12);
    vst1q_f32(wp + 40, vc13);
    vst1q_f32(wp + 44, vc14);
    vst1q_f32(wp + 48, vc15);
    vst1q_f32(wp + 52, vc16);

    vst1q_f32(wp + 56, vc20);
    vst1q_f32(wp + 60, vc21);
    vst1q_f32(wp + 64, vc22);
    vst1q_f32(wp + 68, vc23);
    vst1q_f32(wp + 72, vc24);
    vst1q_f32(wp + 76, vc25);
    vst1q_f32(wp + 80, vc26);
}


void TensorGEMMInnerKernel3x6x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03, vc04, vc05;
    float32x4_t vc10, vc11, vc12, vc13, vc14, vc15;
    float32x4_t vc20, vc21, vc22, vc23, vc24, vc25;
    float32x4_t u0, u1, u2;
    float32x4_t v0, v1, v2, v3, v4, v5;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);
    vc05 = vdupq_n_f32(0.f);
	
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);
    vc15 = vdupq_n_f32(0.f);
    
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);
    vc25 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        up += 12;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        v5 = vld1q_f32(vp + 20);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);
        vc05 = vfmaq_f32(vc05, u0, v5);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);
        vc15 = vfmaq_f32(vc15, u1, v5);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);
        vc24 = vfmaq_f32(vc24, u2, v4);
        vc25 = vfmaq_f32(vc25, u2, v5);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc04);
    vst1q_f32(wp + 20, vc05);

    vst1q_f32(wp + 24, vc10);
    vst1q_f32(wp + 28, vc11);
    vst1q_f32(wp + 32, vc12);
    vst1q_f32(wp + 36, vc13);
    vst1q_f32(wp + 40, vc14);
    vst1q_f32(wp + 44, vc15);

    vst1q_f32(wp + 48, vc20);
    vst1q_f32(wp + 52, vc21);
    vst1q_f32(wp + 56, vc22);
    vst1q_f32(wp + 60, vc23);
    vst1q_f32(wp + 64, vc24);
    vst1q_f32(wp + 68, vc25);
}


void TensorGEMMInnerKernel3x5x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03, vc04;
    float32x4_t vc10, vc11, vc12, vc13, vc14;
    float32x4_t vc20, vc21, vc22, vc23, vc24;
    float32x4_t u0, u1, u2;
    float32x4_t v0, v1, v2, v3, v4;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
    vc04 = vdupq_n_f32(0.f);
	
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    vc14 = vdupq_n_f32(0.f);
    
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    vc24 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        up += 12;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
        vc14 = vfmaq_f32(vc14, u1, v4);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);
        vc24 = vfmaq_f32(vc24, u2, v4);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);
    vst1q_f32(wp + 16, vc04);

    vst1q_f32(wp + 20, vc10);
    vst1q_f32(wp + 24, vc11);
    vst1q_f32(wp + 28, vc12);
    vst1q_f32(wp + 32, vc13);
    vst1q_f32(wp + 36, vc14);

    vst1q_f32(wp + 40, vc20);
    vst1q_f32(wp + 44, vc21);
    vst1q_f32(wp + 48, vc22);
    vst1q_f32(wp + 52, vc23);
    vst1q_f32(wp + 56, vc24);
}


void TensorGEMMInnerKernel3x4x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t u0, u1, u2;
    float32x4_t v0, v1, v2, v3;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
    vc03 = vdupq_n_f32(0.f);
	
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    vc13 = vdupq_n_f32(0.f);
    
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    vc23 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        up += 12;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
        vc23 = vfmaq_f32(vc23, u2, v3);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);
    vst1q_f32(wp + 12, vc03);

    vst1q_f32(wp + 16, vc10);
    vst1q_f32(wp + 20, vc11);
    vst1q_f32(wp + 24, vc12);
    vst1q_f32(wp + 28, vc13);

    vst1q_f32(wp + 32, vc20);
    vst1q_f32(wp + 36, vc21);
    vst1q_f32(wp + 40, vc22);
    vst1q_f32(wp + 44, vc23);
}

void TensorGEMMInnerKernel3x3x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride)
{
    float32x4_t vc00, vc01, vc02;
    float32x4_t vc10, vc11, vc12;
    float32x4_t vc20, vc21, vc22;
    float32x4_t u0, u1, u2;
    float32x4_t v0, v1, v2;
    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);
    vc02 = vdupq_n_f32(0.f);
	
    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);
    vc12 = vdupq_n_f32(0.f);
    
    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);
    vc22 = vdupq_n_f32(0.f);
    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        up += 12;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        vp += wStride;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);
        vc22 = vfmaq_f32(vc22, u2, v2);
    }
    float *wp = WTp;
    vst1q_f32(wp,      vc00);
    vst1q_f32(wp + 4,  vc01);
    vst1q_f32(wp + 8,  vc02);

    vst1q_f32(wp + 12, vc10);
    vst1q_f32(wp + 16, vc11);
    vst1q_f32(wp + 20, vc12);

    vst1q_f32(wp + 24, vc20);
    vst1q_f32(wp + 28, vc21);
    vst1q_f32(wp + 32, vc22);

}


void TensorGEMMInnerKernel3x2x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00, vc01;
    float32x4_t vc10, vc11;
    float32x4_t vc20, vc21;

    float32x4_t u0, u1, u2;
    float32x4_t v0, v1;

    vc00 = vdupq_n_f32(0.f);
    vc01 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);
    vc11 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);
    vc21 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        up += 12;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);

        vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);

        vc20 = vfmaq_f32(vc20, u2, v0);
        vc21 = vfmaq_f32(vc21, u2, v1);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc01);

    vst1q_f32(wp + 8,   vc10);
    vst1q_f32(wp + 12,  vc11);

    vst1q_f32(wp + 16,  vc20);
    vst1q_f32(wp + 20,  vc21);

}

void TensorGEMMInnerKernel3x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00;
    float32x4_t vc10;
    float32x4_t vc20;

    float32x4_t u0, u1, u2;
    float32x4_t v0;

    vc00 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);

    vc20 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        up += 12;
        v0 = vld1q_f32(vp);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);

        vc10 = vfmaq_f32(vc10, u1, v0);

        vc20 = vfmaq_f32(vc20, u2, v0);
    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);

    vst1q_f32(wp + 4,   vc10);

    vst1q_f32(wp + 8,   vc20);

}







void TensorGEMMInnerKernel2x1x4(float* WTp, const float* UTp, float* vp, int inChannels, int wStride) {
    float32x4_t vc00;
    float32x4_t vc10;

    float32x4_t u0, u1;
    float32x4_t v0;

    vc00 = vdupq_n_f32(0.f);

    vc10 = vdupq_n_f32(0.f);

    const float* up = UTp;
    for(int ic = 0; ic < inChannels; ++ic) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        up += 8;
        v0 = vld1q_f32(vp);
        vp += wStride;

        vc00 = vfmaq_f32(vc00, u0, v0);

        vc10 = vfmaq_f32(vc10, u1, v0);

    }
    float* wp = WTp;
    vst1q_f32(wp,       vc00);
    vst1q_f32(wp + 4,   vc10);
}