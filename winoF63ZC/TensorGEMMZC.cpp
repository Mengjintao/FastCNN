#include <stdlib.h>
#include <arm_neon.h>
#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "./TensorGEMMZC.h"

void TensorGEMMInnerKernel4x4x4_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t vc30, vc31, vc32, vc33;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3;
    int step = 4 * tileN;
    float *wp0  = WTp;
    float *wp1  = wp0  + 4;
    float *wp2  = wp1  + 4;
    float *wp3  = wp2  + 4;
    float *wp4  = wp0  + 1 * step;
    float *wp5  = wp4  + 4;
    float *wp6  = wp5  + 4;
    float *wp7  = wp6  + 4;
    float *wp8  = wp0  + 2 * step;
    float *wp9  = wp8  + 4;
    float *wp10 = wp9  + 4;
    float *wp11 = wp10 + 4;
    float *wp12 = wp0  + 3 * step;
    float *wp13 = wp12 + 4;
    float *wp14 = wp13 + 4;
    float *wp15 = wp14 + 4;

    vc00 = vld1q_f32(wp0);
    vc01 = vld1q_f32(wp1);
    vc02 = vld1q_f32(wp2);
    vc03 = vld1q_f32(wp3);
    vc10 = vld1q_f32(wp4);
    vc11 = vld1q_f32(wp5);
    vc12 = vld1q_f32(wp6);
    vc13 = vld1q_f32(wp7);
    vc20 = vld1q_f32(wp8);
    vc21 = vld1q_f32(wp9);
    vc22 = vld1q_f32(wp10);
    vc23 = vld1q_f32(wp11);
    vc30 = vld1q_f32(wp12);
    vc31 = vld1q_f32(wp13);
    vc32 = vld1q_f32(wp14);
    vc33 = vld1q_f32(wp15);

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
        vp += 16;
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
    
    vst1q_f32(wp0,  vc00);
    vst1q_f32(wp1,  vc01);
    vst1q_f32(wp2,  vc02);
    vst1q_f32(wp3,  vc03);
    vst1q_f32(wp4,  vc10);
    vst1q_f32(wp5,  vc11);
    vst1q_f32(wp6,  vc12);
    vst1q_f32(wp7,  vc13);
    vst1q_f32(wp8,  vc20);
    vst1q_f32(wp9,  vc21);
    vst1q_f32(wp10, vc22);
    vst1q_f32(wp11, vc23);
    vst1q_f32(wp12, vc30);
    vst1q_f32(wp13, vc31);
    vst1q_f32(wp14, vc32);
    vst1q_f32(wp15, vc33);
}

void TensorGEMMInnerKernel4x3x4_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3;
    int step = 4 * tileN;
    float *wp0  = WTp;
    float *wp1  = wp0  + 4;
    float *wp2  = wp1  + 4;
    float *wp3  = wp2  + 4;
    float *wp4  = wp0  + 1 * step;
    float *wp5  = wp4  + 4;
    float *wp6  = wp5  + 4;
    float *wp7  = wp6  + 4;
    float *wp8  = wp0  + 2 * step;
    float *wp9  = wp8  + 4;
    float *wp10 = wp9  + 4;
    float *wp11 = wp10 + 4;

    vc00 = vld1q_f32(wp0);
    vc01 = vld1q_f32(wp1);
    vc02 = vld1q_f32(wp2);
    vc03 = vld1q_f32(wp3);
    vc10 = vld1q_f32(wp4);
    vc11 = vld1q_f32(wp5);
    vc12 = vld1q_f32(wp6);
    vc13 = vld1q_f32(wp7);
    vc20 = vld1q_f32(wp8);
    vc21 = vld1q_f32(wp9);
    vc22 = vld1q_f32(wp10);
    vc23 = vld1q_f32(wp11);

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
        vp += 16;
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
    
    vst1q_f32(wp0,  vc00);
    vst1q_f32(wp1,  vc01);
    vst1q_f32(wp2,  vc02);
    vst1q_f32(wp3,  vc03);
    vst1q_f32(wp4,  vc10);
    vst1q_f32(wp5,  vc11);
    vst1q_f32(wp6,  vc12);
    vst1q_f32(wp7,  vc13);
    vst1q_f32(wp8,  vc20);
    vst1q_f32(wp9,  vc21);
    vst1q_f32(wp10, vc22);
    vst1q_f32(wp11, vc23);
}

void TensorGEMMInnerKernel4x2x4_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;

    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3;
    int step = 4 * tileN;
    float *wp0  = WTp;
    float *wp1  = wp0  + 4;
    float *wp2  = wp1  + 4;
    float *wp3  = wp2  + 4;
    float *wp4  = wp0  + 1 * step;
    float *wp5  = wp4  + 4;
    float *wp6  = wp5  + 4;
    float *wp7  = wp6  + 4;

    vc00 = vld1q_f32(wp0);
    vc01 = vld1q_f32(wp1);
    vc02 = vld1q_f32(wp2);
    vc03 = vld1q_f32(wp3);
    vc10 = vld1q_f32(wp4);
    vc11 = vld1q_f32(wp5);
    vc12 = vld1q_f32(wp6);
    vc13 = vld1q_f32(wp7);

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
        vp += 16;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);

	    vc10 = vfmaq_f32(vc10, u1, v0);
        vc11 = vfmaq_f32(vc11, u1, v1);
        vc12 = vfmaq_f32(vc12, u1, v2);
        vc13 = vfmaq_f32(vc13, u1, v3);
    }
    
    vst1q_f32(wp0,  vc00);
    vst1q_f32(wp1,  vc01);
    vst1q_f32(wp2,  vc02);
    vst1q_f32(wp3,  vc03);
    vst1q_f32(wp4,  vc10);
    vst1q_f32(wp5,  vc11);
    vst1q_f32(wp6,  vc12);
    vst1q_f32(wp7,  vc13);
}

void TensorGEMMInnerKernel4x1x4_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN)
{
    float32x4_t vc00, vc01, vc02, vc03;

    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3;
    int step = 4 * tileN;
    float *wp0  = WTp;
    float *wp1  = wp0  + 4;
    float *wp2  = wp1  + 4;
    float *wp3  = wp2  + 4;

    vc00 = vld1q_f32(wp0);
    vc01 = vld1q_f32(wp1);
    vc02 = vld1q_f32(wp2);
    vc03 = vld1q_f32(wp3);

    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        up += 4;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += 16;
        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
    }
    
    vst1q_f32(wp0,  vc00);
    vst1q_f32(wp1,  vc01);
    vst1q_f32(wp2,  vc02);
    vst1q_f32(wp3,  vc03);
}

void TensorGEMMInnerKernel4x4x4_v2(float* WTp, const float* UTp, float* vp, int inChannels, int tileN)
{
    float32x4_t vc00, vc01, vc02, vc03;
    float32x4_t vc10, vc11, vc12, vc13;
    float32x4_t vc20, vc21, vc22, vc23;
    float32x4_t vc30, vc31, vc32, vc33;
    float32x4_t u0, u1, u2, u3, u4, u5, u6, u7;
    float32x4_t v0, v1, v2, v3, v4, v5, v6, v7;

    int step = 4 * tileN;
    float *wp0  = WTp;
    float *wp1  = wp0  + 4;
    float *wp2  = wp1  + 4;
    float *wp3  = wp2  + 4;
    float *wp4  = wp0  + 1 * step;
    float *wp5  = wp4  + 4;
    float *wp6  = wp5  + 4;
    float *wp7  = wp6  + 4;
    float *wp8  = wp0  + 2 * step;
    float *wp9  = wp8  + 4;
    float *wp10 = wp9  + 4;
    float *wp11 = wp10 + 4;
    float *wp12 = wp0  + 3 * step;
    float *wp13 = wp12 + 4;
    float *wp14 = wp13 + 4;
    float *wp15 = wp14 + 4;

    vc00 = vld1q_f32(wp0);
    vc01 = vld1q_f32(wp1);
    vc02 = vld1q_f32(wp2);
    vc03 = vld1q_f32(wp3);
    vc10 = vld1q_f32(wp4);
    vc11 = vld1q_f32(wp5);
    vc12 = vld1q_f32(wp6);
    vc13 = vld1q_f32(wp7);
    vc20 = vld1q_f32(wp8);
    vc21 = vld1q_f32(wp9);
    vc22 = vld1q_f32(wp10);
    vc23 = vld1q_f32(wp11);
    vc30 = vld1q_f32(wp12);
    vc31 = vld1q_f32(wp13);
    vc32 = vld1q_f32(wp14);
    vc33 = vld1q_f32(wp15);

    const float *up = UTp;
    int icStep = inChannels / 2;
    int icRmd  = inChannels % 2;

    if (icStep > 0 || icRmd > 0) {
        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += 16; 
    }

    for (int ic = 0; ic < icStep; ++ic) {
        u4 = vld1q_f32(up);
        u5 = vld1q_f32(up + 4);
        u6 = vld1q_f32(up + 8);
        u7 = vld1q_f32(up + 12);
        up += 16;
        v4 = vld1q_f32(vp);
        v5 = vld1q_f32(vp + 4);
        v6 = vld1q_f32(vp + 8);
        v7 = vld1q_f32(vp + 12);
        vp += 16;

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

        u0 = vld1q_f32(up);
        u1 = vld1q_f32(up + 4);
        u2 = vld1q_f32(up + 8);
        u3 = vld1q_f32(up + 12);
        up += 16;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        vp += 16;

        vc00 = vfmaq_f32(vc00, u4, v4);
        vc01 = vfmaq_f32(vc01, u4, v5);
        vc02 = vfmaq_f32(vc02, u4, v6);
        vc03 = vfmaq_f32(vc03, u4, v7);

	    vc10 = vfmaq_f32(vc10, u5, v4);
        vc11 = vfmaq_f32(vc11, u5, v5);
        vc12 = vfmaq_f32(vc12, u5, v6);
        vc13 = vfmaq_f32(vc13, u5, v7);

        vc20 = vfmaq_f32(vc20, u6, v4);
        vc21 = vfmaq_f32(vc21, u6, v5);
        vc22 = vfmaq_f32(vc22, u6, v6);
        vc23 = vfmaq_f32(vc23, u6, v7);

        vc30 = vfmaq_f32(vc30, u7, v4);
        vc31 = vfmaq_f32(vc31, u7, v5);
        vc32 = vfmaq_f32(vc32, u7, v6);
        vc33 = vfmaq_f32(vc33, u7, v7);
    }

    for (int ic = 0; ic < icRmd; ++ic)
    {
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
    
    vst1q_f32(wp0,  vc00);
    vst1q_f32(wp1,  vc01);
    vst1q_f32(wp2,  vc02);
    vst1q_f32(wp3,  vc03);
    vst1q_f32(wp4,  vc10);
    vst1q_f32(wp5,  vc11);
    vst1q_f32(wp6,  vc12);
    vst1q_f32(wp7,  vc13);
    vst1q_f32(wp8,  vc20);
    vst1q_f32(wp9,  vc21);
    vst1q_f32(wp10, vc22);
    vst1q_f32(wp11, vc23);
    vst1q_f32(wp12, vc30);
    vst1q_f32(wp13, vc31);
    vst1q_f32(wp14, vc32);
    vst1q_f32(wp15, vc33);
}

void TensorGEMMInnerKernel4x4x4_v3(float* WTp, const float* UTp, float* vp, int inChannels, int tileN)
{

	const float* up = UTp;
    int step = 4 * tileN;
    float *wp0 = WTp;
    float *wp1 = wp0 + 1 * step;
    float *wp2 = wp0 + 2 * step;
    float *wp3 = wp0 + 3 * step;
	asm volatile (
		// "movi	v16.2d, #0000000000000000	\n"
		// "movi	v17.2d, #0000000000000000	\n"
		// "movi	v18.2d, #0000000000000000	\n"
		// "movi	v19.2d, #0000000000000000	\n"
		// "movi	v20.2d, #0000000000000000	\n"
		// "movi	v21.2d, #0000000000000000	\n"
		// "movi	v22.2d, #0000000000000000	\n"
		// "movi	v23.2d, #0000000000000000	\n"
		// "movi	v24.2d, #0000000000000000	\n"
		// "movi	v25.2d, #0000000000000000	\n"
		// "movi	v26.2d, #0000000000000000	\n"
		// "movi	v27.2d, #0000000000000000	\n"
		// "movi	v28.2d, #0000000000000000	\n"
		// "movi	v29.2d, #0000000000000000	\n"
		// "movi	v30.2d, #0000000000000000	\n"
		// "movi	v31.2d, #0000000000000000	\n"

        "ld1	{v16.4s, v17.4s, v18.4s, v19.4s}, [%[wp0]]   	\n"
		"ld1	{v20.4s, v21.4s, v22.4s, v23.4s}, [%[wp1]]      \n"
		"ld1	{v24.4s, v25.4s, v26.4s, v27.4s}, [%[wp2]]	    \n"
		"ld1	{v28.4s, v29.4s, v30.4s, v31.4s}, [%[wp3]]      \n"

		"mov	x4, %[ic]		\n"

		"cmp	w4, #0			\n"
		"beq	end			    \n"

		"ld1	{v0.4s, v1.4s, v2.4s, v3.4s}, 	[%[up]], #64	\n"
		"ld1	{v8.4s, v9.4s, v10.4s, v11.4s}, [%[vp]], #64	\n"

		"lsr	x8, x4, 0x1		\n"
		"cmp	w8, #0			\n"
		"beq 	loop2_end		\n"

	"loop2:					    \n"
		"subs	w8, w8, 0x1		\n"

		"prfm 	pldl1keep, [%[up], #512]			\n"
		"ld1	{v4.4s, v5.4s, v6.4s, v7.4s}, [%[up]], #64	\n"
		"fmla	v16.4s, v0.4s, v8.4s	\n"
		"fmla	v17.4s, v0.4s, v9.4s	\n"
		"fmla	v18.4s, v0.4s, v10.4s	\n"
		"fmla	v19.4s, v0.4s, v11.4s	\n"

		"fmla	v20.4s, v1.4s, v8.4s	\n"
		"fmla	v21.4s, v1.4s, v9.4s	\n"
		"fmla	v22.4s, v1.4s, v10.4s	\n"
		"fmla	v23.4s, v1.4s, v11.4s	\n"

		"prfm 	pldl1keep, [%[vp], #256]			\n"
		"ld1	{v12.4s, v13.4s, v14.4s, v15.4s}, [%[vp]], #64\n"
		"fmla	v24.4s, v2.4s, v8.4s	\n"
		"fmla	v25.4s, v2.4s, v9.4s	\n"
		"fmla	v26.4s, v2.4s, v10.4s	\n"
		"fmla	v27.4s, v2.4s, v11.4s	\n"

		"fmla	v28.4s, v3.4s, v8.4s	\n"
		"fmla	v29.4s, v3.4s, v9.4s	\n"
		"fmla	v30.4s, v3.4s, v10.4s	\n"
		"fmla	v31.4s, v3.4s, v11.4s	\n"

		"ld1	{v0.4s, v1.4s, v2.4s, v3.4s}, [%[up]], #64	\n"
		"fmla	v16.4s, v4.4s, v12.4s	\n"
		"fmla	v17.4s, v4.4s, v13.4s	\n"
		"fmla	v18.4s, v4.4s, v14.4s	\n"
		"fmla	v19.4s, v4.4s, v15.4s	\n"

		"fmla	v20.4s, v5.4s, v12.4s	\n"
		"fmla	v21.4s, v5.4s, v13.4s	\n"
		"fmla	v22.4s, v5.4s, v14.4s	\n"
		"fmla	v23.4s, v5.4s, v15.4s	\n"
		"ld1	{v8.4s, v9.4s, v10.4s, v11.4s}, [%[vp]], #64 \n"

		"fmla	v24.4s, v6.4s, v12.4s	\n"
		"fmla	v25.4s, v6.4s, v13.4s	\n"
		"fmla	v26.4s, v6.4s, v14.4s	\n"
		"fmla	v27.4s, v6.4s, v15.4s	\n"

		"fmla	v28.4s, v7.4s, v12.4s	\n"
		"fmla	v29.4s, v7.4s, v13.4s	\n"
		"fmla	v30.4s, v7.4s, v14.4s	\n"
		"fmla	v31.4s, v7.4s, v15.4s	\n"

		"bne	loop2			\n"

	"loop2_end:				\n"
		"ands	x9, x4, 0x1		\n"
		"beq	save_result		\n"
		
	"loop1:					\n"	
		"fmla	v16.4s, v0.4s, v8.4s	\n"
		"fmla	v17.4s, v0.4s, v9.4s	\n"
		"fmla	v18.4s, v0.4s, v10.4s	\n"
		"fmla	v19.4s, v0.4s, v11.4s	\n"

		"fmla	v20.4s, v1.4s, v8.4s	\n"
		"fmla	v21.4s, v1.4s, v9.4s	\n"
		"fmla	v22.4s, v1.4s, v10.4s	\n"
		"fmla	v23.4s, v1.4s, v11.4s	\n"

		"fmla	v24.4s, v2.4s, v8.4s	\n"
		"fmla	v25.4s, v2.4s, v9.4s	\n"
		"fmla	v26.4s, v2.4s, v10.4s	\n"
		"fmla	v27.4s, v2.4s, v11.4s	\n"

		"fmla	v28.4s, v3.4s, v8.4s	\n"
		"fmla	v29.4s, v3.4s, v9.4s	\n"
		"fmla	v30.4s, v3.4s, v10.4s	\n"
		"fmla	v31.4s, v3.4s, v11.4s	\n"

	"save_result:				\n"
		"st1	{v16.4s, v17.4s, v18.4s, v19.4s}, [%[wp0]]   	\n"
		"st1	{v20.4s, v21.4s, v22.4s, v23.4s}, [%[wp1]]      \n"
		"st1	{v24.4s, v25.4s, v26.4s, v27.4s}, [%[wp2]]  	\n"
		"st1	{v28.4s, v29.4s, v30.4s, v31.4s}, [%[wp3]]      \n"
	
	"end:								\n"

		:[up]"+r"(up),
		 [vp]"+r"(vp),
		 [wp0]"+r"(wp0),
		 [wp1]"+r"(wp1),
		 [wp2]"+r"(wp2),
		 [wp3]"+r"(wp3),
		 [ic]"+r"(inChannels)
		:
		: "cc", "memory", "x4", "x8", "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", 
        		"v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        		"v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

}

void TensorGEMMInnerKernel4x4x5_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN)
{
    float32x4_t vc00, vc01, vc02, vc03, vc04;
    float32x4_t vc10, vc11, vc12, vc13, vc14;
    float32x4_t vc20, vc21, vc22, vc23, vc24;
    float32x4_t vc30, vc31, vc32, vc33, vc34;
    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3, v4;

    int step = 4 * tileN;
    float *wp0  = WTp;
    float *wp1  = wp0  + 4;
    float *wp2  = wp1  + 4;
    float *wp3  = wp2  + 4;
    float *wp4  = wp3  + 4;
    float *wp5  = wp0  + 1 * step;
    float *wp6  = wp5  + 4;
    float *wp7  = wp6  + 4;
    float *wp8  = wp7  + 4;
    float *wp9  = wp8  + 4;
    float *wp10 = wp0  + 2 * step;
    float *wp11 = wp10 + 4;
    float *wp12 = wp11 + 4;
    float *wp13 = wp12 + 4;
    float *wp14 = wp13 + 4;
    float *wp15 = wp0  + 3 * step;
    float *wp16 = wp15 + 4;
    float *wp17 = wp16 + 4;
    float *wp18 = wp17 + 4;
    float *wp19 = wp18 + 4;

    vc00 = vld1q_f32(wp0);
    vc01 = vld1q_f32(wp1);
    vc02 = vld1q_f32(wp2);
    vc03 = vld1q_f32(wp3);
    vc04 = vld1q_f32(wp4);
    vc10 = vld1q_f32(wp5);
    vc11 = vld1q_f32(wp6);
    vc12 = vld1q_f32(wp7);
    vc13 = vld1q_f32(wp8);
    vc14 = vld1q_f32(wp9);
    vc20 = vld1q_f32(wp10);
    vc21 = vld1q_f32(wp11);
    vc22 = vld1q_f32(wp12);
    vc23 = vld1q_f32(wp13);
    vc24 = vld1q_f32(wp14);
    vc30 = vld1q_f32(wp15);
    vc31 = vld1q_f32(wp16);
    vc32 = vld1q_f32(wp17);
    vc33 = vld1q_f32(wp18);
    vc34 = vld1q_f32(wp19);

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
        v4 = vld1q_f32(vp + 16);
        vp += 20;

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
    
    vst1q_f32(wp0,  vc00);
    vst1q_f32(wp1,  vc01);
    vst1q_f32(wp2,  vc02);
    vst1q_f32(wp3,  vc03);
    vst1q_f32(wp4,  vc04);
    vst1q_f32(wp5,  vc10);
    vst1q_f32(wp6,  vc11);
    vst1q_f32(wp7,  vc12);
    vst1q_f32(wp8,  vc13);
    vst1q_f32(wp9,  vc14);
    vst1q_f32(wp10, vc20);
    vst1q_f32(wp11, vc21);
    vst1q_f32(wp12, vc22);
    vst1q_f32(wp13, vc23);
    vst1q_f32(wp14, vc24);
    vst1q_f32(wp15, vc30);
    vst1q_f32(wp16, vc31);
    vst1q_f32(wp17, vc32);
    vst1q_f32(wp18, vc33);
    vst1q_f32(wp19, vc34);
}

void TensorGEMMInnerKernel4x3x5_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN) {
    float32x4_t vc00, vc01, vc02, vc03, vc04;
    float32x4_t vc10, vc11, vc12, vc13, vc14;
    float32x4_t vc20, vc21, vc22, vc23, vc24;

    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3, v4;

    int step = 4 * tileN;
    float *wp0  = WTp;
    float *wp1  = wp0  + 4;
    float *wp2  = wp1  + 4;
    float *wp3  = wp2  + 4;
    float *wp4  = wp3  + 4;
    float *wp5  = wp0  + 1 * step;
    float *wp6  = wp5  + 4;
    float *wp7  = wp6  + 4;
    float *wp8  = wp7  + 4;
    float *wp9  = wp8  + 4;
    float *wp10 = wp0  + 2 * step;
    float *wp11 = wp10 + 4;
    float *wp12 = wp11 + 4;
    float *wp13 = wp12 + 4;
    float *wp14 = wp13 + 4;

    vc00 = vld1q_f32(wp0);
    vc01 = vld1q_f32(wp1);
    vc02 = vld1q_f32(wp2);
    vc03 = vld1q_f32(wp3);
    vc04 = vld1q_f32(wp4);
    vc10 = vld1q_f32(wp5);
    vc11 = vld1q_f32(wp6);
    vc12 = vld1q_f32(wp7);
    vc13 = vld1q_f32(wp8);
    vc14 = vld1q_f32(wp9);
    vc20 = vld1q_f32(wp10);
    vc21 = vld1q_f32(wp11);
    vc22 = vld1q_f32(wp12);
    vc23 = vld1q_f32(wp13);
    vc24 = vld1q_f32(wp14);

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
        vp += 20;

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
    
    vst1q_f32(wp0,  vc00);
    vst1q_f32(wp1,  vc01);
    vst1q_f32(wp2,  vc02);
    vst1q_f32(wp3,  vc03);
    vst1q_f32(wp4,  vc04);
    vst1q_f32(wp5,  vc10);
    vst1q_f32(wp6,  vc11);
    vst1q_f32(wp7,  vc12);
    vst1q_f32(wp8,  vc13);
    vst1q_f32(wp9,  vc14);
    vst1q_f32(wp10, vc20);
    vst1q_f32(wp11, vc21);
    vst1q_f32(wp12, vc22);
    vst1q_f32(wp13, vc23);
    vst1q_f32(wp14, vc24);
}

void TensorGEMMInnerKernel4x2x5_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN) {
    float32x4_t vc00, vc01, vc02, vc03, vc04;
    float32x4_t vc10, vc11, vc12, vc13, vc14;

    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3, v4;

    int step = 4 * tileN;
    float *wp0  = WTp;
    float *wp1  = wp0  + 4;
    float *wp2  = wp1  + 4;
    float *wp3  = wp2  + 4;
    float *wp4  = wp3  + 4;
    float *wp5  = wp0  + 1 * step;
    float *wp6  = wp5  + 4;
    float *wp7  = wp6  + 4;
    float *wp8  = wp7  + 4;
    float *wp9  = wp8  + 4;


    vc00 = vld1q_f32(wp0);
    vc01 = vld1q_f32(wp1);
    vc02 = vld1q_f32(wp2);
    vc03 = vld1q_f32(wp3);
    vc04 = vld1q_f32(wp4);
    vc10 = vld1q_f32(wp5);
    vc11 = vld1q_f32(wp6);
    vc12 = vld1q_f32(wp7);
    vc13 = vld1q_f32(wp8);
    vc14 = vld1q_f32(wp9);

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
        vp += 20;

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
    
    vst1q_f32(wp0,  vc00);
    vst1q_f32(wp1,  vc01);
    vst1q_f32(wp2,  vc02);
    vst1q_f32(wp3,  vc03);
    vst1q_f32(wp4,  vc04);
    vst1q_f32(wp5,  vc10);
    vst1q_f32(wp6,  vc11);
    vst1q_f32(wp7,  vc12);
    vst1q_f32(wp8,  vc13);
    vst1q_f32(wp9,  vc14);
}

void TensorGEMMInnerKernel4x1x5_v1(float* WTp, const float* UTp, float* vp, int inChannels, int tileN) {
    float32x4_t vc00, vc01, vc02, vc03, vc04;

    float32x4_t u0, u1, u2, u3;
    float32x4_t v0, v1, v2, v3, v4;

    int step = 4 * tileN;
    float *wp0  = WTp;
    float *wp1  = wp0  + 4;
    float *wp2  = wp1  + 4;
    float *wp3  = wp2  + 4;
    float *wp4  = wp3  + 4;

    vc00 = vld1q_f32(wp0);
    vc01 = vld1q_f32(wp1);
    vc02 = vld1q_f32(wp2);
    vc03 = vld1q_f32(wp3);
    vc04 = vld1q_f32(wp4);

    const float *up = UTp;
    for (int ic = 0; ic < inChannels; ++ic)
    {
        u0 = vld1q_f32(up);
        up += 4;
        v0 = vld1q_f32(vp);
        v1 = vld1q_f32(vp + 4);
        v2 = vld1q_f32(vp + 8);
        v3 = vld1q_f32(vp + 12);
        v4 = vld1q_f32(vp + 16);
        vp += 20;

        vc00 = vfmaq_f32(vc00, u0, v0);
        vc01 = vfmaq_f32(vc01, u0, v1);
        vc02 = vfmaq_f32(vc02, u0, v2);
        vc03 = vfmaq_f32(vc03, u0, v3);
        vc04 = vfmaq_f32(vc04, u0, v4);
    }
    
    vst1q_f32(wp0,  vc00);
    vst1q_f32(wp1,  vc01);
    vst1q_f32(wp2,  vc02);
    vst1q_f32(wp3,  vc03);
    vst1q_f32(wp4,  vc04);
}

void TensorGEMMInnerKernel4x4x5_v2(float* WTp, const float* UTp, float* vp, int inChannels, int tileN)
{

	const float* up = UTp;
    int step = 4 * tileN;
    float *wp0 = WTp;
    float *wp1 = wp0  + 1 * step;
    float *wp2 = wp0  + 2 * step;
    float *wp3 = wp0  + 3 * step;
	asm volatile (
		"ld1	{v12.4s, v13.4s, v14.4s, v15.4s}, [%[wp0]]	        \n"
        "ldr    q16,                              [%[wp0], #64]     \n"
		"ld1	{v17.4s, v18.4s, v19.4s, v20.4s}, [%[wp1]]	        \n"
        "ldr    q21,                              [%[wp1], #64]     \n"
		"ld1	{v22.4s, v23.4s, v24.4s, v25.4s}, [%[wp2]]	        \n"
        "ldr    q26,                              [%[wp2], #64]     \n"
		"ld1	{v27.4s, v28.4s, v29.4s, v30.4s}, [%[wp3]]	        \n"
        "ldr    q31,                              [%[wp3], #64]     \n"

		"mov	x4, %[ic]		\n"

		"cmp	w4, #0			\n"
		"beq	end_4x5			\n"

	"loop1_4x5:				    \n"	
		"subs	w4, w4, 0x1		\n"

		"prfm 	pldl1keep, [%[up], #128]			\n"
		"prfm 	pldl1keep, [%[vp], #128]			\n"
		"ld1 	{v0.4s, v1.4s, v2.4s, v3.4s}, [%[up]], #64	\n"
		"ld1 	{v4.4s, v5.4s, v6.4s, v7.4s}, [%[vp]], #64	\n"
		"ld1	{v8.4s}, [%[vp]], #16	\n"

		"fmla	v12.4s, v0.4s, v4.4s	\n"
		"fmla	v13.4s, v0.4s, v5.4s	\n"
		"fmla	v14.4s, v0.4s, v6.4s	\n"
		"fmla	v15.4s, v0.4s, v7.4s	\n"
		"fmla	v16.4s, v0.4s, v8.4s	\n"

		"fmla	v17.4s, v1.4s, v4.4s	\n"
		"fmla	v18.4s, v1.4s, v5.4s	\n"
		"fmla	v19.4s, v1.4s, v6.4s	\n"
		"fmla	v20.4s, v1.4s, v7.4s	\n"
		"fmla	v21.4s, v1.4s, v8.4s	\n"

		"fmla	v22.4s, v2.4s, v4.4s	\n"
		"fmla	v23.4s, v2.4s, v5.4s	\n"
		"fmla	v24.4s, v2.4s, v6.4s	\n"
		"fmla	v25.4s, v2.4s, v7.4s	\n"
		"fmla	v26.4s, v2.4s, v8.4s	\n"

		"fmla	v27.4s, v3.4s, v4.4s	\n"
		"fmla	v28.4s, v3.4s, v5.4s	\n"
		"fmla	v29.4s, v3.4s, v6.4s	\n"
		"fmla	v30.4s, v3.4s, v7.4s	\n"
		"fmla	v31.4s, v3.4s, v8.4s	\n"
		
		"bne	loop1_4x5		\n"

	"save_result_4x5:			\n"
		"st1	{v12.4s, v13.4s, v14.4s, v15.4s}, [%[wp0]]	        \n"
        "str    q16,                              [%[wp0], #64]     \n"
		"st1	{v17.4s, v18.4s, v19.4s, v20.4s}, [%[wp1]]	        \n"
        "str    q21,                              [%[wp1], #64]     \n"
		"st1	{v22.4s, v23.4s, v24.4s, v25.4s}, [%[wp2]]	        \n"
        "str    q26,                              [%[wp2], #64]     \n"
		"st1	{v27.4s, v28.4s, v29.4s, v30.4s}, [%[wp3]]	        \n"
        "str    q31,                              [%[wp3], #64]     \n"
	
	"end_4x5:							\n"

		:[up]"+r"(up),
		 [vp]"+r"(vp),
		 [wp0]"+r"(wp0),
		 [wp1]"+r"(wp1),
		 [wp2]"+r"(wp2),
		 [wp3]"+r"(wp3),
		 [ic]"+r"(inChannels)
		:
		: "cc", "memory", "x4", "x8", "x9", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", 
        		"v8", "v9", "v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20",
        		"v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28", "v29", "v30", "v31");

}