//
//  BasicConv.hpp
//  WinogradConvolution
//
//  Created by LanHaidong on 17/08/2017.
//  Copyright © 2017 Tencent. All rights reserved.
//

#ifndef BasicConv_hpp
#define BasicConv_hpp

#include <stdio.h>

struct nnp_size {
    int width;
    int height;
};

struct nnp_padding {
    int top;
    int right;
    int bottom;
    int left;
};
void conv3x3s1_neon(float *rawInput, int input_channels, struct nnp_size inputDim,
                        float* kernel, struct nnp_size kernelDim,
                        float* output, int output_channels, struct nnp_size outputDim,
                        struct nnp_padding paddingSize, struct nnp_size subsampling);
#endif /* BasicConv_hpp */
