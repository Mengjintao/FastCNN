//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2018 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.

#pragma once

#include "../layer.h"
#include "./arm/helper.h"
#include <assert.h>
#include <stdio.h>

namespace feather
{
class ConvLayer:
{
    public:
        ConvLayer(float *input, float *kernel, float *biasw, size_t ic, size_t ih, size_t iw, size_t oc, size_t kh=3, size_t kw=3, size_t sh=1, size_t sw=1, size_t pad_left=0, size_t pad_right=0, size_t pad_top=0, size_t pad_bottom=0, size_t g=1, bool bias=0)
        {
	    //Input
            input_channels = ic;
            input_height   = ih;
            input_width    = iw;
	    input_data     = input;	    
	    
	    //Kernel
	    output_channels=oc;	    
	    kernel_height = kh;
            kernel_width  = kw;
	    kernel_data   = kernel;
	    
	    //Bias
	    bias_term = bias;
	    if(bias_term)	bias_data = biasw;
	    else		bias_data = NULL;

	    group = g;
            stride_height = sh;
            stride_width  = sw;
	    
	    padding_left   = pad_left;
            padding_top    = pad_top;
            padding_right  = pad_right;
            padding_bottom = pad_bottom;

	    if(group == 0 || stride_width  == 0 || stride_height == 0)
		assert(!(group == 0 || stride_width  == 0 || stride_height == 0));
	    
	    //Output
            output_width = (input_width + padding_left + padding_right - kernel_width) / stride_width + 1;
            output_height = (input_height + padding_top + padding_bottom - kernel_height) / stride_height + 1;
   	    output_data = (float *) _mm_malloc(output_channels * output_width * output_height * sizeof(float)); 
        }
/*
        int GenerateTopBlobs()
        {
            //Conv layer has and only has one bottom blob.
            const Blob<float> *bottom_blob = _bottom_blobs[_bottom[0]];
	    printf("bottom name %s ptr 0x%lx\n", _bottom[0].c_str(), bottom_blob);
            input_width = bottom_blob->width();
            input_height = bottom_blob->height();
            input_channels = bottom_blob->channels();

            output_width = (input_width + padding_left + padding_right - kernel_width) / stride_width + 1;
            output_height = (input_height + padding_top + padding_bottom - kernel_height) / stride_height + 1;
#if 0
            printf("input channels %d\n", input_channels);
            assert(input_channels == bottom_blob->channels());
            printf("input w %lu\n", input_width);
            printf("padding_left %lu\n", padding_left);
            printf("padding_top %lu\n", padding_top);
            printf("stride_width %lu\n", stride_width);
            printf("stride_height %lu\n", stride_height);
            printf("output %ld %ld\n", output_width, output_height);
#endif
            _top_blobs[_top[0]] = new Blob<float>(1, output_channels, output_height, output_width);
            _top_blobs[_top[0]]->Alloc();
            return 0;
        }
*/
        virtual int Forward()
        {
            return -1;
        }

	virtual int Tuning()
	{
	    return -1;
	}	
	
    protected:
        size_t input_channels;
        size_t input_width;
        size_t input_height;

        size_t output_channels;
        size_t output_width;
        size_t output_height;

        size_t kernel_width;
        size_t kernel_height;

        size_t stride_width;
        size_t stride_height;

        size_t padding_left;
        size_t padding_right;
        size_t padding_top;
        size_t padding_bottom;

        size_t group;
        bool bias_term;

	float *input_data;
        float *kernel_data;
        float *bias_data;
	float *output_data;
};
};
