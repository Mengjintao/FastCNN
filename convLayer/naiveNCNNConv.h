#include "./ncnn/ConvKernels.h"

class ConvNaiveNEONLayer : public ConvLayer
{

    public:
    ConvNaiveNEONLayer(float *input, float *kernel, float *biasw, float *output_ref, size_t ic, size_t ih, size_t iw, size_t oc,
                        size_t kh=3, size_t kw=3, size_t sh=1, size_t sw=1, size_t pad_left=1, size_t pad_right=1, size_t pad_top=1, size_t pad_bottom=1,
                        size_t g=1, bool bias=0, size_t nt=1, size_t iter=10)
	    : ConvLayer(input, kernel, biasw, output_ref, ic, ih, iw, oc, kh, kw, sh, sw, pad_left, pad_right, pad_top, pad_bottom, g, bias, nt, iter)
    {

    }

    int Init()
    {
        return -1;
    }

    int Tuning()
    {
	return -1;
    }	


    int Forward()
    {
        nnp_size inputDim, kernelDim, outputDim, subsampling;

	inputDim.height = input_height;
	inputDim.width  = input_width;
	
	kernelDim.height = kernel_height;
	kernelDim.width  = kernel_width;

	outputDim.height = output_height;
	outputDim.width  = output_width;

        nnp_padding paddings;
        paddings.top     = padding_top;
        paddings.bottom  = padding_bottom;
        paddings.left    = padding_left;
        paddings.right   = padding_right;

    	subsampling.height = 1;
    	subsampling.width  = 1;
	
	conv3x3s1_neon(input_data, input_channels,inputDim, kernel_data, kernelDim, output_data, output_channels, outputDim, paddings, subsampling); 
        return 1;
    }

};

