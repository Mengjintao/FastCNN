
class ConvNaiveLayer : public ConvLayer
{

    public:
    ConvNaiveLayer(float *input, float *kernel, float *biasw, size_t ic, size_t ih, size_t iw, size_t oc, size_t kh=3, size_t kw=3, size_t sh=1, size_t sw=1, size_t pad_left=1, size_t pad_right=1, size_t pad_top=1, size_t pad_bottom=1, size_t g=1, bool bias=0)
	    : ConvLayer(input, kernel, biasw, ic, ih, iw, oc, kh, kw, sh, sw, pad_left, pad_right, pad_top, pad_bottom, g, bias)
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
        int output_page = output_height * output_width;
        int input_page = input_height * input_width;
        int kernel_page = kw * kh;
        int kernel_page_row = input_channels * kernel_page;
        // support group modify here
        // int kernel_page_row = input_channels / group * kernel_page;

        // output_data
        // input_data
        // kernel_data
        // bias_data

        // group

        // padding_top
        // padding_left
        // padding_right
        // padding_bottom

        // stride_height
        // stride_width

        for(int oc=0; oc<output_channels; oc++)
        {
            for(int tile=0; tile<output_height * output_width; tile++)
            {
                // find the start point position in input_page
                int startH = (tile/(output_width)) * stride_height - padding_top;
                int startW = (tile%(output_width)) * stride_width - padding_left;

                float sum = 0;
                for(int ic=0; ic<input_channels; ic++)
                {
                    // find the first data in input_data and kernel_data
                    float *ip = input_data  + ic * input_page;
                    float *kp = kernel_data + oc * kernel_page_row + ic * kernel_page;
                    // support group modify here
                    // float *kp = kernel_data + oc * kernel_page_row + ic / group * kernel_page;
                    
                    float total = 0;
                    for(int i=0; i<kh; i++)	for(int j=0; j<kw; j++)
                    {
            			int x = startH+i;
			            int y = startW+j;
			            float value = 0;
                        if(x>=0 && x<input_height && y>=0 && y<input_width)	value = *(ip + x*input_width + y);
                        else continue;
                        float *tkp = kp + i*kw + j;
                        total += value * (*tkp);
                    }
                    sum += total;
                }
                if(biasw) sum = sum + *(bias_data + oc*output_page + tile);
                *(output_data + oc*output_page + tile) = sum;
            }
        }

        return 1;
    }
    private:

};

