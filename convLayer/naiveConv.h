
class ConvWinogradLayer : public ConvLayer
{
    public:
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
        int output_height = input_height + padding_top  + padding_bottom -2; 
        int output_width  = input_width  + padding_left + padding_right  -2;
        int output_page   = output_height * output_width;

        float *baseResult = output_data; 
	float *testInput  = input_data;
        float *testKernel = kernel_data;
	float *testBias   = bias_data;

        	
        for(int oc=0; oc<output_channels; oc++)
        {
            for(int tile=0; tile<output_height * output_width; tile++)
            {
                int startH = tile/(output_width);
                int startW = tile%(output_width);

                float sum = 0;
                for(int ic=0; ic<input_channels; ic++)
                {
                //float *ip = testInput  + ic*(inputHeight*inputWidth) + startH * inputWidth + startW;
                    float *ip = input_data  + ic * (input_height * input_width);
                    float *kp = kernel_data + oc * (input_channels*9) + ic*9;

                    float total = 0;
                    for(int i=0; i<3; i++)	for(int j=0; j<3; j++)
                    {
			int x = startH+i - padding_top;
			int y = startW+j - padding_left;
			float value = 0;
			if(x>=0 && x<input_height && y>=0 && y<input_width)	value = *(ip + x*input_width + y);
			else value = 0;	
//                        float *tip = ip + i*(inputWidth) + j;
                        float *tkp = kp + i*3            + j;
//                        total += (*tip) * (*tkp);
                        total += value * (*tkp);
                    }
                    sum += total;
                }
                *(baseResult + oc*output_page + tile) = sum;
            }
        }

        return 1;
    }
};

