
class ConvWinoF63Layer : public ConvLayer
{
    public:
    ConvWinoF63Layer(float *input, float *kernel, float *biasw, size_t ic, size_t ih, size_t iw, size_t oc, size_t kh=3, size_t kw=3, size_t sh=1, size_t sw=1, size_t pad_left=1, size_t pad_right=1, size_t pad_top=1, size_t pad_bottom=1, size_t g=1, bool bias=0)
	    : ConvLayer(input, kernel, biasw, ic, ih, iw, oc, kh, kw, sh, sw, pad_left, pad_right, pad_top, pad_bottom, g, bias)
    {

    }

    int Init()
    {
    	icBlock = input_channels;
	enableOffKernel = 0;

	tileBlock = 16;
	ocBlock   = 16;
	icBlock = input_channels;

	ocRegBlock = 4;
	tileRegBlock = 4;
	num_threads = 1;
    	
    	testKernel      = (float *) malloc(sizeof(float) * kernel_height * kernel_width * input_channels * output_channels);
	inputBuf      = new float [icBlock*tileBlock*64]; 
    	gemmBuf       = new float [ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64];
    	kernelBuf;
    	if(enableOffKernel)	
	    kernelBuf = new float [input_channels * output_channels * 64];  	
    	else	       	
            kernelBuf = new float [icBlock * ocBlock * 64];

    	retransformKernel(testKernel, output_channels, input_channels, ocRegBlock);
    	if(enableOffKernel)
    	    offlineKernelTransform(kernelBuf, testKernel, output_channels, input_channels, ocBlock, ocRegBlock);
    	printf("kernelBuf %d KB\n", icBlock*ocBlock*64*4/1024);   
    	printf("gemmBuf %d KB\n",   (ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*4/1024);   
    	printf("inputBuf %d KB\n", tileBlock*icBlock*64*4/1024);  
    	printf("L1 Cache used %d KB\n", (tileBlock*ocBlock*48 + icBlock*ocBlock*64 + tileBlock*icBlock*64)*4/1024);
	return -1;
    }

    int Tuning()
    {
	return -1;



    }	

    int Forward()
    {
        float *baseResult = output_data; 
	float *testInput  = input_data;
        float *testKernel = kernel_data;
	float *testBias   = bias_data;
     /*  
        printf("%d %d %d %d\n", input_channels, output_channels, input_height, input_width);
	printf("%d %d %d %d\n", padding_left, padding_top, stride_width, stride_height);
        printf("%d %d %d %d %d %d\n",tileBlock, ocBlock, icBlock, tileRegBlock, ocRegBlock, enableOffKernel);	
       */
       	winoF63(baseResult, testInput, testKernel, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads);

        return 1;
    }
    private:
	int tileBlock;
	int ocBlock;
    	int icBlock;

	int ocRegBlock;
    	int tileRegBlock;
	bool enableOffKernel;

	float *testKernel;
	float *inputBuf;
    	float *gemmBuf;
    	float *kernelBuf;
};

