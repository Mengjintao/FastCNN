
class ConvWinoF63Layer : public ConvLayer
{
    public:
    ConvWinoF63Layer(float *input, float *kernel, float *biasw, size_t ic, size_t ih, size_t iw, size_t oc, size_t kh=3, size_t kw=3, size_t sh=1, size_t sw=1, size_t pad_left=1, size_t pad_right=1, size_t pad_top=1, size_t pad_bottom=1, size_t g=1, bool bias=0)
	    : ConvLayer(input, kernel, biasw, ic, ih, iw, oc, kh, kw, sh, sw, pad_left, pad_right, pad_top, pad_bottom, g, bias)
    {
    }

    ~ConvWinoF63Layer()
    {
	free(inputBuf);
	free(gemmBuf);
	free(kernelBuf);
	inputBuf  = NULL;
	gemmBuf   = NULL;
	kernelBuf = NULL;
    }

    int Tuning()
    {

        return -1;
    }


    int Init()
    {
    	icBlock = input_channels;
	enableOffKernel = 0;

	tileBlock = 16;
	ocBlock   = 20;
	icBlock = input_channels;

	ocRegBlock = 5;
	tileRegBlock = 4;
	num_threads = 1;
    	
	inputBuf      = (float *) malloc(icBlock*tileBlock*64*sizeof(float)); 
    	gemmBuf       = (float *) malloc((ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*sizeof(float));
    	if(enableOffKernel)	
	    kernelBuf = (float *) malloc(input_channels * output_channels * 64 * sizeof(float));  	
    	else	       	
            kernelBuf = (float *) malloc(icBlock * ocBlock * 64 * sizeof(float));

    	retransformKernel(kernel_data, output_channels, input_channels, ocRegBlock);
    	if(enableOffKernel)
    	    offlineKernelTransform(kernelBuf, kernel_data, output_channels, input_channels, ocBlock, ocRegBlock);
    	printf("kernelBuf %d KB\n", icBlock*ocBlock*64*4/1024);   
    	printf("gemmBuf %d KB\n",   (ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*4/1024);   
    	printf("inputBuf %d KB\n", tileBlock*icBlock*64*4/1024);  
    	printf("L1 Cache used %d KB\n", (tileBlock*ocBlock*48 + icBlock*ocBlock*64 + tileBlock*icBlock*64)*4/1024);
	return -1;
    }


    int Forward()
    {
       	winoF63(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads);

        return 1;
    }
    private:
	int tileBlock;
	int ocBlock;
    	int icBlock;

	int ocRegBlock;
    	int tileRegBlock;
	bool enableOffKernel;

	float *inputBuf;
    	float *gemmBuf;
    	float *kernelBuf;
};

