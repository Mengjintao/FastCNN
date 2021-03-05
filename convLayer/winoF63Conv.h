#include "../utility/helper.h"

class ConvWinoF63Layer : public ConvLayer
{
    public:
    ConvWinoF63Layer(float *input, float *kernel, float *biasw, size_t ic, size_t ih, size_t iw, size_t oc, size_t kh=3, size_t kw=3, size_t sh=1, size_t sw=1, size_t pad_left=1, size_t pad_right=1, size_t pad_top=1, size_t pad_bottom=1, size_t g=1, bool bias=0)
	    : ConvLayer(input, kernel, biasw, ic, ih, iw, oc, kh, kw, sh, sw, pad_left, pad_right, pad_top, pad_bottom, g, bias)
    {
    }

    ~ConvWinoF63Layer()
    {
	if(inputBuf) 	free(inputBuf);
	if(gemmBuf)	free(gemmBuf);
	if(kernelBuf)	free(kernelBuf);
	inputBuf  = NULL;
	gemmBuf   = NULL;
	kernelBuf = NULL;
    }

    int Tuning(float *Res=NULL)
    {
    	icBlock = input_channels;
	num_threads = 1;
	enableOffKernel = 0;

	tileBlock    = 20;
	tileRegBlock = 5;
	ocBlock      = 60;
	ocRegBlock   = 4;

	for(ocRegBlock=4;ocRegBlock<7;ocRegBlock++)
    	for(tileRegBlock=1;tileRegBlock<5;tileRegBlock++)
	{
		for(int kt=1;kt<100;kt++)
		{
			float *kernel_temp = (float *) malloc(input_channels * output_channels * kernel_width * kernel_height  * sizeof(float));
			memcpy(kernel_temp, kernel_data, input_channels * output_channels * kernel_width * kernel_height  * sizeof(float));

			tileBlock = kt*tileRegBlock;
    			int tileH = (output_height + 5)/6;
    			int tileW = (output_width  + 5)/6;
    			int tileN = tileH*tileW;
        		printf("----%d %d %d\n", tileBlock, tileRegBlock, ocRegBlock);	
			if(tileBlock>tileN)	break;

			inputBuf = (float *) malloc(icBlock*tileBlock*64*sizeof(float)); 
    			gemmBuf  = (float *) malloc((ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*sizeof(float));
    			if(enableOffKernel)	
	    			kernelBuf = (float *) malloc(input_channels * output_channels * 64 * sizeof(float));  	
    			else	       	
            			kernelBuf = (float *) malloc(icBlock * ocBlock * 64 * sizeof(float));

    			retransformKernel(kernel_temp, output_channels, input_channels, ocRegBlock);
    			if(enableOffKernel)
    	    			offlineKernelTransform(kernelBuf, kernel_temp, output_channels, input_channels, ocBlock, ocRegBlock);

/*    			printf("kernelBuf %d KB\n", icBlock*ocBlock*64*4/1024);   
    			printf("gemmBuf %d KB\n",   (ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*4/1024);   
    			printf("inputBuf %d KB\n", tileBlock*icBlock*64*4/1024);  
    			printf("L1 Cache used %d KB\n", (tileBlock*ocBlock*48 + icBlock*ocBlock*64 + tileBlock*icBlock*64)*4/1024);
*/
       			winoF63(output_data, input_data, kernel_temp, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads);

			diff(Res, output_data, output_channels * output_height * output_width);

			free(inputBuf);
			free(gemmBuf);
			free(kernelBuf);
			free(kernel_temp);
			inputBuf  = NULL;
			gemmBuf   = NULL;
			kernelBuf = NULL;
			kernel_temp = NULL;
		}
	}
        return -1;
    }


    int Init()
    {
    	icBlock = input_channels;
	enableOffKernel = 0;

	tileBlock = 20;
	ocBlock   = 24;
	icBlock = input_channels;

	ocRegBlock = 6;
	tileRegBlock = 5;
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

