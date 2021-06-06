#include "../utility/helper.h"
#include "./winoF63/winoF63.h"

class ConvWinoF63Layer : public ConvLayer
{
    public:
    ConvWinoF63Layer(float *input, float *kernel, float *biasw, float* output_ref, size_t ic, size_t ih, size_t iw, size_t oc,
	 				size_t kh=3, size_t kw=3, size_t sh=1, size_t sw=1, size_t pad_left=1, size_t pad_right=1, size_t pad_top=1, size_t pad_bottom=1,
					size_t g=1, bool bias=0, size_t nt=1, size_t iter=10)
	    : ConvLayer(input, kernel, biasw, output_ref, ic, ih, iw, oc, kh, kw, sh, sw, pad_left, pad_right, pad_top, pad_bottom, g, bias, nt, iter)
    {
	inputBuf  = NULL;
	gemmBuf   = NULL;
	kernelBuf = NULL;
	ocBlock_best     =256;
	tileBlock_best   = 16;
	ocRegBlock_best  = 4;
	tileRegBlock_best= 4;
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

    int Tuning()
    {
    	timespec start, stop;
	int warmup = 5;
	int nloop = 10;
	    
	icBlock = input_channels;
	num_threads = 1;
	enableOffKernel = 1;


	tileBlock    = 4;
	tileRegBlock = 4;
	ocBlock      = 4;
	ocRegBlock   = 4;

    	int tileH = (output_height + 5)/6;
    	int tileW = (output_width  + 5)/6;
    	int tileN = tileH*tileW;
	float *kernel_temp = (float *) malloc(input_channels * (output_channels+7) * kernel_width * kernel_height  * sizeof(float));

	int InnerK_oc[6]={3,4,5,6,7,4};	
	int InnerK_tl[6]={7,5,4,3,3,4};	

	double minTimeusage=1e100;

	int max_tuning_num =1024;	//
	int oc_num = (output_channels-1)/ ocRegBlock + 1;
	int tl_num = (tileN -1) / tileRegBlock + 1;
	if(oc_num*tl_num>max_tuning_num)	{
		//OC will be divided at most 64 parts.
		if(oc_num>=64)	          oc_num = 64;	
		//All the others will be used on tile
		if(tl_num>max_tuning_num/oc_num)  tl_num = max_tuning_num/oc_num; 
	}

	int ik=3;
	for(int ik=0;ik<6;ik++)
	{
		ocRegBlock=InnerK_oc[ik];
		tileRegBlock=InnerK_tl[ik];

		int oc_step = (output_channels-1) / (ocRegBlock  *oc_num) + 1;
		int tl_step =          (tileN -1) / (tileRegBlock*tl_num) + 1;

		for(int u=1;u<=oc_num;u++)	for(int v=1;v<=tl_num;v++)
		{
		    enableOffKernel=1;
		    ocBlock = u*oc_step*ocRegBlock;
		    tileBlock = v*tl_step*tileRegBlock;

		    if(ocBlock>output_channels)	ocBlock = output_channels;
		    if(tileBlock>tileN)		tileBlock = tileN;
		    if(ocBlock%ocRegBlock)		continue;
		    if(tileBlock%tileRegBlock)	continue;
//		    printf("ocb=%d tb=%d ocr%d tbr%d\n", ocBlock, tileBlock, ocRegBlock, tileRegBlock);

		    for(enableOffKernel=3;enableOffKernel>=0;enableOffKernel--)
		    {
			memcpy(kernel_temp, kernel_data, input_channels * output_channels * kernel_width * kernel_height  * sizeof(float));
			inputBuf = (float *) malloc(icBlock*tileBlock*64*sizeof(float)); 
    			gemmBuf  = (float *) malloc((ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*sizeof(float));
    			if(enableOffKernel)	
	    			kernelBuf = (float *) malloc(input_channels * (output_channels+7) * 64 * sizeof(float));  	
    			else	       	
            			kernelBuf = (float *) malloc(icBlock * ocBlock * 64 * sizeof(float));

    			retransformKernel(kernel_temp, output_channels, input_channels, ocRegBlock);
    			if(enableOffKernel)
    	    			offlineKernelTransform(kernelBuf, kernel_temp, output_channels, input_channels, ocBlock, ocRegBlock);

//    			printf("kernelBuf %d KB\n", icBlock*ocBlock*64*4/1024);   
/*    			printf("gemmBuf %d KB\n",   (ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*4/1024);   
    			printf("inputBuf %d KB\n", tileBlock*icBlock*64*4/1024);  
    			printf("L1 Cache used %d KB\n", (tileBlock*ocBlock*48 + icBlock*ocBlock*64 + tileBlock*icBlock*64)*4/1024);
*/
			Timer inputTran, kernelTran, GEMM, outputTran;
			for(int i=0;i<warmup;i++)
       			    winoF63(output_data, input_data, kernel_temp, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads);

			clock_gettime(CLOCK_MONOTONIC, &start);
			for(int i=0;i<nloop;i++)	
       			    winoF63(output_data, input_data, kernel_temp, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads);

			clock_gettime(CLOCK_MONOTONIC, &stop);
			double elapsedTime = ((stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_nsec - start.tv_nsec) / 1000000.0)/nloop;

			float dis = diff(output_ref, output_data, output_channels * output_height * output_width);
			printf("ocb=%d tb=%d(tN=%d,tS=%d) ocr%d tbr%d,eoffK%d time=%.3f, diff=%.3f\n", ocBlock, tileBlock, tileN,tl_step, ocRegBlock, tileRegBlock, enableOffKernel, elapsedTime, dis);
			if(minTimeusage>elapsedTime)	
			{
				ocBlock_best = ocBlock;  
				tileBlock_best = tileBlock;
				ocRegBlock_best = ocRegBlock;
				tileRegBlock_best = tileRegBlock;
				enableOffKernel_best = enableOffKernel;
				minTimeusage = elapsedTime;
			}
			free(inputBuf);
			free(gemmBuf);
			free(kernelBuf);
			inputBuf  = NULL;
			gemmBuf   = NULL;
			kernelBuf = NULL;
		}
	    }
	}
	free(kernel_temp);
	kernel_temp = NULL;

	printf("Best Config: (%d %d %d) ocb=%d tb=%d ocr%d tbr%d, eoffK%d time=%.3f\n", input_channels, output_channels, input_width, ocBlock_best, tileBlock_best, ocRegBlock_best, tileRegBlock_best, enableOffKernel_best, minTimeusage);
        return -1;
    }


    int Init()
    {

		icBlock = input_channels;
		enableOffKernel = 1;

	tileBlock = tileBlock_best;
	ocBlock   = ocBlock_best;
	tileRegBlock = tileRegBlock_best;
	ocRegBlock = ocRegBlock_best;
	// enableOffKernel = enableOffKernel_best;
	printf("enableoffkernel = %d\n", enableOffKernel);
	num_threads = 1;
    	
	printf("Testing ic=%d oc=%d width=%d tileBlock=%d ocBlock=%d icBlock=%d threads=%d\n", input_channels, output_channels, input_width, tileBlock, ocBlock, icBlock, num_threads);
	inputBuf      = (float *) malloc(icBlock*tileBlock*64*sizeof(float)); 
    	gemmBuf       = (float *) malloc((ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*sizeof(float));
    	if(enableOffKernel)	
	    kernelBuf = (float *) malloc(input_channels * (output_channels+7) * 64 * sizeof(float));  	
    	else	       	
            kernelBuf = (float *) malloc(icBlock * ocBlock * 64 * sizeof(float));

    	retransformKernel(kernel_data, output_channels, input_channels, ocRegBlock);
    	if(enableOffKernel)
    	    offlineKernelTransform(kernelBuf, kernel_data, output_channels, input_channels, ocBlock, ocRegBlock);
    	printf("kernelBuf %d KB\n", icBlock*ocBlock*64*4/1024);   
    	printf("gemmBuf   %d KB\n",   (ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*4/1024);   
    	printf("inputBuf  %d KB\n", tileBlock*icBlock*64*4/1024);  
    	printf("L1 Cache used %d KB\n", (tileBlock*ocBlock*48 + icBlock*ocBlock*64 + tileBlock*icBlock*64)*4/1024);
	
	return -1;
    }

    int Forward()
    {
	for(int i=0;i<1;i++)
		winoF63(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel+4, num_threads);

    	// printf("kernelBuf %d KB\n", icBlock*ocBlock*64*4/1024);   
    	// printf("gemmBuf   %d KB\n",   (ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*4/1024);   
    	// printf("inputBuf  %d KB\n", tileBlock*icBlock*64*4/1024);  
    	// printf("L1 Cache used %d KB\n", (tileBlock*ocBlock*48 + icBlock*ocBlock*64 + tileBlock*icBlock*64)*4/1024);
        return 1;
    }
    private:
	int tileBlock;
	int ocBlock;
    	int icBlock;

	int ocRegBlock;
    	int tileRegBlock;
	int enableOffKernel;

	float *inputBuf;
    	float *gemmBuf;
    	float *kernelBuf;

	int ocBlock_best;
	int tileBlock_best;
	int ocRegBlock_best;
	int tileRegBlock_best;
	int enableOffKernel_best;
};

