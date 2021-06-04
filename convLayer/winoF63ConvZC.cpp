#include "./winoF63ConvZC.h"
#include "./winoF63ZC/winoF63ZC.h"
#include <omp.h>

ConvWinoF63ZCLayer::ConvWinoF63ZCLayer(float *input, float *kernel, float *biasw, size_t ic, size_t ih, size_t iw, size_t oc, int thread_num, size_t kh, size_t kw, size_t sh, size_t sw, size_t pad_left, size_t pad_right, size_t pad_top, size_t pad_bottom, size_t g, bool bias)
    : ConvLayer(input, kernel, biasw, ic, ih, iw, oc, thread_num, kh, kw, sh, sw, pad_left, pad_right, pad_top, pad_bottom, g, bias) {
        inputBuf  = NULL;
        gemmBuf   = NULL;
        kernelBuf = NULL;
        ocBlock_best     = 4;
        tileBlock_best   = 48;
        ocRegBlock_best  = 4;
        tileRegBlock_best= 4;
        icBlock_best =8;
        scheduling_best = 0;
    }

ConvWinoF63ZCLayer::~ConvWinoF63ZCLayer() {
    if(inputBuf) 	_mm_free(inputBuf);
    if(gemmBuf)	    _mm_free(gemmBuf);
    if(kernelBuf)	_mm_free(kernelBuf);
    inputBuf  = NULL;
    gemmBuf   = NULL;
    kernelBuf = NULL;
}

int ConvWinoF63ZCLayer::Tuning(float* res){
    timespec start, stop;
    int warmup = 0;
    int nloop = 5;
    
    enableOffKernel = 0;

    tileBlock    = 4;
    tileRegBlock = 4;
    ocBlock      = 4;
    ocRegBlock   = 4;

    int tileH = (output_height + 5)/6;
    int tileW = (output_width  + 5)/6;
    int tileN = tileH*tileW;
    float *kernel_temp = (float *) malloc(input_channels * (output_channels+7) * kernel_width * kernel_height  * sizeof(float));
    // float *kernel_temp = static_cast<float*>(_mm_malloc((64 * output_channels * ic_block + 128) * sizeof(float)));

    int InnerK_oc[2]={4, 4};	
    int InnerK_tl[2]={4, 5};	

    double minTimeusage=1e100;

    int max_tuning_num = 1024;	//
    

    for(int ik=0;ik<2;ik++)
    {
        ocRegBlock=InnerK_oc[ik];
        tileRegBlock=InnerK_tl[ik];

        int oc_num = (output_channels-1)/ ocRegBlock + 1;
        int tl_num = (tileN -1) / tileRegBlock + 1;
        int ic_num = 8;
        if(oc_num*tl_num>max_tuning_num)	{
            //OC will be divided at most 64 parts.
            if(oc_num>=64)	          oc_num = 64;	
            //All the others will be used on tile
            if(tl_num>max_tuning_num/oc_num)  tl_num = max_tuning_num/oc_num; 
        }

        int oc_step = (output_channels-1) / (ocRegBlock  *oc_num) + 1;
        int tl_step =          (tileN -1) / (tileRegBlock*tl_num) + 1;
        int ic_step = (input_channels-1)  / 16 + 1;

        for(int u=1;u<=oc_num;u++)	for(int v=1;v<=tl_num;v++)  for (int w=1;w<=ic_num;w++)
        {
            // enableOffKernel=1;
            ocBlock = u*oc_step*ocRegBlock;
            // ocBlock = 128;
            tileBlock = v*tl_step*tileRegBlock;
            icBlock = w*ic_step;

            if(ocBlock - ocRegBlock > output_channels)	ocBlock = output_channels;
            if(tileBlock - tileRegBlock >tileN)		tileBlock = tileN;
            if(icBlock > input_channels)  icBlock = input_channels;
            if(ocBlock%ocRegBlock)		continue;
            if(tileBlock%tileRegBlock)	continue;
//		    printf("ocb=%d tb=%d ocr%d tbr%d\n", ocBlock, tileBlock, ocRegBlock, tileRegBlock);

            for(enableOffKernel = 0; enableOffKernel >= 0; enableOffKernel -= 2)
            {
                // memset(output_data, 0, output_channels * output_width * output_height * sizeof(float));
                // memcpy(kernel_temp, kernel_data, input_channels * output_channels * kernel_width * kernel_height  * sizeof(float));
                inputBuf = (float *) _mm_malloc((64 * icBlock * align_ceil(tileN, tileBlock) + 128) * sizeof(float));
                gemmBuf  = (float *) _mm_malloc((64 * output_channels * tileN + 128) * sizeof(float));
                // inputBuf  = (float*) _mm_malloc(sizeof(float) * (64 * align_ceil(tileN, tileBlock) * icBlock + 128));
                // gemmBuf   = (float*) _mm_malloc(sizeof(float) * (64 * align_ceil(output_channels, ocBlock) * align_ceil(tileN, tileBlock) + 128));
                if(enableOffKernel)	{
                    // kernelBuf = (float*) _mm_malloc(sizeof(float) * (64 * ocBlock * icBlock + 128));
                    kernelBuf = (float *) _mm_malloc((64 * align_ceil(output_channels, ocBlock) * align_ceil(input_channels, icBlock) + 128) * sizeof(float));  	
                    // printf("\n");
                } else {
                    kernelBuf = (float *) _mm_malloc((64 * icBlock * align_ceil(output_channels, ocBlock) + 128) * sizeof(float));
                    // memset(kernelBuf, 0, (64 * output_channels * icBlock + 128) * sizeof(float));
                }
                // memset(inputBuf, 0, (64 * icBlock * align_ceil(tileBlock, tileRegBlock) + 128) * sizeof(float));
                memset(gemmBuf, 0, sizeof(float) * (64 * align_ceil(output_channels, ocBlock) * align_ceil(tileN, tileBlock) + 128));
                memset(output_data, 0, output_channels * output_width * output_height * sizeof(float));
                // retransformKernel(kernel_temp, output_channels, input_channels, ocRegBlock);
                // if(enableOffKernel)
                //         offlineKernelTransform(kernelBuf, kernel_temp, output_channels, input_channels, ocBlock, ocRegBlock);

//    			printf("kernelBuf %d KB\n", icBlock*ocBlock*64*4/1024);   
/*    			printf("gemmBuf %d KB\n",   (ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*4/1024);   
                printf("inputBuf %d KB\n", tileBlock*icBlock*64*4/1024);  
                printf("L1 Cache used %d KB\n", (tileBlock*ocBlock*48 + icBlock*ocBlock*64 + tileBlock*icBlock*64)*4/1024);
*/
                Timer inputTran, kernelTran, GEMM, outputTran;
                for(int i=0;i<warmup;i++) {
                    if (num_threads == 1) {
                        winoF63_v1_single(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                                    inputTran, kernelTran, GEMM, outputTran);
                        // winoF63_v1_multi_v1(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                        //                 inputTran, kernelTran, GEMM, outputTran);
                    } else {
                        // winoF63_v1_multi_v1(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                        //                 inputTran, kernelTran, GEMM, outputTran);
                        winoF63_v1_multi(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                                        inputTran, kernelTran, GEMM, outputTran);
                    }
                }
                    

                clock_gettime(CLOCK_MONOTONIC, &start);
                for(int i=0;i<nloop;i++) {
                    if (num_threads == 1) {
                        winoF63_v1_single(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                                    inputTran, kernelTran, GEMM, outputTran);
                        // winoF63_v1_multi_v1(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                        //                 inputTran, kernelTran, GEMM, outputTran);
                    } else {
                        winoF63_v1_multi(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                                    inputTran, kernelTran, GEMM, outputTran);
                        // winoF63_v1_multi_v1(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                        //                 inputTran, kernelTran, GEMM, outputTran);
                    }
                }	

                clock_gettime(CLOCK_MONOTONIC, &stop);
                double elapsedTime = ((stop.tv_sec - start.tv_sec) * 1000.0 + (stop.tv_nsec - start.tv_nsec) / 1000000.0)/nloop;

                float dis = diff(res, output_data, output_channels * output_height * output_width);
                printf("ocb=%d tb=%d(tN=%d,tS=%d) icb=%d ocr%d tbr%d,eoffK%d time=%.3f, diff=%.3f\n", ocBlock, tileBlock, tileN, tl_step, icBlock, ocRegBlock, tileRegBlock, enableOffKernel, elapsedTime, dis);
                // printf("ocb=%d tb=%d(tN=%d,tS=%d) icb=%d ocr%d tbr%d,eoffK%d time=%.3f, diff=%.3f\n", ocBlock, tileBlock, tileN, tl_step, icBlock, ocRegBlock, tileRegBlock, enableOffKernel, elapsedTime);
                if(minTimeusage>elapsedTime)	
                {
                    ocBlock_best = ocBlock;  
                    tileBlock_best = tileBlock;
                    ocRegBlock_best = ocRegBlock;
                    tileRegBlock_best = tileRegBlock;
                    icBlock_best = icBlock;
                    scheduling_best = enableOffKernel;
                    // enableOffKernel_best = enableOffKernel;
                    minTimeusage = elapsedTime;
                }
                _mm_free(inputBuf);
                _mm_free(gemmBuf);
                _mm_free(kernelBuf);
                
            }
        }
    }
    free(kernel_temp);
    kernel_temp = NULL;

    printf("Best Config: (%d %d %d) ocb=%d tb=%d icb=%d ocr%d tbr%d, eoffK%d time=%.3f\n", input_channels, output_channels, input_width, ocBlock_best, tileBlock_best, icBlock_best, ocRegBlock_best, tileRegBlock_best, scheduling_best, minTimeusage);
        return -1;
}

int ConvWinoF63ZCLayer::Init() {
    icBlock = icBlock_best;
    ocBlock   = ocBlock_best;
    tileBlock = tileBlock_best;
    ocRegBlock = ocRegBlock_best;
    tileRegBlock = tileRegBlock_best;
    scheduling = scheduling_best;
    enableOffKernel = scheduling % 2;
    // enableOffKernel = 0;
    // num_threads = 1;

    int tileH = (output_height + 5)/6;
    int tileW = (output_width  + 5)/6;
    int tileN = tileH*tileW;
    
    printf("Testing ic=%d oc=%d width=%d tileBlock=%d ocBlock=%d icBlock=%d threads=%d\n", input_channels, output_channels, input_width, tileBlock, ocBlock, icBlock, num_threads);
    // memset(output_data, 0, output_channels * output_width * output_height * sizeof(float));
    // memcpy(kernel_temp, kernel_data, input_channels * output_channels * kernel_width * kernel_height  * sizeof(float));
    inputBuf = (float *) _mm_malloc((64 * icBlock * align_ceil(tileN, tileBlock) + 128) * sizeof(float));
    gemmBuf  = (float *) _mm_malloc((64 * output_channels * tileN + 128) * sizeof(float));
    // inputBuf  = (float*) _mm_malloc(sizeof(float) * (64 * align_ceil(tileN, tileBlock) * icBlock + 128));
    // gemmBuf   = (float*) _mm_malloc(sizeof(float) * (64 * align_ceil(output_channels, ocBlock) * align_ceil(tileN, tileBlock) + 128));
    if(enableOffKernel)	{
        // kernelBuf = (float*) _mm_malloc(sizeof(float) * (64 * ocBlock * icBlock + 128));
        kernelBuf = (float *) _mm_malloc((64 * output_channels * align_ceil(input_channels, icBlock) + 128) * sizeof(float));  	
        // printf("\n");
    } else {
        kernelBuf = (float *) _mm_malloc((64 * icBlock * align_ceil(output_channels, ocBlock) + 128) * sizeof(float));
        // memset(kernelBuf, 0, (64 * output_channels * icBlock + 128) * sizeof(float));
    }	       	

    // memset(inputBuf, 0, (64 * icBlock * align_ceil(tileBlock, tileRegBlock) + 128) * sizeof(float));
    // memset(gemmBuf, 0, (64 * output_channels * tileN + 128) * sizeof(float));
    // memset(output_data, 0, output_channels * output_width * output_height * sizeof(float));
    if(enableOffKernel) {
        offline_kernel_transform_v1(kernelBuf, kernel_data, output_channels, input_channels, ocBlock, ocRegBlock, icBlock);
        printf("offline kernel transform.\n");
    }
    // printf("kernelBuf %d KB\n", icBlock*ocBlock*64*4/1024);   
    // printf("gemmBuf   %d KB\n",   (ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*4/1024);   
    // printf("inputBuf  %d KB\n", tileBlock*icBlock*64*4/1024);  
    // printf("L1 Cache used %d KB\n", (tileBlock*ocBlock*48 + icBlock*ocBlock*64 + tileBlock*icBlock*64)*4/1024);
    return -1;
}

int ConvWinoF63ZCLayer::Forward()  {
    int n_loop = 1;
    Timer inputTran, kernelTran, GEMM, outputTran;
    for(int i = 0; i < n_loop; i++) {
        // memset(output_data, 0, output_channels * output_height * output_width * sizeof(float));
        if (num_threads == 1) {
            winoF63_v1_single(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                        inputTran, kernelTran, GEMM, outputTran);
            // winoF63_v1_multi_v1(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
            //                 inputTran, kernelTran, GEMM, outputTran);
            
        } else {
            winoF63_v1_multi(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
                        inputTran, kernelTran, GEMM, outputTran);
            // winoF63_v1_multi_v1(output_data, input_data, kernel_data, input_channels, output_channels, input_height, input_width, padding_left, padding_top, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads,
            //                 inputTran, kernelTran, GEMM, outputTran);
        }
    }
    inputTran.printBench("inputTran", n_loop);
    kernelTran.printBench("kernelTran", n_loop);
    GEMM.printBench("TensorGEMM", n_loop);
    outputTran.printBench("outputTran", n_loop);
    return 1;
}