#include "./ncnn/ConvKernels.hpp"
//#include "winograd_kernels.h"
//#include "winograd_kernels_F63.h"
#include "./utility/helper.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <arm_neon.h>
#include "./winoF63/winoF63.h"
#include <math.h>
#include "./conv_layer.h"
#include "./convLayer/naiveConv.h"

void fillTestInput(float* target, int inChannels, nnp_size inputDim){
    float* pChannel = target;
    for(int i = 0; i < inChannels; ++i){
        
    for(int m = 0; m < inputDim.height; ++m){
	    float value = 1.0f;
            for(int n = 0; n < inputDim.width; ++n){
//                value += 0.1f;
                value =  256 *rand() / double(RAND_MAX);
                *(pChannel++) = value;
            }
        }
    }
}

void fillTestKernel(float* target, int inChannels, int outChannels, nnp_size kernelDim){
    float* pChannel = target;
    
    for(int i = 0; i < inChannels; ++i){
        for(int j = 0; j < outChannels; ++j){
        float value = 1.f;
        for(int m = 0; m < kernelDim.height; ++m){
            for(int n = 0; n < kernelDim.width; ++n){
//                value += 0.01f;
                value =  1.0* rand() / double(RAND_MAX);
                *(pChannel++) = value;
            }
        }
        }
    }
}

void printMatrix(float* matrix, int row, int col)
{
	printf("--------------\n");
	for(int i=0;i<row;i++)	
	{
		for(int j=0;j<col;j++)
			printf("%3.3f ", *(matrix+i*col+j));
		printf("\n");
	}
}


int main(int argc, char* argv[]){
    srand((unsigned)time(NULL)); 


    char testName[256];
    Timer timer;
    int inputChannels, outputChannels;
    nnp_size inputDim, outputDim, kernelDim, subsampling;
    nnp_padding paddings;
    int tileBlock = 1;
    int ocBlock   = 1;
    int icBlock   = 1;
    int ocRegBlock = 4;
    int tileRegBlock = 4;
#if 0
    inputChannels = 64;
    inputDim.height = 226;
    inputDim.width = 226;
#else
    inputChannels = 256;
    inputDim.height = 58;
    inputDim.width = 58;
#endif
    kernelDim.height = 3;
    kernelDim.width = 3;
    
   
    subsampling.height = 1;
    subsampling.width = 1;
    
    paddings.top = 1;
    paddings.bottom = 1;
    paddings.left = 1;
    paddings.right = 1;

    outputChannels = 64;
    int num_threads = 1; 

    int pad_width = 1;
    int pad_height = 1;
    int stride_width = 0;
    int stride_height = 0;
    bool enableOffKernel = 0;

    if(argc != 11){
	printf("%d\n", argc);
	printf("usage: ./wingorad_dev [in_channels] [out_channels] [image dim] [tileBlock] [ocBlock] [icBlock] [#threads]\n");
	return 0;
    } else {
	inputChannels   = atoi(argv[1]);
//	outputChannels  = inputChannels;
	outputChannels  = atoi(argv[2]);
	inputDim.width  = atoi(argv[3]);
	inputDim.height = inputDim.width;
	tileBlock       = atoi(argv[4]);
	ocBlock         = atoi(argv[5]);
    	icBlock         = atoi(argv[6]);

	ocRegBlock      = atoi(argv[7]);
    	tileRegBlock    = atoi(argv[8]);
	enableOffKernel = bool(atoi(argv[9]));
    	num_threads     = atoi(argv[10]);
	
	printf("Testing ic=%d oc=%d width=%d tileBlock=%d ocBlock=%d icBlock=%d threads=%d\n", inputChannels, outputChannels, inputDim.width, tileBlock, ocBlock, icBlock, num_threads);
    } 
 
    outputDim.height = inputDim.height - kernelDim.height + 1 + 2*pad_height;
    outputDim.width  = inputDim.width  - kernelDim.width  + 1 + 2*pad_width;

    float* testInput       = (float *) malloc(sizeof(float) * inputDim.height  * inputDim.width  * inputChannels);
    float* paddedInput     = (float *) malloc(sizeof(float) * inputDim.height  * inputDim.width  * inputChannels);
    float* testKernel      = (float *) malloc(sizeof(float) * kernelDim.height * kernelDim.width * inputChannels * outputChannels);
    float* baseResult      = (float *) malloc(sizeof(float) * outputDim.height * outputDim.width * outputChannels);
    float* naiveResult     = (float *) malloc(sizeof(float) * outputDim.height * outputDim.width * outputChannels);
/*
    float* ncnnResult      = (float *) malloc(sizeof(float) * outputDim.height * outputDim.width * outputChannels);
    float* winogradResult  = (float *) malloc(sizeof(float) * outputDim.height * outputDim.width * outputChannels);
    float* winogradResult2 = (float *) malloc(sizeof(float) * outputDim.height * outputDim.width * outputChannels);
  */  
    fillTestInput(testInput, inputChannels, inputDim);
    fillTestKernel(testKernel, inputChannels, outputChannels, kernelDim);

//    ConvLayer conv(testInput, testKernel, NULL, inputChannels, inputDim.height, inputDim.width, outputChannels);
//    conv.Forward();


//    printMatrix(testKernel, 8, 9);
    
    winoF63_naive(naiveResult, testInput, testKernel, inputChannels, outputChannels, inputDim.height, inputDim.width, pad_width, pad_height, num_threads);
//    timer.startBench();
//    conv3x3s1_neon(testInput, inputChannels, inputDim, testKernel, kernelDim, ncnnResult, outputChannels, outputDim, paddings, subsampling);
//    timer.endBench("NCNN wall clock: ");
/*
    float *WT = (float *) malloc(sizeof(float) * 64 * (inputDim.width / 2 - 1) * (inputDim.height / 2 - 1) * outputChannels);
    float *VT = (float *) malloc(sizeof(float) * 64 * (inputDim.width / 2 - 1) * (inputDim.height / 2 - 1) * inputChannels);
    float *UT = (float *) malloc(sizeof(float) * 64 * inputChannels * outputChannels);
    float *ST = (float *) malloc(sizeof(float) * 64 * inputChannels * outputChannels);

    size_t packArraySize = getPackArraySize_F6x6_3x3(inputChannels, num_threads);
    float *packArray = (float *) malloc(sizeof(float) * packArraySize);

    EP::transformKernel(UT, testKernel, inputChannels, outputChannels, ST);
    EP::winogradNonFusedTransform(winogradResult, outputChannels, WT, VT, UT, testInput, inputChannels, inputDim.width, inputDim.height, None, NULL, num_threads);

    float diffRet2 =diff(winogradResult, naiveResult, outputChannels* outputDim.height * outputDim.width);
*/
    int warmup = 5;
    int nloop = 30;
    icBlock = inputChannels;

    float *inputBuf      = new float [icBlock*tileBlock*64]; 
    float *gemmBuf       = new float [ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64];
    float *kernelBuf;
    if(enableOffKernel)	
	kernelBuf = new float [inputChannels * outputChannels * 64];  	
    else	       	
        kernelBuf = new float [icBlock * ocBlock * 64];

    retransformKernel(testKernel, outputChannels, inputChannels, ocRegBlock);
    if(enableOffKernel)
    	offlineKernelTransform(kernelBuf, testKernel, outputChannels, inputChannels, ocBlock, ocRegBlock);
    printf("kernelBuf %d KB\n", icBlock*ocBlock*64*4/1024);   
    printf("gemmBuf %d KB\n",   (ocRegBlock*tileRegBlock*36 + ocBlock*tileBlock*64)*4/1024);   
    printf("inputBuf %d KB\n", tileBlock*icBlock*64*4/1024);  
    printf("L1 Cache used %d KB\n", (tileBlock*ocBlock*48 + icBlock*ocBlock*64 + tileBlock*icBlock*64)*4/1024);

    for(int i=0;i<warmup;i++)
    winoF63(baseResult, testInput, testKernel, inputChannels, outputChannels, inputDim.height, inputDim.width, pad_width, pad_height, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads);

    timer.startBench();
    for(int i = 0; i < nloop; ++i){
    	winoF63(baseResult, testInput, testKernel, inputChannels, outputChannels, inputDim.height, inputDim.width, pad_width, pad_height, stride_width, stride_height, tileBlock, gemmBuf, ocBlock, kernelBuf, icBlock, inputBuf, tileRegBlock, ocRegBlock, enableOffKernel, num_threads);
    }
    sprintf(testName, "ic %d oc %d img %d ocB %d tB %d oRB %d tRB %d offKernel %d winoF63 ", inputChannels, outputChannels,inputDim.width, ocBlock,  tileBlock, ocRegBlock, tileRegBlock, enableOffKernel);
    timer.endBench(testName, (double) nloop);
    
    fflush(stdout); 
    delete  [] gemmBuf;
    delete  [] kernelBuf;
    delete  [] inputBuf;
    gemmBuf   = NULL;
    kernelBuf = NULL;
    inputBuf  = NULL;
     
    printf("%d %d %d\n", outputChannels, outputDim.height, outputDim.width); 
    float diffRet = diff(baseResult, naiveResult, outputChannels* outputDim.height * outputDim.width);
    printf("%s difference is %5.3f\n", testName,  diffRet);

    fflush(stdout); 
//    diff(naiveResult, ncnnResult, outputChannels * outputDim.height * outputDim.width);

    for(int index=0;index<0;index++)
    {
    	printMatrix(baseResult+index*19*19,  19, 19);
    	printMatrix(naiveResult+index*19*19, 19, 19);
    }
/*
    transformKernel_F6x6_3x3(UT, testKernel, inputChannels, outputChannels);
    for(int i=0;i<warmup;i++)
    {pad_input(paddedInput, testInput, inputChannels, inputDim.width, inputDim.height, pad_width, pad_height, pad_width, pad_height);
    winogradNonFusedTransform_F6x6_3x3(winogradResult2, outputChannels, WT, VT, UT, paddedInput, inputChannels, inputDim.width+pad_width*2, inputDim.height+pad_height*2, None, NULL, packArray, num_threads);
    }
//  winogradNonFusedTransform_F6x6_3x3(winogradResult2, outputChannels, WT, VT, UT, testInput, inputChannels, inputDim.width, inputDim.height, None, NULL, packArray, num_threads);

    timer.startBench();
    for(int i = 0; i < nloop; ++i){
//    	winogradNonFusedTransform_F6x6_3x3(winogradResult2, outputChannels, WT, VT, UT, testInput, inputChannels, inputDim.width, inputDim.height, None, NULL, packArray, num_threads);
        pad_input(paddedInput, testInput, inputChannels, inputDim.width, inputDim.height, pad_width, pad_height, pad_width, pad_height);
        winogradNonFusedTransform_F6x6_3x3(winogradResult2, outputChannels, WT, VT, UT, paddedInput, inputChannels, inputDim.width+pad_width*2, inputDim.height+pad_height*2, None, NULL, packArray, num_threads);
    }

    sprintf(testName, "ic %d oc %d img %d EPF63", inputChannels, outputChannels,inputDim.width);
    timer.endBench(testName, (double) nloop);

    diffRet = diff(winogradResult2, naiveResult, outputChannels* outputDim.height * outputDim.width);
    printf("%s difference is %5.3f\n", testName,  diffRet);
*/
    fflush(stdout); 
    free(testInput);
    free(testKernel);
    free(baseResult);
    free(naiveResult);
//    free(winogradResult);
//    free(VT);
//    free(ST);
//    free(UT);
    return 0;
}