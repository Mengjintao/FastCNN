#ifndef WINOF63CONVZC_H_
#define WINOF63CONVZC_H_

#include "../utility/helper.h"
#include "./conv_layer.h"

#define WINOGRAD_V1

class ConvWinoF63ZCLayer : public ConvLayer
{
public:
    ConvWinoF63ZCLayer(float *input, float *kernel, float *biasw, float *output_ref, size_t ic, size_t ih, size_t iw, size_t oc, 
						size_t kh=3, size_t kw=3, size_t sh=1, size_t sw=1, size_t pad_left=1, size_t pad_right=1, size_t pad_top=1, size_t pad_bottom=1,
						size_t g=1, bool bias=0, size_t nt=1, size_t iter=10);
    ~ConvWinoF63ZCLayer();

    int Tuning();
    int Init();
    int Forward();
    
private:
	int tileBlock;
	int ocBlock;
    int icBlock;
	int scheduling;

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
	int icBlock_best;
	int enableOffKernel_best;
	int scheduling_best;
};

#endif