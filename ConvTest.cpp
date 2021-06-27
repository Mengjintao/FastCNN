//#include "winograd_kernels.h"
//#include "winograd_kernels_F63.h"
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <arm_neon.h>
#include "./winoF63/winoF63.h"
#include <math.h>
#include "./conv_layer.h"
#include "./ncnn/ConvKernels.h"
#include "./convLayer/naiveConv.h"
#include "./convLayer/naiveNCNNConv.h"
#include "./convLayer/winoF63Conv.h"
#include "./convLayer/im2colConv.h"
#include "./convLayer/winoF63ConvZC.h"
#include "./utility/helper.h"

void fillTestInput(float* target, int inChannels, nnp_size inputDim){
    float* pChannel = target;
    for(int i = 0; i < inChannels; ++i){
	    float value = 1.0f;
    	for(int m = 0; m < inputDim.height; ++m){
            for(int n = 0; n < inputDim.width; ++n){
            //    value += 1.0f;
                value =  256 *rand() / double(RAND_MAX);
                *(pChannel++) = value;
            }
        }
    }
}

void fillTestKernel(float* target, int inChannels, int outChannels, nnp_size kernelDim){
    float* pChannel = target;
        for(int j = 0; j < outChannels; ++j){
    		for(int i = 0; i < inChannels; ++i){
				float value = 1.f;
				for(int m = 0; m < kernelDim.height; ++m){
					for(int n = 0; n < kernelDim.width; ++n){
					//    value += 0.1f;
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

struct options {
	size_t input_channels;
	size_t output_channels;
	struct nnp_size input_size;
	size_t input_padding;
	struct nnp_size kernel_size;
	struct nnp_size output_subsampling;
	enum nnp_convolution_algorithm algorithm;
	enum nnp_convolution_transform_strategy transform_strategy;
	enum nnp_convolution_tuning_strategy tuning_strategy;
	size_t threads;
	size_t iterations;
};

void print_options_help(const char* program_name) {
	printf(
        "%s parameters...\n"
        "Required parameters:\n"
        "  -ic  --input-channels     The number of input channels\n"
        "  -oc  --output-channels    The number of output channels\n"
        "  -is  --input-size         Input height and width\n"
        "  -ks  --kernel-size        Kernel height and width\n"
        "Optional parameters:\n"
        "  -a   --algorithm          The algorithm (auto, direct, im2col, or winograd) for computing convolution (default: auto)\n"
        "  -ts  --transform-strategy The transformation strategy (online, or offline) for kernel transformation (default: online)\n"
        "  -s   --output-subsampling The size of a output subsampling region, AKA stride (default: 1x1)\n"
        "  -ip  --input-padding      Implicit input padding (default: 1)\n"
		"  -tn  --tuning             whether to tuning (default: no_tuning)\n"
        "  -t   --threads            The number of threads (default: 1)\n"
        "  -i   --iterations         # iterations (default: 10)\n",
	program_name);
}

struct options parse_options(int argc, char** argv) {
	struct options options = {
		.input_channels = 0,
		.output_channels = 0,
		.input_size = { 0, 0 },
		.input_padding = 1,
		.kernel_size = { 0, 0 },
		.output_subsampling = { 1, 1 },
		.algorithm = nnp_convolution_algorithm_auto,
		.transform_strategy = nnp_convolution_transform_strategy_online,
		.tuning_strategy = nnp_convolution_tuning_strategy_no_tuning,
		.threads = 1,
		.iterations = 10
	};

	for (int i = 1; i < argc; i += 1) {
	    if ((strcmp(argv[i], "--input-channels") == 0) || (strcmp(argv[i], "-ic") == 0)) {
			if (i + 1 == argc) {
				fprintf(stderr, "Error: expected input channels value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 1], "%zu", &options.input_channels) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.input_channels == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of input channels: positive value expected\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			i += 1;
		} else if ((strcmp(argv[i], "--output-channels") == 0) || (strcmp(argv[i], "-oc") == 0)) {
			if (i + 1 == argc) {
				fprintf(stderr, "Error: expected output channels value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 1], "%zu", &options.output_channels) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.output_channels == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of output channels: positive value expected\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			i += 1;
		} else if ((strcmp(argv[i], "--input-size") == 0) || (strcmp(argv[i], "-is") == 0)) {
			if (argc - i < 2) {
				fprintf(stderr, "Error: expected two input size values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 1], "%zu", &options.input_size.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.input_size.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the input height: positive value expected\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 2], "%zu", &options.input_size.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 2]);
				exit(EXIT_FAILURE);
			}
			if (options.input_size.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the input width: positive value expected\n", argv[i + 2]);
				exit(EXIT_FAILURE);
			}
			i += 2;
		} else if ((strcmp(argv[i], "--kernel-size") == 0) || (strcmp(argv[i], "-ks") == 0)) {
			if (argc - i < 2) {
				fprintf(stderr, "Error: expected two kernel size values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 1], "%zu", &options.kernel_size.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.kernel_size.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the kernel height: positive value expected\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 2], "%zu", &options.kernel_size.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 2]);
				exit(EXIT_FAILURE);
			}
			if (options.kernel_size.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the kernel width: positive value expected\n", argv[i + 2]);
				exit(EXIT_FAILURE);
			}
			i += 2;
		} else if ((strcmp(argv[i], "--input-padding") == 0) || (strcmp(argv[i], "-ip") == 0)) {
			if (i + 1 == argc) {
				fprintf(stderr, "Error: expected padding value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 1], "%zu", &options.input_padding) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			i += 1;
		} else if ((strcmp(argv[i], "--output-subsampling") == 0) || (strcmp(argv[i], "-s") == 0)) {
			if (argc - i < 2) {
				fprintf(stderr, "Error: expected two output subsampling values\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 1], "%zu", &options.output_subsampling.height) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.output_subsampling.height == 0) {
				fprintf(stderr, "Error: invalid value %s for the output subsampling height: positive value expected\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 2], "%zu", &options.output_subsampling.width) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 2]);
				exit(EXIT_FAILURE);
			}
			if (options.output_subsampling.width == 0) {
				fprintf(stderr, "Error: invalid value %s for the output subsampling width: positive value expected\n", argv[i + 2]);
				exit(EXIT_FAILURE);
			}
			i += 2;
		} else if ((strcmp(argv[i], "--algorithm") == 0) || (strcmp(argv[i], "-a") == 0)) {
			if (i + 1 == argc) {
				fprintf(stderr, "Error: expected convolution algorithm name\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[i + 1], "auto") == 0) {
				options.algorithm = nnp_convolution_algorithm_auto;
			} else if (strcmp(argv[i + 1], "direct") == 0) {
				options.algorithm = nnp_convolution_algorithm_direct;
            } else if (strcmp(argv[i + 1], "im2col") == 0) {
				options.algorithm = nnp_convolution_algorithm_im2col;
            } else if (strcmp(argv[i + 1], "winograd") == 0) {
				options.algorithm = nnp_convolution_algorithm_winograd;
			} else {
				fprintf(stderr, "Error: invalid convolution algorithm name %s\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			i += 1;
		} else if ((strcmp(argv[i], "--transform-strategy") == 0) || (strcmp(argv[i], "-ts") == 0)) {
			if (i + 1 == argc) {
				fprintf(stderr, "Error: expected transformation strategy name\n");
				exit(EXIT_FAILURE);
			}
			if (strcmp(argv[i + 1], "online") == 0) {
				options.transform_strategy = nnp_convolution_transform_strategy_online;
			} else if (strcmp(argv[i + 1], "offline") == 0) {
				options.transform_strategy = nnp_convolution_transform_strategy_offline;
			} else {
				fprintf(stderr, "Error: invalid trasnformation strategy name %s\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			i += 1;
		} else if ((strcmp(argv[i], "--threads") == 0) || (strcmp(argv[i], "-t") == 0)) {
			if (i + 1 == argc) {
				fprintf(stderr, "Error: expected number of threads value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 1], "%zu", &options.threads) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.threads == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of threads: positive value expected\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			i += 1;
		} else if ((strcmp(argv[i], "--iterations") == 0) || (strcmp(argv[i], "-i") == 0)) {
			if (i + 1 == argc) {
				fprintf(stderr, "Error: expected iterations value\n");
				exit(EXIT_FAILURE);
			}
			if (sscanf(argv[i + 1], "%zu", &options.iterations) != 1) {
				fprintf(stderr, "Error: can not parse %s as an unsigned integer\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			if (options.iterations == 0) {
				fprintf(stderr, "Error: invalid value %s for the number of iterations: positive value expected\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			i += 1;
		} else if ((strcmp(argv[i], "--tuning") == 0) || (strcmp(argv[i], "-tn") == 0)) {
			if (i + 1 == argc) {
				fprintf(stderr, "Error: expected iterations value\n");
				exit(EXIT_FAILURE);
			} 
			if (strcmp(argv[i + 1], "no_tuning") == 0) {
				options.tuning_strategy = nnp_convolution_tuning_strategy_no_tuning;
			} else if (strcmp(argv[i + 1], "tuning") == 0) {
				options.tuning_strategy = nnp_convolution_tuning_strategy_grid_search;
			} else {
				fprintf(stderr, "Error: invalid tuning name %s\n", argv[i + 1]);
				exit(EXIT_FAILURE);
			}
			i += 1;
		} else if ((strcmp(argv[i], "--help") == 0) || (strcmp(argv[i], "-h") == 0)) {
			print_options_help(argv[0]);
			exit(EXIT_SUCCESS);
		} else {
			fprintf(stderr, "Error: unknown argument '%s'\n", argv[i]);
			print_options_help(argv[0]);
			exit(EXIT_FAILURE);
		}
	}
	if (options.input_channels == 0) {
		fprintf(stderr, "Error: the number of input channels is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.output_channels == 0) {
		fprintf(stderr, "Error: the number of output channels is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.input_size.width == 0) {
		fprintf(stderr, "Error: the input size is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (options.kernel_size.width == 0) {
		fprintf(stderr, "Error: the kernel size is not specified\n");
		print_options_help(argv[0]);
		exit(EXIT_FAILURE);
	}
	return options;
}

enum nnp_convolution_algorithm select_algorithm(struct nnp_size kernel_size, struct nnp_size output_subsampling, struct nnp_size output_size) {
	if (std::max(output_subsampling.height, output_subsampling.width) == 1) {
		// Stride-1 convolution: consider fast convolution algorithm and direct 1x1
		if (std::max(kernel_size.height, kernel_size.width) == 1) {
			return nnp_convolution_algorithm_direct;
		} else if (kernel_size.height == 3 && kernel_size.width == 3) {
			return nnp_convolution_algorithm_winograd;
		}
	}

	/* Fall-back algorithm */
	return nnp_convolution_algorithm_im2col;

}

int main(int argc, char* argv[]){
    srand((unsigned)time(NULL)); 

	struct options options = parse_options(argc, argv);

	const size_t input_channels = options.input_channels;
	const size_t output_channels = options.output_channels;
	const struct nnp_padding input_padding = { options.input_padding, options.input_padding, options.input_padding, options.input_padding };
	const struct nnp_size input_size = options.input_size;
	const struct nnp_size kernel_size = options.kernel_size;
	const struct nnp_size output_subsampling = options.output_subsampling;
	const struct nnp_size output_size = {
		.width = (input_padding.left + input_size.width + input_padding.right - kernel_size.width) / output_subsampling.width + 1,
		.height = (input_padding.top + input_size.height + input_padding.bottom - kernel_size.height) / output_subsampling.height + 1
	};

	printf("Input channels: %zu\n", input_channels);
	printf("Output channels: %zu\n", output_channels);
	printf("Input: %zux%zu with implicit padding %zu\n", input_size.height, input_size.width, options.input_padding);
	printf("Kernel: %zux%zu\n", kernel_size.height, kernel_size.width);
	printf("Subsampling: %zux%zu\n", output_subsampling.height, output_subsampling.width);

    float* test_input  = (float *) malloc(sizeof(float) * input_size.height  * input_size.width  * input_channels);
    float* test_kernel = (float *) malloc(sizeof(float) * kernel_size.height * kernel_size.width * input_channels * output_channels);
    fillTestInput(test_input, input_channels, input_size);
    fillTestKernel(test_kernel, input_channels, output_channels, kernel_size);

	ConvNaiveLayer conv_reference(test_input, test_kernel, NULL, NULL, input_channels, input_size.height, input_size.width, output_channels);
    conv_reference.Forward();

	ConvLayer* conv_test;

	if (options.algorithm == nnp_convolution_algorithm_auto)
		options.algorithm = select_algorithm(kernel_size, output_subsampling, output_size);

	switch (options.algorithm) {
		case nnp_convolution_algorithm_direct: {

		} break;

		case nnp_convolution_algorithm_im2col: {
			// conv_test = new ConvIm2colLayer(test_input, test_kernel, NULL, conv_reference.output_data,
			// 	 							input_channels, input_size.height, input_size.width, output_channels,
			// 								kernel_size.height, kernel_size.width, output_subsampling.height, output_subsampling.width,
			// 								input_padding.left, input_padding.right, input_padding.top, input_padding.bottom,
			// 								1, false,
			// 								options.threads, options.iterations);
			ConvIm2colLayer conv(test_input, test_kernel, NULL, NULL,
								input_channels, input_size.height, input_size.width, output_channels,
								kernel_size.height, kernel_size.width, output_subsampling.height, output_subsampling.width,
								input_padding.left, input_padding.right, input_padding.top, input_padding.bottom,
								1, false,
								options.threads, options.iterations);	

			if (options.tuning_strategy != nnp_convolution_tuning_strategy_no_tuning)	
				conv.Tuning();
			conv.Init();
			conv.Forward();	
		} break;

		case nnp_convolution_algorithm_winograd: {
			// conv_test = new ConvWinoF63ZCLayer(test_input, test_kernel, NULL, conv_reference.output_data,
			// 								input_channels, input_size.height, input_size.width, output_channels,
			// 								kernel_size.height, kernel_size.width, output_subsampling.height, output_subsampling.width,
			// 								input_padding.left, input_padding.right, input_padding.top, input_padding.bottom,
			// 								1, false,
			// 								options.threads, options.iterations);
			ConvWinoF63ZCLayer conv(test_input, test_kernel, NULL, conv_reference.output_data,
									input_channels, input_size.height, input_size.width, output_channels,
									kernel_size.height, kernel_size.width, output_subsampling.height, output_subsampling.width,
									input_padding.left, input_padding.right, input_padding.top, input_padding.bottom,
									1, false,
									options.threads, options.iterations);
									
			if (options.tuning_strategy != nnp_convolution_tuning_strategy_no_tuning)	
				conv.Tuning();
			conv.Init();
			conv.Forward();

		} break;
	}

	// if (options.tuning_strategy != nnp_convolution_tuning_strategy_no_tuning) {
	// 	conv_test->Tuning();
	// }
	// conv_test->Init();
	// conv_test->Forward();

	// free(conv_test);

    return 0;
}
