#ifndef IM2COLCONV_H
#define IM2COLCONV_H

#include "../conv_layer.h"
#include "../im2col/gemm_pack.h"
#include "../im2col/gemm_kernel.h"


class ConvIm2colLayer : public ConvLayer {

public:
    ConvIm2colLayer(float *input, float *kernel, float *biasw, float *output_ref,
                    size_t ic, size_t ih, size_t iw, size_t oc,
                    size_t kh=3, size_t kw=3, size_t sh=1, size_t sw=1, 
                    size_t pad_left=1, size_t pad_right=1, size_t pad_top=1, size_t pad_bottom=1,
                    size_t g=1, bool bias=0, size_t nt=1, size_t iter=10,
                    int mc=8, int nc=256, int kc=384, int gemm_version=1, int row_batch=8, int col_batch=8,
                    int pack_c_version=0, int mt_pack_b_version=2, int prefetch_a=256, int prefetch_b=256, int prefetch_c=0)
	                : ConvLayer(input, kernel, biasw, output_ref, ic, ih, iw, oc, kh, kw, sh, sw, pad_left, pad_right, pad_top, pad_bottom, g, bias, nt, iter),
                    mc(mc), nc(nc), kc(kc), row_batch(row_batch), col_batch(col_batch),
                    pack_c_version(pack_c_version), mt_pack_b_version(mt_pack_b_version), prefetch_a(prefetch_a), prefetch_b(prefetch_b), prefetch_c(prefetch_c) {
        
        this->M = output_channels;
        this->N = output_height * output_width;
        this->K = input_channels * kernel_height * kernel_width;
	    this->num_threads = num_threads;
        this->transform_input_data = static_cast<float*>(malloc(sizeof(float) * output_height * output_width * input_channels * kernel_height * kernel_width));
        this->output_data = static_cast<float*>(realloc(this->output_data ,sizeof(float) * (output_height * output_width * output_channels + 128))); 
        if (this->num_threads > 1) {
            this->gemm_version = GEMM_BLOCKS_MULTI_THREADS;
        } else {
            if (M <= 32 && N <= 32 && K <= 32)
                this->gemm_version = GEMM_NO_BLOCKS;
            else
                this->gemm_version = GEMM_BLOCKS_SINGLE_THREAD;
        }
    }

    ~ConvIm2colLayer() {
        free(this->transform_input_data);
    }
        
    int Init();
    int Tuning();
    int Forward();

    void im2col();
    void im2col_v1();

    void sgemm(); 
    void GEMM(float* A, float* B, float* C);
    void GEMM_v2(float* A, float* B, float* C);
    void GEMM_multithread(float* A, float* B, float* C);

protected:
    typedef void (*PackA)(int, int, float *, int, float *, int, int, int, const int, const int);
    typedef void (*PackB)(int, int, float *, int, float *, const int, const int);
    typedef void (*PackBMT)(int, int, int, int, float*, int, float*, int, int, const int, const int);
    typedef void (*PackC)(int, int, int, float*, float*, int, const int, const int);
    typedef void (*UnpackC)(int, int, int, float*, float*, int, const int, const int);
    typedef void (*InnerKernel)(int, float *, float *, float *, int, const int, const int, const int);
    typedef void (*InnerKernelForCorner)(int, float *, float *, float *, int);

    PackA pack_a;
    PackB pack_b;
    PackBMT pack_b_mt;
    PackC pack_c;
    UnpackC unpack_c;
    InnerKernel inner_kernel;
    InnerKernelForCorner inner_kernel_for_corner;

    int M;
    int N;
    int K;
    int mc;
    int nc;
    int kc;
    int row_batch;
    int col_batch;
    int pack_c_version;
    int mt_pack_b_version;
    int prefetch_a;
    int prefetch_b;
    int prefetch_c;
    int gemm_version;
    float *transform_input_data;

    static const int GEMM_NO_BLOCKS = 0;
    static const int GEMM_BLOCKS_SINGLE_THREAD = 1;
    static const int GEMM_BLOCKS_MULTI_THREADS = 2;
    static const int GEMM_BLOCKS_MULTI_THREADS_2D = 3;

    void set_pack_a();
    void set_pack_b();
    void set_pack_b_mt();
    void set_pack_c();
    void set_unpack_c();
    void set_inner_kernel();
    void set_inner_kernel_for_corner(int k);

    void select_tuning_range_for_mnk(size_t&, size_t&, int&, int&, int&, int&, int&, int&, int&, int&, int&);
    void select_tuning_range_for_pack(int&, int&, int&, int&, int&, int&);
    void select_tuning_range_for_prefetch(int&, int&, int&, int&, int&, int&, int&, int&, int&);

    bool search_log_file_and_entry(const char *log_path);
    void search_best_param(int &best_mc, int &best_nc, int &best_kc, int &best_rb, int &best_cb, int &best_pc, int &best_pb, int &best_pre_a, int &best_pre_b, int &best_pre_c);
    void write_best_param(const char *log_path, int &best_mc, int &best_nc, int &best_kc, int &best_rb, int &best_cb, int &best_pc, int &best_pb, int &best_pre_a, int &best_pre_b, int &best_pre_c);
};

#endif 
