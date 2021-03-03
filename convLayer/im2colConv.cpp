#include "im2colConv.h"
#include "../utility/common.h"
#include <omp.h>

int ConvIm2colLayer::Init() {
    this->set_pack_a();
    this->set_pack_b();
    this->set_pack_c();
    this->set_unpack_c();
    this->set_inner_kernel();
    this->set_inner_kernel_for_corner(output_channels % row_batch);
    if (gemm_version == GEMM_BLOCKS_MULTI_THREADS)
        this->set_pack_b_mt();
    return 1;
}

int ConvIm2colLayer::Forward() {
    this->im2col();
    // this->im2col_v1();
    this->sgemm();
    return 1;
}

int ConvIm2colLayer::Tuning() {
    return 1;
}

void ConvIm2colLayer::im2col() {
    int stride = output_height * output_width * kernel_height * kernel_width;
    float* ret = transform_input_data;
    for (int ic = 0; ic < input_channels; ic++) {
        int ret_idx = ic * stride;
        for (int u = 0; u < kernel_height; u++) {
            for (int v = 0; v < kernel_width; v++) {
                for (int i = 0; i < output_height; i++) {
                    for (int j = 0; j < output_width; j++) {
                        int row = u - padding_top + i * stride_height;
                        int col = v - padding_left + j * stride_width;
                        if (row < 0 || row >= input_height || col < 0 || col >= input_width) {
                            ret[ret_idx] = 0;
                        }
                        else {
                            int idx = ic * input_height * input_width + row * input_width + col;
                            ret[ret_idx] = input_data[idx];
                        }
                        ++ret_idx;
                    }
                }
            }
        }
    }
}

void ConvIm2colLayer::im2col_v1() {
    int padding_input_width  = input_width  + padding_left + padding_right;
    int padding_input_height = input_height + padding_top  + padding_bottom;
    int padding_input_size   = padding_input_height * padding_input_width;
    int input_size = input_height * input_width;

    float* padding_input_data = static_cast<float*>(_mm_malloc(sizeof(float) * padding_input_size * input_channels, 64));
    memset(padding_input_data, 0, sizeof(float) * padding_input_size * input_channels);

    for (int ic = 0; ic < input_channels; ic++) {
        float* padding_input_ptr = padding_input_data + ic * padding_input_size + padding_top * padding_input_width + padding_left;
        float* input_ptr         = input_data + ic * input_size;
        for (int row = 0; row < input_height; row++) {
            memcpy(padding_input_ptr, input_ptr, sizeof(float) * input_width);
            padding_input_ptr += padding_input_width;
            input_ptr += input_width;
        }
    }
    int transform_input_height = input_channels * kernel_height * kernel_width;
    int transform_input_width  = output_height  * output_width;
    int transform_input_size = transform_input_height * transform_input_width;
    float* in, *in_ptr;
    float* transform_in_ptr = transform_input_data;

    int padding_width = padding_left + padding_right;
    int step = (stride_height - 1) * padding_input_width - stride_width + padding_width;
    // input transform
    for (int ic = 0; ic < input_channels; ic++) {
        in = padding_input_data + ic*padding_input_size;
        for (int u = 0; u < kernel_height; u++) {
            for (int v = 0; v < kernel_width; v++) {
                in_ptr = in + u*padding_input_width + v;
                for (int i = 0; i < output_height; i++) {
                    for (int j = 0; j < output_width; j++) {
                        *transform_in_ptr = *in_ptr;
                        ++transform_in_ptr;
                        in_ptr += stride_width;
                    }
                    // in换行
                    in_ptr += step;
                }
            }
        }
    }

    _mm_free(padding_input_data);
}

void ConvIm2colLayer::sgemm() {
    switch(gemm_version) {
        case GEMM_NO_BLOCKS:
            GEMM(kernel_data, transform_input_data, output_data);
        break;

        case GEMM_BLOCKS_SINGLE_THREAD:
            GEMM_v2(kernel_data, transform_input_data, output_data);
        break;

        case GEMM_BLOCKS_MULTI_THREADS:
            GEMM_multithread(kernel_data, transform_input_data, output_data);
        break;
    }
}

void ConvIm2colLayer::GEMM(float* A, float* B, float* C) {
    int A_row = M;
    int A_col = K;
    int B_row = K;
    int B_col = N;
    int C_row = A_row;
    int C_col = B_col;

    // 这个版本虽然访存连续，但是矩阵B会被重复读取outputChannel次，增加了访存次数
    // 因此需要考虑分块，使矩阵中每一个块只需要被读一次，在访存连续的情况下，减少访存次数
   float* B_ptr;
   for (int k = 0; k < A_row; k++)
   {
       B_ptr = B;
       for (int i = 0; i < A_col; i++) 
       {
           float temp = *(A + k*A_col + i);
           for (int j = 0; j < B_col; j++)
           {
               *(C + k*C_col + j) += temp * *B_ptr;
               ++B_ptr;
           }
       }
   }

    // for(int i=0;i<A_row;i++){
    //     for(int j=0;j<B_col;j++){
    //         for(int k=0;k<A_col;k++){
    //             *(C + i * C_col + j ) += *(A + i * A_col + k) * *(B + k * B_col + j);
    //         }
    //     }
    // }
}

void ConvIm2colLayer::GEMM_v2(float* A, float* B, float* C) {
    int lda = K;
    int ldb = N;
    int ldc = N;

    Timer packA_timer;
    Timer packB_timer;
    Timer kernel_timer;

    // 保证row_batch 能被 mc整除，否则会有bug

    int packB_width;
    int packB_height;
    int packBC_width;
    int packBC_height;

    packBC_height = kc + mc;
    packBC_width = align_ceil(nc, col_batch);
    packB_height = kc;
    packB_width = packBC_width;
    // float *packB = new float[(kc + mc) * nc_ceil];
    // float *packC = packB + kc * nc_ceil;

    // float *packA = new float[M * kc]; 
    // float *packB = new float[packBC_height * packBC_width];
    float *packA = static_cast<float*>(_mm_malloc(M * kc * sizeof(float))); 
    float *packB = static_cast<float*>(_mm_malloc(packBC_height * packBC_width * sizeof(float)));
    float *packC = C;
    if (this->is_pack_c != 0)
        packC = packB + packB_height * packB_width;

    // A中的一大整列M * kc，B中的一大整行kc * N
    for (int kt = 0; kt < K; kt += kc)
    {
        int kc_adjust = min(kc, K - kt);
        // packA_timer.startBench();
        // 对A的一大整列做packing
        this->pack_a(M, kc_adjust, A + kt, lda, packA, mc, kc, nc, row_batch, col_batch);
        // packA_timer.accumBench();
        // B中的一大块kc * nc
        for (int nt = 0; nt < N; nt += nc)
        {
            int nc_adjust = min(nc, N - nt);
            int nc_ceil = align_ceil(nc_adjust, col_batch);

            float *packB_copy = packB;

	    // packB_timer.startBench();
            // 对B的一大块kc * nc做packing
            this->pack_b(kc_adjust, nc_adjust, B + kt*ldb + nt, ldb, packB_copy, row_batch, col_batch);
            // packB_timer.accumBench();

            // 接着就是对A的一大列M * kc和B的一大块kc * nc进行计算
            // 选择A的一大列的一小块
            for (int mt = 0; mt < M; mt += mc)
            {
                int mc_adjust = min(mc, M - mt);
                float *C_copy = C + mt * ldc + nt;
                float *packC_copy;
                int packC_ptr_step;
                int packC_copy_step;
                int packC_col;

                if (this->is_pack_c == 0) {
                    packC_ptr_step = 1;
                    packC_copy_step = ldc;
                    packC_copy = C_copy;
                    packC_col = ldc;
                }
                else if (this->is_pack_c == 1) {
                    packC_ptr_step = 1;
                    packC_copy_step = nc_ceil;
                    packC_copy = packC;
                    packC_col = nc_ceil;
                }
                else if (this->is_pack_c == 2) {
                    packC_ptr_step = mc_adjust;
                    packC_copy_step = col_batch;
                    packC_copy = packC;
                    packC_col = nc_ceil;
                }

                // kernel_timer.startBench();
                if (this->pack_c != nullptr)
                    this->pack_c(mc_adjust, nc_adjust, nc_ceil, C_copy, packC_copy, ldc, row_batch, col_batch);

                int start_remain_row = mc_adjust - mc_adjust % row_batch;
                int step = col_batch * kc_adjust;
                // 每次计算 row_batch 行
                int i = 0;
                for (; i < start_remain_row; i += row_batch) 
                {
                    float *packA_ptr = packA + (mt + i) * kc_adjust;
                    float *packB_ptr = packB_copy;
                    float *packC_copy_copy = packC_copy + i * packC_copy_step;
                    float *packC_ptr;
                    // 每次计算 col_batch 列。不足 col_batch 列的补0处理，因此也当成 col_batch 列进行计算
                    for (int j = 0; j < nc_ceil; j += col_batch)
                    {
                        packC_ptr = packC_copy_copy + j * packC_ptr_step;
                        this->inner_kernel(kc_adjust, packA_ptr, packB_ptr, packC_ptr, packC_col, prefetch_a, prefetch_b, this->prefetch_c);
                        packB_ptr += step;
                    }
                }

                // 剩余不足 row_batch 行，单独处理
                if (i < mc_adjust) {
                    float *packA_ptr = packA + (mt + i) * kc_adjust;
                    float *packB_ptr = packB_copy;
                    float *packC_copy_copy = packC_copy + i * packC_copy_step;
                    float *packC_ptr;
                    int num_remain_row = mc_adjust - start_remain_row;
                    //  每次计算 col_batch 列。不足 col_batch 列的补0处理，因此也当成 col_batch 列进行计算
                    for (int j = 0; j < nc_ceil; j += col_batch)
                    {
                        packC_ptr = packC_copy_copy + j * packC_ptr_step;
                        this->inner_kernel_for_corner(kc_adjust, packA_ptr, packB_ptr, packC_ptr, packC_col);
                        packB_ptr += step;
                    }
                }
                if (this->unpack_c != nullptr)
                    this->unpack_c(mc_adjust, nc_adjust, nc_ceil, C_copy, packC_copy, ldc, row_batch, col_batch);
            
                // kernel_timer.accumBench();
            }
        }
    }
    // packA_timer.printBench("packA time", 1);
    // packB_timer.printBench("packB time", 1);
    // kernel_timer.printBench("kernel time", 1);

    _mm_free(packA);
    _mm_free(packB);
}

// 多线程版本
void ConvIm2colLayer::GEMM_multithread(float* A, float* B, float* C) {
    int lda = K;
    int ldb = N;
    int ldc = N;

    int packB_width;
    int packB_height;
    int packBC_width;
    int packBC_height;

    int num_threads = this->num_threads;
    
    packBC_height = kc + num_threads * mc;
    packBC_width = align_ceil(nc, col_batch);
    packB_height = kc;
    packB_width = packBC_width;

    float *packA = new float[M * kc];
    float *packB = new float[packBC_height * packBC_width];
    // float *packA = _mm_malloc(M * kc * sizeof(float)); 
    // float *packB = _mm_malloc(packBC_height * packBC_width * sizeof(float));
    float *packC = packB + packB_height * packB_width;

    // 确定每个线程所负责的A中分块
    int m_num_block = (M + (mc - 1)) / mc;
    int m_num_block_per_thread = m_num_block / num_threads;
    int m_num_remain_block = m_num_block % num_threads;
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int m_interval = m_num_block_per_thread * mc;
        int m_from;
        int m_to;
        if (tid < m_num_remain_block) {
            m_from = tid * (m_num_block_per_thread + 1) * mc;
            m_to = m_from + (m_num_block_per_thread + 1) * mc;
        }
        else{
            m_from = (tid * m_num_block_per_thread + m_num_remain_block) * mc;
            m_to = m_from + m_num_block_per_thread * mc;
        }
        if (m_to > M) 
            m_to = M;

        if (m_to - m_from > 0) {
            m_interval = m_to - m_from;
            for (int kt = 0; kt < K; kt += kc) {
                int kc_adjust = min(kc, K - kt);

                float *packA_copy = packA + m_from * kc;
                this->pack_a(m_interval, kc_adjust, A + m_from*lda + kt, lda, packA_copy, mc, kc, nc, row_batch, col_batch);
        
                for (int nt = 0; nt < N; nt += nc) {
                    int nc_adjust = min(nc, N - nt);
                    int nc_ceil = align_ceil(nc_adjust, col_batch);

                    float *packB_copy = packB;
                    this->pack_b_mt(kc_adjust, nc_adjust, num_threads, tid, B, ldb, packB_copy, kt, nt, row_batch, col_batch);

                    for (int mt = m_from; mt < m_to; mt += mc) {
                        int mc_adjust = min(mc, m_to - mt);
                        float *C_copy = C + mt * ldc + nt;
                        float *packC_copy;
                        int packC_ptr_step;
                        int packC_copy_step;
                        int packC_col;
                        
                        if (this->is_pack_c == 0) {
                            packC_ptr_step = 1;
                            packC_copy_step = ldc;
                            packC_copy = C_copy;
                            packC_col = ldc;
                        }
                        else if (this->is_pack_c == 1) {
                            packC_ptr_step = 1;
                            packC_copy_step = nc_ceil;
                            packC_copy = packC + tid * mc * packBC_width;
                            packC_col = nc_ceil;
                        }
                        else if (this->is_pack_c == 2) {
                            packC_ptr_step = mc_adjust;
                            packC_copy_step = col_batch;
                            packC_copy = packC + tid * mc * packBC_width;
                            packC_col = nc_ceil;
                        }

                        if (this->pack_c != nullptr)
                            this->pack_c(mc_adjust, nc_adjust, nc_ceil, C_copy, packC_copy, ldc, row_batch, col_batch);

                        int start_remain_row = mc_adjust - mc_adjust % row_batch;
                        int step = col_batch * kc_adjust;

                        // 每次计算8行
                        for (int i = 0; i < start_remain_row; i += row_batch) {
                            float *packA_ptr = packA_copy + (mt - m_from + i) * kc_adjust;
                            float *packB_ptr = packB_copy;
                            float *packC_copy_copy = packC_copy + i * packC_copy_step;
                            float *packC_ptr;
                            // 每次计算8列。不足8列的补0处理，因此也当成8列进行计算
                            for (int j = 0; j < nc_ceil; j += col_batch) {
                                packC_ptr = packC_copy_copy + j * packC_ptr_step;
                                // kernel_8x8(kc_adjust, packA_ptr, packB_ptr, packC_ptr, nc_ceil);
                                this->inner_kernel(kc_adjust, packA_ptr, packB_ptr, packC_ptr, packC_col, this->prefetch_a, this->prefetch_b, this->prefetch_c);
                                packB_ptr += step;
                            }
                        
                        }

                        // 不足8行，单独处理
                        int i = start_remain_row;
                        if (i < mc_adjust) {
                            float *packA_ptr = packA_copy + (mt - m_from + i) * kc_adjust;
                            float *packB_ptr = packB_copy;
                            float *packC_copy_copy = packC_copy + i * packC_copy_step;
                            float *packC_ptr;
                            int num_remain_row = mc_adjust - start_remain_row;
                            // 每次计算8列。不足8列的补0处理，因此也当成8列进行计算
                            for (int j = 0; j < nc_ceil; j += col_batch) {
                                packC_ptr = packC_copy_copy + j * packC_ptr_step;
                                // kernel_Nx8_for_8x8_template(num_remain_row, kc_adjust, packA_ptr, packB_ptr, packC_ptr, nc_ceil);
                                this->inner_kernel_for_corner(kc_adjust, packA_ptr, packB_ptr, packC_ptr, packC_col);
                                packB_ptr += step;
                            }
                        }
                        if (this->unpack_c != nullptr)
                            this->unpack_c(mc_adjust, nc_adjust, nc_ceil, C_copy, packC_copy, ldc, row_batch, col_batch);
                    }
                }
            }
        }
    }

    delete [] packA;
    delete [] packB;
    // delete [] packC;
}

void ConvIm2colLayer::set_pack_a() {
    this->pack_a = pack_a_v2;
}

void ConvIm2colLayer::set_pack_b() {
    if (row_batch == 8 && col_batch == 8) 
        this->pack_b = pack_b_v2_8x8;
    else if (row_batch == 8 && col_batch == 12)
        this->pack_b = pack_b_v2_8x12;
    else if (row_batch == 4 && col_batch == 16)
        this->pack_b = pack_b_v2_4x16;
}

void ConvIm2colLayer::set_pack_b_mt() {
    if (row_batch == 8 && col_batch == 8) {
        switch (gemm_version) {
            case 0:
                this->pack_b_mt = pack_b_multithread_8x8_v1;
            case 1:
                this->pack_b_mt = pack_b_multithread_8x8_v2;
            case 2:
                this->pack_b_mt = pack_b_multithread_8x8_v3;
            default:
                break;
        }
    }
    else if (row_batch == 8 && col_batch == 12) {
        switch (gemm_version) {
            case 0:
                this->pack_b_mt = pack_b_multithread_8x12_v1;
            case 1:
                this->pack_b_mt = pack_b_multithread_8x12_v2;
            case 2:
                this->pack_b_mt = pack_b_multithread_8x12_v3;
            default:
                break;
        }
        
    }
    else if (row_batch == 4 && col_batch == 16) {
        switch (gemm_version) {
            case 0:
                this->pack_b_mt = pack_b_multithread_4x16_v1;
            case 1:
                this->pack_b_mt = pack_b_multithread_4x16_v2;
            case 2:
                this->pack_b_mt = pack_b_multithread_4x16_v3;
            default:
                break;
        }
    }
}

void ConvIm2colLayer::set_pack_c() {
    if (row_batch == 8 && col_batch == 8) {
        if (is_pack_c == 0)
            this->pack_c = nullptr;
        else if (is_pack_c == 1)
            this->pack_c = load_c_v2_8x8;
        else
            this->pack_c = load_c_v2_8x8_pack;
    } else if (row_batch == 8 && col_batch == 12) {
        if (is_pack_c == 0)
            this->pack_c = nullptr;
        else if (is_pack_c == 1)
            this->pack_c = load_c_v2_8x12;
        else
            this->pack_c = load_c_v2_8x12_pack;
    } else if (row_batch == 4 && col_batch == 16) {
        if (is_pack_c == 0)
            this->pack_c = nullptr;
        else if (is_pack_c == 1)
            this->pack_c = load_c_v2_4x16;
        else
            this->pack_c = load_c_v2_4x16_pack;
    }
}

void ConvIm2colLayer::set_unpack_c() {
    if (row_batch == 8 && col_batch == 8) {
        if (is_pack_c == 0)
            this->unpack_c = nullptr;
        else if (is_pack_c == 1)
            this->unpack_c = write_c_v2_8x8;
        else
            this->unpack_c = write_c_v2_8x8_unpack;
    } else if (row_batch == 8 && col_batch == 12) {
        if (is_pack_c == 0)
            this->unpack_c = nullptr;
        else if (is_pack_c == 1)
            this->unpack_c = write_c_v2_8x12;
        else
            this->unpack_c = write_c_v2_8x12_unpack;
    } else if (row_batch == 4 && col_batch == 16) {
        if (is_pack_c == 0)
            this->unpack_c = nullptr;
        else if (is_pack_c == 1)
            this->unpack_c = write_c_v2_4x16;
        else
            this->unpack_c = write_c_v2_4x16_unpack;
    }
}

void ConvIm2colLayer::set_inner_kernel() {
    if (row_batch == 8 && col_batch == 8) {
        if (is_pack_c == 0 || is_pack_c == 1)
            this->inner_kernel = kernel_8x8;
        else
            this->inner_kernel = kernel_8x8_packC;
    } else if (row_batch == 8 && col_batch == 12) {
        if (is_pack_c == 0 || is_pack_c == 1)
            this->inner_kernel = kernel_8x12;
        else
            this->inner_kernel = kernel_8x12_packC;
    } else if (row_batch == 4 && col_batch == 16) {
        if (is_pack_c == 0 || is_pack_c == 1)
            this->inner_kernel = kernel_4x16;
        else
            this->inner_kernel = kernel_4x16_packC;
    }
}

void ConvIm2colLayer::set_inner_kernel_for_corner(int k) {
    if (row_batch == 8 && col_batch == 8) {
        if (is_pack_c == 0 || is_pack_c == 1)
            this->inner_kernel_for_corner = get_kernel_Nx8(k);
        else
            this->inner_kernel_for_corner = get_kernel_Nx8_packC(k);
    } else if (row_batch == 8 && col_batch == 12) {
        if (is_pack_c == 0 || is_pack_c == 1)
            this->inner_kernel_for_corner = get_kernel_Nx12(k);
        else
            this->inner_kernel_for_corner = get_kernel_Nx12_packC(k);
    } else if (row_batch == 4 && col_batch == 16) {
        if (is_pack_c == 0 || is_pack_c == 1)
            this->inner_kernel_for_corner = get_kernel_Nx16(k);
        else
            this->inner_kernel_for_corner = get_kernel_Nx16_packC(k);
    }
}
