#ifndef GEMM_PACKING_H_
#define GEMM_PACKING_H_

void pack_a_v1(int, int, float *, int, float *);

void pack_b_v1(int, int, float *, int, float *);

void pack_a_v2(int, int, float *, int, float *, int, int, int, const int, const int);

void pack_b_v2_8x8(int, int, float *, int, float *, const int, const int);

void pack_b_v2_8x12(int, int, float *, int, float *, const int, const int);

void pack_b_v2_4x16(int, int, float *, int, float *, const int, const int);

void pack_b_multithread_8x8_v1(int, int, int, int, float*, int, float*, int, int, const int, const int);

void pack_b_multithread_8x8_v2(int, int, int, int, float*, int, float*, int, int, const int, const int);

void pack_b_multithread_8x8_v3(int, int, int, int, float*, int, float*, int, int, const int, const int);

void pack_b_multithread_8x12_v1(int, int, int, int, float*, int, float*, int, int, const int, const int);

void pack_b_multithread_8x12_v2(int, int, int, int, float*, int, float*, int, int, const int, const int);

void pack_b_multithread_8x12_v3(int, int, int, int, float*, int, float*, int, int, const int, const int);

void pack_b_multithread_4x16_v1(int, int, int, int, float*, int, float*, int, int, const int, const int);

void pack_b_multithread_4x16_v2(int, int, int, int, float*, int, float*, int, int, const int, const int);

void pack_b_multithread_4x16_v3(int, int, int, int, float*, int, float*, int, int, const int, const int);

void load_c_v2_8x8(int, int, int, float*, float*, int, const int, const int);

void load_c_v2_8x12(int, int, int, float*, float*, int, const int, const int);

void load_c_v2_4x16(int, int, int, float*, float*, int, const int, const int);

void load_c_v2_8x8_pack(int, int, int, float*, float*, int, const int, const int);

void load_c_v2_8x12_pack(int, int, int, float*, float*, int, const int, const int);

void load_c_v2_4x16_pack(int, int, int, float*, float*, int, const int, const int);

void write_c_v2_8x8(int, int, int, float*, float*, int, const int, const int);

void write_c_v2_8x12(int, int, int, float*, float*, int, const int, const int);

void write_c_v2_4x16(int, int, int, float*, float*, int, const int, const int);

void write_c_v2_8x8_unpack(int, int, int, float*, float*, int, const int, const int);

void write_c_v2_8x12_unpack(int, int, int, float*, float*, int, const int, const int);

void write_c_v2_4x16_unpack(int, int, int, float*, float*, int, const int, const int);

void PackMatrixA_multithread(int, int, float *, int, float *,  int, int, int, const int, const int);

// void pack_b_multithread_kernel_8x8_v2(int, int, int, float *, int, float *, const int, const int);

// void pack_b_multithread_kernel_8x8_v3(int, int, int, int, float *, int, float *, const int, const int);

#endif