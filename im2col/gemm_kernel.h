#ifndef GEMM_KERNEL_H_
#define GEMM_KERNEL_H_

typedef void (*InnerKernel)(int, float *, float *, float *, int, const int, const int, const int);
typedef void (*InnerKernelForCorner)(int, float *, float *, float *, int);

void kernel_4x4(int, int, int, float *, float *, float *, int);

void kernel_8x8(int, float *, float *, float *, int, const int, const int, const int);

void kernel_8x8_packC(int, float *, float *, float *, int, const int, const int, const int);

void kernel_4x16(int, float *, float *, float *, int, const int, const int, const int);

void kernel_4x16_packC(int, float *, float *, float *, int, const int, const int, const int);

void kernel_8x12(int, float *, float *, float *, int, const int, const int, const int);

void kernel_8x12_packC(int, float *, float *, float *, int, const int, const int, const int);

template <int N>
void kernel_Nx8_for_8x8_template(int, float *, float *, float *, int);

template <int N>
void kernel_Nx8_for_8x8_packC_template(int, float *, float *, float *, int);

template <int N>
void kernel_Nx12_for_8x12_template(int, float *, float *, float *, int);

template <int N>
void kernel_Nx12_for_8x12_packC_template(int, float *, float *, float *, int);

template <int N>
void kernel_Nx16_for_4x16_template(int, float *, float *, float *, int);

template <int N>
void kernel_Nx16_for_4x16_packC_template(int, float *, float *, float *, int);

InnerKernelForCorner get_kernel_Nx8(int k);

InnerKernelForCorner get_kernel_Nx8_packC(int k);

InnerKernelForCorner get_kernel_Nx12(int k);

InnerKernelForCorner get_kernel_Nx12_packC(int k);

InnerKernelForCorner get_kernel_Nx16(int k);

InnerKernelForCorner get_kernel_Nx16_packC(int k);

#endif