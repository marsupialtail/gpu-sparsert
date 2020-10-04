#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cuda.h>
#include <builtin_types.h>

#include <cnpy.h>
#include <cuda_fp16.h>

#include <vector>
#include <fstream>
#include <cmath>
#include <cublas_v2.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>
// This will output the proper CUDA error strings
// in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

//#define A_dim 64
//#define B_dim 256
//#define C_dim 3136

//#define A_Blocks 2
//#define C_Blocks 98

#define Tsy 1
#define Tsz (C_dim / C_Blocks)
#define ST 1
#define Fx 1
#define Fy (Tsz/Fx)

#define Usy (Tsy * Fy)
#define Gsy (Usy)

//#define Gy 2
#define Block_size (Gy * Gsy)

#ifndef HALF
#define HALF 0
#endif

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line );
        exit(-1);
    }
}

// --- global variables ----------------------------------------------------
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;
size_t     totalGlobalMem;

char       *module_file = (char*) "testing.cubin";
#if RESIDUAL
char       *kernel_name = (char*) "_Z2mmPKfS0_Pf";
#else
char       *kernel_name = (char*) "_Z2mmPKfPf";
#endif

// --- functions -----------------------------------------------------------
void initCUDA() {
    int deviceCount = 0;
    CUresult err = cuInit(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "Error: no devices supporting CUDA\n");
        exit(-1);
    }

    // get first CUDA device
    checkCudaErrors(cuDeviceGet(&device, 0));
    char name[100];
    cuDeviceGetName(name, 100, device);
    printf("> Using device 0: %s\n", name);

    // get compute capabilities and the devicename
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, device));

    err = cuCtxCreate(&context, 0, device);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.\n");
        cuCtxDetach(context);
        exit(-1);
    }
}

void initfunction() {
	auto err = cuModuleLoad(&module, module_file);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error loading the module %s\n", module_file);
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleGetFunction(&function, module, kernel_name);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error getting kernel function %s\n", kernel_name);
        cuCtxDetach(context);
        exit(-1);
    }
}

void finalizeCUDA()
{
    cuCtxDetach(context);
}

void setupDeviceMemory(CUdeviceptr *d_a, CUdeviceptr *d_b, int a, int b)
{
    checkCudaErrors( cuMemAlloc(d_a, a) );
    checkCudaErrors( cuMemAlloc(d_b, b) );
}

void releaseDeviceMemory(CUdeviceptr d_a, CUdeviceptr d_b)
{
    checkCudaErrors( cuMemFree(d_a) );
    checkCudaErrors( cuMemFree(d_b) );
}

void runKernel(CUdeviceptr d_BC, CUdeviceptr d_residual, CUdeviceptr d_AC)
{
#if RESIDUAL
    void *args[3] = { &d_BC, &d_residual, &d_AC};
#else
    void *args[2] = { &d_BC, &d_AC};
#endif

    // grid for kernel: <<<N, 1>>>

    CUevent start, stop;
    //cudaProfilerStart();
    cuEventCreate(&start,0);
    cuEventCreate(&stop,0);
    cuEventRecord(start,0);

#if HALF
    std::cout << A_Blocks << " " << C_Blocks << " " << Block_size/2 << std::endl;
#else
    std::cout << A_Blocks << " " << C_Blocks << " " << Block_size << std::endl;
#endif

#if HALF
    for(int i = 0;i < 1000; i ++){
    checkCudaErrors( cuLaunchKernel(function, A_Blocks, C_Blocks, 1,  // Nx1x1 blocks
                                    Block_size/2, 1, 1,            // 1x1x1 threads
                                    0, 0, args, 0) );
    }
#else
    for(int i = 0;i < 1000; i ++){
    checkCudaErrors( cuLaunchKernel(function, A_Blocks, C_Blocks, 1,  // Nx1x1 blocks
                                    Block_size, 1, 1,            // 1x1x1 threads
                                    0, 0, args, 0) );
    }
#endif
    cuEventRecord(stop,0);
    //cudaProfilerStop();
    cuEventSynchronize(stop);
    float time;
    cuEventElapsedTime(&time,start,stop);
    cuEventDestroy(start);
    cuEventDestroy(stop);
    std::cout << "kernel used " << time / 1000.0 << std::endl;

}

void float2half(float * in, __half * out, int n) {
    for(int i=0; i<n; i++){
        out[i] = __float2half(in[i]);
    }
}

void half2float(__half * in, float * out, int n) {
    for(int i=0; i<n; i++){
        out[i] = __half2float(in[i]);
    }
}


int main(int argc, char **argv)
{

    cnpy::NpyArray arr1 = cnpy::npy_load("BC.npy");
    float * BC = arr1.data<float>();
    assert(arr1.word_size = sizeof(float));
    assert(arr1.shape.size()==2 && arr1.shape[0] == B_dim && arr1.shape[1] == C_dim);

    cnpy::NpyArray arr2 = cnpy::npy_load("ref.npy");
    float * AC = arr2.data<float>();
    assert(arr2.word_size = sizeof(float));
    assert(arr2.shape.size()==2 && arr2.shape[0] == A_dim && arr2.shape[1] == C_dim);

#if RESIDUAL
    cnpy::NpyArray arr3 = cnpy::npy_load("residual.npy");
    float * residual = arr3.data<float>();
    assert(arr3.word_size = sizeof(float));
    assert(arr3.shape.size()==2 && arr3.shape[0] == A_dim && arr3.shape[1] == C_dim);

    __half * residual_h;
    residual_h = (__half *)malloc(A_dim * C_dim * 2);
    float2half(residual,residual_h,A_dim * C_dim);
    std::cout << residual_h[0] << std::endl;
#endif

    __half * BC_h, * AC_h;
    BC_h = (__half *)malloc(C_dim * B_dim *2);
    AC_h = (__half *)malloc(A_dim * C_dim *2);
    float2half(BC,BC_h,B_dim * C_dim);

    CUdeviceptr d_BC, d_AC, d_residual;

    initCUDA();

    initfunction();

    // allocate memory
#if HALF
    setupDeviceMemory(&d_BC, &d_AC, 2 * B_dim * C_dim, 2 * A_dim * C_dim);
#if RESIDUAL
    checkCudaErrors( cuMemAlloc(&d_residual, 2 * A_dim * C_dim) );
    checkCudaErrors (cuMemcpyHtoD(d_residual, residual_h, 2 * A_dim * C_dim) );
#endif
    checkCudaErrors( cuMemcpyHtoD(d_BC, BC_h,2 * B_dim * C_dim) );
#else
    setupDeviceMemory(&d_BC, &d_AC, sizeof(float) * B_dim * C_dim, sizeof(float) * A_dim * C_dim);
#if RESIDUAL
    checkCudaErrors( cuMemAlloc(&d_residual, 4 * A_dim * C_dim) );
    checkCudaErrors (cuMemcpyHtoD(d_residual, residual, 4 * A_dim * C_dim) );
#endif
    checkCudaErrors( cuMemcpyHtoD(d_BC, BC, sizeof(float) * B_dim * C_dim) );
#endif


    // run
    //printf("# Running the kernel...\n");
    runKernel(d_BC, d_residual, d_AC);
    //printf("# Kernel complete.\n");

    // copy results to host and report
    float *result;
    result = (float *)malloc(A_dim * C_dim *sizeof(float));
#if HALF
    checkCudaErrors( cuMemcpyDtoH(AC_h, d_AC, 2 * A_dim * C_dim) );
    half2float(AC_h,result,A_dim * C_dim);
#else
    checkCudaErrors( cuMemcpyDtoH(result, d_AC, sizeof(float) * A_dim * C_dim) );
#endif

    float error = 0;
    for(int i = 0 ; i < A_dim * C_dim; i ++)
    {
        auto diff = abs(result[i] - ((float*)AC)[i]);
	if (diff > 0.1)
	{
	//	std::cout << i << " " << result[i] << " " << ((float*)AC)[i] << std::endl;
	}
	error += diff;
    }

    std::cout << result[0] << result[1] << result[2] << std::endl;
    cnpy::npy_save("ptx_result.npy",&result[0],{A_dim,C_dim},"w");
    std::cout << "error: " << error << std::endl;
    // finish
    //printf("- Finalizing...\n");
    releaseDeviceMemory(d_AC, d_BC);
    finalizeCUDA();
    return 0;
}
