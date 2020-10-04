#include <cnpy.h>
#include <vector>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cublas_v2.h>
#include <time.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>

#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>

#define BATCH_SIZE 1
#define FILTER_X 3
#define FILTER_Y 3
#define WITH_RELU 0
//#define IC 32 //128
//#define OC 32 //128
//#define IMAGE_DIM 14 //28

#define A_dim OC
#define B_dim IC * FILTER_X * FILTER_Y

#ifndef HALF
#define HALF 0
#endif

#if HALF
#define C_dim (IMAGE_DIM * IMAGE_DIM / 2)
#else
#define C_dim IMAGE_DIM * IMAGE_DIM
#endif

//#define Tsx 3
#define Tsy 1
#define Tsz (C_dim / C_Blocks)

#define Fx 1
#define Fy (Tsz/Fx)

#define Usy (Tsy * Fy) // this better be a multiple of 2, ideally 32
#define Gsy (Usy)

//#define Gx 36
#define Block_size (Gy * Gsy)


#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors( CUresult err, const char *file, const int line )
{
    if( CUDA_SUCCESS != err) {
        fprintf(stderr,
                "CUDA Driver API error = %04d from file <%s>, line %i.",
                err, file, line );
        exit(-1);
    }
}
CUdevice   device;
CUcontext  context;
CUmodule   module;
CUfunction function;
size_t     totalGlobalMem;

char       *module_file = (char*) "testing_conv.cubin";

#if RESIDUAL
char       *kernel_name = (char*) "_Z2mmPKfS0_Pf";
#else
char       *kernel_name = (char*) "_Z2mmPKfPf";
#endif


// --- functions -----------------------------------------------------------
void initCUDA()
{
    int deviceCount = 0;
    CUresult err = cuInit(0);
    int major = 0, minor = 0;

    if (err == CUDA_SUCCESS)
        checkCudaErrors(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0) {
        fprintf(stderr, "Error: no devices supporting CUDA");
        exit(-1);
    }

    // get first CUDA device
    checkCudaErrors(cuDeviceGet(&device, 0));
    char name[100];
    cuDeviceGetName(name, 100, device);
    printf("> Using device 0: %s", name);

    // get compute capabilities and the devicename
    checkCudaErrors( cuDeviceComputeCapability(&major, &minor, device) );
    printf("> GPU Device has SM %d.%d compute capability", major, minor);
    err = cuCtxCreate(&context, 0, device);
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error initializing the CUDA context.");
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleLoad(&module, module_file);
    std::cout << err << std::endl;
    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error loading the module %s", module_file);
        cuCtxDetach(context);
        exit(-1);
    }

    err = cuModuleGetFunction(&function, module, kernel_name);

    if (err != CUDA_SUCCESS) {
        fprintf(stderr, "* Error getting kernel function %s", kernel_name);
        cuCtxDetach(context);
        exit(-1);
    }
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
    cuEventCreate(&start,0);
    cuEventCreate(&stop,0);
    cuEventRecord(start,0);

    std::cout << A_Blocks << " " << C_Blocks << " " << Block_size << std::endl;
    for(int i = 0;i < 100; i ++){
        checkCudaErrors( cuLaunchKernel(function, A_Blocks, C_Blocks, 1,  // Nx1x1 blocks
                                        Block_size, 1, 1,            // 1x1x1 threads
                                        0, 0, args, 0) );
    }
    cuEventRecord(stop,0);
    cuEventSynchronize(stop);
    float time;
    cuEventElapsedTime(&time,start,stop);
    cuEventDestroy(start);
    cuEventDestroy(stop);
    std::cout << "kernel used " << time / 100.0 << std::endl;

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

float get_diff(float * t1, float * t2, int size)
{
    float diff = 0;
    for(int i = 0; i < size; i ++){
        diff += abs(t1[i] - t2[i]);
//        if(t1[i] - t2[i] > 3){
//            std::cout << i << std::endl;
//        }
    }
    return diff;
}

void transpose(float * input, float * output, int dim1, int dim2) {
    for(int i = 0; i < dim2; i ++)
        for (int j = 0; j < dim1; j ++)
        {
            output[i * dim1 + j] = input[j * dim2 + i];
        }
}

void fillvector(float *data, int n) {
    for(int i=0; i<n; i++){
        data[i] = float(rand() % 10);
    }
}

// filter shape in OC, K, K, IC
// input shape in H, W, IC
// output shape in H, W, OC
// each block is responsible for a number of output channels
#define CPB 2
#define WIDTH (IC / 2) // aka block size

__device__ inline void atomicAddWarpToArray(__half2 *array, int index, __half2 val)
{

    val += __shfl_down_sync(-1, val, 16);
    val += __shfl_down_sync(-1, val, 8);
    val += __shfl_down_sync(-1, val, 4);
    val += __shfl_down_sync(-1, val, 2);
    val += __shfl_down_sync(-1, val, 1);
    if(threadIdx.x % 32 == 0) {
      array[index] = val;
    }
}

__device__ inline __half2 atomicAddWarp(__half2 val)
{

    val += __shfl_down_sync(-1, val, 16);
    val += __shfl_down_sync(-1, val, 8);
    val += __shfl_down_sync(-1, val, 4);
    val += __shfl_down_sync(-1, val, 2);
    val += __shfl_down_sync(-1, val, 1);
    return val;
}


__global__ void direct(__half2* input, __half2 * filter, __half * output)
{
    int my_oc_start = blockIdx.x * CPB;
    register __half2 my_kernel[CPB][FILTER_X][FILTER_Y];
    __shared__ __half2 reduction_buffer[32];
    //__shared__ __half2 coalesce_buffer[]
    if(threadIdx.x < 32)
    {
        reduction_buffer[threadIdx.x] =  __float2half2_rn(0.0f);
    }
    __syncthreads();
    for(int oc = 0; oc < CPB; oc ++)
    {
        int filter_offset = my_oc_start * WIDTH * FILTER_X * FILTER_Y + oc * WIDTH * FILTER_X * FILTER_Y;
        my_kernel[oc][0][0] = filter[filter_offset + threadIdx.x];
        my_kernel[oc][0][1] = filter[filter_offset + WIDTH + threadIdx.x];
        my_kernel[oc][0][2] = filter[filter_offset + WIDTH * 2 + threadIdx.x];
        my_kernel[oc][1][0] = filter[filter_offset + WIDTH * 3 + threadIdx.x];
        my_kernel[oc][1][1] = filter[filter_offset + WIDTH * 4 + threadIdx.x];
        my_kernel[oc][1][2] = filter[filter_offset + WIDTH * 5 + threadIdx.x];
        my_kernel[oc][2][0] = filter[filter_offset + WIDTH * 6 + threadIdx.x];
        my_kernel[oc][2][1] = filter[filter_offset + WIDTH * 7 + threadIdx.x];
        my_kernel[oc][2][2] = filter[filter_offset + WIDTH * 8 + threadIdx.x];
    }

    __half2 intermediate[CPB];

    __half2 input_10 = __float2half2_rn(0.0f);
    __half2 input_11 = __float2half2_rn(0.0f);
    __half2 input_12 = input[threadIdx.x];
    __half2 input_20 = __float2half2_rn(0.0f);
    __half2 input_21 = __float2half2_rn(0.0f);
    __half2 input_22 = input[IMAGE_DIM * WIDTH + threadIdx.x];
    for (int col = 0; col < IMAGE_DIM; col++) {
        input_10 = input_11;
        input_11 = input_12;
        input_12 = (col == (IMAGE_DIM - 1)) ? __float2half2_rn(0.0f): input[(col + 1) * WIDTH + threadIdx.x];
        input_20 = input_21;
        input_21 = input_22;
        input_22 = (col == (IMAGE_DIM - 1)) ? __float2half2_rn(0.0f): input[IMAGE_DIM * WIDTH + (col + 1) * WIDTH + threadIdx.x];
        for (int oc = 0; oc < CPB; oc++) {
            intermediate[oc] = __hmul2(input_10,my_kernel[oc][1][0]);
            intermediate[oc] = __hfma2(input_11 , my_kernel[oc][1][1],intermediate[oc]);
            intermediate[oc] = __hfma2(input_12 , my_kernel[oc][1][2],intermediate[oc]);
            intermediate[oc] = __hfma2(input_20 , my_kernel[oc][2][0],intermediate[oc]);
            intermediate[oc] = __hfma2(input_21 , my_kernel[oc][2][1],intermediate[oc]);
            intermediate[oc] = __hfma2(input_22 , my_kernel[oc][2][2],intermediate[oc]);
            atomicAddWarpToArray(reduction_buffer, threadIdx.x / 32, intermediate[oc]);
            __syncthreads();

            __half2 result;
            if (threadIdx.x < WIDTH / 32) {
                result = atomicAddWarp(reduction_buffer[threadIdx.x]);
            }
            if (threadIdx.x == 0) {
                output[col * OC + my_oc_start + oc] = __high2half(result) + __low2half(result);
            }
        }
    }




    for(int row = 1; row < IMAGE_DIM - 1; row ++) {
        __half2 input_00 = __float2half2_rn(0.0f);
        __half2 input_01 = __float2half2_rn(0.0f);
        __half2 input_02 = input[(row - 1) * IMAGE_DIM * WIDTH + threadIdx.x];
        __half2 input_10 = __float2half2_rn(0.0f);
        __half2 input_11 = __float2half2_rn(0.0f);
        __half2 input_12 = input[(row ) * IMAGE_DIM * WIDTH + threadIdx.x];
        __half2 input_20 = __float2half2_rn(0.0f);
        __half2 input_21 = __float2half2_rn(0.0f);
        __half2 input_22 = input[(row + 1) * IMAGE_DIM * WIDTH + threadIdx.x];
        for (int col = 0; col < IMAGE_DIM; col++) {
            input_00 = input_01;
            input_01 = input_02;
            input_02 = (col == (IMAGE_DIM - 1)) ? __float2half2_rn(0.0f): input[(row - 1) * IMAGE_DIM * WIDTH + (col + 1) * WIDTH + threadIdx.x];
            input_10 = input_11;
            input_11 = input_12;
            input_12 = (col == (IMAGE_DIM - 1)) ? __float2half2_rn(0.0f): input[(row ) * IMAGE_DIM * WIDTH + (col + 1) * WIDTH + threadIdx.x];
            input_20 = input_21;
            input_21 = input_22;
            input_22 = (col == (IMAGE_DIM - 1)) ? __float2half2_rn(0.0f): input[(row + 1) * IMAGE_DIM * WIDTH + (col + 1) * WIDTH + threadIdx.x];
            for (int oc = 0; oc < CPB; oc++) {
                intermediate[oc] = __hmul2(input_00 ,my_kernel[oc][0][0]);
                intermediate[oc] = __hfma2(input_01 , my_kernel[oc][0][1],intermediate[oc]);
                intermediate[oc] = __hfma2(input_02 , my_kernel[oc][0][2],intermediate[oc]);
                intermediate[oc] = __hfma2(input_10 , my_kernel[oc][1][0],intermediate[oc]);
                intermediate[oc] = __hfma2(input_11 , my_kernel[oc][1][1],intermediate[oc]);
                intermediate[oc] = __hfma2(input_12 , my_kernel[oc][1][2],intermediate[oc]);
                intermediate[oc] = __hfma2(input_20 , my_kernel[oc][2][0],intermediate[oc]);
                intermediate[oc] = __hfma2(input_21 , my_kernel[oc][2][1],intermediate[oc]);
                intermediate[oc] = __hfma2(input_22 , my_kernel[oc][2][2],intermediate[oc]);
                atomicAddWarpToArray(reduction_buffer, threadIdx.x / 32, intermediate[oc]);
                __syncthreads();
                __half2 result;
                if (threadIdx.x < WIDTH / 32) {
                    result = atomicAddWarp(reduction_buffer[threadIdx.x]);
                }
                if (threadIdx.x == 0) {
                    output[row * IMAGE_DIM * OC + col * OC + my_oc_start + oc] = __high2half(result) + __low2half(result);
                }
            }
        }
    }

    int row = IMAGE_DIM - 1;
    __half2 input_00 = __float2half2_rn(0.0f);
    __half2 input_01 = __float2half2_rn(0.0f);
    __half2 input_02 = input[(row - 1) * IMAGE_DIM * WIDTH + threadIdx.x];
    input_10 = __float2half2_rn(0.0f);
    input_11 = __float2half2_rn(0.0f);
    input_12 = input[(row ) * IMAGE_DIM * WIDTH + threadIdx.x];

    for (int col = 0; col < IMAGE_DIM; col++) {
        input_00 = input_01;
        input_01 = input_02;
        input_02 = (col == (IMAGE_DIM - 1)) ? __float2half2_rn(0.0f): input[(row - 1) * IMAGE_DIM * WIDTH + (col + 1) * WIDTH + threadIdx.x];
        input_10 = input_11;
        input_11 = input_12;
        input_12 = (col == (IMAGE_DIM - 1)) ? __float2half2_rn(0.0f): input[(row ) * IMAGE_DIM * WIDTH + (col + 1) * WIDTH + threadIdx.x];

        for (int oc = 0; oc < CPB; oc++) {
            intermediate[oc] = __hmul2(input_00 , my_kernel[oc][0][0]);
            intermediate[oc] = __hfma2(input_01 , my_kernel[oc][0][1],intermediate[oc]);
            intermediate[oc] = __hfma2(input_02 , my_kernel[oc][0][2],intermediate[oc]);
            intermediate[oc] = __hfma2(input_10 , my_kernel[oc][1][0],intermediate[oc]);
            intermediate[oc] = __hfma2(input_11 , my_kernel[oc][1][1],intermediate[oc]);
            intermediate[oc] = __hfma2(input_12 , my_kernel[oc][1][2],intermediate[oc]);

            atomicAddWarpToArray(reduction_buffer, threadIdx.x / 32, intermediate[oc]);
            __half2 result;
            __syncthreads();

            if (threadIdx.x < WIDTH / 32) {
                result = atomicAddWarp(reduction_buffer[threadIdx.x]);
            }
            if (threadIdx.x == 0) {
                output[row * IMAGE_DIM * OC + col * OC + my_oc_start + oc] = __high2half(result) + __low2half(result);
            }
        }
    }

}

int main(int argc, char const *argv[]) {
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));


//    fillvector((float*)input, IC * IMAGE_DIM * IMAGE_DIM);
//    fillvector((float*)kernel, IC * OC * FILTER_Y * FILTER_X);
    cnpy::NpyArray arr = cnpy::npy_load("filter.npy");
    float * kernel = arr.data<float>();
    assert(arr.word_size = sizeof(float));
    assert(arr.shape.size()==2 && arr.shape[0] == OC && arr.shape[1] == IC * FILTER_X * FILTER_Y); //reference kernel for cudnn

    cnpy::NpyArray arr2 = cnpy::npy_load("filter_KFFC.npy");
    float * kernel_KFFC = arr2.data<float>();
    assert(arr2.word_size = sizeof(float));
    assert(arr2.shape.size()==4 && arr2.shape[0] == OC && arr2.shape[1] == FILTER_X && arr2.shape[2] == FILTER_Y && arr2.shape[3] == IC); //reference kernel for cudnn

    cnpy::NpyArray arr1 = cnpy::npy_load("input.npy");
    float * input = arr1.data<float>();
    assert(arr1.word_size = sizeof(float));
    assert(arr1.shape.size()==3 && arr1.shape[0] == IC && arr1.shape[1] == IMAGE_DIM && arr1.shape[2] == IMAGE_DIM); //reference kernel for cudnn

#if RESIDUAL
    cnpy::NpyArray arr3 = cnpy::npy_load("residual.npy");
    float * residual = arr3.data<float>();
    assert(arr3.word_size = sizeof(float));
    assert(arr1.shape.size()==3 && arr1.shape[0] == OC && arr1.shape[1] == IMAGE_DIM && arr1.shape[2] == IMAGE_DIM);

    __half * residual_h;
    residual_h = (__half *)malloc(OC * IMAGE_DIM * IMAGE_DIM * 2);
    float2half(residual,residual_h,OC * IMAGE_DIM * IMAGE_DIM);
    std::cout << residual[0] << std::endl;
#endif

    auto transposed_input = (float * ) malloc(IC * IMAGE_DIM * IMAGE_DIM * 4);
    transpose(input, transposed_input, IC , IMAGE_DIM * IMAGE_DIM);
    initCUDA();

#if HALF
    __half * kernel_h, *input_h, *kernel_KFFC_h, *transposed_input_h;
    kernel_h = (__half *)malloc(IC * OC * FILTER_X * FILTER_Y *2);
    kernel_KFFC_h = (__half *)malloc(IC * OC * FILTER_X * FILTER_Y * 2);
    input_h = (__half *)malloc(IC * IMAGE_DIM * IMAGE_DIM *2);
    transposed_input_h = (__half *) malloc(IC * IMAGE_DIM * IMAGE_DIM * 2);
    float2half(transposed_input, transposed_input_h, IC * IMAGE_DIM * IMAGE_DIM);
    float2half(kernel, kernel_h, IC * OC * FILTER_X * FILTER_Y);
    float2half(kernel_KFFC,kernel_KFFC_h, IC * OC * FILTER_X * FILTER_Y);
    float2half(input, input_h, IC * IMAGE_DIM * IMAGE_DIM);
#endif

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
#if HALF
            CUDNN_DATA_HALF,
#else
            /*dataType=*/CUDNN_DATA_FLOAT,
#endif
            /*batch_size=*/BATCH_SIZE,
            /*channels=*/IC,
            /*image_height=*/IMAGE_DIM,
            /*image_width=*/IMAGE_DIM));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
#if HALF
                       CUDNN_DATA_HALF,
#else
                       /*dataType=*/CUDNN_DATA_FLOAT,
#endif
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/OC,
            /*in_channels=*/IC,
            /*kernel_height=*/FILTER_X,
            /*kernel_width=*/FILTER_Y));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
#if HALF
                       CUDNN_DATA_HALF,
#else
                       /*dataType=*/CUDNN_DATA_FLOAT,
#endif
            /*batch_size=*/BATCH_SIZE,
            /*channels=*/OC,
            /*image_height=*/IMAGE_DIM,
            /*image_width=*/IMAGE_DIM));

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
            /*pad_height=*/1,
            /*pad_width=*/1,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/1,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
#if HALF
                       CUDNN_DATA_HALF
#else
                       /*dataType=*/CUDNN_DATA_FLOAT
#endif

            ));


    cudnnConvolutionFwdAlgo_t convolution_algorithm;
    checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                input_descriptor,
                                                kernel_descriptor,
                                                convolution_descriptor,
                                                output_descriptor,
                                                CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                    /*memoryLimitInBytes=*/0,
                                                &convolution_algorithm));

    std::cout << "picked algorithm: " << convolution_algorithm << std::endl;
    size_t workspace_bytes = 0;
    //convolution_algorithm = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       input_descriptor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       convolution_algorithm,
                                                       &workspace_bytes));
    std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
              << std::endl;


#if HALF
    int input_bytes = 1 * IC * IMAGE_DIM * IMAGE_DIM * 2;
    int output_bytes = 1 * OC * IMAGE_DIM * IMAGE_DIM * 2;
    int kernel_bytes = IC * OC * FILTER_Y * FILTER_X * 2;
    __half * d_residual{nullptr};
    std::cout << RESIDUAL << std::endl;

#if RESIDUAL

    cudaMalloc((void **)&d_residual,output_bytes) ;
    cudaMemcpy(d_residual, residual_h,output_bytes,cudaMemcpyHostToDevice);
#endif
    __half * d_input{nullptr};
    cudaMalloc((void **)&d_input, input_bytes);
    cudaMemcpy(d_input, input_h, input_bytes, cudaMemcpyHostToDevice);

    __half * d_output{nullptr};
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_output, 0, output_bytes);

    __half * d_transposed_input{nullptr};
    cudaMalloc((void **)&d_transposed_input, input_bytes);
    cudaMemcpy(d_transposed_input, transposed_input_h, input_bytes, cudaMemcpyHostToDevice);

    __half * d_kernel{nullptr};
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMemcpy(d_kernel, kernel_h, kernel_bytes, cudaMemcpyHostToDevice);

    __half * d_kernel_KFFC{nullptr};
    cudaMalloc(&d_kernel_KFFC, kernel_bytes);
    cudaMemcpy(d_kernel_KFFC, kernel_KFFC_h, kernel_bytes, cudaMemcpyHostToDevice);
#else
    int input_bytes = 1 * IC * IMAGE_DIM * IMAGE_DIM * sizeof(float);
    int output_bytes = 1 * OC * IMAGE_DIM * IMAGE_DIM * sizeof(float);
    int kernel_bytes = IC * OC * FILTER_Y * FILTER_X * sizeof(float);

    float* d_input{nullptr};
    cudaMalloc((void **)&d_input, input_bytes);
    cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

    float* d_output{nullptr};
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_output, 0, output_bytes);

    float* d_kernel{nullptr};
    cudaMalloc(&d_kernel, kernel_bytes);
    cudaMemcpy(d_kernel, kernel, kernel_bytes, cudaMemcpyHostToDevice);

    float * d_residual{nullptr};
#endif

    void* d_workspace{nullptr};
    cudaMalloc(&d_workspace, workspace_bytes);


    const float alpha = 1, beta = 0;

    cudaDeviceSynchronize();

    for(int i = 0; i < 10; i ++) {
        checkCUDNN(cudnnConvolutionForward(cudnn,
                                           &alpha,
                                           input_descriptor,
                                           d_input,
                                           kernel_descriptor,
                                           d_kernel,
                                           convolution_descriptor,
                                           convolution_algorithm,
                                           d_workspace,
                                           workspace_bytes,
                                           &beta,
                                           output_descriptor,
                                           d_output));
    }

    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(int i = 0; i < 10; i ++) {
        checkCUDNN(cudnnConvolutionForward(cudnn,
                                           &alpha,
                                           input_descriptor,
                                           d_input,
                                           kernel_descriptor,
                                           d_kernel,
                                           convolution_descriptor,
                                           convolution_algorithm,
                                           d_workspace,
                                           workspace_bytes,
                                           &beta,
                                           output_descriptor,
                                           d_output));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time,start,stop);
    std::cout << "baseline used " << time / 10 << std::endl;
    cudaDeviceSynchronize();

    float* h_output = (float *)malloc(OC * IMAGE_DIM * IMAGE_DIM * 4);
#if HALF
    __half * h_output_h = (__half *)malloc(output_bytes);
    cudaMemcpy(h_output_h, d_output, output_bytes, cudaMemcpyDeviceToHost);
    half2float(h_output_h,h_output,output_bytes / 2);
#else
    cudaMemcpy(h_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
#endif

    cudaDeviceSynchronize();

    std::cout << h_output[0] << std::endl;
    cnpy::npy_save("cudnn_output.npy",&h_output[0],{OC, IMAGE_DIM, IMAGE_DIM},"w");

    cudaFree(d_output);
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_output, 0, output_bytes);

    cudaProfilerStart();
    cudaEventRecord(start);
//    for(int i = 0; i < 10; i ++) {
//        direct << < OC / CPB, WIDTH >> > ((__half2 * )
//        d_transposed_input, (__half2 * )
//        d_kernel_KFFC, d_output);
//    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaProfilerStop();
    cudaEventElapsedTime(&time,start,stop);
    std::cout << "direct used " << time / 10<< std::endl;
    cudaDeviceSynchronize();

    float* h_direct_output = (float *)malloc(OC * IMAGE_DIM * IMAGE_DIM * 4);
    float* h_direct_output_CHW = (float *)malloc(OC * IMAGE_DIM * IMAGE_DIM * 4);
#if HALF
    __half * h_direct_output_h = (__half *)malloc(output_bytes);
    cudaMemcpy(h_direct_output_h, d_output, output_bytes, cudaMemcpyDeviceToHost);
    half2float(h_direct_output_h,h_direct_output,output_bytes / 2);
#else
    cudaMemcpy(h_direct_output, d_output, output_bytes, cudaMemcpyDeviceToHost);
#endif

    cudaDeviceSynchronize();
    transpose(h_direct_output, h_direct_output_CHW, IMAGE_DIM * IMAGE_DIM, OC);
    std::cout << h_direct_output_CHW[0] << std::endl;
    cnpy::npy_save("direct_output.npy",&h_direct_output_CHW[0],{OC, IMAGE_DIM, IMAGE_DIM},"w");

    cudaFree(d_output);
    cudaMalloc(&d_output, output_bytes);
    cudaMemset(d_output, 0, output_bytes);

    runKernel(reinterpret_cast<CUdeviceptr>(d_input),reinterpret_cast<CUdeviceptr>(d_residual), reinterpret_cast<CUdeviceptr>(d_output));

    float* h_output_kernel = (float *)malloc(OC * IMAGE_DIM * IMAGE_DIM * 4);

#if HALF
    __half * h_output_kernel_h = (__half *)malloc(output_bytes);
    cudaMemcpy(h_output_kernel_h, d_output, output_bytes, cudaMemcpyDeviceToHost);
    half2float(h_output_kernel_h,h_output_kernel,output_bytes/2);
#else
    cudaMemcpy(h_output_kernel, d_output, output_bytes, cudaMemcpyDeviceToHost);
#endif

    cnpy::npy_save("kernel_output.npy",&h_output_kernel[0],{OC, IMAGE_DIM, IMAGE_DIM},"w");

    std::cout << h_output_kernel[0] << std::endl;
    std::cout << "Difference: " << get_diff(h_output,h_output_kernel,OC * IMAGE_DIM * IMAGE_DIM) << std::endl;

// Do something with h_output ...

    delete[] h_output;
    cudaFree(d_kernel);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_workspace);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(kernel_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);

    cudnnDestroy(cudnn);

}
