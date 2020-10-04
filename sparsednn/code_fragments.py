START_CONV = """
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

#define IC IC_P
#define OC OC_P
#define IMAGE_DIM IMAGE_DIM_P

//#define A_BLOCKS 16 //blockDim.x
//#define C_BLOCKS 32 //blockDim.y

#define Tsy 1
#define Tsz (C_dim / C_BLOCKS)
#define ST ST_VAL
#define Fx 1
#define Fy Tsz

//#define Ny (A_dim / A_BLOCKS / Tsy)

#define Usy (Tsy * Fy)
#define Gsy Usy

#define Gy GY
#define Block_size (Gy * Gsy)
#define RESIDUAL RESIDUAL_DEF


#define checkCUDNN(expression)                               \
{ \
    cudnnStatus_t status = (expression); \
    if (status != CUDNN_STATUS_SUCCESS) { \
    std::cerr << "Error on line " << __LINE__ << ": " \
         << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE); \
    } \
    }

namespace cg = cooperative_groups;

#if RESIDUAL
__global__ void mm(const float * __restrict__ input, const float * __restrict__ residual, float * AC) // does A * AB = B
#else
__global__ void mm(const float * __restrict__ input, float * AC) // does A * AB = B
#endif
{
    register float ACC[Ny] = {0.0};
register float RC = {0.0};
__shared__ float result[Ny][Tsz];
#if(Gy > 1)
for(int i = threadIdx.x; i < Ny * Tsz; i += Block_size)
{
    ((float*)result)[i] = 0.0;
}
__syncthreads();
#endif
// int A_offset = blockIdx.x * (A_dim / A_BLOCKS);
int C_offset = blockIdx.y * (C_dim / C_BLOCKS);
int groupId = threadIdx.x / (Gsy);
int lane = threadIdx.x % (Gsy);
int lane2 = threadIdx.x % Fy;
int C_idx = C_offset + lane2;
int c_x  = C_idx / IMAGE_DIM;
int c_y  = C_idx % IMAGE_DIM;



"""

START_NONFUSED="""
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
// we are doing AC = AB * BC, reduce across the B dimension
// binding B to the x dimension, A to the y dimension and C to the z dimension

#define Tsy 1
#define Tsz (C_dim / C_BLOCKS)
#define Gsy Tsz
#define Gy GY
#define Block_size (Gy * Gsy)
#define In_Format 'INPUT_FORMAT'
#define Out_Format 'OUTPUT_FORMAT'

namespace cg = cooperative_groups;

__global__ void mm(const float * __restrict__ BC, float * AC)
{
    register float ACC[Ny] = {0.0};
	register float RC = 0.0;
#if Gy > 1	
        __shared__ float result[Ny][Tsz];
	for(int i = threadIdx.x; i < Ny * Tsz; i += Block_size)
	{
		((float*)result)[i] = 0.0;
	}
	__syncthreads();
#endif
#if In_Format == 'NHWC'
	__shared__ float smem_cache[Tsz][TSB+1];
#endif
#if Out_Format == 'NHWC'
	__shared__ float smem_result[Tsz][Ny+1];
#endif

	int A_offset = blockIdx.x * (A_dim / A_BLOCKS);
	int C_offset = blockIdx.y * (C_dim / C_BLOCKS);
	int groupId = threadIdx.x / (Gsy);
	int lane = threadIdx.x % (Gsy);
"""


START_NONFUSED_RESIDUAL="""
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
// we are doing AC = AB * BC, reduce across the B dimension
// binding B to the x dimension, A to the y dimension and C to the z dimension

#define Tsy 1
#define Tsz (C_dim / C_BLOCKS)
#define Gsy Tsz
#define Gy GY
#define Block_size (Gy * Gsy)
#define In_Format 'INPUT_FORMAT'
#define Out_Format 'OUTPUT_FORMAT'
#define RESIDUAL 1

namespace cg = cooperative_groups;

__global__ void mm(const float * __restrict__ BC, const float * __restrict__ residual, float * AC)
{
    register float ACC[Ny] = {0.0};
	register float RC = 0.0;
#if Gy > 1	
        __shared__ float result[Ny][Tsz];
	for(int i = threadIdx.x; i < Ny * Tsz; i += Block_size)
	{
		((float*)result)[i] = 0.0;
	}
	__syncthreads();
#endif
#if In_Format == 'NHWC'
	__shared__ float smem_cache[Tsz][TSB+1];
#endif
#if Out_Format == 'NHWC'
	__shared__ float smem_result[Tsz][Ny+1];
#endif

	int A_offset = blockIdx.x * (A_dim / A_BLOCKS);
	int C_offset = blockIdx.y * (C_dim / C_BLOCKS);
	int groupId = threadIdx.x / (Gsy);
	int lane = threadIdx.x % (Gsy);
"""

START_FUSED="""
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
// we are doing AC = AB * BC, reduce across the B dimension
// binding B to the x dimension, A to the y dimension and C to the z dimension

#define Tsy 1
#define Tsz (C_dim / C_BLOCKS)
#define ST ST_VAL
#define Fx FX
#define Fy (Tsz/Fx)

//#define Ny (A_dim / A_BLOCKS / Tsy)

#define Usy (Tsy * Fy)
//#define Gsy 32 //Usy

#define Gy GY
#define Block_size (Gy * Gsy)

namespace cg = cooperative_groups;

__global__ void mm(const float * __restrict__ BC, float * AC)
{
    register float ACC[Ny][Fx] = {0.0};
	register float RC[ST][Fx] = {0.0};
	
	__shared__ float result[Ny][Tsz];

	for(int i = threadIdx.x; i < Ny * Tsz; i += Block_size)
	{
		((float*)result)[i] = 0.0;
	}
	__syncthreads();

	int A_offset = blockIdx.x * (A_dim / A_BLOCKS);
	int C_offset = blockIdx.y * INPUT_PADDED_W * STRIDE * ROWS;
	int groupId = threadIdx.x / (Gsy);
	int lane = threadIdx.x % (Gsy);
	float k00, k01, k02, k10, k11, k12, k20, k21, k22;
"""

GEN_LOAD="""
RC = BC[0 + C_offset + lane];
"""

GEN_LOAD_STRIDE="""
RC[0][0] = BC[0 + C_offset + lane * STRIDE + 0 * Fy];
"""

GEN_ACC = """
ACC[I] += RC * 0.0I1f;"""

GEN_END = """
asm("//END;");
"""


GEN_LANDMARK_PTX="""
asm("//BIGJ;START");
"""

GROUP_CONTROL_START = """
if(groupId == GROUP)
{
"""

GROUP_CONTROL_END = """
}
"""

BLOCK_CONTROL_START = """
if(blockIdx.x == BLOCK)
{

"""

BLOCK_CONTROL_END = """
}
"""

GEN_LANDMARK="""
asm("//B1G0;");
"""

BLOCK_END_NHWC="""
   
#if Gy == 1
    #pragma unroll
    for(int j =0; j < Ny; j++)
    {
        smem_result[lane][j] = ACC[j];
    }

    #pragma unroll
    for(int j = 0; j < Gsy; j++)
	{
	    #pragma unroll
	    for(int i = lane; i < Ny; i+= Gsy)
	    {
            AC[(C_offset + j) * A_dim + A_offset + i] = smem_result[j][i];
        }
    
    }
    
#else
    exit()
#endif       
       
"""

BLOCK_END_REDUCTION_NO_RELU="""
        AC[OFFSET + C_offset  + lane] = ACC[IDX] + BIASf;
"""

BLOCK_END_REDUCTION="""
        AC[OFFSET + C_offset  + lane] = max(ACC[IDX] + BIASf,0.0f);
"""

BLOCK_END_REDUCTION_RESIDUAL="""
        AC[OFFSET + C_offset  + lane] = residual[OFFSET + C_offset  + lane] + max(ACC[IDX] + BIASf,0.0f);
"""

BLOCK_END_REDUCTION_RESIDUAL_2="""
        AC[OFFSET + C_offset  + lane] = max(residual[OFFSET + C_offset  + lane] + ACC[IDX] + BIASf,0.0f);
"""

BLOCK_END="""
   
#if Gy == 1
    for(int i = 0; i < Ny; i++)
	{
    
        AC[(A_offset + i) * C_dim + C_offset + lane] = ACC[i];

    }
    
#else
    for(int i = 0; i < Ny; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * Ny * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[A_offset + row][C_offset + col] = result[row][col];
		AC[(A_offset + row) * C_dim + C_offset + col] = result[row][col];
	}
#endif       
       
"""

BLOCK_END_RESIDUAL="""
   
#if Gy == 1
    for(int i = 0; i < Ny; i++)
	{
    
        AC[(A_offset + i) * C_dim + C_offset + lane] = residual[(A_offset + i) * C_dim + C_offset + lane] + ACC[i];

    }
    
#else
    for(int i = 0; i < Ny; i++)
	{
	    for(int j = 0; j < Fx; j ++)
        {
		    atomicAdd(&result[i][lane], ACC[i]);
        }
	
        }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * Ny * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[A_offset + row][C_offset + col] = result[row][col];
		AC[(A_offset + row) * C_dim + C_offset + col] += result[row][col];
	}
#endif       
       
"""

END_NONFUSED="""
 
}
int main()
{

	std::cout << "Group size " << Gsy << std::endl;

	cnpy::NpyArray arr = cnpy::npy_load("AB_sparse_tidy.npy");
	float * AB = arr.data<float>();
	assert(arr.word_size = sizeof(float));
	assert(arr.shape.size()==2 && arr.shape[0] == B_dim && arr.shape[1] == A_dim); //transposed

	cnpy::NpyArray arr1 = cnpy::npy_load("BC.npy");
	float * BC = arr1.data<float>();
	assert(arr1.word_size = sizeof(float));
#if In_Format == 'NHWC'
	assert(arr1.shape.size()==2 && arr1.shape[0] == C_dim && arr1.shape[1] == B_dim);
#else
	assert(arr1.shape.size()==2 && arr1.shape[0] == B_dim && arr1.shape[1] == C_dim);
#endif
    cnpy::NpyArray arr2 = cnpy::npy_load("ref.npy");
	float * AC = arr2.data<float>();
    std::cout << AC[0] << std::endl;

	float *d_BC, *d_AC, *d_residual;
	cudaMalloc((void**)&d_BC, B_dim * C_dim *sizeof(float));
	cudaMalloc((void**)&d_AC, A_dim * C_dim *sizeof(float));


	cudaMemcpy( d_BC,BC, B_dim * C_dim *sizeof(float), cudaMemcpyHostToDevice);

	float *result;
	result = (float *)malloc(A_dim * C_dim *sizeof(result));

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 GS(A_BLOCKS,C_BLOCKS);

     std::cout << "warning: sometimes you might want to fix the launch dimensions to 32" << std::endl;

    for(int i = 0;i < 1000;i ++){
#if RESIDUAL
	    mm<<<GS,Gsy>>>(d_BC,d_residual,d_AC);
#else
        mm<<<GS,Gsy>>>(d_BC,d_AC);
#endif
    }

	cudaProfilerStart();
	cudaEventRecord(start);

	for(int i = 0;i < 1000;i ++){
#if RESIDUAL
	    mm<<<GS,Gsy>>>(d_BC,d_residual,d_AC);
#else
        mm<<<GS,Gsy>>>(d_BC,d_AC);
#endif
    }
	cudaEventRecord(stop);
	cudaProfilerStop();
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << "kernel used " << time / 1000.0 << std::endl;

	cudaMemcpy(result, d_AC, A_dim * C_dim *sizeof(float), cudaMemcpyDeviceToHost);

	float error = 0;
	for(int i = 0 ; i < A_dim * C_dim; i ++)
	{
        error += abs(result[i] - AC[i]);
	}
	
	#if Out_Format == 'NCHW'
        cnpy::npy_save("result.npy",&result[0],{A_dim,C_dim},"w");
    #else
        cnpy::npy_save("result.npy",&result[0],{C_dim,A_dim},"w");
    #endif

	std::cout << result[0] << result[1] << result[2] << std::endl;
	std::cout << error << std::endl;
	cudaFree(d_BC);
	cudaFree(d_AC);
}
"""

END_FUSED="""
 
}
int main()
{

	std::cout << "Group size " << Gsy << std::endl;
	std::cout << "===========================WARNING========================" << std::endl;
	std::cout << "NEED TO CHANGE BLOCK_END_START CONDITIONS MANUALLY" << Gsy << std::endl;
	

	cnpy::NpyArray arr1 = cnpy::npy_load("padded_input_image.npy");
    float * padded_image = arr1.data<float>();
    assert(arr1.word_size = sizeof(float));
    assert(arr1.shape.size()==3 && arr1.shape[0] == IC && arr1.shape[1] == INPUT_PADDED_H);

   // loads in a transposed way, wierd
    /*cnpy::NpyArray arr2 = cnpy::npy_load("result.npy");
    float * AC = arr2.data<float>();
    assert(arr2.word_size = sizeof(float));
    assert(arr2.shape.size()==2 && arr2.shape[0] == A_dim && arr2.shape[1] == C_dim);
*/
    int padded_image_bytes = (IC * INPUT_PADDED_H * INPUT_PADDED_W ) *sizeof(float);
	float *d_padded_image, *d_AC;
	cudaMalloc((void**)&d_padded_image,padded_image_bytes + 2 * sizeof(float) );
	cudaMemset(d_padded_image, 0,padded_image_bytes + 2 * sizeof(float));
	
        
    int padded_output_bytes = (OC * OUTPUT_PADDED_H * OUTPUT_PADDED_W) * sizeof(float);
        
        cudaMalloc((void**)&d_AC, padded_output_bytes + 2 * sizeof(float));
        cudaMemset(d_AC, 0, padded_output_bytes + 2 * sizeof(float));

    // this is very important. we are copying starting from the second element!!
	cudaMemcpy( d_padded_image + 1,padded_image,padded_image_bytes, cudaMemcpyHostToDevice);

	float *result;
	result = (float *)malloc(A_dim * C_dim *sizeof(result));

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 GS(A_BLOCKS,REAL_CBLOCKS);

    for(int i = 0;i < 1000;i ++){
	    mm<<<GS,Block_size>>>(d_padded_image + INPUT_PADDED_W+1,d_AC+OUTPUT_PADDED_W+1);
    }

	cudaProfilerStart();
	cudaEventRecord(start);

	for(int i = 0;i < 1000;i ++){
	    mm<<<GS,Block_size>>>(d_padded_image + INPUT_PADDED_W+1,d_AC+OUTPUT_PADDED_W+1);
    }
	cudaEventRecord(stop);
	cudaProfilerStop();
	cudaEventSynchronize(stop);
	float time;
	cudaEventElapsedTime(&time,start,stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	std::cout << "kernel used " << time / 1000.0 << std::endl;

	cudaMemcpy(result, d_AC + 1, padded_output_bytes, cudaMemcpyDeviceToHost);
    cnpy::npy_save("output.npy",&result[0],{OC,OUTPUT_PADDED_H,OUTPUT_PADDED_W},"w");
/*
	float error = 0;
	for(int i = 0 ; i < A_dim * C_dim; i ++)
	{
		float diff = abs(result[i] - ((float*)AC)[i]);
		error += diff;
		if (diff > 0.3){
		    std::cout << i << " " << diff << " " << result[i] << " " <<  ((float*)AC)[i] << std::endl;
		
		}
	}
*/
	std::cout << result[0] << result[1] << result[2] << std::endl;
	cudaFree(d_padded_image);
	cudaFree(d_AC);
}
"""
