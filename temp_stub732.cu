
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
#define Tsz (12544 / 256)
#define Gsy Tsz
#define Gy 1
#define Block_size (Gy * Gsy)
#define In_Format 'INPUT_FORMAT'
#define Out_Format 'OUTPUT_FORMAT'

namespace cg = cooperative_groups;

__global__ void mm(const float * __restrict__ BC, float * AC)
{
    register float ACC[8] = {0.0};
	register float RC = 0.0;
#if Gy > 1	
        __shared__ float result[8][Tsz];
	for(int i = threadIdx.x; i < 8 * Tsz; i += Block_size)
	{
		((float*)result)[i] = 0.0;
	}
	__syncthreads();
#endif
#if In_Format == 'NHWC'
	__shared__ float smem_cache[Tsz][TSB+1];
#endif
#if Out_Format == 'NHWC'
	__shared__ float smem_result[Tsz][8+1];
#endif

	int A_offset = blockIdx.x * (64 / 8);
	int C_offset = blockIdx.y * (12544 / 256);
	int groupId = threadIdx.x / (Gsy);
	int lane = threadIdx.x % (Gsy);


if(blockIdx.x == 0)
{



	if(groupId == 0)
	{


		asm("//B0G0;START");

		RC = BC[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		ACC[3] += RC * 0.031f;
		ACC[4] += RC * 0.041f;
		ACC[5] += RC * 0.051f;
		ACC[6] += RC * 0.061f;
		ACC[7] += RC * 0.071f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 8; i++)
	{
    
        AC[(0 + i) * 12544 + C_offset + lane] = ACC[i];

    }
    
#else
    for(int i = 0; i < 8; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 8 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[0 + row][C_offset + col] = result[row][col];
		AC[(0 + row) * 12544 + C_offset + col] = result[row][col];
	}
#endif       
       


}

if(blockIdx.x == 1)
{



	if(groupId == 0)
	{


		asm("//B1G0;START");

		RC = BC[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		ACC[3] += RC * 0.031f;
		ACC[4] += RC * 0.041f;
		ACC[5] += RC * 0.051f;
		ACC[6] += RC * 0.061f;
		ACC[7] += RC * 0.071f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 8; i++)
	{
    
        AC[(8 + i) * 12544 + C_offset + lane] = ACC[i];

    }
    
#else
    for(int i = 0; i < 8; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 8 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[8 + row][C_offset + col] = result[row][col];
		AC[(8 + row) * 12544 + C_offset + col] = result[row][col];
	}
#endif       
       


}

if(blockIdx.x == 2)
{



	if(groupId == 0)
	{


		asm("//B2G0;START");

		RC = BC[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		ACC[3] += RC * 0.031f;
		ACC[4] += RC * 0.041f;
		ACC[5] += RC * 0.051f;
		ACC[6] += RC * 0.061f;
		ACC[7] += RC * 0.071f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 8; i++)
	{
    
        AC[(16 + i) * 12544 + C_offset + lane] = ACC[i];

    }
    
#else
    for(int i = 0; i < 8; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 8 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[16 + row][C_offset + col] = result[row][col];
		AC[(16 + row) * 12544 + C_offset + col] = result[row][col];
	}
#endif       
       


}

if(blockIdx.x == 3)
{



	if(groupId == 0)
	{


		asm("//B3G0;START");

		RC = BC[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		ACC[3] += RC * 0.031f;
		ACC[4] += RC * 0.041f;
		ACC[5] += RC * 0.051f;
		ACC[6] += RC * 0.061f;
		ACC[7] += RC * 0.071f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 8; i++)
	{
    
        AC[(24 + i) * 12544 + C_offset + lane] = ACC[i];

    }
    
#else
    for(int i = 0; i < 8; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 8 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[24 + row][C_offset + col] = result[row][col];
		AC[(24 + row) * 12544 + C_offset + col] = result[row][col];
	}
#endif       
       


}

if(blockIdx.x == 4)
{



	if(groupId == 0)
	{


		asm("//B4G0;START");

		RC = BC[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		ACC[3] += RC * 0.031f;
		ACC[4] += RC * 0.041f;
		ACC[5] += RC * 0.051f;
		ACC[6] += RC * 0.061f;
		ACC[7] += RC * 0.071f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 8; i++)
	{
    
        AC[(32 + i) * 12544 + C_offset + lane] = ACC[i];

    }
    
#else
    for(int i = 0; i < 8; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 8 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[32 + row][C_offset + col] = result[row][col];
		AC[(32 + row) * 12544 + C_offset + col] = result[row][col];
	}
#endif       
       


}

if(blockIdx.x == 5)
{



	if(groupId == 0)
	{


		asm("//B5G0;START");

		RC = BC[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		ACC[3] += RC * 0.031f;
		ACC[4] += RC * 0.041f;
		ACC[5] += RC * 0.051f;
		ACC[6] += RC * 0.061f;
		ACC[7] += RC * 0.071f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 8; i++)
	{
    
        AC[(40 + i) * 12544 + C_offset + lane] = ACC[i];

    }
    
#else
    for(int i = 0; i < 8; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 8 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[40 + row][C_offset + col] = result[row][col];
		AC[(40 + row) * 12544 + C_offset + col] = result[row][col];
	}
#endif       
       


}

if(blockIdx.x == 6)
{



	if(groupId == 0)
	{


		asm("//B6G0;START");

		RC = BC[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		ACC[3] += RC * 0.031f;
		ACC[4] += RC * 0.041f;
		ACC[5] += RC * 0.051f;
		ACC[6] += RC * 0.061f;
		ACC[7] += RC * 0.071f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 8; i++)
	{
    
        AC[(48 + i) * 12544 + C_offset + lane] = ACC[i];

    }
    
#else
    for(int i = 0; i < 8; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 8 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[48 + row][C_offset + col] = result[row][col];
		AC[(48 + row) * 12544 + C_offset + col] = result[row][col];
	}
#endif       
       


}

if(blockIdx.x == 7)
{



	if(groupId == 0)
	{


		asm("//B7G0;START");

		RC = BC[0 + C_offset + lane];

		ACC[0] += RC * 0.001f;
		ACC[1] += RC * 0.011f;
		ACC[2] += RC * 0.021f;
		ACC[3] += RC * 0.031f;
		ACC[4] += RC * 0.041f;
		ACC[5] += RC * 0.051f;
		ACC[6] += RC * 0.061f;
		ACC[7] += RC * 0.071f;
		asm("//END;");

	}



   
#if Gy == 1
    for(int i = 0; i < 8; i++)
	{
    
        AC[(56 + i) * 12544 + C_offset + lane] = ACC[i];

    }
    
#else
    for(int i = 0; i < 8; i++)
	{
        atomicAdd(&result[i][lane], ACC[i]);
    }
    
	__syncthreads();

	for(int i = threadIdx.x; i < Tsy * 8 * Tsz; i+= Block_size)
	{
		int row = i / Tsz;
		int col = i % Tsz;
		//AC[56 + row][C_offset + col] = result[row][col];
		AC[(56 + row) * 12544 + C_offset + col] = result[row][col];
	}
#endif       
       


}

 
}
int main()
{

	std::cout << "Group size " << Gsy << std::endl;

	cnpy::NpyArray arr = cnpy::npy_load("mobilenet/contraction_1x1_0_transposed.npy");
	float * AB = arr.data<float>();
	assert(arr.word_size = sizeof(float));
	assert(arr.shape.size()==2 && arr.shape[0] == 32 && arr.shape[1] == 64); //transposed

	cnpy::NpyArray arr1 = cnpy::npy_load("BC.npy");
	float * BC = arr1.data<float>();
	assert(arr1.word_size = sizeof(float));
#if In_Format == 'NHWC'
	assert(arr1.shape.size()==2 && arr1.shape[0] == 12544 && arr1.shape[1] == 32);
#else
	assert(arr1.shape.size()==2 && arr1.shape[0] == 32 && arr1.shape[1] == 12544);
#endif
    cnpy::NpyArray arr2 = cnpy::npy_load("ref.npy");
	float * AC = arr2.data<float>();
    std::cout << AC[0] << std::endl;

	float *d_BC, *d_AC, *d_residual;
	cudaMalloc((void**)&d_BC, 32 * 12544 *sizeof(float));
	cudaMalloc((void**)&d_AC, 64 * 12544 *sizeof(float));


	cudaMemcpy( d_BC,BC, 32 * 12544 *sizeof(float), cudaMemcpyHostToDevice);

	float *result;
	result = (float *)malloc(64 * 12544 *sizeof(result));

	cudaEvent_t start,stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	dim3 GS(8,256);

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

	cudaMemcpy(result, d_AC, 64 * 12544 *sizeof(float), cudaMemcpyDeviceToHost);

	float error = 0;
	for(int i = 0 ; i < 64 * 12544; i ++)
	{
        error += abs(result[i] - AC[i]);
	}
	
	#if Out_Format == 'NCHW'
        cnpy::npy_save("result.npy",&result[0],{64,12544},"w");
    #else
        cnpy::npy_save("result.npy",&result[0],{12544,64},"w");
    #endif

	std::cout << result[0] << result[1] << result[2] << std::endl;
	std::cout << error << std::endl;
	cudaFree(d_BC);
	cudaFree(d_AC);
}
