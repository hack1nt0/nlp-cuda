#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

#ifndef NLP_CUDA_TRANSPOSE_CU
#define NLP_CUDA_TRANSPOSE_CU

const int N = 1024;		// matrix size is NxN
const int K = 16;	    	// TODO, set K to the correct value and tile size will be KxK


							// to be launched with one thread per element, in KxK threadblocks
							// thread (x,y) in grid writes element (i,j) of output matrix 
__global__ void
transpose_parallel_per_element(float in[], float out[])
{
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;
	//printf("%d %d\n", r, c);
	if (r >= N || c >= N) return;
	//printf("%d %d\n", r, c);
	out[c * N + r] = in[r * N + c];
}

//The following functions and kernels are for your reference
void
transpose_CPU(float in[], float out[])
{
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched on a single thread
__global__ void
transpose_serial(float in[], float out[])
{
	for (int j = 0; j < N; j++)
		for (int i = 0; i < N; i++)
			out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

// to be launched with one thread per row of output matrix
__global__ void
transpose_parallel_per_row(float in[], float out[])
{
	int i = threadIdx.x;
	//printf("%d\n", i);
	for (int j = 0; j < N; j++)
		out[j + i*N] = in[i + j*N]; // out(j,i) = in(i,j)
}

#endif