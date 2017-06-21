#include <stdio.h>
#include "utils.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <time.h>

using namespace std;

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

int main(int argc, char **argv)
{
	int numbytes = N * N * sizeof(float);

	float *in = (float *)malloc(numbytes);
	float *out = (float *)malloc(numbytes);
	float *gold = (float *)malloc(numbytes);

	fill(in, N);
	long long start = clock();

	transpose_CPU(in, gold);

	long long finish = clock();
	double duration = (double)(finish - start) / CLOCKS_PER_SEC * 1000;
	printf("transpose_cpu: %.6f ms\n", duration);

	float *d_in, *d_out;

	cudaMalloc(&d_in, numbytes);
	cudaMalloc(&d_out, numbytes);
	cudaMemcpy(d_in, in, numbytes, cudaMemcpyHostToDevice);

	GpuTimer timer;

	/*
	* Now time each kernel and verify that it produces the correct result.
	*
	* To be really careful about benchmarking purposes, we should run every kernel once
	* to "warm" the system and avoid any compilation or code-caching effects, then run
	* every kernel 10 or 100 times and average the timings to smooth out any variance.
	* But this makes for messy code and our goal is teaching, not detailed benchmarking.
	*/


	timer.Start();
	transpose_serial<<<1, 1>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_serial: %.3f ms.\nVerifying transpose...%s\n",
		timer.Elapsed(), equals(out, gold, N) ? "Success" : "Failed");

	timer.Start();
	cudaMemset(d_out, 0, N * N);
	transpose_parallel_per_row<<<1, N>>>(d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_row: %g ms.\nVerifying transpose...%s\n",
		timer.Elapsed(), equals(out, gold, N) ? "Success" : "Failed");

	dim3 threads(K, K); // TODO, you need to define the correct blocks per grid
	dim3 blocks((N + K - 1) / K, (N + K - 1) / K);	// TODO, you need to define the correct threads per block
	timer.Start();
	cudaMemset(d_out, 0, N * N);
	transpose_parallel_per_element << <blocks, threads >> > (d_in, d_out);
	timer.Stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
	printf("transpose_parallel_per_element: %g ms.\nVerifying transpose...%s\n",
		timer.Elapsed(), equals(out, gold, N) ? "Success" : "Failed");

	cudaFree(d_in);
	cudaFree(d_out);
}