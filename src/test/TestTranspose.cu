//
// Created by DY on 17-6-21.
//
#include <matrix/transpose.cuh>
#include <common_headers.h>
#include <cu_common_headers.cu>
//using namespace cutils;

int main() {
	int numbytes = N * N * sizeof(float);

	float *in = (float *)malloc(numbytes);
	float *out = (float *)malloc(numbytes);
	float *gold = (float *)malloc(numbytes);

    CpuTimer cpuTimer;

    printf("transpose_cpu:\n");
    cpuTimer.start();
	transpose_CPU(in, gold);
    cpuTimer.stop();

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


    printf("transpose_serial:\n");
	timer.start();
	transpose_serial<<<1, 1>>>(d_in, d_out);
	timer.stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    checkResultsExact(out, gold, N);

    printf("transpose_parallel_per_row:\n");
	timer.start();
	cudaMemset(d_out, 0, N * N);
	transpose_parallel_per_row<<<1, N>>>(d_in, d_out);
	timer.stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    checkResultsExact(out, gold, N);

    printf("transpose_parallel_per_element:\n");
	dim3 threads(K, K); // TODO, you need to define the correct blocks per grid
	dim3 blocks((N + K - 1) / K, (N + K - 1) / K);	// TODO, you need to define the correct threads per block
	timer.start();
	cudaMemset(d_out, 0, N * N);
	transpose_parallel_per_element << <blocks, threads >> > (d_in, d_out);
	timer.stop();
	cudaMemcpy(out, d_out, numbytes, cudaMemcpyDeviceToHost);
    checkResultsExact(out, gold, N);

	cudaFree(d_in);
	cudaFree(d_out);
    return 0;
}

