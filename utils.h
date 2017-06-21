#pragma once
#ifndef __GPU_TIMER_H__

#define __GPU_TIMER_H__

#include "cuda_runtime.h"
#include <iostream>

struct GpuTimer {
	cudaEvent_t start;
	cudaEvent_t stop;

	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start() {
		cudaEventRecord(start, 0);
	}

	void Stop() {
		cudaEventRecord(stop, 0);
	}

	float Elapsed() {
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed;
	}
};

template <class T>
void fill(T* arr, int size) {
	for (int i = 0; i < size; ++i) arr[i] = (T)rand();
}


template <class T>
bool equals(T* arrA, T* arrB, int size) {
	for (int i = 0; i < size; ++i) if (abs(arrA[i] - arrB[i]) > 1e-8) return false;
	return true;
}

template <class T>
bool printSquare(T* arrA, int size) {
	int n = (int)sqrt(size);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) std::cout << arrA[i * n + j] << '\t';
		std::cout << std::endl;
	}
}

#endif  /* __GPU_TIMER_H__ */