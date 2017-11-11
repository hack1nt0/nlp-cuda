//
// Created by DY on 17-10-25.
//

#ifndef NLP_CUDA_DENSE_MATRIX_J_H
#define NLP_CUDA_DENSE_MATRIX_J_H

#include "CuDenseExprJ.cu"
#include <common_headers.h>
#include <cu_common_headers.cu>

namespace JNI {

template <typename T>
__global__
void assignKernel(T* dst, const CuDenseExprJ<T>* src) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i == 0) printf("0x%p  0x%p\n", dst, src);
//    if (i >= src->nrow() * src->ncol()) return;
//    dst[i] = src->at(i);
}

template <typename T>
__global__
void assignKernel(T* dst, T value) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
//    if (i == 0) printf("0x%p  0x%p\n", dst, src);
//    if (i >= src->nrow() * src->ncol()) return;
    if (i >= 1) return;
    dst[i] = value;
}

template<typename T>
struct CuDenseMatrixJ : CuDenseExprJ<T> {
    T* data;
//    vector<T> data;
    int rows, cols;

    virtual ~CuDenseMatrixJ() { printf("destructing...\n"); }

    CuDenseMatrixJ(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        cudaMalloc(&data, sizeof(T) * (rows * cols));
        checkCudaErrors(cudaDeviceSynchronize());
//            this->data.resize(rows * cols);
//            for (int i = 0; i < rows * cols; ++i) data[i] = rand() % 100;
    }

    CuDenseMatrixJ<T>& operator=(const CuDenseExprJ<T>* ep) {
        printf("%d %d \n", this->rows, this->cols);
        fflush(stdout);
//        printf("%d %d %d %d \n", this->rows, ep->nrow(), this->cols, ep->ncol());
//        fflush(stdout);
//        assert(this->rows == ep->nrow() && this->cols == ep->ncol());
        assign(ep);
    }

    void assign(const CuDenseExprJ<T>* ep) {
        int threads = 16 * 16;
        int blocks = (rows * cols + threads - 1) / threads;
        assignKernel<<<blocks, threads>>>(this->data, ep);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    CuDenseMatrixJ<T>& operator=(const T& value) {
        assign(ConstDenseExprJ<T>(value));
    }

    void assign(const T& value) {
        int threads = 16 * 16;
        int blocks = (rows * cols + threads - 1) / threads;
        assignKernel<<<blocks, threads>>>(this->data, value);
        checkCudaErrors(cudaDeviceSynchronize());
    }

    __device__
    inline T at(int r, int c) const {
        return data[r * this->cols + c];
    }

    __device__
    inline T at(int i) const {
        return data[i];
    }

    __device__ __host__
    inline int nrow() const { return this->rows; }

    __device__ __host__
    inline int ncol() const { return this->cols; }

    void print(int r, int c) {
        T* h_data = new T[rows * cols];
        cudaMemcpy(h_data, this->data, sizeof(T) * (rows * cols), cudaMemcpyDeviceToHost);
        printf("JniDenseMatrix [rows, cols] = [%d, %d]\n", this->nrow(), this->ncol());
        for (int i = 0; i < min(10, this->nrow()); ++i) {
            for (int j = 0; j < min(10, this->ncol()); ++j) {
                printf("%e\t", h_data[i * cols + j]);
            }
            printf("\n");
        }
    }

    inline void print() { print(this->nrow(), this->ncol()); }
};
}

#endif
