//
// Created by DY on 17-10-14.
//

#ifndef NLP_CUDA_CUMATRIX_H
#define NLP_CUDA_CUMATRIX_H

#include "CuDenseExpr.cu"
#include "CuDenseMatrixHeader.cu"
#include "CDenseMatrix.h"

/*
 * Row major, Shadow copy
 */
template <typename T>
CuDenseMatrix<T> CuDenseMatrix<T>::operator~() {

    if (this->rows == this->cols) {
        transposeDevice(*this);
        checkCudaErrors(cudaDeviceSynchronize());
    } else {
        CDenseMatrix<T> h_matrix(this->rows, this->cols);
        h_matrix = *this;
        h_matrix.t();
        *this = h_matrix;
    }
    return *this;//todo
};

template <typename T>
CuDenseMatrix<T>& CuDenseMatrix<T>::operator=(const CDenseMatrix<T>& o) {
    assert(this->rows * this->cols == o.rows * o.cols);
    this->rows = o.rows;
    this->cols = o.cols;
    checkCudaErrors(cudaMemcpy(this->data, o.data, sizeof(T) * (this->rows * this->cols), cudaMemcpyHostToDevice));
    return *this;
};

template <typename T>
__global__
void printKernel(CuDenseMatrix<T> m, int rows, int cols) {
    printf("CuDenseMatrix[rows=%d, cols=%d]:\n", rows, cols);
    for (int i = 0; i < min(m.rows, rows); ++i) {
        for (int j = 0; j < min(m.cols, cols); ++j) {
            printf("%.3e\t", m.data[j]);
        }
        printf("\n");
    }
    printf("--------------------------------");
};

template <typename T>
void CuDenseMatrix<T>::print(int rows, int cols) {
    printKernel<<<1, 1>>>(*this, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize());
};

#endif //NLP_CUDA_CUMATRIX_H
