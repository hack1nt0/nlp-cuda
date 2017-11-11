//
// Created by DY on 17-8-5.
//

#ifndef NLP_CUDA_CU_SPARSE_MATRIX_H
#define NLP_CUDA_CU_SPARSE_MATRIX_H

#include "SparseMatrix.h"
#include "CuSparseMatrixHeader.cu"

template <typename T>
CuSparseMatrix<T>& CuSparseMatrix<T>::operator=(const SparseMatrix<T>& o) {
    assert(rows==o.rows && nnz==o.nnz);
    this->rows = o.rows;
    this->cols = o.cols;
    this->nnz = o.nnz;
    checkCudaErrors(cudaMemcpy(row_ptr, o.row_ptr, sizeof(int)*(o.rows + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(index, o.index, sizeof(int)*o.nnz, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(data, o.data, sizeof(T)*o.nnz, cudaMemcpyHostToDevice));
    this->needFree = true;
    return *this;
};

template <typename T>
CuSparseMatrix<T> CuSparseMatrix<T>::operator~() {
    SparseMatrix<T> h = *this;
    *this = ~h;
    return *this;
};

template <typename T>
CuDenseMatrix<T> CuSparseMatrix<T>::dp(const CuDenseMatrix<T>& o) {
    CuDenseMatrix<T> C;
};

template <typename T>
__global__
void printKernel(CuSparseMatrix<T> m, int rows, int cols) {
    printf("CuSparseMatrix[rows=%d, cols=%d, nnz=%d]:\n", m.rows, m.cols, m.nnz);
    for (int i = 0; i < min(m.rows, rows); ++i) {
        int c = 0;
        for (int j = m.row_ptr[i]; j < m.row_ptr[i + 1] && c < cols; ++j, ++c) {
            printf("(%d, %.3e)\t", m.index[j], m.data[j]);
        }
        printf("\n");
    }
    printf("--------------------------------");
}

template <typename T>
void CuSparseMatrix<T>::print(int rows, int cols) {
    printKernel<<<1, 1>>>(*this, rows, cols);
    checkCudaErrors(cudaDeviceSynchronize());
}

#endif


