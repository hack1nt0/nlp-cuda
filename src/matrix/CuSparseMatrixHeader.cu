//
// Created by DY on 17-8-5.
//

#ifndef NLP_CUDA_CU_SPARSE_MATRIX_HEADER_H
#define NLP_CUDA_CU_SPARSE_MATRIX_HEADER_H

#include "CuSparseExpr.cu"

template <typename T>
struct CuDenseMatrix;

template <typename T>
struct SparseMatrix;

template <typename T>
struct CuSparseMatrix : CuSparExpr<T, CuSparseMatrix<T> > {
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    int* row_ptr = NULL;
    int* index = NULL;
    T* data = NULL;
    bool needFree;
    string type;
    cusparseMatDescr_t descr; //todo

    virtual ~CuSparseMatrix() {
        if (needFree) {
            checkCudaErrors(cudaFree(row_ptr));
            checkCudaErrors(cudaFree(index));
            checkCudaErrors(cudaFree(data));
            checkCusparseErrors(cusparseDestroyMatDescr(descr));
        }
    }

    CuSparseMatrix(const CuSparseMatrix<T>& o) : type(typeid(T).name()) {
        this->rows = o.rows;
        this->cols = o.cols;
        this->nnz = o.nnz;
        this->row_ptr = o.row_ptr;
        this->index = o.index;
        this->data = o.data;
        this->descr = o.descr;
        this->needFree = false;
    }

    CuSparseMatrix(int rows, int cols, int nnz) : type(typeid(T).name()) {
        this->rows = rows;
        this->cols = cols;
        this->nnz = nnz;
        checkCudaErrors(cudaMalloc((void **) &this->row_ptr, sizeof(int) * (rows + 1)));
        checkCudaErrors(cudaMalloc((void **) &this->index, sizeof(int) * nnz));
        checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * nnz));
        this->needFree = true;
        checkCusparseErrors(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    }

    CuSparseMatrix(int rows, int cols, int nnz,
                   int* row_ptr, int* index, T* data) : type(typeid(T).name())  {
        this->rows = rows;
        this->cols = cols;
        this->nnz = nnz;
        this->row_ptr = row_ptr;
        this->index = index;
        this->data = data;
        this->needFree = false;
        checkCusparseErrors(cusparseCreateMatDescr(&descr));
        cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
        cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
    }

    template <class ETYPE>
    CuSparseMatrix<T>& operator=(const CuSparExpr<T, ETYPE> &expr) {
        foreachDevice(data, expr, nnz);
        checkCudaErrors(cudaDeviceSynchronize());
        return *this;
    }

    CuSparseMatrix<T>& operator=(const CuSparseMatrix& o) {
        assert(rows == o.rows && nnz == o.nnz);
        this->rows = o.rows;
        this->cols = o.cols;
        this->nnz = o.nnz;
        checkCudaErrors(cudaMemcpy(row_ptr, o.row_ptr, sizeof(int) * (o.rows + 1), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(index, o.index, sizeof(int) * o.nnz, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(data, o.data, sizeof(T) * o.nnz, cudaMemcpyDeviceToDevice));
        this->needFree = true;
        return *this;
    }

    CuSparseMatrix<T>& operator=(const SparseMatrix<T>& o);

    CuSparseMatrix<T> operator~();

    //dot product
    CuDenseMatrix<T> dp(const CuDenseMatrix<T>& o);

    __device__
    T at(size_t i) const {
        return data[i];
    }

    __device__ __host__ inline
    int nrow() { return rows; }

    __device__ __host__ inline
    int ncol() { return cols; }

    __device__ __host__ inline
    int getNnz() { return nnz; }

    void print(int rows = 10, int cols = 10);
};

#endif


