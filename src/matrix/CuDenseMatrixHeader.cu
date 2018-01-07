//
// Created by DY on 17-10-14.
//

#ifndef NLP_CUDA_CUMATRIX_HEADER_H
#define NLP_CUDA_CUMATRIX_HEADER_H

template <typename T>
struct CDenseMatrix;

#include "CuDenseExpr.cu"

/*
 * Row major, Shadow copy
 */
template <typename T>
struct CuDenseMatrix : CuDenseExpr<T, CuDenseMatrix<T> > {
    T *data;
    bool needFree;
    int rows, cols;

    virtual ~CuDenseMatrix() {
        if (needFree) {
            checkCudaErrors(cudaFree(data));
        }
    }

    void initData(const T *data, int rows, int cols) {
        checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * (rows * cols)));
        checkCudaErrors(cudaMemcpy(this->data, data, sizeof(T) * (rows * cols), cudaMemcpyHostToDevice));
    }

    CuDenseMatrix(const T* data, int rows, int cols) {
        initData(data, rows, cols);
        this->rows = rows;
        this->cols = cols;
        this->needFree = true;
    }

    CuDenseMatrix(int rows, int cols) {
        checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * (rows * cols)));
        this->rows = rows;
        this->cols = cols;
        this->needFree = true;
    }

    CuDenseMatrix(const CuDenseMatrix &that) {
        this->data = that.data;
        this->rows = that.rows;
        this->cols = that.cols;
        this->needFree = false;
    }

    CuDenseMatrix& operator=(const CuDenseMatrix &that) {
        if (this != &that) {
            if (this->rows * this->cols != that.rows * this->cols) {
                this->~CuDenseMatrix();
            }
            this->data = that.data;
            this->rows = that.rows;
            this->cols = that.cols;
            this->needFree = true;
        }
        return *this;
    }

    CuDenseMatrix<T>& operator=(const CDenseMatrix<T>& o);

    CuDenseMatrix<T>& operator=(T value) {
        fillDevice(data, getNnz(), CuDenseConstExpr<T>(value));
        checkCudaErrors(cudaDeviceSynchronize());
        return *this;
    }

    template <class ETYPE>
    CuDenseMatrix<T>& operator=(const CuDenseExpr<T, ETYPE> &e) {
        fillDevice(data, getNnz(), e);
        checkCudaErrors(cudaDeviceSynchronize());
        return *this;
    }

    template <class OP, class LHS>
    CuDenseMatrix& operator=(const CuDenseZipExpr<OP, LHS, T> &e) {
        fillDevice(data, getNnz(), e);
        checkCudaErrors(cudaDeviceSynchronize());
        return *this;
    }

    template <class E>
    CuDenseMatrix& operator+=(E expr) {
        return *this = *this + expr;
    }

    template <class E>
    CuDenseMatrix& operator-=(E expr) {
        return *this = *this - expr;
    }

    template <class E>
    CuDenseMatrix& operator*=(E expr) {
        return *this = *this * expr;
    }

    template <class E>
    CuDenseMatrix& operator/=(E expr) {
        return *this = *this / expr;
    }

    void reshape(int rows, int cols) {
        assert(rows * cols == this->rows * this->cols && rows > 0 && cols > 0);
        this->rows = rows;
        this->cols = cols;
    }

    CuDenseMatrix<T> operator~();

    __device__ inline
    T& at(int r, int c) const {
        r %= rows; // broad-casting
        c %= cols;
        return data[r * cols + c];
    }

    __device__ inline
    T& at(int i) const {
        return data[i];
    }

    __device__ __host__ inline
    int nrow() const {
        return rows;
    }

    __device__ __host__ inline
    int ncol() const {
        return cols;
    }

    __device__ __host__ inline
    int getNnz() const {
        return rows * cols;
    }

    void print(int rows = 10, int cols = 10);
};

template <typename T>
T sum(const CuDenseMatrix<T>& m) {
    thrust::device_ptr<T> devicePtr(m.data);
    return thrust::reduce(devicePtr, devicePtr + m.getNnz());
}

#endif //NLP_CUDA_CUMATRIX_H
