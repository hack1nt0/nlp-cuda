//
// Created by DY on 17-8-5.
//

#ifndef NLP_CUDA_DEVICE_MATRIX_H
#define NLP_CUDA_DEVICE_MATRIX_H

#include <istream>
#include <vector>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>
#include <CudaUtils.cu>
#include <cassert>
#include <ostream>
#include <expression_templates.h>
#include <matrix.h>
#include "matrix.h"

using namespace std;

//namespace cutils {
    class CudaBlasContext {
    public:
        cublasHandle_t handle;

        CudaBlasContext() {
            cublasCreate(&handle);
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        }

        virtual ~CudaBlasContext() {
            cublasDestroy(handle);
        }
    };

    const CudaBlasContext cudaBlasContext;

    class CudaSparseContext {
    public:
        cusparseHandle_t handle;

        CudaSparseContext() {
            cusparseCreate(&handle);
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
        }

        virtual ~CudaSparseContext() {
            cusparseDestroy(handle);
        }
    };

    const CudaSparseContext cudaSparseContext;

template <typename T>
    class DeviceSparseMatrix : public BaseExpr<T, DeviceSparseMatrix<T> > {
    public:
        T *data = 0;
        int *index = 0;
        int *row_ptr = 0;
        int copyDepth = -1;
        int rows = 0;
        int cols = 0;
        int nnz = 0;
        cusparseMatDescr_t descr; //todo

        DeviceSparseMatrix() {}

        DeviceSparseMatrix(int rows, int cols, int nnz) {
            checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * nnz));
            checkCudaErrors(cudaMalloc((void **) &this->index, sizeof(int) * nnz));
            checkCudaErrors(cudaMalloc((void **) &this->row_ptr, sizeof(int) * (rows + 1)));
            this->rows = rows;
            this->cols = cols;
            this->nnz = nnz;
            this->copyDepth = 0;
            checkCusparseErrors(cusparseCreateMatDescr(&descr));
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        }

        DeviceSparseMatrix(const T* data, const int* index, const int* row_ptr, int rows,
                           int cols, int nnz) {
            checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * nnz));
            checkCudaErrors(cudaMemcpy(this->data, data, sizeof(T) * nnz, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMalloc((void **) &this->index, sizeof(int) * nnz));
            checkCudaErrors(cudaMemcpy(this->index, index, sizeof(int) * nnz, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMalloc((void **) &this->row_ptr, sizeof(int) * (rows + 1)));
            checkCudaErrors(cudaMemcpy(this->row_ptr, row_ptr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice));
            this->rows = rows;
            this->cols = cols;
            this->nnz = nnz;
            this->copyDepth = 0;
            checkCusparseErrors(cusparseCreateMatDescr(&descr));
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        }

        DeviceSparseMatrix(const DeviceSparseMatrix &that) {
            this->data = that.data;
            this->index = that.index;
            this->row_ptr = that.row_ptr;
            this->rows = that.rows;
            this->cols = that.cols;
            this->nnz = that.nnz;
            this->descr = that.descr;
            this->copyDepth = that.copyDepth + 1;
        }

        DeviceSparseMatrix& operator=(const DeviceSparseMatrix &that) {
            if (this->nnz != that.nnz) {
                if (this->data != 0) {
                    checkCudaErrors(cudaFree(this->data));
                    checkCudaErrors(cudaFree(this->index));
                }
                checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * that.nnz));
                checkCudaErrors(cudaMalloc((void **) &this->index, sizeof(int) * that.nnz));
            }
            if (this->rows != that.rows) {
                if (this->data != 0) checkCudaErrors(cudaFree(this->row_ptr));
                checkCudaErrors(cudaMalloc((void **) &this->row_ptr, sizeof(int) * (that.rows + 1)));
            }
            checkCudaErrors(cudaMemcpy(this->data, that.data, sizeof(T) * that.nnz, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(this->index, that.index, sizeof(int) * that.nnz, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(this->row_ptr, that.row_ptr, sizeof(int) * (that.rows + 1), cudaMemcpyDeviceToDevice));
            this->rows = that.rows;
            this->cols = that.cols;
            this->nnz = that.nnz;
            this->copyDepth = 0;
            return *this;
        }

        virtual ~DeviceSparseMatrix() {
            if (copyDepth == 0) {
                if (data != 0) {
                    cudaFree(data);
                    cudaFree(index);
                    cudaFree(row_ptr);
                    checkCusparseErrors(cusparseDestroyMatDescr(descr));
                }
            }
        }

        friend ostream &operator<<(ostream &os, const DeviceSparseMatrix &d_matrix) {
            SparseMatrix<T> matrix;
            matrix = d_matrix;
            os << "DeviceSparseMatrix [rows, cols, nnz] = [" << matrix.rows << ", " << matrix.cols << ", " << matrix.nnz << "]" << endl;
            for (int i = 0; i < matrix.rows; ++i) {
                int from = matrix.row_ptr[i], to = matrix.row_ptr[i + 1];
                for (int j = 0; j < matrix.cols; ++j) {
                    if (from < to && j == matrix.index[from]) {
                        printf("%10.3f\t", matrix.data[from++]);
                    } else {
                        printf("%10s\t", ".");
                    }
                }
                os << endl;
            }
//            HostArray<T> data(matrix.nnz);
//            checkCudaErrors(cudaMemcpy(data.data, matrix.data, sizeof(T) * data.size, cudaMemcpyDeviceToHost));
//            HostArray<int> index(matrix.nnz);
//            checkCudaErrors(cudaMemcpy(index.data, matrix.index, sizeof(int) * index.size, cudaMemcpyDeviceToHost));
//            HostArray<int> row_ptr(matrix.rows + 1);
//            checkCudaErrors(
//                    cudaMemcpy(row_ptr.data, matrix.row_ptr, sizeof(int) * row_ptr.size, cudaMemcpyDeviceToHost));
//            std::cout << matrix.rows << '\t' << matrix.cols << '\t' << matrix.nnz << std::endl;
//            data.print("data", 100);
//            index.print("index", 100);
//            row_ptr.print("row_ptr", 100);
            return os;
        }

        __device__
        T at(size_t i) const {
            return data[i];
        }

        template <class ETYPE>
        DeviceSparseMatrix& operator=(const BaseExpr<T, ETYPE> &expr) {
            fillDevice(data, expr.self(), nnz);
            checkCudaErrors(cudaDeviceSynchronize());
            return *this;
        }

        DeviceSparseMatrix& operator=(const SparseMatrix<T> &h_matrix) {
            if (this->nnz != h_matrix.nnz) {
                if (this->data != 0) {
                    checkCudaErrors(cudaFree(this->data));
                    checkCudaErrors(cudaFree(this->index));
                }
                checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * h_matrix.nnz));
                checkCudaErrors(cudaMalloc((void **) &this->index, sizeof(int) * h_matrix.nnz));
            }
            if (this->rows != h_matrix.rows) {
                if (this->data != 0) checkCudaErrors(cudaFree(this->row_ptr));
                checkCudaErrors(cudaMalloc((void **) &this->row_ptr, sizeof(int) * (h_matrix.rows + 1)));
            }
            checkCudaErrors(cudaMemcpy(this->data, h_matrix.data, sizeof(T) * h_matrix.nnz, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(this->index, h_matrix.index, sizeof(int) * h_matrix.nnz, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(this->row_ptr, h_matrix.row_ptr, sizeof(int) * (h_matrix.rows + 1), cudaMemcpyHostToDevice));
            this->rows = h_matrix.rows;
            this->cols = h_matrix.cols;
            this->nnz = h_matrix.nnz;
            this->copyDepth = 0;
            return *this;
        }

        DeviceSparseMatrix& operator=(const TransExpr<T, DeviceSparseMatrix> &expr) {
            SparseMatrix<T> matrix;
            matrix = expr.lhs;
            matrix = ~matrix;
            *this = matrix;
            return *this;
        }
    };

    class TestExpr;

    /*
     * Row major, Shadow copy
     * when factory method return new instance, the \copyDepth MUST BE -1.
     */
    template <typename T>
    class DeviceDenseMatrix : public BaseExpr<T, DeviceDenseMatrix<T> > {
    public:
        T *data = 0;
        int copyDepth = -1;
        int rows, cols;

        void init(const T* data, int rows, int cols) {
            checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * (rows * cols)));
            checkCudaErrors(cudaMemcpy(this->data, data, sizeof(T) * (rows * cols), cudaMemcpyHostToDevice));
            this->rows = rows;
            this->cols = cols;
            this->copyDepth = 0;
        }

        DeviceDenseMatrix(const T* data, int rows, int cols) {
            init(data, rows, cols);
        }

        DeviceDenseMatrix(int rows, int cols) {
            vector<T> vec(rows * cols);
            for (int i = 0; i < (int) vec.size(); ++i) vec[i] = (rand() % 100) / 10.f;
            init(vec.data(), rows, cols);
        }

        DeviceDenseMatrix(const DeviceDenseMatrix &that) {
            this->data = that.data;
            this->rows = that.rows;
            this->cols = that.cols;
            this->copyDepth = that.copyDepth + 1;
        }

        DeviceDenseMatrix& operator=(const DeviceDenseMatrix &that) {
            if (this != &that) {
                if (this->rows * this->cols != that.rows * this->cols) {
                    this->~DeviceDenseMatrix();
                }
                this->data = that.data;
                this->rows = that.rows;
                this->cols = that.cols;
                this->copyDepth = that.copyDepth + 1;
            }
            return *this;
        }

        void toHost(T* h_data) {
            checkCudaErrors(cudaMemcpy(h_data, data, sizeof(T) * (rows * cols), cudaMemcpyDeviceToHost));
        }

//        template <typename T>
//        friend T* operator=(T* h_data, const DeviceDenseMatrix& d_data) {
//            checkCudaErrors(cudaMemcpy(h_data, d_data.data, sizeof(T) * (d_data.rows * d_data.cols), cudaMemcpyDeviceToHost));
//        }

        virtual ~DeviceDenseMatrix() {
            if (copyDepth == 0) {
                checkCudaErrors(cudaFree(data));
            }
        }

        /*
         * C = alpha * A * B + beta * C
         * */
        static void cudaSparseMultiplyDense(DeviceDenseMatrix &C, double beta,
                                            double alpha, const DeviceSparseMatrix<double> &A, bool transposeA,
                                            const DeviceDenseMatrix<double> &B, bool transposeB) {
            cusparseOperation_t transA = transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
            cusparseOperation_t transB = transposeB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
            checkCusparseErrors(
                    cusparseDcsrmm2(cudaSparseContext.handle, //todo
                                    transA,
                                    transB,
                                    A.rows,
                                    B.cols,
                                    A.cols,
                                    A.nnz,
                                    &alpha,
                                    A.descr,
                                    A.data,
                                    A.row_ptr,
                                    A.index,
                                    B.data,
                                    B.rows,
                                    &beta,
                                    C.data,
                                    C.rows)
            );
        }

        void reshape(int rows, int cols) {
            assert(rows * cols == this->rows * this->cols && rows > 0 && cols > 0);
            this->rows = rows;
            this->cols = cols;
        }

        friend ostream &operator<<(ostream &os, const DeviceDenseMatrix &matrix) {
            int size = matrix.rows * matrix.cols;
            T *data = new T[size];
            checkCudaErrors(cudaMemcpy(data, matrix.data, sizeof(T) * size, cudaMemcpyDeviceToHost));
            os << "DeviceDenseMatrix [rows, cols] = [" << matrix.rows << ", " << matrix.cols << "]" << endl;
            for (int i = 0; i < min(matrix.rows, 10); ++i) {
                for (int j = 0; j < min(matrix.cols, 10); ++j) {
//                    os << data[i * matrix.cols + j] <<"\t";
                    printf("%e\t", data[i * matrix.cols + j]);
                }
                os << endl;
            }
            delete[] data;
            return os;
        }

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

        template <class ETYPE>
        DeviceDenseMatrix& operator=(const BaseExpr<T, ETYPE> &expr) {
            fillDevice(data, expr.self(), rows, cols);
            checkCudaErrors(cudaDeviceSynchronize());
            return *this;
        }

        DeviceDenseMatrix& operator=(T value) {
            fillDevice(data, ConstViewer<T>(value), rows, cols);
            checkCudaErrors(cudaDeviceSynchronize());
            return *this;
        }

        template <class E>
        DeviceDenseMatrix& operator+=(E expr) {
            return *this = *this + expr;
        }

        template <class E>
        DeviceDenseMatrix& operator-=(E expr) {
            return *this = *this - expr;
        }

        template <class E>
        DeviceDenseMatrix& operator*=(E expr) {
            return *this = *this * expr;
        }

        template <class E>
        DeviceDenseMatrix& operator/=(E expr) {
            return *this = *this / expr;
        }

        template <class OP, class LHS>
        DeviceDenseMatrix& operator=(const ZipExpr<OP, LHS, T> &expr) {
            fillDevice(data, expr, rows * cols);
            checkCudaErrors(cudaDeviceSynchronize());
            return *this;
        }

        __device__ inline
        void set(int row, int col, T value) {
            //todo bound checking
            data[row * cols + col] = value;
        }

        __device__ inline
        void set(int i, T value) {
            data[i] = value;
        }
    };


//}

#endif


