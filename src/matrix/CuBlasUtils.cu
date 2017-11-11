//
// Created by DY on 17-11-10.
//

#ifndef NLP_CUDA_CUBLASUTILS_H
#define NLP_CUDA_CUBLASUTILS_H

#include "CuDenseMatrix.cu"
#include "CuSparseMatrix.cu"

struct CudaSparseContext {
    cusparseHandle_t handle;

    CudaSparseContext() {
        cusparseCreate(&handle);
        cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    }

    virtual ~CudaSparseContext() {
        cusparseDestroy(handle);
    }
};
CudaSparseContext cudaSparseContext;

/*
 * C = alpha * A * B + beta * C
 * */
template <typename T>
void cuSparseMultiplyDense(CuDenseMatrix<T> &C, T beta,
                           T alpha, const CuSparseMatrix<T> &A, bool transposeA,
                           const CuDenseMatrix<T> &B, bool transposeB) {

    cusparseOperation_t transA = transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t transB = transposeB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    if (A.type == "d") {
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
    } else if (A.type == "f") {
        checkCusparseErrors(
            cusparseScsrmm2(cudaSparseContext.handle, //todo
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
}

struct CudaBlasContext {
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

#endif //NLP_CUDA_CUBLASUTILS_H
