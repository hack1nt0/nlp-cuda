//
// Created by DY on 17-10-9.
//

#include <common_headers.h>
#include <cu_common_headers.cu>
#include "matrix/SparseMatrix.h"
#include "matrix/CuSparseMatrix.cu"
#include "matrix/DenseMatrix.h"
#include "dist.h"

template <typename T>
__global__
void distKernel(T* D, int d_DSize, int from, int h_DSize, CuSparseMatrix<T> csr) {
    int ti = threadIdx.x + blockIdx.x * blockDim.x;
    int di = from + ti;
    if (di >= h_DSize || ti >= d_DSize) return;
    int i, j;
    dist<T>::unzip(i, j, di, csr.rows);
    D[ti] = csr.distRow(i, j);
//    if (from > 0) printf("%d %d %e\n", i, j, D[ti]);
}

template <typename T>
void dist<T>::init(const SparseMatrix<T> &dtm, bool verbose) {
    GpuTimer timer;
    if (verbose) {
        printf("dist...");
        fflush(stdout);
    }
    T* d_D;
    int d_size = int(1e7 * 2);
    checkCudaErrors(cudaMalloc(&d_D, sizeof(T) * d_size));
    CuSparseMatrix<T> d_m(dtm.data, dtm.index, dtm.row_ptr, dtm.rows, dtm.cols, dtm.nnz);
    int from = 0;
    while (from < nnz) {
        int threads = 16 * 16;
        int blocks = (min(d_size, nnz - from) + threads - 1) / threads;
        distKernel<<<blocks, threads>>>(d_D, d_size, from, nnz, d_m);
        checkCudaErrors(cudaDeviceSynchronize()); //todo
        cudaMemcpy(this->data + from, d_D, sizeof(double) * min(d_size, nnz - from), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaDeviceSynchronize()); //todo
        from += d_size;
        if (verbose) {
            printf("\rdist...%d%%", min(100, int(double(from) / nnz * 100)));
            fflush(stdout);
        }
    }
    cudaFree(d_D);
    if (verbose) {
        printf("\rdist...done. %e ms\n", timer.elapsed());
        fflush(stdout);
    }
}

template class dist<double>;

