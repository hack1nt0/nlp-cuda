//
// Created by DY on 17-10-9.
//

#include "SymmetricDistance.h"
#include <common_headers.h>
#include <matrix/CuSparseMatrix.h>

template <typename T>
__global__
void distKernel(T* D, int d_DSize, int from, int h_DSize, DeviceSparseMatrix<T> csr) {
    int ti = threadIdx.x + blockIdx.x * blockDim.x;
    int di = from + ti;
    if (di >= h_DSize || ti >= d_DSize) return;
    /*
       s = ti + 1
       r = rows-1
       (r+r-i+1)*i/2 >= s
       (2*r-i+1)*i >= 2*s
       (2*r+1)i-i^2 >= 2*s
       i^2-(2*r+1)*i+2*s <= 0
       a = 1, b = -2*r-1, c = 2*s
       (-b-sqrt(b^2-4*a*c))/2/a <= i
     */

    int s = di + 1;
    int r = csr.rows - 1;
    double a = 1;
    double b = -2. * r - 1;
    double c = 2. * s;
    int i = (int)ceil((-b - sqrt(b * b - 4 * a * c)) / 2 / a);
    //avoid float error
    i = max(1, min(i, csr.rows));
    if ((2 * r - (i - 1) + 1) * (i - 1) / 2 >= s) i--;
    if ((2 * r - i + 1) * i / 2 < s) i++;
    int rect = (2 * r - (i - 1) + 1) * (i - 1) / 2;
    int j = i + 1 + s - rect - 1;
    i--; j--;
    if (i == csr.rows - 2) printf("%d %d %d\n", i, j, ti);
    D[ti] = csr.distRow(i, j);
//    if (from > 0) printf("%d %d %e\n", i, j, D[ti]);
}

template <typename T>
void SymmetricDistance<T>::SymmetricDistance(const SparseMatrix<T> &dtm, bool verbose) {
    GpuTimer timer;
    if (verbose) {
        printf("dist...");
        timer.start();
    }
    int h_DSize = dtm.nrow() * (dtm.nrow() - 1) / 2;
    T* d_D;
    int d_DSize = int(1e7 * 2);
    checkCudaErrors(cudaMalloc(&d_D, sizeof(T) * d_DSize));
    DeviceSparseMatrix<T> d_m(dtm.data, dtm.index, dtm.row_ptr, dtm.rows, dtm.cols, dtm.nnz);
    int from = 0;
    while (from < h_DSize) {
        int threads = 16 * 16;
        int blocks = (min(d_DSize, h_DSize - from) + threads - 1) / threads;
        distKernel<<<blocks, threads>>>(d_D, d_DSize, from, h_DSize, d_m);
        checkCudaErrors(cudaDeviceSynchronize()); //todo
        cudaMemcpy(this->data + from, d_D, sizeof(double) * min(d_DSize, h_DSize - from), cudaMemcpyDeviceToHost);
        checkCudaErrors(cudaDeviceSynchronize()); //todo
        from += d_DSize;
        if (verbose) printf("\rdist...%d%%", min(100, int(double(from) / h_DSize * 100)));
    }
    cudaFree(d_D);
    if (verbose) {
        printf("\rdist...done. %e ms\n", timer.elapsed());
    }
}

template class SymmetricDistance<double>;

