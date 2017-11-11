//
// Created by DY on 17-9-9.
//

#ifndef NLP_CUDA_KMEANS_H
#define NLP_CUDA_KMEANS_H

#include "kmeans.h"
#include <common_headers.h>
#include <cu_common_headers.cu>
#include "matrix/CuSparseMatrix.cu"
#include <CpuTimer.h>

using namespace std;

template <typename T>
__global__
void calculateDist(DeviceDenseMatrix<T> dist, CuSparseMatrix<T> dtm, DeviceDenseMatrix<T> mean, DeviceDenseMatrix<T> distConst) {
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int di = tidx / dist.cols;
    int ki = tidx % dist.cols;
    if (di >= dtm.rows) return;
    T d = 0;
    int from = dtm.row_ptr[di];
    int to   = dtm.row_ptr[di + 1];
    for (int i = from; i < to; ++i) {
        d += dtm.data[i] * dtm.data[i] - 2 * dtm.data[i] * mean.at(ki, dtm.index[i]);
    }
    dist.at(di, ki) = d + distConst.at(ki);
}

template <typename T>
__global__
void calculateDistConst(DeviceDenseMatrix<T> distConst, DeviceDenseMatrix<T> mean) {
    int ki = threadIdx.x + blockDim.x * blockIdx.x;
    if (ki >= mean.rows) return;
    T c = 0;
    for (int i = 0; i < mean.cols; ++i) c += mean.at(ki, i) * mean.at(ki, i);
    distConst.at(ki) = c;
}

template <typename T>
__global__
void zeroSumDist(DeviceDenseMatrix<T> dist, DeviceDenseMatrix<T> minDist, DeviceDenseMatrix<int> belongTo) {
    int di = threadIdx.x + blockDim.x * blockIdx.x;
    if (di >= dist.rows) return;
    T minD = dist.at(di, 0);
    int minIdx = 0;
    for (int i = 1; i < dist.cols; ++i) {
        T d = dist.at(di, i);
        if (d < minD) {
            minIdx = i;
            minD = d;
        }
    }
    minDist.at(di) = minD;
    for (int i = 0; i < dist.cols; ++i) {
        if (i != minIdx) dist.at(di, i) = 0;
        else {
            dist.at(di, i) = 1;
            belongTo.at(di) = i;
        }
    }
}


template <typename T>
vector<T> kmeans(T* h_mean, int* h_class,
                   const T* data, const int* index, const int* row_ptr,
                   unsigned int rows, unsigned int cols, unsigned int nnz,
                   unsigned int k, unsigned int max_itr, unsigned int seed) {
    assert(0 < k);
    GpuTimer gpuTimer;

    DeviceDenseMatrix<T> d_dist(rows, k);
    DeviceDenseMatrix<T> d_distConst(k, 1);
    DeviceSparseMatrix<T> d_dtm(data, index, row_ptr, rows, cols, nnz);
    DeviceDenseMatrix<T> d_mean(h_mean, k, cols);
    DeviceDenseMatrix<T> d_classCapacity(k, 1);
    DeviceDenseMatrix<T> d_minDist(rows, 1);
    DeviceDenseMatrix<int> d_class(rows, 1);
    vector<T> distances;

    for (int itr = 0; itr < max_itr; ++itr) {
        gpuTimer.start();
        int threads0 = min(k, 16 * 16);
        int blocks0 = (k + threads0 - 1) / threads0;
        calculateDistConst <<< blocks0, threads0 >>> (d_distConst, d_mean);
        checkCudaErrors(cudaDeviceSynchronize());

        int threads1 = 16 * 16;
        int blocks1 = (rows * k + threads1 - 1) / threads1;
        calculateDist <<< blocks1, threads1 >>> (d_dist, d_dtm, d_mean, d_distConst);
        checkCudaErrors(cudaDeviceSynchronize());

        int threads2 = 16 * 16;
        int blocks2 = (rows + threads2 - 1) / threads2;
        zeroSumDist <<< blocks2, threads2 >>> (d_dist, d_minDist, d_class);
        checkCudaErrors(cudaDeviceSynchronize());

        thrust::device_ptr<T> dev_ptr(d_minDist.data);
        T cur_totDist = thrust::reduce(dev_ptr, dev_ptr + rows);
        printf("%5d\t%30e\t%20.3f\t", itr, cur_totDist, gpuTimer.elapsed() / 1000);
        if (distances.size() > 0 && abs(cur_totDist - distances[distances.size() - 1]) <= 1e-9) break;
        distances.push_back(cur_totDist);
        /**
         * mean[k, i] = \Sum_d data[d, i] * dist[d, k]
         *
         */

        gpuTimer.start();
        d_classCapacity = sum(d_dist, 0);
        d_classCapacity = maximum(d_classCapacity, 1.); //!
//        cout << d_classCapacity << endl;
        d_dtm.t();
        d_mean.reshape(cols, k);
        d_dist.t();
        d_dist.reshape(rows, k);
        DeviceDenseMatrix<T>::cudaSparseMultiplyDense(d_mean, 0., 1., d_dtm, false, d_dist, false);
        d_dtm.t();
        d_mean.reshape(k, cols);
        d_mean /= d_classCapacity;
        printf("%10.3f\n", gpuTimer.elapsed() / 1000);
    }

    d_mean.toHost(h_mean);
    d_class.toHost(h_class);

    return distances;
}

template
vector<double> kmeans(double* h_mean, int* h_class,
                 const double* data, const int* index, const int* row_ptr,
                 unsigned int rows, unsigned int cols, unsigned int nnz,
                 unsigned int k, unsigned int max_itr, unsigned int seed);

#endif
