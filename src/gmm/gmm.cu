//
// Created by DY on 17-6-18.
//
#ifndef NLP_CUDA_GMM_CU
#define NLP_CUDA_GMM_CU

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include <cuda_common_headers.cu>
#include <thrust/device_vector.h>
#include <cmath>
#include <fstream>
#include <matrix/CuSparseMatrix.h>
#include <ds/io.h>
#include "gmm.h"
#include <kmeans.h>

using namespace std;
//using namespace cutils;

template <typename T>
__global__
void expectKernel(DeviceDenseMatrix<T> resp, DeviceSparseMatrix<T> dtm, DeviceDenseMatrix<T> mean, DeviceDenseMatrix<T> conv,
DeviceDenseMatrix<T> class_weight, DeviceDenseMatrix<T> respConst) {
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   if (idx >= resp.rows * resp.cols) return;
   int d = idx / resp.cols; //todo
   int k = idx % resp.cols;
   /*
    * resp[d,k]       ~    w[k] * P(d|k) 
    * log(resp[d,k])  ~    log(w[k]) - 0.5 * ( M * log(2*PI) 
    * + log(prob(cov[k])) + SUM_i { (dtm[d,i]-mean[k,i])^2 / cov[k,i] } )
    *
    */
   T result = log(2. * 3.14) * mean.cols + respConst.at(k);
   int from = dtm.row_ptr[d];
   int to   = dtm.row_ptr[d + 1];
   for (int i = from; i < to; ++i) {
       int   m    = dtm.index[i];
       T data = dtm.data[i];
       result += (data * data - 2. * data * mean.at(k, m)) / conv.at(k, m);
   }
   result = log(class_weight.at(k)) - .5 * result;
   resp.at(d, k) = result;
}

template <typename T>
__global__
void respConstKernel(DeviceDenseMatrix<T> respConst, DeviceDenseMatrix<T> mean, DeviceDenseMatrix<T> cov) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= mean.rows) return;
    T result = 0;
    for (int i = 0; i < mean.cols; ++i) {
        result += mean.at(k, i) / cov.at(k, i) * mean.at(k, i);
        result += log(cov.at(k, i));
    }
    respConst.at(k) = result;
}

template <typename T>
__global__
void normRespKernel(DeviceDenseMatrix<T> resp, DeviceDenseMatrix<T> logLikelihood) {
    int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= resp.rows) return;
    T maxLogResp = -3.4e+38;
    for (int i = 0; i < resp.cols; ++i) {
        maxLogResp = max(maxLogResp, resp.at(d, i));
    }
    T sumExp = 0.;
    for (int i = 0; i < resp.cols; ++i) {
        sumExp += exp(resp.at(d, i) - maxLogResp);
    }
    T logSumExp = maxLogResp + log(sumExp);
    logLikelihood.set(d, logSumExp);
    for (int i = 0; i < resp.cols; ++i) {
//        resp.set(d, i, exp(resp.at(d, i) - logSumExp));
        resp.at(d, i) = exp(resp.at(d, i) - logSumExp);
    }
}

template <typename T>
__global__
void varKernel(DeviceDenseMatrix<T> var, DeviceDenseMatrix<T> resp, DeviceSparseMatrix<T> dtm, DeviceDenseMatrix<T> mean,
               DeviceDenseMatrix<T> class_weight, T smoothing) {
    int k = threadIdx.x;
    if (k >= var.rows) return;
    for (int r = 0; r < dtm.rows; ++r) {
        int from = dtm.row_ptr[r];
        int to   = dtm.row_ptr[r + 1];
        for (int i = from; i < to; ++i) {
            int m = dtm.index[i];
            var.at(k, m) += resp.at(r, k) * (dtm.data[m] - mean.at(k, m)) * (dtm.data[m] - mean.at(k, m));
        }
    }
    for (int m = 0; m < var.cols; ++m) {
        var.at(k, m) = var.at(k, m) / class_weight.at(k) + smoothing;
    }
}

template <typename T>
void gmmInit(T* h_mean, T* h_covar, T* h_class_weight,
             const T* data, const int* index, const int* row_ptr,
             int rows, int cols, int nnz,
             unsigned int k, unsigned int seed, T beta) {
    initKmeans(h_mean, k, data, index, row_ptr, rows, cols, nnz, seed);
    srand(seed);
    for (int i = 0; i < cols * k; ++i) {
        h_covar[i] = (T) rand() / RAND_MAX;
		h_covar[i] = max(h_covar[i], beta);
    }
    for (int i = 0; i < k; ++i) {
        h_class_weight[i] = 1.0f / k;
    }
}

template <typename T>
vector<T> gmm(T* h_resp, T* h_mean, T* h_covar, T* h_class_weight,
                 const T* data, const int* index, const int* row_ptr,
                 unsigned int rows, unsigned int cols, unsigned int nnz,
                 unsigned int k, unsigned int max_itr, unsigned int seed,
                 T alpha, T beta) {
    T class_weight_smoothing = alpha;
    T variance_smoothing = beta;
    DeviceSparseMatrix<T> d_dtm(data, index, row_ptr, rows, cols, nnz);
    DeviceSparseMatrix<T> d_dtm_pow_2(data, index, row_ptr, rows, cols, nnz);
    d_dtm_pow_2 = d_dtm_pow_2 ^ 2.;
    d_dtm_pow_2 = ~d_dtm_pow_2;

    DeviceDenseMatrix<T> d_mean(h_mean, k, cols);
    DeviceDenseMatrix<T> d_covar(h_covar, k, cols);
    DeviceDenseMatrix<T> d_conv_tmp(k, cols);
    DeviceDenseMatrix<T> d_respect(rows, k);
    DeviceDenseMatrix<T> d_respect_const(1, k);
    DeviceDenseMatrix<T> d_respect_col_major(rows, k);
    DeviceDenseMatrix<T> d_class_weight(h_class_weight, k, 1);
    DeviceDenseMatrix<T> d_doc_likelihood(k, 1);

    DeviceDenseMatrix<T> tmp(cols, k);

    T pre_likelihood = -3.4e+38; //todo
    vector<T> h_likelihood;
    GpuTimer gpuTimer;

    printf("Iteration\t(Average).Log.likelihood\tExpectation(s)\tMaximization(s)\n");
    unsigned int valid_itr;
    for (valid_itr = 0; valid_itr < max_itr; ++valid_itr) {
        gpuTimer.start();

        int threads0 = min(16 * 16, k);
        int blocks0 = (k + threads0 - 1) / threads0;
        respConstKernel <<< blocks0, threads0 >>> (d_respect_const, d_mean, d_covar);
        checkCudaErrors(cudaDeviceSynchronize());
        int threads1 = min(16 * 16, k);
        int blocks1 = rows * ((k + threads1 - 1) / threads1);
        expectKernel <<< blocks1, threads1 >>>
                                   (d_respect, d_dtm, d_mean, d_covar, d_class_weight, d_respect_const);
        checkCudaErrors(cudaDeviceSynchronize());
        int threads2 = min(16 * 16, rows);
        int blocks2 = (rows + threads2 - 1) / threads2;
        normRespKernel <<< blocks2, threads2 >>> (d_respect, d_doc_likelihood);
        checkCudaErrors(cudaDeviceSynchronize());


        thrust::device_ptr<T> dev_ptr(d_doc_likelihood.data);
        T cur_likelihood = thrust::reduce(dev_ptr, dev_ptr + rows) / rows;
        printf("%5d\t%30e\t%20.3f\t", valid_itr, cur_likelihood, gpuTimer.elapsed() / 1000);

        if (cur_likelihood != cur_likelihood || abs(cur_likelihood - pre_likelihood) <= 1e-4) break;
        h_likelihood.push_back(cur_likelihood);
        pre_likelihood = cur_likelihood;

        //Maximization
        gpuTimer.start();

        /**
         * mean[ki, mi] = \SUM_di (resp[di, ki] * data[di, mi])
         *
         * var[ki, mi] = \SUM_di (data[di, mi] - mean[ki, mi]) ^ 2 * resp[di, ki]
         *             = \SUM_di (mean[ki, mi] ^ 2 + data[di, mi] ^ 2 - data[di, mi] * mean[ki, mi] * 2) * resp[di, ki]
         *
         * Everything is for effective, Time resorting to cublas & cusparse & expression template, Space BY HAND.
         * Expression Template only used in by element matrix evaluation, e.g. +, - or .*(Multiply by element),
         * others are work of Cublas and Cusparse.
         * Allocating space BY HAND is the most controlled and programmer-intuitive approach AFAIK
         *
         */

        d_class_weight = sum(d_respect, 0);
        d_dtm = ~d_dtm;
        tmp.reshape(cols, k);
        d_respect_col_major.rows = k;
        d_respect_col_major.cols = rows;
        d_respect_col_major = ~d_respect;
        d_respect_col_major.rows = rows;
        d_respect_col_major.cols = k;
        DeviceDenseMatrix<T>::cudaSparseMultiplyDense(tmp, 0., 1., d_dtm, false, d_respect_col_major, false);
        tmp.reshape(k, cols);
        d_dtm = ~d_dtm;
        d_class_weight = maximum(d_class_weight, class_weight_smoothing); //todo
        d_mean = tmp / d_class_weight;

        /*
         * Ill-conditional covariance matrix(var(x, x) << 0, since we use diagonal covariance matrix) when using Matrix
         * -based approach to estimate the vars.
         */
//        d_conv = (d_mean ^ 2.f) * d_class_weight;
//        d_conv -= tmp * d_mean * 2.f;
//        tmp.reshape(cols, k);
//        DeviceDenseMatrix::cudaSparseMultiplyDense(tmp, 0.f, 1.f, d_dtm_pow_2, false, d_respect_col_major, false);
//        tmp.reshape(k, cols);
//        d_conv += tmp;
//        d_conv = d_conv / d_class_weight + variance_smoothing;

        d_covar = tmp * tmp / d_class_weight;
        tmp.reshape(cols, k);
        DeviceDenseMatrix<T>::cudaSparseMultiplyDense(tmp, 0.f, 1.f, d_dtm_pow_2, false, d_respect_col_major, false);
        tmp.reshape(k, cols);
        d_covar = tmp / d_class_weight - d_covar / d_class_weight;
        d_covar = maximum(d_covar, variance_smoothing);

//        int varKernelThreads = min(16 * 16, k);
//        int varKernelBlocks = (k + varKernelThreads - 1) / varKernelThreads;
//        d_conv = 0.f;
//        varKernel<<<varKernelBlocks, varKernelThreads>>>(d_conv, d_respect, d_dtm, d_mean, d_class_weight, variance_smoothing);

//        thrust::device_ptr<double> dev_ptr2(d_conv.data);
//        cout << "max var " << thrust::reduce(dev_ptr2, dev_ptr2 + k * cols, MIN_double, thrust::maximum<double>());
//        thrust::device_ptr<double> dev_ptr3(d_conv.data);
//        cout << " min var " << thrust::reduce(dev_ptr3, dev_ptr3 + k * cols, -MIN_double, thrust::minimum<double>());

        thrust::device_ptr<T> tmp_ptr(d_class_weight.data);
        T sum_class_weight = thrust::reduce(tmp_ptr, tmp_ptr + k);
        d_class_weight /= sum_class_weight;

        printf("%10.3f\n", gpuTimer.elapsed() / 1000.);
    }

    d_respect.toHost(h_resp);
    d_mean.toHost(h_mean);
    d_covar.toHost(h_covar);
    d_class_weight.toHost(h_class_weight);

    return h_likelihood;
}

template void gmmInit<double>(double* h_mean, double* h_covar, double* h_class_weight,
                                 const double* data, const int* index, const int* row_ptr,
                                 int rows, int cols, int nnz,
                                 unsigned int k, unsigned int seed, double beta);

template vector<double> gmm(double* h_resp, double* h_mean, double* h_covar, double* h_class_weight,
                   const double* data, const int* index, const int* row_ptr,
                   unsigned int rows, unsigned int cols, unsigned int nnz,
                   unsigned int k, unsigned int max_itr, unsigned int seed,
                   double alpha, double beta);
#endif
