//
// Created by DY on 17-6-18.
//

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <CudaUtils.cu>
#include <thrust/device_vector.h>
#include <cmath>
#include <fstream>
#include <device_matrix.h>
#include "gmm.h"

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
//   resp.set(d, k, result);
    resp.at(d, k) = result;
}

template <typename T>
__global__
void respConstKernel(DeviceDenseMatrix<T> respConst, DeviceDenseMatrix<T> mean, DeviceDenseMatrix<T> cov) {
    int k = threadIdx.x + blockIdx.x * blockDim.x;
    if (k >= mean.rows) return;
    T result = 0.f;
    for (int i = 0; i < mean.cols; ++i) {
        result += mean.at(k, i) / cov.at(k, i) * mean.at(k, i);
        result += log(cov.at(k, i));
    }
//    respConst.set(k,  result);
    respConst.at(k) = result;
}

template <typename T>
__global__
void normRespKernel(DeviceDenseMatrix<T> resp, DeviceDenseMatrix<T> logLikelihood) {
    int d = threadIdx.x + blockIdx.x * blockDim.x;
    if (d >= resp.rows) return;
    double maxLogResp = -3.4e+38;
    for (int i = 0; i < resp.cols; ++i) {
        maxLogResp = max(maxLogResp, resp.at(d, i));
    }
    double sumExp = 0.f;
    for (int i = 0; i < resp.cols; ++i) {
        sumExp += exp(resp.at(d, i) - maxLogResp);
    }
    double logSumExp = maxLogResp + log(sumExp);
    logLikelihood.set(d, logSumExp);
    for (int i = 0; i < resp.cols; ++i) {
//        resp.set(d, i, exp(resp.at(d, i) - logSumExp));
        resp.at(d, i) = exp(resp.at(d, i) - logSumExp);
    }
}

template <typename T>
__global__
void varKernel(DeviceDenseMatrix<T> var, DeviceDenseMatrix<T> resp, DeviceSparseMatrix<T> dtm, DeviceDenseMatrix<T> mean,
               DeviceDenseMatrix<T> class_weight, double smoothing) {
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

void gmmInit(double* h_mean, double* h_conv, double* h_class_weight, unsigned int k, unsigned int cols, unsigned int seed) {
    srand(seed);
    rand();
    for (int i = 0; i < cols * k; ++i) {
        h_mean[i] = (double) rand() / RAND_MAX;
        h_conv[i] = (double) rand() / RAND_MAX * 50;
    }
    for (int i = 0; i < k; ++i) {
        h_class_weight[i] = 1.0f / k;
    }
}

GmmModel gmm(const double* data, const int* index, const int* row_ptr,
                 unsigned int rows, unsigned int cols, unsigned int nnz,
                 unsigned int k, unsigned int max_itr, unsigned int seed) {
    GmmModel model(rows, cols, k, max_itr, -1, seed);
    gmmInit(model.mean, model.conv, model.class_weight, k, cols, seed);
    model.valid_itr = gmm(model.resp, model.mean, model.conv, model.class_weight, model.likelihood,
                          data, index, row_ptr,
                          rows, cols, nnz,
                          k, max_itr, seed, 1e-5, 1e-5);
    return model;
}


//data may be changed
unsigned int gmm(double* h_resp, double* h_mean, double* h_conv, double* h_class_weight, double* h_likelihood,
                 const double* data, const int* index, const int* row_ptr,
                 unsigned int rows, unsigned int cols, unsigned int nnz,
                 unsigned int k, unsigned int max_itr, unsigned int seed,
                 double alpha, double beta) {
    /*
     * normalize
     * todo norm2 or whitening?
     */
    double class_weight_smoothing = alpha;
    double variance_smoothing = beta;
    DeviceSparseMatrix<double> d_dtm(data, index, row_ptr, rows, cols, nnz);
    DeviceSparseMatrix<double> d_dtm_pow_2(data, index, row_ptr, rows, cols, nnz);
    d_dtm_pow_2 = d_dtm_pow_2 ^ 2.;
    d_dtm_pow_2 = ~d_dtm_pow_2;

    DeviceDenseMatrix<double> d_mean(h_mean, k, cols);
    DeviceDenseMatrix<double> d_conv(h_conv, k, cols);
    DeviceDenseMatrix<double> d_conv_tmp(k, cols);
    DeviceDenseMatrix<double> d_respect(rows, k);
    DeviceDenseMatrix<double> d_respect_const(1, k);
    DeviceDenseMatrix<double> d_respect_col_major(rows, k);
    DeviceDenseMatrix<double> d_class_weight(h_class_weight, k, 1);
    DeviceDenseMatrix<double> d_doc_likelihood(k, 1);

    DeviceDenseMatrix<double> tmp(cols, k);

    double pre_likelihood = -3.4e+38; //todo
    GpuTimer gpuTimer;

    printf("Iteration\t(Average).Log.likelihood\tExpectation(s)\tMaximization(s)\n");
    unsigned int valid_itr;
    for (valid_itr = 0; valid_itr < max_itr; ++valid_itr) {
        gpuTimer.start();

        int threads0 = min(16 * 16, k);
        int blocks0 = (k + threads0 - 1) / threads0;
		cout << threads0 << '\t' << blocks0 << endl;
        respConstKernel <<< blocks0, threads0 >>> (d_respect_const, d_mean, d_conv);
        checkCudaErrors(cudaDeviceSynchronize());
//        cout << "gauss const " << endl << d_respect_const << endl;
        int threads1 = min(16 * 16, k);
        int blocks1 = rows * ((k + threads1 - 1) / threads1);
		cout << threads1 << '\t' << blocks1 << endl;
        expectKernel <<< blocks1, threads1 >>>
                                   (d_respect, d_dtm, d_mean, d_conv, d_class_weight, d_respect_const);
        checkCudaErrors(cudaDeviceSynchronize());
//        cout << "log d_respect" << endl << d_respect << endl;
        int threads2 = min(16 * 16, rows);
        int blocks2 = (rows + threads2 - 1) / threads2;
        normRespKernel <<< blocks2, threads2 >>> (d_respect, d_doc_likelihood);
        checkCudaErrors(cudaDeviceSynchronize());
//        cout << "d_respect" << endl << d_respect << endl;
//        cout << "d_doc_likelihood" << endl << d_doc_likelihood << endl;


        thrust::device_ptr<double> dev_ptr(d_doc_likelihood.data);
        double cur_likelihood = thrust::reduce(dev_ptr, dev_ptr + rows) / rows;
        h_likelihood[valid_itr] = cur_likelihood;
//        double min_likelihood = thrust::reduce(dev_ptr, dev_ptr + rows, -MIN_double, thrust::minimum<double>());
//        cout << "min likelihood " << min_likelihood << endl;
        printf("%5d\t%30e\t%20.3f\t", valid_itr, cur_likelihood, gpuTimer.elapsed() / 1000);

        if (cur_likelihood != cur_likelihood || abs(cur_likelihood - pre_likelihood) <= 1e-4) break;
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
        DeviceDenseMatrix<double>::cudaSparseMultiplyDense(tmp, 0., 1., d_dtm, false, d_respect_col_major, false);
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

        d_conv = tmp * tmp / d_class_weight;
        tmp.reshape(cols, k);
        DeviceDenseMatrix<double>::cudaSparseMultiplyDense(tmp, 0.f, 1.f, d_dtm_pow_2, false, d_respect_col_major, false);
        tmp.reshape(k, cols);
        d_conv = tmp / d_class_weight - d_conv / d_class_weight;
        d_conv = maximum(d_conv, variance_smoothing);

//        int varKernelThreads = min(16 * 16, k);
//        int varKernelBlocks = (k + varKernelThreads - 1) / varKernelThreads;
//        d_conv = 0.f;
//        varKernel<<<varKernelBlocks, varKernelThreads>>>(d_conv, d_respect, d_dtm, d_mean, d_class_weight, variance_smoothing);

//        thrust::device_ptr<double> dev_ptr2(d_conv.data);
//        cout << "max var " << thrust::reduce(dev_ptr2, dev_ptr2 + k * cols, MIN_double, thrust::maximum<double>());
//        thrust::device_ptr<double> dev_ptr3(d_conv.data);
//        cout << " min var " << thrust::reduce(dev_ptr3, dev_ptr3 + k * cols, -MIN_double, thrust::minimum<double>());

        thrust::device_ptr<double> tmp_ptr(d_class_weight.data);
        double sum_class_weight = thrust::reduce(tmp_ptr, tmp_ptr + k);
        d_class_weight /= sum_class_weight;

        printf("%10.3f\n", gpuTimer.elapsed() / 1000.);
    }

    d_respect.toHost(h_resp);
    d_mean.toHost(h_mean);
    d_conv.toHost(h_conv);
    d_class_weight.toHost(h_class_weight);

	printf("finished itr...\n");

    return valid_itr;
}


int main(int argc, char* argv[]) {
//    int k = atoi(argv[1]);
//    int max_itr = atoi(argv[2]);
//    int seed = atoi(argv[3]);
	int k = 2;
	int max_itr = 2;
	int seed = 1003;
    ifstream cin("/Users/dy/TextUtils/data/train/spamsms.dtm");
    int rows, cols, size;
    double sparsity, constructionCost;
    vector<double> data;
    vector<int> index;
    vector<int> row_ptr;

    cin >> rows;
    cin >> cols;
    cin >> size;
    cin >> sparsity;
    cin >> constructionCost;
    string word;
    getline(cin, word); // todo
    for (int i = 0; i < cols; ++i) {
        getline(cin, word);
//        cout << i << '\t' << word << endl;
    }
    double idf;
    for (int i = 0; i < cols; ++i) {
        cin >> idf;
    }
    for (int i = 0; i <= rows; ++i) {
        int d;
        cin >> d;
//        cout << i << '\t' << d << endl;
        row_ptr.push_back(d);
    }
    for (int i = 0; i < size; ++i) {
        int d;
        cin >> d;
        index.push_back(d);
    }
    for (int i = 0; i < size; ++i) {
        double d;
        cin >> d;
//        cout << i << '\t' << d << endl;
        data.push_back(d);
    }
	cin.close();
	cout << "here." << endl;
//
//    rows = 2;
//    cols = 4;
//    row_ptr.push_back(0);
//    row_ptr.push_back(2);
//    row_ptr.push_back(4);
//    index.push_back(0);
//    index.push_back(1);
//    index.push_back(2);
//    index.push_back(3);
//    data.push_back(10000);
//    data.push_back(1);
//    data.push_back(10000);
//    data.push_back(1);

    gmm(data.data(), index.data(), row_ptr.data(), rows, cols, size, k, max_itr, seed);

//
//    cout << endl << endl;
//    double dtm[2][2] =  {{0.f,   1.f},
//                       {0.f,    1.f}};
//    double mean[2][2] = {{
//                                0.000000e+00,2.886728e+01
//                        },
//                        {
//                                0.000000e+00,9.999949e-01
//                        }};
//    double resp[2][2] = {{
//                                1.731565e-02,9.826841e-01
//                        },
//                        {
//                                9.826841e-01,9.826841e-01
//                        }};
//    double class_w[2] = {
//            3.464129e-02,
//            1.965378e+00
//    };
//
//    double tmp[2][2];
//    for (int k = 0; k < 2; ++k) {
//        for (int m = 0; m < 2; ++m) {
//            tmp[k][m] = 0.f;
//            for (int d = 0; d < 2; ++d) tmp[k][m] += dtm[d][m] * resp[d][k];
//            printf("%e\t", tmp[k][m]);
//        }
//        cout << endl;
//    }
//    cout << endl;
//    double var[2][2];
//    for (int k = 0; k < 2; ++k) {
//        for (int m = 0; m < 2; ++m)  {
//            var[k][m] = 0.f;
//            for (int d = 0; d < 2; ++d) var[k][m] += resp[d][k] * dtm[d][m] * dtm[d][m];
//            for (int d = 0; d < 2; ++d) var[k][m] += resp[d][k] * powf(mean[k][m], 2.f);
//            for (int d = 0; d < 2; ++d) var[k][m] -= resp[d][k] * dtm[d][m] * mean[k][m] * 2.f;
//            var[k][m] = var[k][m] / class_w[k] + 1e-5f;
//            printf("%e\t", var[k][m]);
//        }
//        cout << endl;
//    }
    return 0;
}

