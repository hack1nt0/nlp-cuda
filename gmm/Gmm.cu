//
// Created by DY on 17-6-18.
//

#include "../dtm/DocumentTermMatrix.h"
#include "DiagNormalPDF.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
using namespace std;

typedef struct {
    int dn;
    int tn;
    int* idx;
    float* weight;
    int* stride;
} DeviceDtm;

__global__
void expect(const DeviceDtm& d_dtm, const float* d_prior, const DiagNormalPDF* d_centroids, float* d_resp, int n, int k, int itr) {

}

__global__
void maximize(const DeviceDtm& d_dtm, float* d_prior, DiagNormalPDF* d_centroids, const float* d_resp, int n, int k, int ki) {

}

class Gmm {
public:
    DocumentTermMatrix* h_dtm;
    DeviceDtm d_dtm;
    DiagNormalPDF* d_centorids;
    float *d_resp, *d_prior, *d_loglikelihood;
    int n, m, k;

    Gmm(DocumentTermMatrix& dtm, int k) {
        this->n = dtm.dn;
        this->m = dtm.indexer.term2id.size();
        this->k = k;
        h_dtm = &dtm;
        d_dtm.dn = dtm.dn;
        d_dtm.tn = dtm.tn;
        int size = dtm.indexer.term2id.size();
        cudaMalloc((void**)&d_dtm.idx, sizeof(int) * size);
        cudaMalloc((void**)&d_dtm.weight, sizeof(float) * size);
        cudaMalloc((void**)&d_dtm.stride, sizeof(int) * size);
        int offset = 0;
        for (int i = 0; i < dtm.dn; ++i) {
            int n = dtm.matrix[i].first.size();
            d_dtm.stride[i] = n;
            cudaMemcpy(d_dtm.idx + offset, dtm.matrix[i].first.data(), sizeof(int) * n, cudaMemcpyHostToDevice);
            cudaMemcpy(d_dtm.idx + offset, dtm.matrix[i].second.data(), sizeof(float) * n, cudaMemcpyHostToDevice);
            offset += n;
        }
        cudaMalloc((void**)&d_centorids, sizeof(DiagNormalPDF) * k);
        for (int i = 0; i < k; ++i) {
            d_centorids[i].n = m;
            cudaMalloc((void**)&d_centorids[i].mean, sizeof(float) * m);
            cudaMalloc((void**)&d_centorids[i].variance, sizeof(float) * m);
        }
        cudaMalloc((void**)&d_resp, sizeof(float) * n * k);
        cudaMalloc((void**)&d_prior, sizeof(float) * k);
    }


    void cluster(int maxItr) {
        cudaMalloc((void**)&d_loglikelihood, sizeof(float) * maxItr);
        for (int itr = 0; itr < maxItr; ++itr) {
            dim3 threads(1024);
            dim3 blocks((n + threads.x - 1) / threads.x);
            expect<<<blocks, threads>>>(d_dtm, d_prior, d_centorids, d_resp, d_dtm.dn, k, itr);
            cudaDeviceSynchronize();

            printf("iter %d, log-likelihood %.5f\n", itr, d_loglikelihood[itr]);

            for (int ki = 0; ki < k; ++ki) {
                maximize<<<blocks, threads>>>(d_dtm, d_prior, d_centorids, d_resp, d_dtm.dn, k, ki);
            }
            cudaDeviceSynchronize();
        }
    }
};
