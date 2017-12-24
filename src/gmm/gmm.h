//
// Created by DY on 17-9-27.
//

#ifndef NLP_CUDA_GMM_H
#define NLP_CUDA_GMM_H

#include "../utils/utils.h"

template <typename T>
void gmmInit(T* h_mean, T* h_covar, T* h_class_weight,
             const T* data, const int* index, const int* row_ptr,
             int rows, int cols, int nnz,
             unsigned int k, unsigned int seed, T beta);


template <typename T>
vector<T> gmm(T* h_resp, T* h_mean, T* h_covar, T* h_class_weight,
              const T* data, const int* index, const int* row_ptr,
              unsigned int rows, unsigned int cols, unsigned int nnz,
              unsigned int k, unsigned int max_itr, unsigned int seed,
              T alpha, T beta);

#endif //NLP_CUDA_GMM_H
