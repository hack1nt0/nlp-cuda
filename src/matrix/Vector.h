//
// Created by DY on 2018/1/6.
//

#ifndef NLP_CUDA_VECTOR_H
#define NLP_CUDA_VECTOR_H

#include "DenseVector.h"
#include "SparseVector.h"

template<typename V, typename I>
DenseVector<V, I>& DenseVector<V, I>::operator=(const SparseVector<V, I> &sv) {
    //memset(value, 0, sizeof(value_t) * nnz);
    for (index_t i = 0; i < nnz; ++i) at(i) = 0;
    for (index_t i = 0; i < sv.nnz; ++i) at(sv.index[i]) = sv.at(i);
    return *this;
}

template<typename V, typename I>
DenseVector<V, I>& DenseVector<V, I>::operator+=(const SparseVector<V, I> &sv) {
    for (index_t i = 0; i < sv.nnz; ++i) {
        at(sv.index[i]) += sv.at(i);
    }
    return *this;
}
#endif //NLP_CUDA_VECTOR_H
