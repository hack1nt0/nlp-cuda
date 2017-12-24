//
// Created by DY on 17/12/18.
//

#ifndef NLP_CUDA_MATRIX_H
#define NLP_CUDA_MATRIX_H

#include "SparseMatrix.h"
#include "DenseMatrix.h"

template<class Mat>
static typename Mat::value_t norm(const Mat &v, int L = 2) {
    typedef typename Mat::value_t value_t;
    typedef typename Mat::index_t index_t;
    value_t r = 0;
    switch (L) {
        case 1: {
            for (index_t i = 1; i < v.nnz; ++i) r += std::abs(v.at(i));
            break;
        }
        case 2: {
            for (index_t i = 0; i < v.nnz; ++i) r += v.at(i)*v.at(i);
            r = sqrt(r);
            break;
        }
        default: {
            throw std::runtime_error("Unknown normlaization.");
        }
    }
    return r;
}

template<class Mat>
static void capply(Mat &o, std::function<void(typename Mat::Vector &)> f) {
    typedef typename Mat::value_t value_t;
    typedef typename Mat::index_t index_t;
#pragma omp parallel for
    for (index_t i = 0; i < o.ncol(); ++i) {
        auto v = o.col(i);
        f(v);
    }
}

template<class Mat>
static void rapply(Mat &o, std::function<void(typename Mat::Vector &)> f) {
    typedef typename Mat::value_t value_t;
    typedef typename Mat::index_t index_t;
#pragma omp parallel for
    for (index_t i = 0; i < o.nrow(); ++i) {
        auto v = o.row(i);
        f(v);
    }
}

template<class Mat>
inline static void cnormalize(Mat &o, int L = 2) {
    capply(o, [=](typename Mat::Vector &v) { v /= norm(v, L); });
}

template<class Mat>
inline static void rnormalize(Mat &o, int L = 2) {
    rapply(o, [=](typename Mat::Vector &v) { v /= norm(v, L); });
}

template<class Mat>
static void creduce(Mat& output, Mat &input, std::function<void(const typename Mat::Vector &)> f) {
    typedef typename Mat::value_t value_t;
    typedef typename Mat::index_t index_t;
#pragma omp parallel for
    for (index_t i = 0; i < input.ncol(); ++i) {
        output.at(i) = f(input.col(i));
    }
}

#endif //NLP_CUDA_MATRIX_H
