//
// Created by DY on 17/12/18.
//

#ifndef NLP_CUDA_MATRIX_H
#define NLP_CUDA_MATRIX_H

#include "SparseMatrix.h"
#include "CDenseMatrix.h"
#include "RDenseMatrix.h"
#include <mkl.h>
#include <functional>

template <typename V, typename I>
struct MatrixUtils {
    typedef V              value_t;
    typedef I              index_t;
    typedef SparseMatrix<V, I> spm_t;
    typedef RDenseMatrix<V, I> rdm_t;
    typedef CDenseMatrix<V, I> cdm_t;
    typedef DenseVector<V, I>  dv_t;
    typedef SparseVector<V, I> sv_t;


    static value_t dot(const dv_t& a, const sv_t& b) {
        value_t r = 0;
        for (index_t i = 0; i < b.nnz; ++i) r += a[b.index[i]] * b[i];
        return r;
    }

    static void mm(cdm_t& C,
            double alpha,
            const spm_t& A,
            const cdm_t& B,
            double beta) {
#pragma omp parallel for
        for (int i = 0; i < A.nrow(); ++i)
            for (int j = 0; j < B.ncol(); ++j) {
                double r = alpha * dot(B.col(j), A.row(i));
                C.at(i, j) = r + beta * C.at(i, j);
            }
    }
};

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
    capply(o, [=](typename Mat::Vector &v) {
      auto nm = norm(v, L);
      if (nm != 0) v /= nm;
    });
}

template<class Mat>
inline static void rnormalize(Mat &o, int L = 2) {
    rapply(o, [=](typename Mat::Vector &v) {
      auto nm = norm(v, L);
      if (nm != 0) v /= nm;
    });
}

template<class Mat, typename R>
void creduce(Mat& output, const Mat &input, std::function<void(typename Mat::value_t&, const typename Mat::Vector &)> f) {
    typedef typename Mat::value_t value_t;
    typedef typename Mat::index_t index_t;
#pragma omp parallel for
    for (index_t i = 0; i < input.ncol(); ++i) {
        f(output.at(i), input.col(i));
    }
}

template<class Mat>
inline void csum(Mat& output, const Mat &input) {
    typedef typename Mat::value_t value_t;
    typedef typename Mat::index_t index_t;
    typedef typename Mat::Vector  col_t;
    creduce<Mat, value_t>(output, input, [](value_t& r, const col_t& col) {
      r = 0;
      for (index_t i = 0; i < col.getNnz(); ++i) r += col[i];
    });
}


struct BlasWrapper {
    typedef MKL_INT index_t;
    typedef SparseMatrix<double, index_t> DoubleSparseMatrix;
    typedef RDenseMatrix<double, index_t> DoubleRDenseMatrix;
    typedef CDenseMatrix<double, index_t> DoubleCDenseMatrix;
    typedef SparseMatrix<float,  index_t> FloatSparseMatrix;
    typedef RDenseMatrix<float,  index_t> FloatRDenseMatrix;
    typedef CDenseMatrix<float,  index_t> FloatCDenseMatrix;
    typedef DenseVector<double,  index_t> DoubleDenseVector;
    typedef DenseVector<float,   index_t> FloatDenseVector;
    typedef SparseVector<double, index_t> DoubleSparseVector;
    typedef SparseVector<float,  index_t> FloatSparseVector;

    char transa;
    char* matdescra;

    BlasWrapper() : transa('N'), matdescra(new char[6]) {
        matdescra[0] = 'G';
        matdescra[1] = 'x';
        matdescra[2] = 'x';
        matdescra[3] = 'F';
    }

    ~BlasWrapper() { delete[] matdescra;}

    inline void mm(DoubleRDenseMatrix &C,
                   double alpha,
                   const DoubleSparseMatrix &A,
                   const DoubleRDenseMatrix &B,
                   double beta) {
        matdescra[3] = 'C';
        index_t m = A.nrow();
        index_t n = B.ncol();
        index_t k = A.ncol();
        assert(k == B.nrow());
        mkl_dcsrmm(&transa, &m, &n, &k, &alpha, matdescra, A.getCsrVal(), A.getCsrInd(), A.getCsrPtr(), A.getCsrPtr() + 1, B.getValue(), &n, &beta, C.getValue(), &n);
    };

    inline void mm(FloatRDenseMatrix &C,
                   float alpha,
                   const FloatSparseMatrix &A,
                   const FloatRDenseMatrix &B,
                   float beta) {
        matdescra[3] = 'C';
        index_t m = A.nrow();
        index_t n = B.ncol();
        index_t k = A.ncol();
        assert(k==B.nrow());
        mkl_scsrmm(&transa, &m, &n, &k, &alpha, matdescra, A.getCsrVal(), A.getCsrInd(), A.getCsrPtr(), A.getCsrPtr() + 1, B.getValue(), &n, &beta, C.getValue(), &n);
    }

    inline void mm(DoubleCDenseMatrix &C,
                   double alpha,
                   const DoubleSparseMatrix &A,
                   const DoubleCDenseMatrix &B,
                   double beta) {
        matdescra[3] = 'F';
        to1BaseIndexing(A);
        index_t m = A.nrow();
        index_t n = B.ncol();
        index_t k = A.ncol();
        assert(k == B.nrow());
        mkl_dcsrmm(&transa, &m, &n, &k, &alpha, matdescra, A.getCsrVal(), A.getCsrInd(), A.getCsrPtr(), A.getCsrPtr() + 1, B.getValue(), &k, &beta, C.getValue(), &m);
        to0BaseIndexing(A);
    }

    inline void mm(FloatCDenseMatrix &C,
                   float alpha,
                   const FloatSparseMatrix &A,
                   const FloatCDenseMatrix &B,
                   float beta) {
        matdescra[3] = 'F';
        to1BaseIndexing(A);
        index_t m = A.nrow();
        index_t n = B.ncol();
        index_t k = A.ncol();
        assert(k == B.nrow());
        mkl_scsrmm(&transa, &m, &n, &k, &alpha, matdescra, A.getCsrVal(), A.getCsrInd(), A.getCsrPtr(), A.getCsrPtr() + 1, B.getValue(), &k, &beta, C.getValue(), &m);
        to0BaseIndexing(A);
    }

    template <class SparseMatrix>
    void to1BaseIndexing(SparseMatrix& m) {
        typedef typename SparseMatrix::value_t value_t;
        index_t *ptr = m.getCsrPtr();
        index_t *ind = m.getCsrInd();
        if (ptr[0] == 0) {
            for (index_t i = 0; i < m.nrow() + 1; ++i) ++ptr[i];
            for (index_t i = 0; i < m.getNnz(); ++i) ++ind[i];
        }
    }

    template <class SparseMatrix>
    void to0BaseIndexing(SparseMatrix& m) {
        typedef typename SparseMatrix::value_t value_t;
        index_t *ptr = m.getCsrPtr();
        index_t *ind = m.getCsrInd();
        if (ptr[0] == 1) {
            for (index_t i = 0; i < m.nrow() + 1; ++i) --ptr[i];
            for (index_t i = 0; i < m.getNnz(); ++i) --ind[i];
        }
    }

    inline double dot(const DoubleSparseVector& a, const DoubleDenseVector& b) {
        return cblas_ddoti(a.nnz, a.value, a.index, b.value);
    }

    inline double dot(const DoubleDenseVector& a, const DoubleSparseVector& b) {
        return dot(b, a);
    }
};

#endif //NLP_CUDA_MATRIX_H
