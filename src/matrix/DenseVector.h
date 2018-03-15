//
// Created by DY on 2018/1/6.
//

#ifndef NLP_CUDA_DENSEVECTOR_H
#define NLP_CUDA_DENSEVECTOR_H

#include "DenseExpr.h"
#include <cassert>
#include <iostream>

using namespace std;

template <typename V, typename I>
struct SparseVector;

template <typename V, typename I>
struct DenseVector : DenseExpr<V, I, DenseVector<V, I> > {
    typedef DenseVector<V, I> self_t;
    typedef V              value_t;
    typedef I              index_t;
    index_t ld; // leading dimension
    index_t nnz;
    value_t* value;

    DenseVector(index_t ld, index_t nnz, value_t *value) : nnz(nnz), ld(ld), value(value) {}

    DenseVector(index_t nnz, value_t *value) : nnz(nnz), ld(1), value(value) {}

    inline index_t nrow() const { return ld == 1 ? nnz : 1; }
    inline index_t ncol() const { return ld == 1 ? 1 : nnz; }
    inline index_t getNnz() const { return nnz; }

    inline const value_t& at(index_t i) const {
        assert(i < nnz);
        return value[i * ld];
    }

    inline value_t& at(index_t i) {
        assert(i < nnz);
        return value[i * ld];
    }

    inline const value_t&operator[](index_t i) const {
        assert(i < nnz);
        return value[i * ld];
    }

    inline value_t&operator[](index_t i) {
        assert(i < nnz);
        return value[i*ld];
    }

    template <class EType>
    self_t& operator=(const DenseExpr<V, I, EType> &expr) {
        const EType& e = expr.self();
        for (index_t i = 0; i < nnz; ++i) at(i) = e.at(i);
        return *this;
    }

    self_t& operator=(V v) {
        *this = ConstDenseExpr<V, I>(v);
        return *this;
    }

    self_t& operator=(const SparseVector<V, I>& sv);

    template <class Etype>
    self_t& operator+=(const DenseExpr<V, I, Etype>& e) {
        *this = *this + e;
        return *this;
    }

    self_t& operator+=(const SparseVector<V, I>& sv);

    template <class Etype>
    self_t& operator-=(const DenseExpr<V, I, Etype>& e) {
        *this = *this - e;
        return *this;
    }

    template <class Etype>
    self_t& operator*=(const DenseExpr<V, I, Etype>& e) {
        *this = *this * e;
        return *this;
    }

    template <class Etype>
    self_t& operator/=(const DenseExpr<V, I, Etype>& e) {
        *this = *this / e;
        return *this;
    }

    self_t& operator/=(V v) {
        *this = *this / ConstDenseExpr<V, I>(v);
        return *this;
    }

    void print(bool head = true) const {
        if (head) cout << "DenseVector " << ld << " * " << nnz << endl;
        for (index_t i = 0; i < nnz; ++i) cout << at(i) << '\t';
        cout << endl;
    }

};

#endif //NLP_CUDA_DENSEVECTOR_H
