//
// Created by DY on 2018/1/6.
//

#ifndef NLP_CUDA_SPARSEVECTOR_H
#define NLP_CUDA_SPARSEVECTOR_H

#include "SparseExpr.h"
#include <cassert>
#include <iostream>

using namespace std;

template <typename V, typename I>
struct SparseVector : SparExpr<V, I, SparseVector<V, I> > {
    typedef SparseVector<V, I> self_t;
    typedef I                index_t;
    typedef V                value_t;
    index_t size = 0;
    const index_t nnz = 0;
    const index_t* index = nullptr;
    value_t* value = nullptr;

    SparseVector() = default; //todo

    SparseVector(const index_t size, const index_t nnz, const index_t *index, value_t *value)
        : size(size), nnz(nnz), index(index), value(value) {}

    inline value_t& at(index_t i) {
        assert(i < nnz);
        return value[i];
    }

    inline const value_t& at(index_t i) const {
        assert(i < nnz);
        return value[i];
    }

    inline const value_t&operator[](index_t i) const {
        assert(i < nnz);
        return value[i];
    }

    inline value_t&operator[](index_t i) {
        assert(i < nnz);
        return value[i];
    }

    template <class EType>
    self_t& operator=(const SparExpr<V, I, EType> &expr) {
        const EType& e = expr.self();
        for (index_t i = 0; i < nnz; ++i) this->at(i) = e.at(i);
        return *this;
    }

    template <class Etype>
    self_t& operator*=(const SparExpr<V, I, Etype>& e) {
        *this = *this * e;
        return *this;
    }

    self_t& operator*=(V v) {
        *this = *this * ConstSparExpr<V, I>(v);
        return *this;
    }

    template <class Etype>
    self_t& operator/=(const SparExpr<V, I, Etype>& e) {
        *this = *this / e;
        return *this;
    }

    self_t& operator/=(V v) {
        *this = *this / ConstSparExpr<V, I>(v);
        return *this;
    }

    inline index_t getNnz() const { return nnz; }

    void print(bool head = true) const {
        if (head) cout << "SparseVector " << nnz << " of " << size << endl;
        for (index_t i = 0, j = 0; i < size; ++i) {
            if (j < nnz && index[j] == i) cout << at(j++) << '\t';
            else cout << ".\t";
        }
        cout << endl;
    }

};

#endif //NLP_CUDA_SPARSEVECTOR_H
