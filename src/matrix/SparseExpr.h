//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_SPARSE_EXPRESSION_TEMPLATES_H
#define NLP_CUDA_SPARSE_EXPRESSION_TEMPLATES_H


#include <iostream>
#include <cassert>
#include <host_defines.h>
#include "functors.h"
#include "expression_templates.h"

template<typename T, typename ETYPE>
class SparExpr {
public:
    inline const ETYPE& self(void) const {
        return *static_cast<const ETYPE *>(this);
    }
};

template <class OP, class LHS, typename T>
struct UnarySparExpr : public SparExpr<T, UnarySparExpr<OP, LHS, T> > {
    LHS lhs;

    UnarySparExpr(const LHS &lhs) : lhs(lhs) {}

    __device__ __host__
    inline T at(int i) const {
        return OP::apply(lhs.at(i));
    }
};

template <class LHS, typename T>
UnarySparExpr<Sqrt<T>, LHS, T> sqrt(const SparExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnarySparExpr<Sqrt<T>, LHS, T>(e);
}

template <class LHS, typename T>
UnarySparExpr<Neg<T>, LHS, T> operator-(const SparExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnarySparExpr<Neg<T>, LHS, T>(e);
}

template <class LHS, typename T>
UnarySparExpr<Exp<T>, LHS, T> exp(const SparExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnarySparExpr<Exp<T>, LHS, T>(e);
}

template <class LHS, typename T>
UnarySparExpr<Log<T>, LHS, T> log(const SparExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnarySparExpr<Log<T>, LHS, T>(e);
}

template <typename T, class LHS>
struct TransSparExpr {
    LHS lhs;

    TransSparExpr(const LHS &lhs) : lhs(lhs) {}
};

template <typename T, class LHS> inline
TransSparExpr<T, LHS> operator~(const SparExpr<T, LHS> &lhs) {
    return TransSparExpr<T, LHS>(lhs.self());
};

template <class OP, class LHS, class RHS, typename T>
struct BinSparExpr : public SparExpr<T, BinSparExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    BinSparExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    __device__ __host__
    inline T at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }
};

template <class LHS, typename T>
BinSparExpr<Mul<T>, LHS, ConstViewer<T>, T> operator*(const SparExpr<T, LHS> &lhs, T rhs) {
    return BinSparExpr<Mul<T>, LHS, ConstViewer<T>, T>(lhs.self(), ConstViewer<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinSparExpr<Mul<T>, LHS, RHS, T> operator*(const SparExpr<T, LHS> &lhs, const SparExpr<T, RHS> &rhs) {
    return BinSparExpr<Mul<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinSparExpr<Div<T>, LHS, ConstViewer<T>, T> operator/(const SparExpr<T, LHS> &lhs, T rhs) {
    return BinSparExpr<Div<T>, LHS, ConstViewer<T>, T>(lhs.self(), ConstViewer<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinSparExpr<Div<T>, LHS, RHS, T> operator/(const SparExpr<T, LHS> &lhs, const SparExpr<T, RHS> &rhs) {
    return BinSparExpr<Div<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinSparExpr<Pow<T>, LHS, ConstViewer<T>, T> operator^(const SparExpr<T, LHS> &lhs, T rhs) {
    return BinSparExpr<Pow<T>, LHS, ConstViewer<T>, T>(lhs.self(), ConstViewer<T>(rhs));
};



#endif //NLP_CUDA_SPARSE_EXPRESSION_TEMPLATES_H
