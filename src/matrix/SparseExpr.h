//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_SPARSE_EXPR_H
#define NLP_CUDA_SPARSE_EXPR_H

#include "functors.h"

template<typename V, typename I, typename ETYPE>
struct SparExpr {
    inline const ETYPE& self(void) const {
        return *static_cast<const ETYPE *>(this);
    }
};

template <class OP, class LHS, typename V, typename I>
struct UnarySparExpr : public SparExpr<V, I, UnarySparExpr<OP, LHS, V, I> > {
    LHS lhs;

    UnarySparExpr(const LHS& lhs) : lhs(lhs) {}

    inline V at(I i) const {
        return OP::apply(lhs.at(i));
    }

    inline I nrow() const { return lhs.nrow(); }

    inline I ncol() const { return lhs.ncol(); }

    inline I getNnz() const { return lhs.getNnz(); }
};

template <class LHS, typename V, typename I>
UnarySparExpr<Sqrt<V>, LHS, V, I> sqrt(const SparExpr<V, I, LHS> &lhs) {
    return UnarySparExpr<Sqrt<V>, LHS, V, I>(lhs.self());
}

template <class LHS, typename V, typename I>
UnarySparExpr<Neg<V>, LHS, V, I> operator-(const SparExpr<V, I, LHS>& lhs) {
    return UnarySparExpr<Neg<V>, LHS, V, I>(lhs.self());
}

template <class LHS, typename V, typename I>
UnarySparExpr<Exp<V>, LHS, V, I> exp(const SparExpr<V, I, LHS> &lhs) {
    return UnarySparExpr<Exp<V>, LHS, V, I>(lhs.self());
}

template <class LHS, typename V, typename I>
UnarySparExpr<Log<V>, LHS, V, I> log(const SparExpr<V, I, LHS> &lhs) {
    return UnarySparExpr<Log<V>, LHS, V, I>(lhs.self());
}

template <class OP, class LHS, class RHS, typename V, typename I>
struct BinSparExpr : public SparExpr<V, I, BinSparExpr<OP, LHS, RHS, V, I> > {
    LHS lhs;
    RHS rhs;

    BinSparExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    inline V at(I i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }

    inline I nrow() const { return lhs.nrow(); }

    inline I ncol() const { return lhs.ncol(); }

    inline I getNnz() const { return lhs.getNnz(); }
};

template <typename V, typename I>
struct ConstSparExpr : public SparExpr<V, I, ConstSparExpr<V, I> > {
    V v;
    ConstSparExpr(V v) : v(v) {}

    inline V at(I i) const {
        return v;
    }
};

template <class LHS, class RHS, typename LV, typename RV, typename I>
BinSparExpr<Add<LV, RV>, LHS, RHS, LV, I> operator+(const SparExpr<LV, I, LHS> &lhs, const SparExpr<RV, I, RHS> &rhs) {
    return BinSparExpr<Add<LV, RV>, LHS, RHS, LV, I>(lhs.self(), rhs.self());
};
template <class LHS, class RHS, typename LV, typename RV, typename I>
BinSparExpr<Sub<LV, RV>, LHS, RHS, LV, I> operator-(const SparExpr<LV, I, LHS> &lhs, const SparExpr<RV, I, RHS> &rhs) {
    return BinSparExpr<Sub<LV, RV>, LHS, RHS, LV, I>(lhs.self(), rhs.self());
};
template <class LHS, class RHS, typename LV, typename RV, typename I>
BinSparExpr<Mul<LV, RV>, LHS, RHS, LV, I> operator*(const SparExpr<LV, I, LHS> &lhs, const SparExpr<RV, I, RHS> &rhs) {
    return BinSparExpr<Mul<LV, RV>, LHS, RHS, LV, I>(lhs.self(), rhs.self());
};
template <class LHS, class RHS, typename LV, typename RV, typename I>
BinSparExpr<Div<LV, RV>, LHS, RHS, LV, I> operator/(const SparExpr<LV, I, LHS> &lhs, const SparExpr<RV, I, RHS> &rhs) {
    return BinSparExpr<Div<LV, RV>, LHS, RHS, LV, I>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename V, typename I>
BinSparExpr<NEq<V>, LHS, RHS, bool, I> operator!=(const SparExpr<V, I, LHS> &lhs, const SparExpr<V, I, RHS> &rhs) {
    return BinSparExpr<NEq<V>, LHS, RHS, bool, I>(lhs.self(), rhs.self());
};


template <class LHS, typename V, typename I>
V max(const SparExpr<V, I, LHS> &lhs) {
    const LHS& e = lhs.self();
    V ma = e.at(0);
    for (I i = 1; i < e.getNnz(); ++i) ma = std::max(ma, e.at(i));
    return ma;
};

template <class LHS, typename V, typename I>
V min(const SparExpr<V, I, LHS> &lhs) {
    const LHS& e = lhs.self();
    V mi = e.at(0);
    for (I i = 1; i < e.getNnz(); ++i) mi = std::min(mi, e.at(i));
    return mi;
};

template <class LHS, typename V, typename I>
V sum(const SparExpr<V, I, LHS> &lhs) {
    const LHS &e = lhs.self();
    V s = 0;
    for (I i = 0; i < e.getNnz(); ++i) s += e.at(i);
    return s;
};


#endif //NLP_CUDA_SPARSE_EXPRESSION_TEMPLATES_H
