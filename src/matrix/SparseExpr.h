//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_SPARSE_EXPR_H
#define NLP_CUDA_SPARSE_EXPR_H

#include "functors.h"

template<typename T, typename ETYPE>
struct SparExpr {
    inline const ETYPE& self(void) const {
        return *static_cast<const ETYPE *>(this);
    }
};

template <class OP, class LHS, typename T>
struct UnarySparExpr : public SparExpr<T, UnarySparExpr<OP, LHS, T> > {
    LHS lhs;

    UnarySparExpr(const LHS &lhs) : lhs(lhs) {}

    inline T at(int i) const {
        return OP::apply(lhs.at(i));
    }

    inline int nrow() const { return lhs.nrow(); }

    inline int ncol() const { return lhs.ncol(); }

    inline int getNnz() const { return lhs.getNnz(); }
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

template <class OP, class LHS, class RHS, typename T>
struct BinSparExpr : public SparExpr<T, BinSparExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    BinSparExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    inline T at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }

    inline int nrow() const { return lhs.nrow(); }

    inline int ncol() const { return lhs.ncol(); }

    inline int getNnz() const { return lhs.getNnz(); }
};

template <typename T>
struct ConstSparExpr : public SparExpr<T, ConstSparExpr<T> > {
    T v;
    ConstSparExpr(T v) : v(v) {}

    inline T at(int r, int c) const {
        return v;
    }

    inline T at(int i) const {
        return v;
    }
};

template <class LHS, typename T>
BinSparExpr<Mul<T>, LHS, ConstSparExpr<T>, T> operator*(const SparExpr<T, LHS> &lhs, T rhs) {
    return BinSparExpr<Mul<T>, LHS, ConstSparExpr<T>, T>(lhs.self(), ConstSparExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinSparExpr<Mul<T>, LHS, RHS, T> operator*(const SparExpr<T, LHS> &lhs, const SparExpr<T, RHS> &rhs) {
    return BinSparExpr<Mul<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinSparExpr<Div<T>, LHS, ConstSparExpr<T>, T> operator/(const SparExpr<T, LHS> &lhs, T rhs) {
    return BinSparExpr<Div<T>, LHS, ConstSparExpr<T>, T>(lhs.self(), ConstSparExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinSparExpr<Div<T>, LHS, RHS, T> operator/(const SparExpr<T, LHS> &lhs, const SparExpr<T, RHS> &rhs) {
    return BinSparExpr<Div<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinSparExpr<Pow<T>, LHS, ConstSparExpr<T>, T> operator^(const SparExpr<T, LHS> &lhs, T rhs) {
    return BinSparExpr<Pow<T>, LHS, ConstSparExpr<T>, T>(lhs.self(), ConstSparExpr<T>(rhs));
};

template <class LHS, typename T>
BinSparExpr<Sub<T>, LHS, ConstSparExpr<T>, T> operator-(const SparExpr<T, LHS> &lhs, T rhs) {
    return BinSparExpr<Sub<T>, LHS, ConstSparExpr<T>, T>(lhs.self(), ConstSparExpr<T>(rhs));
};

template <class LHS, typename T>
BinSparExpr<Add<T>, LHS, ConstSparExpr<T>, T> operator+(const SparExpr<T, LHS> &lhs, T rhs) {
    return BinSparExpr<Add<T>, LHS, ConstSparExpr<T>, T>(lhs.self(), ConstSparExpr<T>(rhs));
};

template <class LHS, typename T>
BinSparExpr<Max<T>, LHS, ConstSparExpr<T>, T> max(const SparExpr<T, LHS> &lhs, T rhs) {
    return BinSparExpr<Max<T>, LHS, ConstSparExpr<T>, T>(lhs.self(), ConstSparExpr<T>(rhs));
};

template <class OP, class LHS, class RHS, typename T>
struct BoolBinSparExpr : public SparExpr<T, BoolBinSparExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    BoolBinSparExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    inline bool at(int r, int c) const {
        return OP::apply(lhs.at(r, c), rhs.at(r, c));
    }

    inline bool at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }

    inline int nrow() const { return lhs.nrow(); }

    inline int ncol() const { return lhs.ncol(); }

    inline int getNnz() const { return lhs.getNnz(); }
};

template <class LHS, class RHS, typename T>
BoolBinSparExpr<NEq<T>, LHS, RHS, T> operator!=(const SparExpr<T, LHS> &lhs, const SparExpr<T, RHS> &rhs) {
    return BoolBinSparExpr<NEq<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
T max(const SparExpr<T, LHS> &lhs) {
    const LHS& e = lhs.self();
    T ma = e.at(0);
    for (int i = 1; i < e.getNnz(); ++i) ma = max(ma, e.at(i));
    return ma;
};

template <class LHS, typename T>
T min(const SparExpr<T, LHS> &lhs) {
    const LHS& e = lhs.self();
    T mi = e.at(0);
    for (int i = 1; i < e.getNnz(); ++i) mi = min(mi, e.at(i));
    return mi;
};

template <class LHS, typename T>
T sum(const SparExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    T s = 0;
    for (int i = 0; i < e.getNnz(); ++i) s += e.at(i);
    return s;
};


#endif //NLP_CUDA_SPARSE_EXPRESSION_TEMPLATES_H
