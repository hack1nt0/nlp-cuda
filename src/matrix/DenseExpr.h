//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_DENSE_EXPR_H
#define NLP_CUDA_DENSE_EXPR_H

#include "functors.h"

template<typename V, typename I, typename ETYPE>
struct DenseExpr {
    inline const ETYPE& self() const {
        return *static_cast<const ETYPE *>(this);
    }
};

template <typename V, typename I>
struct ConstDenseExpr : public DenseExpr<V, I, ConstDenseExpr<V, I> > {
    V v;
    ConstDenseExpr(V v) : v(v) {}

    inline V at(I r, I c) const {
        return v;
    }

    inline V at(I i) const {
        return v;
    }

    inline I nrow() const { return 1; }

    inline I ncol() const { return 1; }

    inline I getNnz() const { return 1; }
};

template <class OP, class LHS, typename V, typename I>
struct UnaryDenseExpr : DenseExpr<V, I, UnaryDenseExpr<OP, LHS, V, I> > {
    LHS lhs;

    UnaryDenseExpr(const LHS &lhs) : lhs(lhs) {}

    inline V at(I r, I c) const {
        return OP::apply(lhs.at(r, c));
    }

    inline V at(I i) const {
        return OP::apply(lhs.at(i));
    }

    inline I nrow() const { return lhs.nrow(); }

    inline I ncol() const { return lhs.ncol(); }

    inline I getNnz() const { return lhs.getNnz(); }
};

template <class LHS, typename V, typename I>
UnaryDenseExpr<Sqrt<V>, LHS, V, I> sqrt(const DenseExpr<V, I, LHS> &e) {
    return UnaryDenseExpr<Sqrt<V>, LHS, V, I>(e.self());
}

template <class LHS, typename V, typename I>
UnaryDenseExpr<Log<V>, LHS, V, I> log(const DenseExpr<V, I, LHS> &e) {
    return UnaryDenseExpr<Log<V>, LHS, V, I>(e.self());
}

template <class LHS, typename V, typename I>
UnaryDenseExpr<Sign<V>, LHS, int, I> sign(const DenseExpr<V, I, LHS> &e) {
    return UnaryDenseExpr<Sign<V>, LHS, int, I>(e.self());
}

template <class LHS, typename V, typename I>
struct TransDenseExpr : public DenseExpr<V, I, TransDenseExpr<LHS, V, I> > {
    const LHS& lhs;

    TransDenseExpr(const LHS &lhs) : lhs(lhs) {}

    inline V at(I r, I c) const {
        return lhs.at(c, r);
    }

    inline I nrow() const { return lhs.ncol(); }

    inline I ncol() const { return lhs.nrow(); }

    inline I getNnz() const { return lhs.getNnz(); }
};

template <typename LHS, typename V, typename I> inline
TransDenseExpr<LHS, V, I> operator~(const DenseExpr<V, I, LHS> &lhs) {
    return TransDenseExpr<LHS, V, I>(lhs.self());
};

template <class OP, class LHS, class RHS, typename V, typename I>
struct BinDenseExpr : public DenseExpr<V, I, BinDenseExpr<OP, LHS, RHS, V, I> > {
    LHS lhs;
    RHS rhs;

    BinDenseExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    inline V at(I r, I c) const {
        return OP::apply(lhs.at(r, c), rhs.at(r % rhs.nrow(), c % rhs.ncol()));
    }

    inline V at(I i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }

    inline I nrow() const { return lhs.nrow(); }

    inline I ncol() const { return lhs.ncol(); }

    inline I getNnz() const { return lhs.getNnz(); }
};

template <class LHS, class RHS, typename LV, typename RV, typename I> inline
BinDenseExpr<Add<LV, RV>, LHS, RHS, LV, I> operator+(const DenseExpr<LV, I, LHS> &lhs, const DenseExpr<RV, I, RHS> &rhs) {
    return BinDenseExpr<Add<LV, RV>, LHS, RHS, LV, I>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename LV, typename RV, typename I> inline
BinDenseExpr<Sub<LV, RV>, LHS, RHS, LV, I> operator-(const DenseExpr<LV, I, LHS> &lhs, const DenseExpr<RV, I, RHS> &rhs) {
    return BinDenseExpr<Sub<LV, RV>, LHS, RHS, LV, I>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename LV, typename RV, typename I> inline
BinDenseExpr<Mul<LV, RV>, LHS, RHS, LV, I> operator*(const DenseExpr<LV, I, LHS> &lhs, const DenseExpr<RV, I, RHS> &rhs) {
    return BinDenseExpr<Mul<LV, RV>, LHS, RHS, LV, I>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename LV, typename RV, typename I> inline
BinDenseExpr<Div<LV, RV>, LHS, RHS, LV, I> operator/(const DenseExpr<LV, I, LHS> &lhs, const DenseExpr<RV, I, RHS> &rhs) {
    return BinDenseExpr<Div<LV, RV>, LHS, RHS, LV, I>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename LV, typename RV, typename I> inline
BinDenseExpr<Pow<LV, RV>, LHS, RHS, LV, I> operator^(const DenseExpr<LV, I, LHS> &lhs, const DenseExpr<RV, I, RHS> &rhs) {
    return BinDenseExpr<Pow<LV, RV>, LHS, RHS, LV, I>(lhs.self(), rhs.self());
};

template <class LHS, typename LV, typename I> inline
BinDenseExpr<Pow<LV, LV>, LHS, ConstDenseExpr<LV, I>, LV, I> operator^(const DenseExpr<LV, I, LHS> &lhs, LV v) {
    return BinDenseExpr<Pow<LV, LV>, LHS, ConstDenseExpr<LV, I>, LV, I>(lhs.self(), ConstDenseExpr<LV, I>(v));
};

template <class LHS, class RHS, typename V, typename I>
BinDenseExpr<Eq<V>, LHS, RHS, bool, I> operator==(const DenseExpr<V, I, LHS> &lhs, const DenseExpr<V, I, RHS> &rhs) {
    return BinDenseExpr<Eq<V>, LHS, RHS, bool, I>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename V, typename I>
BinDenseExpr<Max<V>, LHS, RHS, V, I> max(const DenseExpr<V, I, LHS> &lhs, const DenseExpr<V, I, RHS> &rhs) {
    return BinDenseExpr<Max<V>, LHS, RHS, V, I>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename V, typename I>
BinDenseExpr<Min<V>, LHS, RHS, V, I> min(const DenseExpr<V, I, LHS> &lhs, const DenseExpr<V, I, RHS> &rhs) {
    return BinDenseExpr<Min<V>, LHS, RHS, V, I>(lhs.self(), rhs.self());
};

/** Shrunk Ops **/

template <class LHS, typename V, typename I> inline
V sum(const DenseExpr<V, I, LHS> &lhs) {
    const LHS& e = lhs.self();
    V r = 0;
    for (I i = 0; i < e.getNnz(); ++i) r += e.at(i);
    return r;
};

#endif //NLP_CUDA_EXPRESSION_TEMPLATES_H
