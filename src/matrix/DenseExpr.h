//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_DENSE_EXPR_H
#define NLP_CUDA_DENSE_EXPR_H

#include "functors.h"

template<typename T, typename ETYPE>
struct DenseExpr {
    inline const ETYPE& self(void) const {
        return *static_cast<const ETYPE *>(this);
    }

};

template <class OP, class LHS, typename T>
struct UnaryDenseExpr : public DenseExpr<T, UnaryDenseExpr<OP, LHS, T> > {
    LHS lhs;

    UnaryDenseExpr(const LHS &lhs) : lhs(lhs) {}

    inline T at(int r, int c) const {
        return OP::apply(lhs.at(r, c));
    }

    inline T at(int i) const {
        return OP::apply(lhs.at(i));
    }

    inline int nrow() const { return lhs.nrow(); }

    inline int ncol() const { return lhs.ncol(); }
};

template <class LHS, typename T>
UnaryDenseExpr<Sqrt<T>, LHS, T> sqrt(const DenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnaryDenseExpr<Sqrt<T>, LHS, T>(e);
}

template <class LHS, typename T>
UnaryDenseExpr<Log<T>, LHS, T> log(const DenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnaryDenseExpr<Log<T>, LHS, T>(e);
}

template <class LHS, typename T>
UnaryDenseExpr<Log2<T>, LHS, T> log2(const DenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnaryDenseExpr<Log2<T>, LHS, T>(e);
}

template <class OP, class LHS, typename T>
struct IntUnaryDenseExpr : public DenseExpr<T, IntUnaryDenseExpr<OP, LHS, T> > {
    LHS lhs;

    IntUnaryDenseExpr(const LHS &lhs) : lhs(lhs) {}

    inline int at(int r, int c) const {
        return OP::apply(lhs.at(r, c));
    }

    inline int at(int i) const {
        return OP::apply(lhs.at(i));
    }
};

template <class LHS, typename T>
IntUnaryDenseExpr<Sign<T>, LHS, T> sign(const DenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return IntUnaryDenseExpr<Sign<T>, LHS, T>(e);
}

template <typename T, class LHS>
struct TransDenseExpr : public DenseExpr<T, TransDenseExpr<T, LHS> > {
    const LHS& lhs;

    TransDenseExpr(const LHS &lhs) : lhs(lhs) {}

    inline T at(int r, int c) const {
        return lhs.at(c, r);
    }
};

template <class OP, class LHS, typename T>
//struct ZipDenseExpr : public BaseDenseExpr<T, ZipDenseExpr<OP, LHS, T> > {
struct ZipDenseExpr : public DenseExpr<T, ZipDenseExpr<OP, LHS, T> > {
    LHS lhs;
    int index;
    int rows;
    int cols;
    T zero;

    ZipDenseExpr(const LHS &lhs, int index, int rows, int cols, T zero) : lhs(lhs), index(index), rows(rows), cols(cols),
                                                              zero(zero) {
        assert(index == 0 || index == 1);
    }

    inline T at(int r, int c) const {
        return lhs.at(c, r);
    }

    inline T at(int i) const {
        T result = zero;
        switch (index) {
            case 0:
                for (int r = 0; r < rows; ++r) result = OP::apply(result, lhs.at(r, i));
                break;
            case 1:
                for (int c = 0; c < cols; ++c) result = OP::apply(result, lhs.at(i, c));
                break;
        }
        return result;
    }
};

template <class LHS, typename T>
ZipDenseExpr<Add<T>, LHS, T> sum(const DenseExpr<T, LHS> &lhs, int index) {
    const LHS &e = lhs.self();
    return ZipDenseExpr<Add<T>, LHS, T>(e, index, e.rows, e.cols, (T)0);
};

template <class LHS, typename T>
T sum(const DenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    T s = 0;
    for (int i = 0; i < e.nrow() * e.ncol(); ++i) s += e.at(i);
    return s;
};

template <typename T, class LHS> inline
TransDenseExpr<T, LHS> operator~(const DenseExpr<T, LHS> &lhs) {
    return TransDenseExpr<T, LHS>(lhs.self());
};

template <class OP, class LHS, class RHS, typename T>
struct BinDenseExpr : public DenseExpr<T, BinDenseExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    BinDenseExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    inline T at(int r, int c) const {
        return OP::apply(lhs.at(r, c), rhs.at(r, c));
    }

    inline T at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }

    inline int nrow() const { return lhs.nrow(); }

    inline int ncol() const { return lhs.ncol(); }
};

template <typename T, class LHS, class RHS> inline
BinDenseExpr<Add<T>, LHS, RHS, T> operator+(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinDenseExpr<Add<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <typename T, class LHS, class RHS> inline
BinDenseExpr<Sub<T>, LHS, RHS, T> operator-(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinDenseExpr<Sub<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <typename T>
struct ConstDenseExpr : public DenseExpr<T, ConstDenseExpr<T> > {
    T v;
    ConstDenseExpr(T v) : v(v) {}

    inline T at(int r, int c) const {
        return v;
    }

    inline T at(int i) const {
        return v;
    }
};



template <class LHS, typename T>
BinDenseExpr<Add<T>, LHS, ConstDenseExpr<T>, T> operator+(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinDenseExpr<Add<T>, LHS, ConstDenseExpr<T>, T>(lhs.self(), ConstDenseExpr<T>(rhs));
};

template <class LHS, typename T>
BinDenseExpr<Sub<T>, LHS, ConstDenseExpr<T>, T> operator-(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinDenseExpr<Sub<T>, LHS, ConstDenseExpr<T>, T>(lhs.self(), ConstDenseExpr<T>(rhs));
};

template <class LHS, typename T>
UnaryDenseExpr<Neg<T>, LHS, T> operator-(const DenseExpr<T, LHS> &lhs) {
    return UnaryDenseExpr<Neg<T>, LHS, T>(lhs.self());
};

template <class LHS, typename T>
BinDenseExpr<Mul<T>, LHS, ConstDenseExpr<T>, T> operator*(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinDenseExpr<Mul<T>, LHS, ConstDenseExpr<T>, T>(lhs.self(), ConstDenseExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinDenseExpr<Mul<T>, LHS, RHS, T> operator*(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinDenseExpr<Mul<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinDenseExpr<Div<T>, LHS, ConstDenseExpr<T>, T> operator/(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinDenseExpr<Div<T>, LHS, ConstDenseExpr<T>, T>(lhs.self(), ConstDenseExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinDenseExpr<Div<T>, LHS, RHS, T> operator/(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinDenseExpr<Div<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinDenseExpr<Pow<T>, LHS, ConstDenseExpr<T>, T> operator^(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinDenseExpr<Pow<T>, LHS, ConstDenseExpr<T>, T>(lhs.self(), ConstDenseExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinDenseExpr<Max<T>, LHS, RHS, T> maximum(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinDenseExpr<Max<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinDenseExpr<Max<T>, LHS, ConstDenseExpr<T>, T> maximum(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinDenseExpr<Max<T>, LHS, ConstDenseExpr<T>, T>(lhs.self(), ConstDenseExpr<T>(rhs));
};

template <class OP, class LHS, class RHS, typename T>
struct BoolBinDenseExpr : public DenseExpr<T, BoolBinDenseExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    BoolBinDenseExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    inline bool at(int r, int c) const {
        return OP::apply(lhs.at(r, c), rhs.at(r, c));
    }

    inline bool at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }

    inline int nrow() const { return lhs.nrow(); }

    inline int ncol() const { return lhs.ncol(); }
};

template <class LHS, class RHS, typename T>
BoolBinDenseExpr<Eq<T>, LHS, RHS, T> operator==(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BoolBinDenseExpr<Eq<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename T>
BoolBinDenseExpr<NEq<T>, LHS, RHS, T> operator!=(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BoolBinDenseExpr<NEq<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BoolBinDenseExpr<LessThan<T>, LHS, ConstDenseExpr<T>, T> operator<(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BoolBinDenseExpr<LessThan<T>, LHS, ConstDenseExpr<T>, T>(lhs.self(), ConstDenseExpr<T>(rhs));
};
#endif //NLP_CUDA_EXPRESSION_TEMPLATES_H
