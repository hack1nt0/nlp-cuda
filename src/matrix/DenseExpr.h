//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_EXPRESSION_TEMPLATES_H
#define NLP_CUDA_EXPRESSION_TEMPLATES_H


#include <iostream>
#include <cassert>
#include <cmath>
#include <host_defines.h>
#include "functors.h"
using namespace std;

//namespace cutils {


template<typename T, typename ETYPE>
class DenseExpr {
public:
    inline const ETYPE& self(void) const {
        return *static_cast<const ETYPE *>(this);
    }
};

template <class OP, class LHS, typename T>
struct UnaryExpr : public DenseExpr<T, UnaryExpr<OP, LHS, T> > {
    LHS lhs;

    UnaryExpr(const LHS &lhs) : lhs(lhs) {}

    __device__ __host__
    inline T at(int r, int c) const {
        return OP::apply(lhs.at(r, c));
    }

    __device__ __host__
    inline T at(int i) const {
        return OP::apply(lhs.at(i));
    }
};

template <class LHS, typename T>
UnaryExpr<Sqrt<T>, LHS, T> sqrt(const DenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnaryExpr<Sqrt<T>, LHS, T>(e);
}

template <class LHS, typename T>
UnaryExpr<Log<T>, LHS, T> log(const DenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnaryExpr<Log<T>, LHS, T>(e);
}

template <class LHS, typename T>
UnaryExpr<Log2<T>, LHS, T> log2(const DenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return UnaryExpr<Log2<T>, LHS, T>(e);
}

template <class OP, class LHS, typename T>
struct IntUnaryExpr : public DenseExpr<T, IntUnaryExpr<OP, LHS, T> > {
    LHS lhs;

    IntUnaryExpr(const LHS &lhs) : lhs(lhs) {}

    __device__ __host__
    inline int at(int r, int c) const {
        return OP::apply(lhs.at(r, c));
    }

    __device__ __host__
    inline int at(int i) const {
        return OP::apply(lhs.at(i));
    }
};

template <class LHS, typename T>
IntUnaryExpr<Sign<T>, LHS, T> sign(const DenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return IntUnaryExpr<Sign<T>, LHS, T>(e);
}

template <typename T, class LHS>
struct TransExpr : public DenseExpr<T, TransExpr<T, LHS> > {
    LHS lhs;

    TransExpr(const LHS &lhs) : lhs(lhs) {}

    __device__ __host__
    inline T at(int r, int c) const {
        return lhs.at(c, r);
    }
};

template <class OP, class LHS, typename T>
//struct ZipExpr : public BaseExpr<T, ZipExpr<OP, LHS, T> > {
struct ZipExpr : public DenseExpr<T, ZipExpr<OP, LHS, T> > {
    LHS lhs;
    int index;
    int rows;
    int cols;
    T zero;

    ZipExpr(const LHS &lhs, int index, int rows, int cols, T zero) : lhs(lhs), index(index), rows(rows), cols(cols),
                                                              zero(zero) {}

    __device__ __host__
    inline T at(int r, int c) const {
        return lhs.at(c, r);
    }

    __device__ __host__
    inline T at(int i) const {
        T result = zero;
        switch (index) {
            case 0:
                for (int r = 0; r < rows; ++r) {
//                    printf("%e\n", lhs.at(r, i));
                    result = OP::apply(result, lhs.at(r, i));
                }
                break;
            case 1:
                for (int c = 0; c < cols; ++c) result = OP::apply(result, lhs.at(i, c));
                break;
            default:
                printf("ZipExpr with uncorrected zip-index.");
        }
        return result;
    }
};

template <class LHS, typename T>
ZipExpr<Add<T>, LHS, T> sum(const DenseExpr<T, LHS> &lhs, int index) {
    const LHS &e = lhs.self();
    return ZipExpr<Add<T>, LHS, T>(e, index, e.rows, e.cols, (T)0);
};

template <typename T, class LHS> inline
TransExpr<T, LHS> operator~(const DenseExpr<T, LHS> &lhs) {
    return TransExpr<T, LHS>(lhs.self());
};

template <class OP, class LHS, class RHS, typename T>
struct BinExpr : public DenseExpr<T, BinExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    BinExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    __device__ __host__
    inline T at(int r, int c) const {
        return OP::apply(lhs.at(r, c), rhs.at(r, c));
    }

    __device__ __host__
    inline T at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }
};

template <typename T, class LHS, class RHS> inline
BinExpr<Add<T>, LHS, RHS, T> operator+(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinExpr<Add<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <typename T, class LHS, class RHS> inline
BinExpr<Sub<T>, LHS, RHS, T> operator-(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinExpr<Sub<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <typename T>
struct ConstViewer : public DenseExpr<T, ConstViewer<T> > {
    T v;
    ConstViewer(T v) : v(v) {}

    __device__ __host__
    inline T at(int r, int c) const {
        return v;
    }

    __device__ __host__
    inline T at(int i) const {
        return v;
    }
};

template <class LHS, typename T>
BinExpr<Add<T>, LHS, ConstViewer<T>, T> operator+(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinExpr<Add<T>, LHS, ConstViewer<T>, T>(lhs.self(), ConstViewer<T>(rhs));
};

template <class LHS, typename T>
BinExpr<Sub<T>, LHS, ConstViewer<T>, T> operator-(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinExpr<Sub<T>, LHS, ConstViewer<T>, T>(lhs.self(), ConstViewer<T>(rhs));
};

template <class LHS, typename T>
UnaryExpr<Neg<T>, LHS, T> operator-(const DenseExpr<T, LHS> &lhs) {
    return UnaryExpr<Neg<T>, LHS, T>(lhs.self());
};

template <class LHS, typename T>
BinExpr<Mul<T>, LHS, ConstViewer<T>, T> operator*(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinExpr<Mul<T>, LHS, ConstViewer<T>, T>(lhs.self(), ConstViewer<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinExpr<Mul<T>, LHS, RHS, T> operator*(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinExpr<Mul<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinExpr<Div<T>, LHS, ConstViewer<T>, T> operator/(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinExpr<Div<T>, LHS, ConstViewer<T>, T>(lhs.self(), ConstViewer<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinExpr<Div<T>, LHS, RHS, T> operator/(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinExpr<Div<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinExpr<Pow<T>, LHS, ConstViewer<T>, T> operator^(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinExpr<Pow<T>, LHS, ConstViewer<T>, T>(lhs.self(), ConstViewer<T>(rhs));
};

template <class LHS, class RHS, typename T>
BinExpr<Max<T>, LHS, RHS, T> maximum(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BinExpr<Max<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
BinExpr<Max<T>, LHS, ConstViewer<T>, T> maximum(const DenseExpr<T, LHS> &lhs, T rhs) {
    return BinExpr<Max<T>, LHS, ConstViewer<T>, T>(lhs.self(), ConstViewer<T>(rhs));
};

template <class OP, class LHS, class RHS, typename T>
struct BoolBinExpr : public DenseExpr<T, BoolBinExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    BoolBinExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    __device__ __host__
    inline bool at(int r, int c) const {
        return OP::apply(lhs.at(r, c), rhs.at(r, c));
    }

    __device__ __host__
    inline bool at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }
};

template <class LHS, class RHS, typename T>
BoolBinExpr<Eq<T>, LHS, RHS, T> operator==(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BoolBinExpr<Eq<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename T>
BoolBinExpr<NEq<T>, LHS, RHS, T> operator!=(const DenseExpr<T, LHS> &lhs, const DenseExpr<T, RHS> &rhs) {
    return BoolBinExpr<NEq<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

#endif //NLP_CUDA_EXPRESSION_TEMPLATES_H
