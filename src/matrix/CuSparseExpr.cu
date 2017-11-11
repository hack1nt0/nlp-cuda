//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_CU_SPARSE_EXPR_H
#define NLP_CUDA_CU_SPARSE_EXPR_H

#include <cu_common_headers.cu>
#include "cufunctors.cu"

template<typename T, typename ETYPE>
struct CuSparExpr {
    inline const ETYPE& self(void) const {
        return *static_cast<const ETYPE *>(this);
    }
};

template <class OP, class LHS, typename T>
struct CuSparUnaryExpr : CuSparExpr<T, CuSparUnaryExpr<OP, LHS, T> > {
    LHS lhs;

    CuSparUnaryExpr(const LHS &lhs) : lhs(lhs) {}

    __device__ __host__
    inline T at(int i) const {
        return OP::apply(lhs.at(i));
    }
};

template <class LHS, typename T>
CuSparUnaryExpr<Sqrt<T>, LHS, T> sqrt(const CuSparExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return CuSparUnaryExpr<Sqrt<T>, LHS, T>(e);
}

template <class LHS, typename T>
CuSparUnaryExpr<Neg<T>, LHS, T> operator-(const CuSparExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return CuSparUnaryExpr<Neg<T>, LHS, T>(e);
}

template <class LHS, typename T>
CuSparUnaryExpr<Exp<T>, LHS, T> exp(const CuSparExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return CuSparUnaryExpr<Exp<T>, LHS, T>(e);
}

template <class LHS, typename T>
CuSparUnaryExpr<Log<T>, LHS, T> log(const CuSparExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return CuSparUnaryExpr<Log<T>, LHS, T>(e);
}

template <class OP, class LHS, class RHS, typename T>
struct CuSparBinExpr : public CuSparExpr<T, CuSparBinExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    CuSparBinExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    __device__ __host__
    inline T at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }
};

template <typename T>
struct CuSparConstExpr : public CuSparExpr<T, CuSparConstExpr<T> > {
    T v;
    CuSparConstExpr(T v) : v(v) {}

    inline T at(int r, int c) const {
        return v;
    }

    inline T at(int i) const {
        return v;
    }
};

template <class LHS, typename T>
CuSparBinExpr<Mul<T>, LHS, CuSparConstExpr<T>, T> operator*(const CuSparExpr<T, LHS> &lhs, T rhs) {
    return CuSparBinExpr<Mul<T>, LHS, CuSparConstExpr<T>, T>(lhs.self(), CuSparConstExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
CuSparBinExpr<Mul<T>, LHS, RHS, T> operator*(const CuSparExpr<T, LHS> &lhs, const CuSparExpr<T, RHS> &rhs) {
    return CuSparBinExpr<Mul<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
CuSparBinExpr<Div<T>, LHS, CuSparConstExpr<T>, T> operator/(const CuSparExpr<T, LHS> &lhs, T rhs) {
    return CuSparBinExpr<Div<T>, LHS, CuSparConstExpr<T>, T>(lhs.self(), CuSparConstExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
CuSparBinExpr<Div<T>, LHS, RHS, T> operator/(const CuSparExpr<T, LHS> &lhs, const CuSparExpr<T, RHS> &rhs) {
    return CuSparBinExpr<Div<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
CuSparBinExpr<Pow<T>, LHS, CuSparConstExpr<T>, T> operator^(const CuSparExpr<T, LHS> &lhs, T rhs) {
    return CuSparBinExpr<Pow<T>, LHS, CuSparConstExpr<T>, T>(lhs.self(), CuSparConstExpr<T>(rhs));
};

template<typename T, typename EType>
void fillDevice(T* target, int size, const CuSparExpr<T, EType>& e) {
    int threads = 16 * 16;
    int blocks = (size + threads - 1) / threads;
    fillKernel<<<blocks, threads>>>(target, size, e.self());
};

#endif //NLP_CUDA_SPARSE_EXPRESSION_TEMPLATES_H
