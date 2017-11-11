//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_CU_DENSE_EXPR_H
#define NLP_CUDA_CU_DENSE_EXPR_H

#include <cu_common_headers.cu>
#include "cufunctors.cu"

template<typename T, typename ETYPE>
struct CuDenseExpr {
    inline const ETYPE& self(void) const {
        return *static_cast<const ETYPE *>(this);
    }
};

template <class OP, class LHS, typename T>
struct CuDenseUnaryExpr : public CuDenseExpr<T, CuDenseUnaryExpr<OP, LHS, T> > {
    LHS lhs;

    CuDenseUnaryExpr(const LHS &lhs) : lhs(lhs) {}

    __device__
    inline T at(int r, int c) const {
        return OP::apply(lhs.at(r, c));
    }

    __device__
    inline T at(int i) const {
        return OP::apply(lhs.at(i));
    }
};

template <class LHS, typename T>
CuDenseUnaryExpr<Sqrt<T>, LHS, T> sqrt(const CuDenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return CuDenseUnaryExpr<Sqrt<T>, LHS, T>(e);
}

template <class LHS, typename T>
CuDenseUnaryExpr<Log<T>, LHS, T> log(const CuDenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return CuDenseUnaryExpr<Log<T>, LHS, T>(e);
}

template <class LHS, typename T>
CuDenseUnaryExpr<Log2<T>, LHS, T> log2(const CuDenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return CuDenseUnaryExpr<Log2<T>, LHS, T>(e);
}

template <class OP, class LHS, typename T>
struct CuDenseIntUnaryExpr : public CuDenseExpr<T, CuDenseIntUnaryExpr<OP, LHS, T> > {
    LHS lhs;

    CuDenseIntUnaryExpr(const LHS &lhs) : lhs(lhs) {}

    __device__
    inline int at(int r, int c) const {
        return OP::apply(lhs.at(r, c));
    }

    __device__
    inline int at(int i) const {
        return OP::apply(lhs.at(i));
    }
};

template <class LHS, typename T>
CuDenseIntUnaryExpr<Sign<T>, LHS, T> sign(const CuDenseExpr<T, LHS> &lhs) {
    const LHS &e = lhs.self();
    return CuDenseIntUnaryExpr<Sign<T>, LHS, T>(e);
}

template <typename T, class LHS>
struct CuDenseTransExpr : public CuDenseExpr<T, CuDenseTransExpr<T, LHS> > {
    LHS lhs;

    CuDenseTransExpr(const LHS &lhs) : lhs(lhs) {}

    __device__
    inline T at(int r, int c) const {
        return lhs.at(c, r);
    }
};

template <class OP, class LHS, typename T>
//struct CuDenseZipExpr : public BaseCuDenseExpr<T, CuDenseZipExpr<OP, LHS, T> > {
struct CuDenseZipExpr : public CuDenseExpr<T, CuDenseZipExpr<OP, LHS, T> > {
    LHS lhs;
    int index;
    int rows;
    int cols;
    T zero;

    CuDenseZipExpr(const LHS &lhs, int index, int rows, int cols, T zero) : lhs(lhs), index(index), rows(rows), cols(cols),
                                                              zero(zero) {
        assert(index == 0 || index == 1);
    }

    __device__
    inline T at(int r, int c) const {
        return lhs.at(c, r);
    }

    __device__
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
CuDenseZipExpr<Add<T>, LHS, T> sum(const CuDenseExpr<T, LHS> &lhs, int index) {
    const LHS &e = lhs.self();
    return CuDenseZipExpr<Add<T>, LHS, T>(e, index, e.rows, e.cols, (T)0);
};

template <typename T, class LHS> inline
CuDenseTransExpr<T, LHS> operator~(const CuDenseExpr<T, LHS> &lhs) {
    return CuDenseTransExpr<T, LHS>(lhs.self());
};

template <class OP, class LHS, class RHS, typename T>
struct CuDenseBinExpr : public CuDenseExpr<T, CuDenseBinExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    CuDenseBinExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    __device__
    inline T at(int r, int c) const {
        return OP::apply(lhs.at(r, c), rhs.at(r, c));
    }

    __device__
    inline T at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }
};

template <typename T, class LHS, class RHS> inline
CuDenseBinExpr<Add<T>, LHS, RHS, T> operator+(const CuDenseExpr<T, LHS> &lhs, const CuDenseExpr<T, RHS> &rhs) {
    return CuDenseBinExpr<Add<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <typename T, class LHS, class RHS> inline
CuDenseBinExpr<Sub<T>, LHS, RHS, T> operator-(const CuDenseExpr<T, LHS> &lhs, const CuDenseExpr<T, RHS> &rhs) {
    return CuDenseBinExpr<Sub<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <typename T>
struct CuDenseConstExpr : public CuDenseExpr<T, CuDenseConstExpr<T> > {
    T v;
    CuDenseConstExpr(T v) : v(v) {}

    __device__
    inline T at(int r, int c) const {
        return v;
    }

    __device__
    inline T at(int i) const {
        return v;
    }
};

template <class LHS, typename T>
CuDenseBinExpr<Add<T>, LHS, CuDenseConstExpr<T>, T> operator+(const CuDenseExpr<T, LHS> &lhs, T rhs) {
    return CuDenseBinExpr<Add<T>, LHS, CuDenseConstExpr<T>, T>(lhs.self(), CuDenseConstExpr<T>(rhs));
};

template <class LHS, typename T>
CuDenseBinExpr<Sub<T>, LHS, CuDenseConstExpr<T>, T> operator-(const CuDenseExpr<T, LHS> &lhs, T rhs) {
    return CuDenseBinExpr<Sub<T>, LHS, CuDenseConstExpr<T>, T>(lhs.self(), CuDenseConstExpr<T>(rhs));
};

template <class LHS, typename T>
CuDenseUnaryExpr<Neg<T>, LHS, T> operator-(const CuDenseExpr<T, LHS> &lhs) {
    return CuDenseUnaryExpr<Neg<T>, LHS, T>(lhs.self());
};

template <class LHS, typename T>
CuDenseBinExpr<Mul<T>, LHS, CuDenseConstExpr<T>, T> operator*(const CuDenseExpr<T, LHS> &lhs, T rhs) {
    return CuDenseBinExpr<Mul<T>, LHS, CuDenseConstExpr<T>, T>(lhs.self(), CuDenseConstExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
CuDenseBinExpr<Mul<T>, LHS, RHS, T> operator*(const CuDenseExpr<T, LHS> &lhs, const CuDenseExpr<T, RHS> &rhs) {
    return CuDenseBinExpr<Mul<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
CuDenseBinExpr<Div<T>, LHS, CuDenseConstExpr<T>, T> operator/(const CuDenseExpr<T, LHS> &lhs, T rhs) {
    return CuDenseBinExpr<Div<T>, LHS, CuDenseConstExpr<T>, T>(lhs.self(), CuDenseConstExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
CuDenseBinExpr<Div<T>, LHS, RHS, T> operator/(const CuDenseExpr<T, LHS> &lhs, const CuDenseExpr<T, RHS> &rhs) {
    return CuDenseBinExpr<Div<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
CuDenseBinExpr<Pow<T>, LHS, CuDenseConstExpr<T>, T> operator^(const CuDenseExpr<T, LHS> &lhs, T rhs) {
    return CuDenseBinExpr<Pow<T>, LHS, CuDenseConstExpr<T>, T>(lhs.self(), CuDenseConstExpr<T>(rhs));
};

template <class LHS, class RHS, typename T>
CuDenseBinExpr<Max<T>, LHS, RHS, T> maximum(const CuDenseExpr<T, LHS> &lhs, const CuDenseExpr<T, RHS> &rhs) {
    return CuDenseBinExpr<Max<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, typename T>
CuDenseBinExpr<Max<T>, LHS, CuDenseConstExpr<T>, T> maximum(const CuDenseExpr<T, LHS> &lhs, T rhs) {
    return CuDenseBinExpr<Max<T>, LHS, CuDenseConstExpr<T>, T>(lhs.self(), CuDenseConstExpr<T>(rhs));
};

template <class OP, class LHS, class RHS, typename T>
struct CuDenseBoolBinExpr : public CuDenseExpr<T, CuDenseBoolBinExpr<OP, LHS, RHS, T> > {
    LHS lhs;
    RHS rhs;

    CuDenseBoolBinExpr(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}

    __device__
    inline bool at(int r, int c) const {
        return OP::apply(lhs.at(r, c), rhs.at(r, c));
    }

    __device__
    inline bool at(int i) const {
        return OP::apply(lhs.at(i), rhs.at(i));
    }
};

template <class LHS, class RHS, typename T>
CuDenseBoolBinExpr<Eq<T>, LHS, RHS, T> operator==(const CuDenseExpr<T, LHS> &lhs, const CuDenseExpr<T, RHS> &rhs) {
    return CuDenseBoolBinExpr<Eq<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template <class LHS, class RHS, typename T>
CuDenseBoolBinExpr<NEq<T>, LHS, RHS, T> operator!=(const CuDenseExpr<T, LHS> &lhs, const CuDenseExpr<T, RHS> &rhs) {
    return CuDenseBoolBinExpr<NEq<T>, LHS, RHS, T>(lhs.self(), rhs.self());
};

template<typename T, typename EType>
void fillDevice(T* target, int size, const CuDenseExpr<T, EType>& e) {
    int threads = 16 * 16;
    int blocks = (size + threads - 1) / threads;
    fillKernel<<<blocks, threads>>>(target, size, e.self());
};
#endif
