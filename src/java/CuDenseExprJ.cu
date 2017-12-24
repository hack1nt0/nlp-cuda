//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_DENSE_EXPR_J_H
#define NLP_CUDA_DENSE_EXPR_J_H

#include <matrix/functors.h>
#include "../utils/utils.h"
#include "../utils/cutils.cu"
#include <stdexcept>

namespace JNI {

    template<typename T>
    struct CuDenseExprJ {

//        __device__ __host__
//        virtual inline int nrow() const { throw std::runtime_error("nrow()"); }
//
//        __device__ __host__
//        virtual inline int ncol() const { throw std::runtime_error("ncol()"); }
//
//        __device__
//        virtual inline T at(int r, int c) const { throw std::runtime_error("at(r,c)"); }
//
//        __device__
//        virtual inline T at(int i) const { throw std::runtime_error("at(i)"); }

        __device__ __host__
        virtual inline int nrow() const { ; }

        __device__ __host__
        virtual inline int ncol() const { ; }

        __device__
        virtual inline T at(int r, int c) const { ; }

        __device__
        virtual inline T at(int i) const { ; }
    };

    template<typename T>
    T sum(const CuDenseExprJ<T> *e) {
        T s = 0;
        for (int i = 0; i < e->nrow(); ++i) for (int j = 0; j < e->ncol(); ++j) s += e->at(i, j);
        return s;
    }

    template<class OP, typename T>
    struct UnaryDenseExprJ : CuDenseExprJ<T> {
        const CuDenseExprJ<T> *lhs;

        UnaryDenseExprJ(const CuDenseExprJ<T> *lhs) : lhs(lhs) {}

        __device__
        inline T at(int r, int c) const {
            return OP::apply(lhs->at(r, c));
        }

        __device__
        inline T at(int i) const {
            return OP::apply(lhs->at(i));
        }

        __device__ __host__
        inline int nrow() const { return lhs->nrow(); }

        __device__ __host__
        inline int ncol() const { return lhs->ncol(); }

    };

    template<typename T>
    UnaryDenseExprJ<Sqrt<T>, T>* sqrt(const CuDenseExprJ<T> *lhs) {
        UnaryDenseExprJ<Sqrt<T>, T>* ep = new UnaryDenseExprJ<Sqrt<T>, T>(lhs);
        UnaryDenseExprJ<Sqrt<T>, T>* d_ep;
        cudaMalloc(&d_ep, sizeof(*ep));
        cudaMemcpy(d_ep, ep, sizeof(*ep), cudaMemcpyHostToDevice);
        delete(ep);
        return d_ep;
    }

    template<typename T>
    UnaryDenseExprJ<Log<T>, T>* log(const CuDenseExprJ<T> *lhs) {
        return new UnaryDenseExprJ<Log<T>, T>(lhs);
    }

//
//template <class LHS, typename T>
//UnaryDenseExprJ<Log2<T>, LHS, T> log2(const CuDenseExprJ<T, LHS> &lhs) {
//    const LHS &e = lhs.self();
//    return UnaryDenseExprJ<Log2<T>, LHS, T>(e);
//}
//
//template <class OP, class LHS, typename T>
//struct IntUnaryDenseExprJ : public CuDenseExprJ<T, IntUnaryDenseExprJ<OP, LHS, T> > {
//    LHS lhs;
//
//    IntUnaryDenseExprJ(const LHS &lhs) : lhs(lhs) {}
//
//    inline int at(int r, int c) const {
//        return OP::apply(lhs.at(r, c));
//    }
//
//    inline int at(int i) const {
//        return OP::apply(lhs.at(i));
//    }
//};
//
//template <class LHS, typename T>
//IntUnaryDenseExprJ<Sign<T>, LHS, T> sign(const CuDenseExprJ<T, LHS> &lhs) {
//    const LHS &e = lhs.self();
//    return IntUnaryDenseExprJ<Sign<T>, LHS, T>(e);
//}
//
//template <typename T, class LHS>
//struct TransDenseExprJ : public CuDenseExprJ<T, TransDenseExprJ<T, LHS> > {
//    LHS lhs;
//
//    TransDenseExprJ(const LHS &lhs) : lhs(lhs) {}
//
//    inline T at(int r, int c) const {
//        return lhs.at(c, r);
//    }
//};
//
//template <class OP, class LHS, typename T>
////struct ZipDenseExprJ : public BaseDenseExprJ<T, ZipDenseExprJ<OP, LHS, T> > {
//struct ZipDenseExprJ : public CuDenseExprJ<T, ZipDenseExprJ<OP, LHS, T> > {
//    LHS lhs;
//    int index;
//    int rows;
//    int cols;
//    T zero;
//
//    ZipDenseExprJ(const LHS &lhs, int index, int rows, int cols, T zero) : lhs(lhs), index(index), rows(rows), cols(cols),
//                                                              zero(zero) {
//        assert(index == 0 || index == 1);
//    }
//
//    inline T at(int r, int c) const {
//        return lhs.at(c, r);
//    }
//
//    inline T at(int i) const {
//        T result = zero;
//        switch (index) {
//            case 0:
//                for (int r = 0; r < rows; ++r) result = OP::apply(result, lhs.at(r, i));
//                break;
//            case 1:
//                for (int c = 0; c < cols; ++c) result = OP::apply(result, lhs.at(i, c));
//                break;
//        }
//        return result;
//    }
//};
//
//template <class LHS, typename T>
//ZipDenseExprJ<Add<T>, LHS, T> sum(const CuDenseExprJ<T, LHS> &lhs, int index) {
//    const LHS &e = lhs.self();
//    return ZipDenseExprJ<Add<T>, LHS, T>(e, index, e.rows, e.cols, (T)0);
//};
//
//template <typename T, class LHS> inline
//TransDenseExprJ<T, LHS> operator~(const CuDenseExprJ<T, LHS> &lhs) {
//    return TransDenseExprJ<T, LHS>(lhs.self());
//};
//
    template<class OP, typename T>
    struct BinDenseExprJ : CuDenseExprJ<T> {
        const CuDenseExprJ<T> *lhs;
        const CuDenseExprJ<T> *rhs;

        BinDenseExprJ(const CuDenseExprJ<T> *lhs, const CuDenseExprJ<T> *rhs) : lhs(lhs), rhs(rhs) {}

        __device__
        inline T at(int r, int c) const {
            return OP::apply(lhs->at(r, c), rhs->at(r, c));
        }

        __device__
        inline T at(int i) const {
            return OP::apply(lhs->at(i), rhs->at(i));
        }

        __device__ __host__
        virtual inline int nrow() const { return lhs->nrow(); }

        __device__ __host__
        virtual inline int ncol() const { return lhs->ncol(); }
    };

    template<typename T>
    inline
    BinDenseExprJ<Add<T>, T>* add(const CuDenseExprJ<T> *lhs, const CuDenseExprJ<T> *rhs) {
        return new BinDenseExprJ<Add<T>, T>(lhs, rhs);
    };

    template<typename T>
    inline
    BinDenseExprJ<Sub<T>, T>* minus(const CuDenseExprJ<T> *lhs, const CuDenseExprJ<T> *rhs) {
        return new BinDenseExprJ<Sub<T>, T>(lhs, rhs);
    };

    template<typename T>
    struct ConstDenseExprJ : CuDenseExprJ<T> {
        T v;

        ConstDenseExprJ(T v) : v(v) {}

        __device__
        inline T at(int r, int c) const {
            return v;
        }

        __device__
        inline T at(int i) const {
            return v;
        }

//        __device__ __host__
//        inline int nrow() const { throw std::runtime_error("ConstDenseExprJ nrow()"); }
//
//        __device__ __host__
//        inline int ncol() const { throw std::runtime_error("ConstDenseExprJ ncol()"); }

        __device__ __host__
        inline int nrow() const { ; }

        __device__ __host__
        inline int ncol() const { ; }
    };


    template<typename T>
    BinDenseExprJ<Add<T>, T>* add(const CuDenseExprJ<T> *lhs, T rhs) {
        return new BinDenseExprJ<Add<T>, T>(lhs, new ConstDenseExprJ<T>(rhs));
    };

//template <class LHS, typename T>
//BinDenseExprJ<Sub<T>, LHS, ConstDenseExprJ<T>, T> operator-(const CuDenseExprJ<T, LHS> &lhs, T rhs) {
//    return BinDenseExprJ<Sub<T>, LHS, ConstDenseExprJ<T>, T>(lhs.self(), ConstDenseExprJ<T>(rhs));
//};
//
//template <class LHS, typename T>
//UnaryDenseExprJ<Neg<T>, LHS, T> operator-(const CuDenseExprJ<T, LHS> &lhs) {
//    return UnaryDenseExprJ<Neg<T>, LHS, T>(lhs.self());
//};
//
//template <class LHS, typename T>
//BinDenseExprJ<Mul<T>, LHS, ConstDenseExprJ<T>, T> operator*(const CuDenseExprJ<T, LHS> &lhs, T rhs) {
//    return BinDenseExprJ<Mul<T>, LHS, ConstDenseExprJ<T>, T>(lhs.self(), ConstDenseExprJ<T>(rhs));
//};
//
//template <class LHS, class RHS, typename T>
//BinDenseExprJ<Mul<T>, LHS, RHS, T> operator*(const CuDenseExprJ<T, LHS> &lhs, const CuDenseExprJ<T, RHS> &rhs) {
//    return BinDenseExprJ<Mul<T>, LHS, RHS, T>(lhs.self(), rhs.self());
//};
//
//template <class LHS, typename T>
//BinDenseExprJ<Div<T>, LHS, ConstDenseExprJ<T>, T> operator/(const CuDenseExprJ<T, LHS> &lhs, T rhs) {
//    return BinDenseExprJ<Div<T>, LHS, ConstDenseExprJ<T>, T>(lhs.self(), ConstDenseExprJ<T>(rhs));
//};
//
//template <class LHS, class RHS, typename T>
//BinDenseExprJ<Div<T>, LHS, RHS, T> operator/(const CuDenseExprJ<T, LHS> &lhs, const CuDenseExprJ<T, RHS> &rhs) {
//    return BinDenseExprJ<Div<T>, LHS, RHS, T>(lhs.self(), rhs.self());
//};
//
//template <class LHS, typename T>
//BinDenseExprJ<Pow<T>, LHS, ConstDenseExprJ<T>, T> operator^(const CuDenseExprJ<T, LHS> &lhs, T rhs) {
//    return BinDenseExprJ<Pow<T>, LHS, ConstDenseExprJ<T>, T>(lhs.self(), ConstDenseExprJ<T>(rhs));
//};
//
//template <class LHS, class RHS, typename T>
//BinDenseExprJ<Max<T>, LHS, RHS, T> maximum(const CuDenseExprJ<T, LHS> &lhs, const CuDenseExprJ<T, RHS> &rhs) {
//    return BinDenseExprJ<Max<T>, LHS, RHS, T>(lhs.self(), rhs.self());
//};
//
//template <class LHS, typename T>
//BinDenseExprJ<Max<T>, LHS, ConstDenseExprJ<T>, T> maximum(const CuDenseExprJ<T, LHS> &lhs, T rhs) {
//    return BinDenseExprJ<Max<T>, LHS, ConstDenseExprJ<T>, T>(lhs.self(), ConstDenseExprJ<T>(rhs));
//};
//
//template <class OP, class LHS, class RHS, typename T>
//struct BoolBinDenseExprJ : public CuDenseExprJ<T, BoolBinDenseExprJ<OP, LHS, RHS, T> > {
//    LHS lhs;
//    RHS rhs;
//
//    BoolBinDenseExprJ(const LHS &lhs, const RHS &rhs) : lhs(lhs), rhs(rhs) {}
//
//    inline bool at(int r, int c) const {
//        return OP::apply(lhs.at(r, c), rhs.at(r, c));
//    }
//
//    inline bool at(int i) const {
//        return OP::apply(lhs.at(i), rhs.at(i));
//    }
//};
//
//template <class LHS, class RHS, typename T>
//BoolBinDenseExprJ<Eq<T>, LHS, RHS, T> operator==(const CuDenseExprJ<T, LHS> &lhs, const CuDenseExprJ<T, RHS> &rhs) {
//    return BoolBinDenseExprJ<Eq<T>, LHS, RHS, T>(lhs.self(), rhs.self());
//};
//
//template <class LHS, class RHS, typename T>
//BoolBinDenseExprJ<NEq<T>, LHS, RHS, T> operator!=(const CuDenseExprJ<T, LHS> &lhs, const CuDenseExprJ<T, RHS> &rhs) {
//    return BoolBinDenseExprJ<NEq<T>, LHS, RHS, T>(lhs.self(), rhs.self());
//};
//
}
#endif //NLP_CUDA_EXPRESSION_TEMPLATES_J_H
