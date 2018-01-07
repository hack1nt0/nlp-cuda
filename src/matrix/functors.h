//
// Created by DY on 17-9-13.
//

#ifndef NLP_CUDA_FUNCTORS_H
#define NLP_CUDA_FUNCTORS_H

#include <cmath>
#include <algorithm>

/** All Ops MUST apply to primary type **/

/** Unary Functors **/

template<typename T>
class Neg {
public:
    inline
    static T apply(T a) { return -a; }
};

template<typename T>
class Sqrt {
public:
    inline
    static T apply(T a) { return sqrt(a); }
};

template<typename T>
class Log {
public:
    inline
    static T apply(T a) { return log(a); }
};

template<typename T>
class Log2 {
public:
    inline
    static T apply(T a) { return log2(a); }
};

template<typename T>
class Log10 {
public:
    inline
    static T apply(T a) { return log10(a); }
};

template<typename T>
class Exp {
public:
    inline
    static T apply(T a) { return exp(a); }
};

template<typename T>
class Sign {
public:
    inline
    static int apply(T a) { return a == 0 ? 0 : (a < 0 ? -1 : +1); }
};

/** Binary Functors **/

template<typename LV, typename RV>
class Add {
  public:
    inline
    static LV apply(LV a, RV b) { return a + b; }
};

template<typename LV, typename RV>
class Sub {
  public:
    inline
    static LV apply(LV a, RV b) { return a - b; }
};
template<typename LV, typename RV>
class Mul {
  public:
    inline
    static LV apply(LV a, RV b) { return a * b; }
};
template<typename LV, typename RV>
class Div {
  public:
    inline
    static LV apply(LV a, RV b) { return a / b; }
};

template<typename LV, typename RV>
struct Pow {
    inline static LV apply(LV a, RV b) { return pow(a, b); }
};

template<typename T>
class Eq {
public:
    inline
    static bool apply(T a, T b) { return a == b; }
};

template<typename T>
class NEq {
public:
    inline
    static bool apply(T a, T b) {return a != b; }
};

template<typename T>
class LessThan {
  public:
    inline
    static bool apply(T a, T b) {return a < b; }
};

template<typename T>
struct Max {
    inline static T apply(T a, T b) { return std::max(a, b); }
};

template<typename T>
struct Min {
    inline static T apply(T a, T b) { return std::min(a, b); }
};

#endif //NLP_CUDA_FUNCTORS_H
