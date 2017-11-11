//
// Created by DY on 17-9-13.
//

#ifndef NLP_CUDA_FUNCTORS_H
#define NLP_CUDA_FUNCTORS_H

#include <cmath>

//using namespace std;

template<typename T>
class Add {
public:
    inline
    static T apply(T a, T b) { return a + b; }
};

template<typename T>
class Mul {
public:
    inline
    static T apply(T a, T b) { return a * b; }
};

template<typename T>
class Div {
public:
    inline
    static T apply(T a, T b) { return a / b; }
};

template<typename T>
class Sub {
public:
    inline
    static T apply(T a, T b) { return a - b; }
};

template<typename T>
class Pow {
public:
    inline
    static T apply(T a, T b) { return pow(a, b); }
};

template<typename T>
class Max {
public:
    inline
    static T apply(T a, T b) { return max(a, b); }
};

template<typename T>
class Identity {
public:
    inline
    static T apply(T a) { return a; }
};

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

#endif //NLP_CUDA_FUNCTORS_H
