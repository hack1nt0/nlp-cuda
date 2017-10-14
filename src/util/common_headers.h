//
// Created by DY on 17-9-27.
//

#ifndef NLP_CUDA_COMMON_HEADERS_H
#define NLP_CUDA_COMMON_HEADERS_H

#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <queue>

using namespace std;


template <typename T1, typename T2>
ostream& operator<< (ostream& out, const pair<T1, T2>& p) {
    cout << '(' << p.first << ", " << p.second << ')';
    return out;
};

//template <class Iterator>
//ostream& operator<< (ostream& out, const Iterator& v) {
//    out << "[";
//    int size = v.size();
//    for (auto i = v.begin(); i != v.end(); ++i) {
//        out << *i << (i < size - 1 ? ", " : "");
//    }
//    out << "]";
//    return out;
//}

template <typename T>
void print(const T* array, int size) {
    cout << '[';
    for (int i = 0; i < size; ++i)
        cout << array[i] << (i < size - 1 ? ", " : "");
    cout << ']';
}

const // It is a const object...
class nullptr_t
{
public:
    template<class T>
    inline operator T*() const // convertible to any type of null non-member pointer...
    { return 0; }

    template<class C, class T>
    inline operator T C::*() const   // or any type of null member pointer...
    { return 0; }

private:
    void operator&() const;  // Can't take address of nullptr

} null = {};

struct CpuTimer {
    clock_t time;
    float duration;

    void start() {
        time = clock();
    }

    void stop() {
        duration = (float)(clock() - time)/ CLOCKS_PER_SEC * 1000;
    }

    float elapsed() {
        stop();
        return duration;
    }
};
#endif
