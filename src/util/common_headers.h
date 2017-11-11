//
// Created by DY on 17-9-27.
//

#ifndef NLP_CUDA_COMMON_HEADERS_H
#define NLP_CUDA_COMMON_HEADERS_H

#include <cmath>
#include <cfloat>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <queue>
#include <map>

//#ifdef __APPLE__
//#include <tr1/unordered_set>
//#include <tr1/unordered_map>
//using namespace std::tr1;
//#endif
//
//#ifdef __linux__
//#include <unordered_map>
//#include <unordered_set>
//#endif
//
//#ifdef _WIN64 //todo??
//#include <unordered_map>
//#include <unordered_set>
//#endif

//#include <omp.h>

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

template <typename T>
void summary(vector<T> data) {
    sort(data.begin(), data.end());
    int size = data.size();
    if (size <= 5) {
        cout << "Too less to summary, data:\n";
        for (int i = 0; i < size; ++i) cout << data[i] << '\t';
        cout << "----------------------" << endl;
        return;
    }
    T stat[6];
    stat[0] = data[0];
    stat[5] = data[size - 1];
    stat[2] = (size & 1) == 0 ?
              (data[size / 2 - 1] + data[size / 2]) / 2. :
              data[size / 2];
    stat[1] = data[size / 4];
    stat[4] = data[size - size / 4];
    stat[3] = 0;
    for (int i = 0; i < size; ++i) stat[3] += data[i];
    stat[3] /= size;
    cout << "Min.      : " << stat[0] << endl;
    cout << "1st.Quat. : " << stat[1] << endl;
    cout << "median    : " << stat[2] << endl;
    cout << "mean      : " << stat[3] << endl;
    cout << "3rd.Quat. : " << stat[4] << endl;
    cout << "Max.      : " << stat[5] << endl;
    cout << "----------------------" << endl;
}

template <typename T>
void table(const vector<T>& data) {
    map<T, int> count;
    for (T value : data) count[value]++;
    typedef pair<T, int> Pair;
    vector<Pair> order(count.begin(), count.end());
    sort(order.begin(), order.end(), [](const Pair& a, const Pair& b) { return a.second > b.second; });
    for (int i = 0; i < order.size(); ++i) {
        cout << order[i].first << "\t: " << order[i].second << endl;
    }
}
#endif
