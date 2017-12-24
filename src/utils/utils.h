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
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <csignal>

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
#include <tbb/concurrent_hash_map.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/parallel_sort.h>

#include <mutex>
#include <omp.h>

using namespace std;


template <typename T1, typename T2>
ostream& operator<< (ostream& out, const pair<T1, T2>& p) {
    cout << '(' << p.first << ", " << p.second << ')';
    return out;
};

//template <class Iterator>
//ostream& operator<< (ostream& out, const Iterator& v) {
//    out << "[";
//    int capacity = v.capacity();
//    for (auto i = v.begin(); i != v.end(); ++i) {
//        out << *i << (i < capacity - 1 ? ", " : "");
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

template <typename T>
std::string toHexString(const T& t) {
    std::string r;
    char* p = (char*)&t;
    for (int i = 0; i < sizeof(t); ++i) {
        char c  = *(p + i);
        int hex = (c & 255 - 15) >> 4;
        hex = hex < 10 ? hex + '0' : 'A' + hex - 10;
        r.push_back((char)hex);
        hex = c & 15;
        hex = hex < 10 ? hex + '0' : 'A' + hex - 10;
        r.push_back((char)hex);
    }
    return r;
}

struct ProgressBar {
    static volatile bool aborted;
    int columns = 0;
    int width = 0;
    atomic<int> w;
    char* c = new char[100];
    bool print;

    virtual ~ProgressBar() {
        delete[] c;
        if (print) {
            if (!aborted) {
                cout << '\r' << setw(3) << 100 << "% |";
                for (int i = 0; i < columns; ++i) cout << "█";
                cout << '|' << endl;
            } else {
                cout << endl;
                aborted = false;
            }
        }
    }

    ProgressBar(int width, bool print = true, double cratio = 0.6) : width(width), w(0), print(print) {
        signal(SIGINT, interrupt);
        columns = int(getColumns() * cratio);
        columns = max(10, columns);
    }

    static void interrupt(int sig) { aborted = true; }

    inline void increase(bool flush = true, int delta = 1) {
        w += delta;
        int ww = w.load();
        if (isMasterThread()) {
            int percentage = int(ww*100.0/width);
            cout << '\r' << setw(3) << percentage << "% |";
            int finished = int(ww*1.0/width*columns);
            for (int i = 0; i < finished; ++i) cout << "█";
            for (int i = finished; i < columns; ++i) cout << ' ';
            cout << '|';
            if (flush) cout.flush();
        }
    }

    inline bool isMasterThread() {
#ifdef _OPENMP
        return omp_get_thread_num() == 0;
#else
        return true;
#endif
    }

    int getColumns() {
        FILE* o = popen("stty size", "r");
        fgets(c, 100-1, o);
        int i = 0; while (c[i] != ' ') ++i;
        int r = atoi(c + i + 1);
        return r;
    }

    inline bool interrupted() { return aborted; }
};

class interrupt_exception : public std::exception {
  public:
    interrupt_exception(std::string message)
        : detailed_message(message)
    {};
    /**
     * Virtual destructor. Needed to avoid "looser throw specification" errors.
     */
    virtual ~interrupt_exception() throw() {};

    virtual const char* what() const throw() {
        return detailed_message.c_str();
    }
    std::string detailed_message;
};

#endif
