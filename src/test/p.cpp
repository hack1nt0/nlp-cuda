#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include <sstream>
#include <iomanip>
#include <bitset>
#include <memory>
#include "b.h"
#include "c.h"

using namespace std;

typedef std::string string_t;

template <typename T>
string toHexString(const T& t) {
    string r;
    char* p = (char*)&t;
    for (int i = 0; i < sizeof(t); ++i) {
        char c   = *(p + i);
        int hex = (c & 255 - 15) >> 4;
        hex = hex < 10 ? hex + '0' : 'A' + hex - 10;
        r.push_back((char)hex);
        hex = c & 15;
        hex = hex < 10 ? hex + '0' : 'A' + hex - 10;
        r.push_back((char)hex);
    }
    return r;
}
/*
struct A {
    A() {}
    A(const A& o) { cout << "A(const A& o)" << endl; }
    A(A&& o) { cout << "A(A&& o)" << endl; }
    A& operator=(const A& o) { cout << "=(const A& o)" << endl; return *this; }
    A& operator=(A&& o) { cout << "=(A&& o)" << endl; return *this; }

    void operator()(int i) { cout << "A()" << endl; }
};*/
 
int main() {
    int n = 10;
    int* a = new int[n];
    memset(a, +0, sizeof(int) * n);
    for (int i = 0; i < n; ++i) cout << a[i] << endl;
    return 0;
}
