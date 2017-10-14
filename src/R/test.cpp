#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

struct A {
    int size;
    int* a;
    A(int size) {
        cout << "manual construct..." << endl;
        this->size = size;
        a = new int[size];
    }

    A(const A& o) {
        cout << "copy construct..." << endl;
        delete[] a;
        this->size = o.size;
        a = new int[size];
        memcpy(a, o.a, sizeof(int) * size);
    }

    A(A&& o) {
        cout << "move construct..." << endl;
        size = o.size;
        a = o.a;
        o.a = NULL;
    }

    ~A() {
        delete[] a;
    }

    A& operator=(const A& o) {
        cout << "copy revalued..." << endl;
        delete[] a;
        this->size = o.size;
        a = new int[size];
        memcpy(a, o.a, sizeof(int) * size);
        return *this;
    }

    A& operator=(A&& o) {
        cout << "move revalued..." << endl;
        size = o.size;
        a = o.a;
        o.a = NULL;
        return *this;
    }
};

A f() {
    A a(10);
    return a;
}

int main() {
//    A a1(10);
//    A a2(a1);
//    A a3(1);
//    a3 = a2;
//    a3 = f();
//    A a4(f());
    int n = 1000000000;
    double r;
    for (int i = 0; i < n; ++i) r = sqrt(sqrt(sqrt(19.)));
    cout << r << endl;
    return 0;
}
