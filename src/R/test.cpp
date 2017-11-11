#include <iostream>
#include <cmath>
#include <algorithm>
using namespace std;

struct A {
    int size;
    int* a;
    A(int size) {
        cout << "A manual construct..." << endl;
        this->size = size;
        a = new int[size];
        memset(a, 0, sizeof(int) * size);
    }

    A(const A& o) {
        cout << "A copy construct..." << endl;
        delete[] a;
        this->size = o.size;
        a = new int[size];
        memcpy(a, o.a, sizeof(int) * size);
    }

    A(A&& o) {
        cout << "A move construct..." << endl;
        size = o.size;
        a = o.a;
        o.a = NULL;
    }

    virtual ~A() {
        cout << "A destructing..." << endl;
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
        cout << "A move revalued..." << endl;
        size = o.size;
        a = o.a;
        o.a = NULL;
        return *this;
    }

    virtual void af() { cout << "af()" << endl; }
};

A f() {
    A a(10);
    return a;
}

struct B : A {
    int a[2] = {1,2};

    B(int size) : A(size) {
        cout << "B manual costructing..." << endl;
    }

    void af() { cout << "B f() " << a[0] << endl; }
    
    ~B() {
        cout << "B destructing..." << endl;
    }
};

int main() {
    A* b = new B(10);
    b->af();
    delete b;
    return 0;
}
