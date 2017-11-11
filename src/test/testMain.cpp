
#include <iostream>
#include <cfloat>
#include <vector>
#include <algorithm>
#include <unordered_map>
#include <string>

using namespace std;

struct B {
    int v;
    void print();
};

struct A {
    int v;
    A() {}
    A(const B& o){ v = o.v; }
    void print() { cout << v << endl; }
};

int main() {
    A a;
    a.print();
    return 0;
}
