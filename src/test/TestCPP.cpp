//
// Created by DY on 17-7-7.
//

#include <iostream>
#include <vector>

using namespace std;

const int N = 40;

template <class SUMMABLE>
void sum( int n, const SUMMABLE array[], SUMMABLE& res = 0) {
    for (int i = 0; i < n; ++i) res += array[i];
}

typedef enum Color{RED, BLUE, GREEN} Color;

class X
{
public:
    X() { std::cout << "X::X()" << std::endl; }
    X( X const & ) { std::cout << "X::X( X const & )" << std::endl; }
    X& operator=( X const & ) { std::cout << "X::operator=(X const &)" << std::endl; }
};

inline X f() {
    X tmp;
    return tmp;
}

template <class F>
void g(F f) {f();};

int main() {
    X x;
    x = f();
    vector<int> v(1);

    cout << static_cast<long>(1.2f) << endl;
    cout << long(1.2f) << endl;

    g([](){cout << "hello" << endl;});
    return 0;
}

