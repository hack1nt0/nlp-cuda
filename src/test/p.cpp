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

struct X {
	int x;
	
    X() { cout << "A()" << endl; }
    X(int x) : x(x) { cout << "A(x)" << endl; }
   
    X(const X& o) { cout << "A(const A& o)" << endl; }
    X(X&& o) { cout << "A(A&& o)" << endl; }
    X& operator=(const X& o) { cout << "=(const A& o)" << endl; return *this; }
    X& operator=(X&& o) { cout << "=(A&& o)" << endl; return *this; }

    void operator()(int i) { cout << "A()" << endl; }
	X clone() { X x(1); x.x = 10; return x; }
};

#include <dlfcn.h>
 
int main() {
	
//    dlopen("/Users/dy/nlp-cuda/src/R/R-package/inst/libs/libcorn.so", RTLD_NOW);
//    void* dll = dlopen("q.so", RTLD_NOW);
//	const char* e;
//    if ((e = dlerror()) != NULL) throw runtime_error(e);
//	else cout << "LOADED SUCCESSFULLY" << endl;
//	void* qaddr = dlsym(dll, "q");
//	if (qaddr == NULL) throw runtime_error("q() NOT FOUND");
//	((void(*)())qaddr)();
//	dlclose(dll);

    unsigned int a = 1;
    cout << typeid(a).name() << endl;
    float b = 1;
    cout << typeid(b).name() << endl;
    double c = 1;
    cout << typeid(c).name() << endl;
    long d = 1;
    cout << typeid(d).name() << endl;
    return 0;
}
