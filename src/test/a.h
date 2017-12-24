#ifndef A_H
#define A_H

#include <iostream>

using namespace std;

struct A {
	//int a;
	//static bool b;

	void f() { cout << "A::f()" << endl; }
	void g() { cout << "A::g()" << endl; }
};
//bool A::b = true;
#endif