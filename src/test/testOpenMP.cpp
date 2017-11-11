//
// Created by DY on 17-10-15.
//

#include <iostream>
using namespace std;

int main() {
    int r = 0;

    int n = (int)1e9;

#pragma omp parallel for
    for (int i = 0; i < n; ++i) r++;

    cout << r << endl;
    return 0;
}
