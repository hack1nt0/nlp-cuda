//
// Created by DY on 17-8-1.
//

#include <iostream>
#include <device_matrix.h>
#include <vector>
using namespace std;
//using namespace cutils;

template <class F>
void g(F f) {f();};

void f() {cout << "hello" << endl;};

int main(){
//    DeviceDenseMatrix matrix(10, 10);
//    DeviceDenseMatrix matrixSum(1, 10);
//    cout << matrix << endl;
//    GpuTimer timer;
//    timer.start();
//    matrixSum = sum(matrix, 0);
//    timer.stop();
//    cout << matrixSum << endl;
//    timer.start();
//    matrixSum = sum(matrix, 1);
//    timer.stop();
//    cout << matrixSum << endl;
//    DeviceSparseMatrix matrix(10, 10, 10);
//    cout << matrix << endl;
//    GpuTimer timer;
//    timer.start();
//    matrix = matrix ^ 2.f;
//    timer.stop();
//    cout << matrix << endl;

//    SparseMatrix<float> matrix(10, 10, 0.6);
//    cout << matrix << endl;
//    matrix = ~matrix;
//    cout << matrix << endl;
//    SparseMatrix<float> matrix2(5, 5, 0.6);
//    cout << matrix2 << endl;
//    matrix = ~matrix2;
//    cout << matrix << endl;

    srand(324234);
    int rows = 2;
    int cols = 2;
    vector<int> row_ptr;
    vector<int> index;
    vector<float> data;
    row_ptr.push_back(0);
    row_ptr.push_back(1);
    row_ptr.push_back(2);
    index.push_back(1);
    index.push_back(1);
    data.push_back(1);
    data.push_back(1);
    DeviceSparseMatrix A(data, index, row_ptr, rows, cols);
    cout << "A" << endl;
    cout << A << endl;
    vector<float> v;
    v.push_back(1.013468e-01f);
    v.push_back(8.462893e-01f);
    v.push_back(5.236398e-02f);
    v.push_back(1.013468e-01f);
    v.push_back(8.462893e-01f);
    v.push_back(5.236398e-02f);
    DeviceDenseMatrix B(v, 2, 3);
    cout << "B" << endl;
    cout << B << endl;
    DeviceDenseMatrix B_col_major(3, 2);
    B_col_major = ~B;
    B_col_major.rows = 2;
    B_col_major.cols = 3;
    cout << "B col major" << endl;
    cout << B_col_major << endl;

    DeviceDenseMatrix C(2, 3);
    DeviceDenseMatrix::cudaSparseMultiplyDense(C, 0.f, 1.f, A, false, B_col_major, false);
    transposeDevice(C);

    cout << "A" << endl;
    cout << A << endl;
    cout << "B" << endl;
    cout << B << endl;
    cout << "C" << endl;
    cout << C << endl;
    return 0;
}

