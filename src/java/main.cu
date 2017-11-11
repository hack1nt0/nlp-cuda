//
// Created by DY on 17-10-25.
//
#include "jni_cuda_CuMatrix.h"
#include "CuDenseMatrixJ.cu"

using namespace JNI;

int main() {
    int rows = 1;
    int cols = 1;
    CuDenseMatrixJ<double>* denseMatrixJ = new CuDenseMatrixJ<double>(rows, cols);
    *denseMatrixJ = 1.;
    denseMatrixJ->print();
    *denseMatrixJ = sqrt(denseMatrixJ);
    denseMatrixJ->print();
    return 0;
}
