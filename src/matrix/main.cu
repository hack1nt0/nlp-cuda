
#include "CuDenseMatrix.cu"
#include "CuSparseMatrix.cu"

int main() {
    int rows = 4;
    int cols = 4;
    int nnz = 4;
    CuDenseMatrix<double> cuDenseMatrix(rows, cols);
    cuDenseMatrix.print();

    CuSparseMatrix<double> cuSparseMatrix(rows, cols, nnz);
    cuSparseMatrix.print();
    return 0;
}