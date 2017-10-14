//
// Created by DY on 17-10-3.
//


#include <matrix/DenseMatrix.h>
#include "knn.h"
#include <common_headers.h>

int main() {
    int n = (int)1000;
    DenseMatrix<double> matrix(n, 2, 1);
    cout << matrix << endl;
    CpuTimer timer;
    timer.start();
    KDTree<double, DenseMatrix<double> > kdTree(matrix);
    timer.stop();
    cout << timer.elapsed() << endl;
    const double point[2] = {1., 1.};
    vector<int> nn;
    timer.start();
//    for (int i = 0; i < n; ++i)
        kdTree.nearestNeighbors(nn, point, 0.5);
    timer.stop();
    cout << timer.elapsed() << endl;
    for (int i = 0; i < nn.size(); ++i) cout << nn[i] << endl;
    return 0;
}
