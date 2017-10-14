//
// Created by DY on 17-10-3.
//


#include <matrix/DenseMatrix.h>
#include <quadtree.h>
#include <common_headers.h>
#include <CpuTimer.h>

int main() {
    int n = (int)100000;
    DenseMatrix<double> matrix(n, 2, 1);
    cout << matrix << endl;
    CpuTimer timer;
    timer.start();
    QuadTree<DenseMatrix<double> > quadTree(matrix);
    timer.stop();
    printf("cost %e ms\n", timer.elapsed());
    printf("nodes %d, leaves %d\n", quadTree.nodes, quadTree.leaves);
    return 0;
}
