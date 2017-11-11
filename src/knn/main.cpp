//
// Created by DY on 17-9-29.
//

#include <common_headers.h>
#include <ds/DocumentTermMatrix.h>
#include "knn.h"
#include "VpTree.h"

int main() {
//    DistMatrix<double> distMatrix; distMatrix.read("dist.double.in");
//    int k = 90;
//    SparseMatrix<double> nn = knn(distMatrix, k);
//    nn.println();
//    nn.save("knn.90.double.in");
//
//    SparseMatrix<double> sparseMatrix; sparseMatrix.read("knn.90.double.in");
//
//    double s = sum(sparseMatrix != nn);
//    if (s != 0) {
//        cout << s << endl;
//        exit(1);
//    }

    DocumentTermMatrix<double> dtm;
    dtm.read("/Users/dy/TextUtils/data/train/spamsms.dtm");
    dtm.normalize();
    VpTree<> vpTree(dtm);
    SparseMatrix<double> nn = knnCollect(dtm, 100, 1, true);
    nn.println();
    return 0;
}
