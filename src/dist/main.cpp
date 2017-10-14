//
// Created by DY on 17-9-29.
//

#include <common_headers.h>
#include <knn.h>
#include "SymmetricDistance.h"

int main() {
    bool verbose = true;
    ifstream in("/Users/dy/TextUtils/data/train/spamsms.dtm");
    DocumentTermMatrix<double> dtm(in);
    in.close();
    dtm.normalize();

    SymmetricDistance<double> D(dtm.rows());
    D.from(*dtm.csr, true);
    for (int i = 0; i < min(10, D.rows); ++i) {
        for (int j = 0; j < min(10, D.rows); ++j) {
            printf("%e\t", D.at(i, j));
        }
        printf("\n");
    }

    int k = 10;
    Knn<double> knn(D, verbose);
    SparseMatrix<double> nn = knn.knn(k);

    cout << nn << endl;
    return 0;
}
