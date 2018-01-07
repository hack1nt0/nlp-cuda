//
// Created by DY on 17-9-22.
//

#include "tsne.h"
#include <utils.h>
#include <matrix/DocumentTermMatrix.h>

int main() {
//    ifstream in;
//    SparseMatrix<double> csr;
//    csr.read(&in);
//    in.close();
//    csr.println();
//    csr = csr + ~csr;
//    csr.println();
//    cout << min(csr) << ' ' << max(csr) << endl;

    DocumentTermMatrix<double> dtm;
    dtm.read("/Users/dy/TextUtils/data/train/spamsms.dtm");
    dtm.normalize();

    int perplexity = 30;
    bool verbose = true;
    int dim = 2;
    double theta = 0.3;
    int maxItr = 1000;
    int seed = 1;
    CDenseMatrix<double> Y = tsne(dtm, dim, maxItr, perplexity, theta, seed, verbose);
    Y.println();

    return 0;
}
