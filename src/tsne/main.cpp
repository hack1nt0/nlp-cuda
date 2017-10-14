//
// Created by DY on 17-9-22.
//

#include "tsne.h"
#include <common_headers.h>

int main() {
    ifstream in("/Users/dy/TextUtils/data/train/spamsms.dtm");
    DocumentTermMatrix<double> dtm(in);
    in.close();
    
    int perplexity = 10;
    bool verbose = true;
    Tsne<double> tsne(*dtm.csr, perplexity, verbose);
    int dim = 2;
    double theta = 0.3;
    int maxItr = 10;
    int seed = 1;
    DenseMatrix<double> Y = tsne.tsne(dim, perplexity, theta, maxItr, seed);
    cout << Y << endl;
    return 0;
}
