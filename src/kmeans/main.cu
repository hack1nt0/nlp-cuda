
#include <iostream>
#include <matrix/DocumentTermMatrix.h>
#include <fstream>
#include "kmeans.h"
#include "../knn/knn.h"

int main(int argc, char* argv[]) {
    int k = atoi(argv[1]);
    int max_itr = atoi(argv[2]);
    int seed = atoi(argv[3]);
    DocumentTermMatrix<double> dtm(std::cin);
//    int k = 10;
//    int max_itr = 10;
//    int seed = 10;
//    ifstream in("/Users/dy/TextUtils/data/train/spamsms.dtm");
//    DocumentTermMatrix dtm(in);
    dtm.normalize();
    cout << "DTM done." << endl;

//    int Dsize = dtm.csr->rows * (dtm.csr->rows - 1) / 2;
//    double* D = new double[Dsize];
//    knn(D, dtm.csr->row_ptr, dtm.csr->index, dtm.csr->data, dtm.csr->rows, dtm.csr->cols, dtm.csr->nnz);
//    for (int i = 0; i < 10; ++i) cout << D[i] << endl;
//    delete[] D;

    int rows = dtm.csr->rows;
    int cols = dtm.csr->cols;
    int nnz = dtm.csr->nnz;
    vector<double> h_mean(k * cols);
    vector<int> h_class(rows);
    initKmeans(h_mean.data(), k,
               dtm.csr->data, dtm.csr->index, dtm.csr->row_ptr,
               rows, cols, nnz, seed);
    vector<double> distances = kmeans(h_mean.data(), h_class.data(),
                                      dtm.csr->data, dtm.csr->index, dtm.csr->row_ptr,
                                      rows, cols, nnz,
                                      k, max_itr, seed);

    return 0;
}
