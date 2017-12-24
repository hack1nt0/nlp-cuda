//
// Created by DY on 17-11-10.
//

//#include "DocumentTermMatrix.h"
#include "../dist/dist.h"

int main() {
//    typedef vector<string> doc_type;
//    typedef DocumentTermMatrix<double, int, doc_type> dtm_type;
//    int n = 4;
//    vector<doc_type> documents(n);
//    documents[0] = {"A", "B"};
//    documents[1] = {"B", "C"};
//    documents[2] = {"E", "F"};
//    documents[3] = {"A", "B"};
//    dtm_type dtm(n, documents.data(), false);
//    dtm.print();

    typedef DenseMatrix<> dm_t;
    typedef SparseMatrix<double, int> sm_t;

    sm_t sm = sm_t::rnorm(2, 2, 0.5, 1);
    sm.print();

    dm_t dm = dm_t::rnorm(2, 2);
    dm.print();
    dm.row(0) = sm.col(0);
    dm.print();
    dm.row(1) = 100.;
    dm.print();
    dm.col(1) += dm_t::C(100.);
    dm.print();
    return 0;
}
