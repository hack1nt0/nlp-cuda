//
// Created by DY on 17-9-15.
//

#include "../matrix/DocumentTermMatrix.h"
#include "gmm.h"

int main(int argc, char* args[]) {
    typedef DocumentTermMatrix<double, int> dtm_t;
    typedef RDenseMatrix<double, int> dm_t;
    typedef SparseMatrix<double, int> sm_t;
    typedef MixtureModel<sm_t>        mxm_t;

    dtm_t dtm;
    dtm.read("../data/spamsms.dtm");
    rnormalize(dtm, 2);
    dtm.updateCsc();

    cout << dtm.ncol() << endl;

    int k = atoi(args[1]);
    int maxItr = atoi(args[2]);
    double lbPrior = 1. / k / 10;
    double lbVar = 1e-5;
    bool verbose = true;
    mxm_t::GmmModel gmmModel(dtm.nrow(), dtm.ncol(), k);
    std::vector<double> logProb;

    mxm_t::gmm(gmmModel, dtm, k, maxItr, 0.1, lbPrior, lbVar, logProb, verbose);


//    BlasWrapper blas;
//
//    sm_t A = sm_t::rnorm(2, 3, 0.6, 1);
//    A.print();
//    dm_t B = dm_t::rnorm(3, 4);
//    B.print();
//    dm_t C(2, 4);
//    blas.mm(C, 1., A, B, 0.);
//    C.print();
//    cout << "---------------" << endl;
//    for (int i = 0; i < 2; ++i) {
//        for (int j = 0; j < 4; ++j) {
//            C.at(i, j) = B.col(j).dot(A.row(i));
//        }
//    }
//    C.print();
//    cout << "---------------" << endl;
//    dm_t A = dm_t::rnorm(2, 2);
//    A.print();
//    A = A * A;
//    A.print();
    return 0;
}