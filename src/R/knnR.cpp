//
// Created by DY on 17-9-17.
//

#include <Rcpp.h>
#include <ds/DocumentTermMatrix.h>
#include <knn/knn.h>

using namespace Rcpp;

RcppExport
SEXP dtm_knn(SEXP xpSEXP, SEXP kSEXP, SEXP verboseSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        XPtr<DocumentTermMatrix<double> > xp(xpSEXP);
        int k = as<int>(kSEXP);
        bool verbose = as<bool>(verboseSEXP);
        int rows = xp->nrow();
        int cols = rows;
        int nnz  = rows * k;
        IntegerVector row_ptr(rows + 1);
        IntegerVector index(nnz);
        NumericVector data(nnz);
        SparseMatrix<double> nn(rows, cols, nnz, row_ptr.begin(), index.begin(), data.begin());
        knnCollect(nn, *xp, k, verbose);
        S4 sparMatrix("dgRMatrix");
        sparMatrix.slot("j") = index;
        sparMatrix.slot("p") = row_ptr;
        sparMatrix.slot("x") = data;
        sparMatrix.slot("Dim") = IntegerVector::create(rows, cols);
        return wrap(sparMatrix);
    END_RCPP
}

