//
// Created by DY on 17-9-17.
//

#include <Rcpp.h>
#include <SymmetricDistance.h>
using namespace Rcpp;
//using namespace std;

Rcpp::NumericVector dist(S4 dtm, bool verbose) {
    IntegerVector dims = as<IntegerVector>(dtm.slot("Dim"));
    int rows = dims[0];
    int cols = dims[1];
    IntegerVector row_ptr = as<IntegerVector>(dtm.slot("p"));
    IntegerVector index = as<IntegerVector>(dtm.slot("j"));
    NumericVector data = as<NumericVector>(dtm.slot("x"));
    int nnz = data.size();
    SparseMatrix M(data.begin(), index.begin(), row_ptr.begin(), rows, cols, nnz);
    /*
     * size of D = rows - 1 + ... + 1 = rows * (rows - 1) / 2
     */
    NumericVector D(rows * (rows - 1) / 2);
    SymmetricDistance symmetricDistance(D);
    symmetricDistance.from(M, verbose);

    D.attr("class") = "dist";
    D.attr("Size") = rows;
    D.attr("Diag") = false;
    D.attr("Upper") = false;
    D.attr("method") = "euclidean";
    D.attr("call") = NULL;
    return D;
}

RcppExport SEXP dist(SEXP dtmSEXP, SEXP verboseSEXP) {
BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter<S4>::type dtm(dtmSEXP);
        Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
        rcpp_result_gen = Rcpp::wrap(dist(dtm, verbose));
        return rcpp_result_gen;
END_RCPP
}
