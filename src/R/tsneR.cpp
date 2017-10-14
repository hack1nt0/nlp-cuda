//
// Created by DY on 17-9-17.
//

#include <Rcpp.h>
#include <tsne.h>
using namespace Rcpp;
//using namespace std;

Rcpp::List tsne(S4 dtm,
                         IntegerVector belong, int classes,
                         int newRows, int newCols,
                         int perplexity, int max_itr,
                         unsigned int seed) {
    IntegerVector dims = as<IntegerVector>(dtm.slot("Dim"));
    int rows = dims[0];
    int cols = dims[1];
    IntegerVector row_ptr = as<IntegerVector>(dtm.slot("p"));
    IntegerVector index = as<IntegerVector>(dtm.slot("j"));
    NumericVector data = as<NumericVector>(dtm.slot("x"));
    int nnz = data.size();
    /*
     * size of D = rows - 1 + ... + 1 = rows * (rows - 1) / 2
     */
    NumericMatrix Y(newCols, newRows);
    IntegerVector landmarks(newRows);
    tsne(Y.begin(), landmarks.begin(), newRows, newCols,
         data.begin(), index.begin(), row_ptr.begin(),
         rows, cols, nnz,
         belong.begin(), classes,
         perplexity, max_itr, seed);
    return Rcpp::List::create(
            Rcpp::Named("y") = Y,
            Rcpp::Named("l") = landmarks);
}

RcppExport SEXP tsne(SEXP dtmSEXP,
                     SEXP belongSEXP, SEXP classesSEXP,
                     SEXP newRowsSEXP, SEXP newColsSEXP,
                     SEXP perplexitySEXP, SEXP max_iterSEXP,
                     SEXP seedSEXP) {
BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter<S4>::type dtm(dtmSEXP);
        Rcpp::traits::input_parameter<IntegerVector>::type belong(belongSEXP);
        Rcpp::traits::input_parameter<int>::type classes(classesSEXP);
        Rcpp::traits::input_parameter<int>::type newRows(newRowsSEXP);
        Rcpp::traits::input_parameter<int>::type newCols(newColsSEXP);
        Rcpp::traits::input_parameter<int>::type perplexity(perplexitySEXP);
        Rcpp::traits::input_parameter<int>::type max_itr(max_iterSEXP);
        Rcpp::traits::input_parameter<unsigned int>::type seed(seedSEXP);
        rcpp_result_gen = Rcpp::wrap(tsne(dtm, belong, classes, newRows, newCols, perplexity, max_itr, seed));
        return rcpp_result_gen;
END_RCPP
}
