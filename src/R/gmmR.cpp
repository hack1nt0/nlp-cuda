//
// Created by DY on 17-8-28.
//

#include <Rcpp.h>
#include "gmm.h"
using namespace Rcpp;
//using namespace std;

Rcpp::List gmm(S4 dtm, int k, int max_itr, int seed, double alpha, double beta) {
    IntegerVector dims = as<IntegerVector>(dtm.slot("Dim"));
    int rows = dims[0];
    int cols = dims[1];
    IntegerVector row_ptr = as<IntegerVector>(dtm.slot("p"));
    IntegerVector index = as<IntegerVector>(dtm.slot("j"));
    NumericVector data = as<NumericVector>(dtm.slot("x"));

    NumericMatrix resp(rows, k);
    NumericMatrix mean(k, cols);
    NumericMatrix conv(k, cols);
    NumericVector class_weight(k);
    gmmInit(mean.begin(), conv.begin(), class_weight.begin(), k, cols, seed, beta);
    std::vector<double> append_likehood = gmm(resp.begin(), mean.begin(), conv.begin(), class_weight.begin(),
                                              data.begin(), index.begin(), row_ptr.begin(),
                                              rows, cols, data.size(), k, max_itr, seed, alpha, beta);
    return Rcpp::List::create(
            Rcpp::Named("resp") = resp,
            Rcpp::Named("mean") = mean,
            Rcpp::Named("conv") = conv,
            Rcpp::Named("class_weight") = class_weight,
            Rcpp::Named("likelihood") = append_likehood);
}

RcppExport SEXP gmm(SEXP dtmSEXP, SEXP kSEXP, SEXP max_itrSEXP, SEXP seedSEXP, SEXP alphaSEXP, SEXP betaSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter<S4>::type dtm(dtmSEXP);
        Rcpp::traits::input_parameter<int>::type k(kSEXP);
        Rcpp::traits::input_parameter<int>::type max_itr(max_itrSEXP);
        Rcpp::traits::input_parameter<int>::type seed(seedSEXP);
        Rcpp::traits::input_parameter<double>::type alpha(alphaSEXP);
        Rcpp::traits::input_parameter<double>::type beta(betaSEXP);
        rcpp_result_gen = Rcpp::wrap(gmm(dtm, k, max_itr, seed, alpha, beta));
        return rcpp_result_gen;
    END_RCPP
}

