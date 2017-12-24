//
// Created by DY on 17-8-28.
//

#include "Rutils.h"
#include <kmeans/kmeans.h>


//struct Model {
//
//
//    Model(index_t rows, index_t cols, index_t k) : center(k, cols), ss(rows, 1), cluster(rows, 1), size(k, 1), itr(0) {}
//};


RcppExport SEXP kmeans_matrix(SEXP trainSEXP, SEXP kSEXP, SEXP max_itrSEXP, SEXP nstartSEXP, SEXP methodSEXP, SEXP seedSEXP, SEXP tolSEXP, SEXP verboseSEXP)  {
    BEGIN_RCPP
//        Rcpp::RObject rcpp_result_gen;
//        Rcpp::RNGScope rcpp_rngScope_gen;
//        typedef KMeans<dm_type> km_t;
//        dm_type train = toDenseMatrix(trainSEXP);
//        int k = as<int>(kSEXP);
//        int max_itr = as<int>(max_itrSEXP);
//        int nstart = as<int>(nstartSEXP);
////        string method = as<string>(methodSEXP);
//        double tol = as<double>(tolSEXP);
//        int seed = as<int>(seedSEXP);
//        bool verbose = as<bool>(verboseSEXP);
//        Rcpp::as()
//
//        NumericMatrix meanR(k, train.ncol());
//        km_t::Mean mean(meanR.nrow(), meanR.ncol(), meanR.begin(), false);
//        NumericVector costR(max_itr);
//        km_t::Cost cost(max_itr, 1, costR.begin(), false);
//        IntegerVector belongR(train.nrow());
//        km_t::Belong belong(train.nrow(), 1, belongR.begin(), false);
//        IntegerVector sizeR(k);
//        km_t::Size size(k, 1, sizeR.begin(), false);
//
//        km_t::go(mean, cost, belong, size, train, max_itr, seed, tol, verbose);
//
//        return List::create(
//            Named("mean") = meanR,
//            Named("cost") = costR,
//            Named("belong") = belongR,
//            Named("size") = sizeR
//        );
    END_RCPP
}

RcppExport SEXP kmeans_dtm(SEXP trainSEXP, SEXP kSEXP, SEXP max_itrSEXP, SEXP nstartSEXP, SEXP methodSEXP, SEXP seedSEXP, SEXP tolSEXP, SEXP verboseSEXP)  {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        typedef KMeans<sm_type> km_t;
        sm_type train = toSparseMatrix(trainSEXP);
        int k = as<int>(kSEXP);
        int max_itr = as<int>(max_itrSEXP);
        int nstart = as<int>(nstartSEXP);
        double tol = as<double>(tolSEXP);
        int seed = as<int>(seedSEXP);
        bool verbose = as<bool>(verboseSEXP);

        NumericMatrix center(k, train.ncol());
        IntegerVector cluster(train.nrow());
        NumericVector ss(train.nrow());
        IntegerVector size(k);

        km_t::Model model(train.nrow(), train.ncol(), k, center.begin(), cluster.begin(), ss.begin(), size.begin());

        km_t::go(model, train, k, max_itr, seed, tol, verbose);

        return List::create(
            Named("centers") = center,
            Named("ss") = ss,
            Named("cluster") = cluster,
            Named("size") = size,
            Named("loss") = model.loss,
            Named("changed") = model.changed,
            Named("itr") = model.itr
        );
    END_RCPP
}
