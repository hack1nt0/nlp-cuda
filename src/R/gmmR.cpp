//
// Created by DY on 17-8-28.
//

#include "Rutils.h"
#include "../gmm/gmm.h"

RcppExport SEXP gmm_dtm(SEXP xSEXP, SEXP kSEXP, SEXP maxItrSEXP, SEXP tolSEXP, SEXP lbPriorSEXP, SEXP lbVarSEXP,
                    SEXP priorSEXP, SEXP meanSEXP, SEXP varSEXP, SEXP verboseSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        sm_t x = RU::toSparseMatrix(xSEXP);
        typedef MixtureModel<sm_t> mxm_t;
        int rows = x.nrow();
        int cols = x.ncol();
        int k = as<int>(kSEXP);
        int maxItr = as<int>(maxItrSEXP);
        double tol = as<double>(tolSEXP);
        double lbPrior = as<double>(lbPriorSEXP);
        double lbVar = as<double>(lbVarSEXP);
        bool verbose = as<bool>(verboseSEXP);

        NumericVector prior(clone(priorSEXP));
        NumericMatrix mean(clone(meanSEXP));
        NumericMatrix var(clone(varSEXP));
        NumericMatrix resp(k, rows);
        std::vector<double> logProb;

        mxm_t::GmmModel gmmModel(rows, cols, k, prior.begin(), mean.begin(), var.begin(), resp.begin());
        mxm_t::gmm(gmmModel, x, k, maxItr, tol, lbPrior, lbVar, logProb, verbose);

        List r = List::create(
            Named("prior") = prior,
            Named("mean") = mean,
            Named("var") = var,
            Named("resp") = resp,
            Named("avg.log.L") = logProb
        );
        r.attr("class") = "xmm";
        return r;
    END_RCPP
}

