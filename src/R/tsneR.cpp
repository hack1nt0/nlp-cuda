//
// Created by DY on 17-9-17.
//

#include <Rcpp.h>
#include <tsne/tsne.h>
#include <matrix/DocumentTermMatrix.h>
#include <matrix/DenseMatrix.h>

using namespace Rcpp;

RcppExport SEXP tsne(SEXP dtmSEXP, SEXP dimSEXP,
                     SEXP maxIterSEXP, SEXP perplexitySEXP,
                     SEXP thetaSEXP,
                     SEXP seedSEXP, SEXP verboseSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        XPtr<DocumentTermMatrix<double> > xp(dtmSEXP);
        const DocumentTermMatrix<double>& dtm = *xp;
        int dim = as<int>(dimSEXP);
        int perplexity = as<int>(perplexitySEXP);
        int maxItr = as<int>(maxIterSEXP);
        double theta = as<double>(thetaSEXP);
        int seed = as<int>(seedSEXP);
        int verbose = as<bool>(verboseSEXP);
        Rcpp::NumericMatrix rY(dim, dtm.nrow());
        DenseMatrix<double> Y(rY.begin(), dtm.nrow(), dim);
        tsne(Y, dtm, dim, maxItr, perplexity, theta, seed, verbose);
        cout << "hi" << endl;
        rY = Rcpp::transpose(rY);
        cout << "hi" << endl;
        rY.attr("class") = "Tsne";
        cout << "hi" << endl;
        return rY;
    END_RCPP
}
