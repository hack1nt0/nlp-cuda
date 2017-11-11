//
// Created by DY on 17-9-17.
//

#include <Rcpp.h>
#include <matrix/dist/DistMatrix.h>
#include <ds/DocumentTermMatrix.h>

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
    SparseMatrix<double> M( rows, cols, nnz, row_ptr.begin(), index.begin(), data.begin());
    /*
     * size of D = rows - 1 + ... + 1 = rows * (rows - 1) / 2
     */
    NumericVector D(rows * (rows - 1) / 2);
    DistMatrix<double> distance(D.begin(), M, verbose);
    D.attr("class") = "dist";
    D.attr("Size") = rows;
    D.attr("Diag") = false;
    D.attr("Upper") = false;
    D.attr("method") = "euclidean";
    D.attr("call") = NULL;
    return D;
}

//RcppExport SEXP dist(SEXP dtmSEXP, SEXP verboseSEXP) {
//BEGIN_RCPP
//        Rcpp::RObject rcpp_result_gen;
//        Rcpp::RNGScope rcpp_rngScope_gen;
//        Rcpp::traits::input_parameter<S4>::type dtm(dtmSEXP);
//        Rcpp::traits::input_parameter<bool>::type verbose(verboseSEXP);
//        rcpp_result_gen = Rcpp::wrap(dist(dtm, verbose));
//        return rcpp_result_gen;
//END_RCPP
//}

RcppExport SEXP Dist_read(SEXP pathSEXP) {
    string path = as<string>(pathSEXP);
    XPtr<DistMatrix<double> > ptr( new DistMatrix<double>(), true);
    ptr->read(path);
    return ptr;
}

RcppExport SEXP Dtm_dist(SEXP dtmSEXP, SEXP verboseSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::XPtr<DocumentTermMatrix<double> > dtmPtr(dtmSEXP);
        bool verbose = as<bool>(verboseSEXP);
//        NumericVector D(dtmPtr->rows * (dtmPtr->rows - 1) / 2);
//        DistMatrix<double> distance(D.begin(), *dtmPtr, verbose);
//        D.attr("class") = "dist";
//        D.attr("Size") = dtmPtr->rows;
//        D.attr("Diag") = false;
//        D.attr("Upper") = false;
//        D.attr("method") = "euclidean";
//        D.attr("call") = NULL;
        XPtr<DistMatrix<double> > ptr( new DistMatrix<double>(*dtmPtr, verbose), true);
        return ptr;
    END_RCPP
}

RcppExport SEXP Dist_at(SEXP xp, SEXP rowSEXP, SEXP colSEXP) {
    BEGIN_RCPP
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::XPtr<DistMatrix<double> > distPtr(xp);
        int row = as<int>(rowSEXP) - 1;
        int col = as<int>(colSEXP) - 1;
        if(!(0 <= row && row < distPtr->rows && 0 <= col && col < distPtr->rows)) return wrap(NumericVector::get_na());
        return wrap(distPtr->at(row, col));
    END_RCPP
}

RcppExport SEXP Dist_summary(SEXP xp, SEXP exactSEXP, SEXP miSEXP, SEXP maSEXP) {
    BEGIN_RCPP
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::XPtr<DistMatrix<double> > distPtr(xp);
        bool exact = as<bool>(exactSEXP);
        double mi = as<double>(miSEXP);
        double ma = as<double>(maSEXP);
        vector<pair<double, double> > buckets = distPtr->summary(exact, mi, ma);
        if (exact) {
            return DataFrame::create(
                    Named("Min.", buckets[0].second),
                    Named("1st Qu.", buckets[1].second),
                    Named("Median", buckets[2].second),
                    Named("3rd Qu.", buckets[3].second),
                    Named("Max.", buckets[4].second)
            );
        } else {
            return DataFrame::create(
                    Named("min", NumericVector::create(buckets[0].first, buckets[0].second)),
                    Named("geom.1.Qt", NumericVector::create(buckets[1].first, buckets[1].second)),
                    Named("median", NumericVector::create(buckets[2].first, buckets[2].second)),
                    Named("geom.3.Qt", NumericVector::create(buckets[3].first, buckets[3].second)),
                    Named("max", NumericVector::create(buckets[4].first, buckets[4].second))
            );
        }
    END_RCPP
}
