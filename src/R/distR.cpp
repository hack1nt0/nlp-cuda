//
// Created by DY on 17-9-17.
//

#include <dist/dist.h>
#include "Rutils.h"

RcppExport SEXP dist_dtm(SEXP xSEXP, SEXP ySEXP, SEXP metricSEXP, SEXP verboseSEXP) {
    BEGIN_RCPP
        sm_type sm = toSparseMatrix(xSEXP);
        unsigned long long rows = sm.nrow();
        CharacterVector metric(metricSEXP);
        int kind = metric[0] == "euclidean" ? 1 : 0;
        bool verbose = as<bool>(verboseSEXP);
        /*
         * capacity of D = rows - 1 + ... + 1 = rows * (rows - 1) / 2
         */
        NumericVector D(rows * (rows - 1) / 2);
        DistMatrix<double, unsigned long long> distMatrix(D.begin(), rows);
        dist(distMatrix, sm, kind, verbose);
        D.attr("class") = "dist";
        D.attr("Size") = rows;
        D.attr("Diag") = false;
        D.attr("Upper") = false;
        D.attr("method") = "euclidean";
        D.attr("call") = NULL;
        return D;
    END_RCPP
}

//RcppExport SEXP dist_dtm(SEXP xpSEXP, SEXP sSEXP, SEXP tSEXP, SEXP methodSEXP, SEXP verboseSEXP) {
//    BEGIN_RCPP
//        Rcpp::RObject rcpp_result_gen;
//        Rcpp::RNGScope rcpp_rngScope_gen;
//        Rcpp::XPtr<dtm_type> ptr(xpSEXP);
//        IntegerVector s(sSEXP);
//        IntegerVector t(tSEXP);
//        bool verbose = as<bool>(verboseSEXP);
//        NumericMatrix r(s.size(), t.size());
//        CharacterVector colnames(t.size());
//        CharacterVector rownames(s.size());
//        if (s!=R_NilValue) {
//            for (int i = 0; i < s.size(); ++i)
//                for (int j = 0; j < t.size(); ++j)
//                    r.at(i, j) = ptr->row(s[i]).dist2(ptr->row(t[j]));
//            for (int i = 0; i < s.size(); ++i) rownames[i] = to_string(s[i]);
//            for (int i = 0; i < t.size(); ++i) colnames[i] = to_string(t[i]);
//            r.attr("dimnames") = List::create(rownames, colnames);
//            return r;
//        } else {
//            return R_NilValue;
//        }
//    END_RCPP
//}

//RcppExport SEXP index_dist(SEXP xp, SEXP rowSEXP, SEXP colSEXP) {
//    BEGIN_RCPP
//        Rcpp::RNGScope rcpp_rngScope_gen;
//        Rcpp::XPtr<DistMatrix<double> > distPtr(xp);
//        int row = as<int>(rowSEXP) - 1;
//        int col = as<int>(colSEXP) - 1;
//        if(!(0 <= row && row < distPtr->rows && 0 <= col && col < distPtr->rows)) return wrap(NumericVector::get_na());
//        return wrap(distPtr->at(row, col));
//    END_RCPP
//}
//
//RcppExport SEXP summary_dist(SEXP xp, SEXP exactSEXP, SEXP miSEXP, SEXP maSEXP) {
//    BEGIN_RCPP
//        Rcpp::RNGScope rcpp_rngScope_gen;
//        Rcpp::XPtr<DistMatrix<double> > distPtr(xp);
//        bool exact = as<bool>(exactSEXP);
//        double mi = as<double>(miSEXP);
//        double ma = as<double>(maSEXP);
//        vector<pair<double, double> > buckets = distPtr->summary(exact, mi, ma);
//        if (exact) {
//            return DataFrame::create(
//                    Named("Min.", buckets[0].second),
//                    Named("1st Qu.", buckets[1].second),
//                    Named("Median", buckets[2].second),
//                    Named("3rd Qu.", buckets[3].second),
//                    Named("Max.", buckets[4].second)
//            );
//        } else {
//            return DataFrame::create(
//                    Named("min", NumericVector::create(buckets[0].first, buckets[0].second)),
//                    Named("geom.1.Qt", NumericVector::create(buckets[1].first, buckets[1].second)),
//                    Named("median", NumericVector::create(buckets[2].first, buckets[2].second)),
//                    Named("geom.3.Qt", NumericVector::create(buckets[3].first, buckets[3].second)),
//                    Named("max", NumericVector::create(buckets[4].first, buckets[4].second))
//            );
//        }
//    END_RCPP
//}
