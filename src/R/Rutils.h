//
// Created by DY on 17/12/22.
//

#ifndef NLP_CUDA_RUTILS_H
#define NLP_CUDA_RUTILS_H

#include <Rcpp.h>
#include "../matrix/matrix.h"
#include <matrix/DocumentTermMatrix.h>

using namespace Rcpp;

typedef Rcpp::List                                Document;
typedef SparseMatrix<double, int>                 sm_t;
typedef CDenseMatrix<double, int>                 dm_t;
typedef SparseVector<double, int>                 sv_t;
typedef DenseVector<double, int>                  dv_t;
typedef DocumentTermMatrix<double, int, Document> dtm_t;

struct RUtils {

    static sm_t toSparseMatrix(const SEXP& e) {
        S4 dgCMatrix(e);
        IntegerVector i(dgCMatrix.slot("i"));
        IntegerVector p(dgCMatrix.slot("p"));
        NumericVector x(dgCMatrix.slot("x"));
        IntegerVector d(dgCMatrix.slot("Dim"));
        return sm_t(d[0], d[1], i.size(), p.begin(), i.begin(), x.begin(), false, false);
    }

    static sv_t toSparseVector(const SEXP& e) {
        S4 dsparseVector(e);
        IntegerVector i(dsparseVector.slot("i"));
        NumericVector x(dsparseVector.slot("x"));
        int length = as<int>(dsparseVector.slot("length"));
        return sv_t(length, i.size(), i.begin(), x.begin());
    }

    static dm_t toDenseMatrix(const SEXP& e) {
        NumericMatrix matrixR(e);
        return dm_t(matrixR.nrow(), matrixR.ncol(), matrixR.begin(), false);
    }

    static dv_t toDenseVector(const SEXP& e) {
        NumericVector vectorR(e);
        return dv_t(vectorR.size(), vectorR.begin());
    }

    static vector<Document> toWordListList(const SEXP& e) {
        Rcpp::List textList(e);
        return std::vector<Document>(textList.begin(), textList.end());
    }

    static S4 toDtmR(dtm_t& dtm) {
        S4 r("dtm");
        r.slot("Dim") = IntegerVector::create(dtm.nrow(), dtm.ncol());
        CharacterVector dict(dtm.ncol());
        for (int i = 0; i < dtm.ncol(); ++i) dict[i] = (*dtm.dict)[i].first;
        r.slot("Dimnames") = List::create(R_NilValue, dict);
        NumericVector idf(dtm.ncol());
        for (int i = 0; i < dtm.ncol(); ++i) idf[i] = std::log((1. + dtm.nrow()) / (*dtm.dict)[i].second);
        r.slot("idf") = idf;
        r.slot("p") = IntegerVector(dtm.getCscPtr(), dtm.getCscPtr() + dtm.ncol() + 1);
        r.slot("i") = IntegerVector(dtm.getCscInd(), dtm.getCscInd() + dtm.getNnz());
        r.slot("x") = NumericVector(dtm.getCscVal(), dtm.getCscVal() + dtm.getNnz());
        r.slot("d.len") = IntegerVector(dtm.nterm.begin(), dtm.nterm.end());
        return r;
    }

    static S4 toDtmR(sm_t& sm) {
        S4 r("dtm");
        r.slot("Dim") = IntegerVector::create(sm.nrow(), sm.ncol());
        r.slot("p") = IntegerVector(sm.getCscPtr(), sm.getCscPtr() + sm.ncol() + 1);
        r.slot("i") = IntegerVector(sm.getCscInd(), sm.getCscInd() + sm.getNnz());
        r.slot("x") = NumericVector(sm.getCscVal(), sm.getCscVal() + sm.getNnz());
        return r;
    }
};

typedef RUtils RU;

#endif //NLP_CUDA_RUTILS_H
