//
// Created by DY on 17/12/22.
//

#include "Rutils.h"

sm_type toSparseMatrix(const SEXP& e) {
    S4 dgCMatrix(e);
    IntegerVector i(dgCMatrix.slot("i"));
    IntegerVector p(dgCMatrix.slot("p"));
    NumericVector x(dgCMatrix.slot("x"));
    IntegerVector d(dgCMatrix.slot("Dim"));
    return sm_type(d[0], d[1], i.size(), p.begin(), i.begin(), x.begin(), false, false);
}

dm_type toDenseMatrix(const SEXP& e) {
    NumericMatrix matrixR(e);
    return dm_type(matrixR.nrow(), matrixR.ncol(), matrixR.begin(), false);
}

vector<Document> toWordListList(const SEXP& e) {
    Rcpp::List textList(e);
    return std::vector<Document>(textList.begin(), textList.end());
}

S4 toDtmR(dtm_type& dtm) {
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
