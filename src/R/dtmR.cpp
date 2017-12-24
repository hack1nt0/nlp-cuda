//
// Created by DY on 17-10-21.
//

#include "Rutils.h"

RcppExport SEXP new_dtm(SEXP trainWordListListSEXP, SEXP testWordListListSEXP, SEXP parSEXP) {
    vector<Document> trainWordListList = toWordListList(trainWordListListSEXP);
    int parallel = as<int>(parSEXP);
    dtm_type dtm(trainWordListList.size(), trainWordListList.data(), parallel);
    if (Rf_isNull(testWordListListSEXP)) {
        return toDtmR(dtm);
    }
    S4 tr("dtm");
    tr.slot("Dim") = IntegerVector::create(dtm.nrow(), dtm.ncol());
    CharacterVector dict(dtm.ncol());
    for (int i = 0; i < dtm.ncol(); ++i) dict[i] = (*dtm.dict)[i].first;
    tr.slot("Dimnames") = List::create(R_NilValue, dict);
    NumericVector idf(dtm.ncol());
    for (int i = 0; i < dtm.ncol(); ++i) idf[i] = std::log((1. + dtm.nrow()) / (*dtm.dict)[i].second);
    tr.slot("idf") = idf;
    tr.slot("p") = IntegerVector(dtm.getCscPtr(), dtm.getCscPtr() + dtm.ncol() + 1);
    tr.slot("i") = IntegerVector(dtm.getCscInd(), dtm.getCscInd() + dtm.getNnz());
    tr.slot("x") = NumericVector(dtm.getCscVal(), dtm.getCscVal() + dtm.getNnz());
    tr.slot("d.len") = IntegerVector(dtm.nterm.begin(), dtm.nterm.end());
    std::vector<Document> testWordListList = toWordListList(testWordListListSEXP);
    dtm_type test(testWordListList.size(), testWordListList.data(), parallel, &dtm);
    S4 te("dtm");
    te.slot("Dim") = IntegerVector::create(test.nrow(), test.ncol());
    te.slot("Dimnames") = List::create(R_NilValue, dict);
    te.slot("idf") = idf;
    te.slot("p") = IntegerVector(test.getCscPtr(), test.getCscPtr() + test.ncol() + 1);
    te.slot("i") = IntegerVector(test.getCscInd(), test.getCscInd() + test.getNnz());
    te.slot("x") = NumericVector(test.getCscVal(), test.getCscVal() + test.getNnz());
    te.slot("d.len") = IntegerVector(test.nterm.begin(), test.nterm.end());

    return List::create(Named("train") = tr, Named("test") = te);
}

RcppExport SEXP test_dtm(SEXP sxp) {
    String s = CharacterVector(sxp)[0];
    Rcout << s.get_cstring() << endl;
    return R_NilValue;
}

RcppExport SEXP row_normalize_dtm(SEXP e, SEXP LSEXP) {
    sm_type sm = toSparseMatrix(e);
    int L = as<int>(LSEXP);
    rnormalize(sm, L);
    return R_NilValue;
}
