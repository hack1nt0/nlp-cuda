//
// Created by DY on 17-10-21.
//

#include <Rcpp.h>
#include <ds/DocumentTermMatrix.h>

using namespace Rcpp;

RcppExport SEXP Dtm_new(SEXP pathSEXP) {
    return R_NilValue;
}

RcppExport SEXP Dtm_read(SEXP pathSEXP) {
    string path = as<string>(pathSEXP);
    XPtr<DocumentTermMatrix<double> > ptr( new DocumentTermMatrix<double>(), true);
    ptr->read(path);
    return ptr;
}

RcppExport SEXP Dtm_normalize(SEXP xp) {
    Rcpp::XPtr<DocumentTermMatrix<double> > dtmPtr(xp);
    dtmPtr->normalize();
    return dtmPtr;
}


RcppExport SEXP Dtm_summary(SEXP xp, SEXP kSEXP) {
    BEGIN_RCPP
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::XPtr<DocumentTermMatrix<double> > dtmPtr(xp);
        int k = min(as<int>(kSEXP), dtmPtr->cols);
        NumericVector dim(4);
        dim.attr("names") = CharacterVector::create("rows", "cols", "nnz", "density(%)");
        dim[0] = dtmPtr->rows;
        dim[1] = dtmPtr->cols;
        dim[2] = dtmPtr->nnz;
        dim[3] = dim[2] / (dim[0] * dim[1]) * 100;

        vector<int> whichCommon = dtmPtr->commonWords(k);
        CharacterVector commonWord(k);
        for (int i = 0; i < k; ++i) commonWord[i] = dtmPtr->dict[whichCommon[i]];
        NumericVector commonIdf(k);
        for (int i = 0; i < k; ++i) commonIdf[i] = dtmPtr->idf[whichCommon[i]];

        vector<int> whichRare = dtmPtr->rareWords(k);
        CharacterVector rareWord(k);
        for (int i = 0; i < k; ++i) rareWord[i] = dtmPtr->dict[whichRare[i]];
        NumericVector rareIdf(k);
        for (int i = 0; i < k; ++i) rareIdf[i] = dtmPtr->idf[whichRare[i]];

        DataFrame typicalWords = DataFrame::create(
                Rcpp::Named("commonWord") = commonWord,
                Rcpp::Named("commonIdf") = commonIdf,
                Rcpp::Named("rareWord") = rareWord,
                Rcpp::Named("rareIdf") = rareIdf
        );
        return Rcpp::List::create(
                Rcpp::Named("dim") = dim,
                Rcpp::Named("typicalWords") = typicalWords
        );
    END_RCPP
}

RcppExport SEXP Dtm_print(SEXP xp, SEXP idsSEXP) {
    BEGIN_RCPP
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::XPtr<DocumentTermMatrix<double> > dtmPtr(xp);
        IntegerVector ids = as<IntegerVector>(idsSEXP);
        List res(ids.size());
        for (int i = 0; i < ids.size(); ++i) {
            int id = ids[i];//todo int or long
            auto r = dtmPtr->row(id);
            NumericVector data(r.nnz);
            CharacterVector words(r.nnz);
            for (int i = 0; i < r.nnz; ++i) data[i] = r.data[i], words[i] = dtmPtr->dict[r.index[i]];
            data.attr("names") = words;
            res[i] = data;
        }
        return res;
    END_RCPP
}
