//
// Created by DY on 17/12/22.
//

#ifndef NLP_CUDA_RUTILS_H
#define NLP_CUDA_RUTILS_H

#include <Rcpp.h>
#include <matrix/DocumentTermMatrix.h>

using namespace Rcpp;

typedef Rcpp::List                                Document;
typedef SparseMatrix<double, int>                 sm_type;
typedef DenseMatrix<double, int>                  dm_type;
typedef DocumentTermMatrix<double, int, Document> dtm_type;

sm_type toSparseMatrix(const SEXP& e);

dm_type toDenseMatrix(const SEXP& e);

vector<Document> toWordListList(const SEXP& e);

S4 toDtmR(dtm_type& dtm);

#endif //NLP_CUDA_RUTILS_H
