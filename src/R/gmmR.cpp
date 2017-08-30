//
// Created by DY on 17-8-28.
//

#include <Rcpp.h>
#include <iostream>
#include <fstream>
#include "gmm.h"
using namespace Rcpp;
using namespace std;

NumericVector _timesTwo(NumericVector x) {

    double *hx = new double[x.size()];
    memcpy(hx, x.begin(), x.size() * sizeof(double));
    for (int i = 0; i < x.size(); ++i) std::cout << hx[i] << std::endl;
    return x * 2;
}

RcppExport SEXP timesTwo(SEXP xSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter<NumericVector>::type x(xSEXP);
        rcpp_result_gen = Rcpp::wrap(_timesTwo(x));
        return rcpp_result_gen;
    END_RCPP
}

void _test_copy() {
    NumericVector A = NumericVector::create(1, 2, 3);
    NumericVector B = A;

    Rcout << "Before: " << std::endl << "A: " << A << std::endl << "B: " << B << std::endl;

    A[1] = 5; // 2 -> 5

    Rcout << "After: " << std::endl << "A: " << A << std::endl << "B: " << B << std::endl;
}

RcppExport SEXP test_copy() {
    BEGIN_RCPP
        Rcpp::RNGScope rcpp_rngScope_gen;
        _test_copy();
    END_RCPP
}

Rcpp::List _gmmR(S4 dtm, int k, int max_itr, int seed) {
    IntegerVector dims = as<IntegerVector>(dtm.slot("Dim"));
    int rows = dims[0];
    int cols = dims[1];
    IntegerVector row_ptr = as<IntegerVector>(dtm.slot("p"));
    IntegerVector index = as<IntegerVector>(dtm.slot("j"));
    NumericVector data = as<NumericVector>(dtm.slot("x"));

    NumericMatrix resp(rows, k);
    NumericMatrix mean(k, cols);
    NumericMatrix conv(k, cols);
    NumericVector class_weight(k);
    NumericVector likelihood(max_itr);
    gmmInit(mean.begin(), conv.begin(), class_weight.begin(), k, cols);
    gmm(resp.begin(), mean.begin(), conv.begin(), class_weight.begin(), likelihood.begin(),
        data.begin(), index.begin(), row_ptr.begin(),
        rows, cols, data.size(), k, max_itr, seed);
    return Rcpp::List::create(
            Rcpp::Named("resp") = resp,
            Rcpp::Named("mean") = mean,
            Rcpp::Named("conv") = conv,
            Rcpp::Named("class_weight") = class_weight,
            Rcpp::Named("likelihood") = likelihood);
}

RcppExport SEXP gmmR(SEXP dtmSEXP, SEXP kSEXP, SEXP max_itrSEXP, SEXP seedSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter<S4>::type dtm(dtmSEXP);
        Rcpp::traits::input_parameter<int>::type k(kSEXP);
        Rcpp::traits::input_parameter<int>::type max_itr(max_itrSEXP);
        Rcpp::traits::input_parameter<int>::type seed(seedSEXP);
        rcpp_result_gen = Rcpp::wrap(_gmmR(dtm, k, max_itr, seed));
        return rcpp_result_gen;
    END_RCPP
}

S4 _normalize_csr(S4 dtm) {
    IntegerVector dims = as<IntegerVector>(dtm.slot("Dim"));
    IntegerVector row_ptr = as<IntegerVector>(dtm.slot("p"));
    IntegerVector index = as<IntegerVector>(dtm.slot("j"));
    NumericVector data = as<NumericVector>(dtm.slot("x"));
    NumericVector new_data(data.size());

    int rows = dims[0];
    for (int r = 0; r < rows; ++r) {
        int from = row_ptr[r];
        int to = row_ptr[r + 1];
        double norm2 = 0.f;
        for (int j = from; j < to; ++j) {
            norm2 += data[j] * data[j];
        }
        norm2 = sqrt(norm2);
        for (int j = from; j < to; ++j) {
            new_data[j] = data[j] /  norm2;
        }
    }
    S4 csr("dgRMatrix");
    csr.slot("x") = new_data;
    csr.slot("j") = index;
    csr.slot("p") = row_ptr;
    csr.slot("Dim") = dims;
    return csr;
}

RcppExport SEXP normalize_csr(SEXP dtmSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter<S4>::type dtm(dtmSEXP);
        rcpp_result_gen = Rcpp::wrap(_normalize_csr(dtm));
        return rcpp_result_gen;
    END_RCPP
}

S4 _read_csr(const std::string& path) {
    int docs, items, nnz;
    double sparsity, constructionCost;
    std::ifstream in(path.c_str());
    in >> docs;
    in >> items;
    in >> nnz;
    in >> sparsity;
    in >> constructionCost;
    NumericVector data(nnz);
    IntegerVector index(nnz);
    IntegerVector row_ptr(docs + 1);
    string word;
    getline(in, word); // todo
    for (int i = 0; i < items; ++i) {
        getline(in, word);
    }
    double idf;
    for (int i = 0; i < items; ++i) {
        in >> idf;
    }
    for (int i = 0; i <= docs; ++i) {
        int d;
        in >> d;
        row_ptr[i] = d;
    }
    for (int i = 0; i < nnz; ++i) {
        int d;
        in >> d;
        index[i] = d;
    }
    for (int i = 0; i < nnz; ++i) {
        double d;
        in >> d;
        data[i] = d;
    }
    in.close();

    S4 csr("dgRMatrix");
    csr.slot("x") = data;
    csr.slot("j") = index;
    csr.slot("p") = row_ptr;
    csr.slot("Dim") = IntegerVector::create(docs, items);
    return csr;
}

RcppExport SEXP read_csr(SEXP pathSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter< const std::string& >::type path(pathSEXP);
        rcpp_result_gen = Rcpp::wrap(_read_csr(path));
        return rcpp_result_gen;
    END_RCPP
}


//int main() {
//    return 0;
//}

