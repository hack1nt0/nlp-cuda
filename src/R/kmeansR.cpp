//
// Created by DY on 17-8-28.
//

#include <Rcpp.h>
#include <kmeans.h>
#include <vector>
using namespace Rcpp;

Rcpp::List kmeans(S4 dtm, int k, int max_itr, int seed, int topics) {
    IntegerVector dims = as<IntegerVector>(dtm.slot("Dim"));
    int rows = dims[0];
    int cols = dims[1];
    IntegerVector row_ptr = as<IntegerVector>(dtm.slot("p"));
    IntegerVector index = as<IntegerVector>(dtm.slot("j"));
    NumericVector data = as<NumericVector>(dtm.slot("x"));
    int nnz = data.size();
    List dimnames = as<List>(dtm.slot("Dimnames"));
    CharacterVector terms = as<CharacterVector>(dimnames.at(1));

    IntegerVector belong(rows);
    NumericMatrix mean(cols, k);
    initKmeans(mean.begin(), k, data.begin(), index.begin(), row_ptr.begin(), rows, cols, nnz, seed);
    std::vector<double> dist = kmeans(mean.begin(), belong.begin(), data.begin(), index.begin(), row_ptr.begin(), rows, cols, nnz, k, max_itr, seed);

    CharacterMatrix topicMatix(topics, k);
    std::vector<int> order(cols);
    for (int i = 0; i < cols; ++i) order[i] = i;
    struct Comp {
        const NumericMatrix::Column& mean;

        Comp(const NumericMatrix::Column& mean) : mean(mean) {}

        bool operator()(int a, int b) const {
            return mean[a] > mean[b];
        }
    };
    for (int i = 0; i < k; ++i) {
        sort(order.begin(), order.end(), Comp(mean.column(i)));
        for (int j = 0; j < topics; ++j) {
            topicMatix.at(j, i) = terms[order[j]];
        }
    }
    IntegerVector capacity(k);
    for (int i = 0; i < k; ++i) capacity[i] = 0;
    for (int i = 0; i < rows; ++i) capacity[belong[i]]++;
    return Rcpp::List::create(
            Rcpp::Named("mean") = mean,
            Rcpp::Named("capacity") = capacity,
            Rcpp::Named("belong") = belong,
            Rcpp::Named("dist") = dist,
            Rcpp::Named("topics") = topicMatix);
}

RcppExport SEXP kmeans(SEXP dtmSEXP, SEXP kSEXP, SEXP max_itrSEXP, SEXP seedSEXP, SEXP topicsSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter<S4>::type dtm(dtmSEXP);
        Rcpp::traits::input_parameter<int>::type k(kSEXP);
        Rcpp::traits::input_parameter<int>::type max_itr(max_itrSEXP);
        Rcpp::traits::input_parameter<int>::type seed(seedSEXP);
        Rcpp::traits::input_parameter<int>::type topics(topicsSEXP);
        rcpp_result_gen = Rcpp::wrap(kmeans(dtm, k, max_itr, seed, topics));
        return rcpp_result_gen;
    END_RCPP
}

