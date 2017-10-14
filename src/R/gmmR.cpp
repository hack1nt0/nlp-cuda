//
// Created by DY on 17-8-28.
//

#include <Rcpp.h>
#include <gmm.h>
using namespace Rcpp;
//using namespace std;

Rcpp::List gmm(S4 dtm, int k, int max_itr, int seed, double alpha, double beta, int topics) {
    IntegerVector dims = as<IntegerVector>(dtm.slot("Dim"));
    int rows = dims[0];
    int cols = dims[1];
    IntegerVector row_ptr = as<IntegerVector>(dtm.slot("p"));
    IntegerVector index = as<IntegerVector>(dtm.slot("j"));
    NumericVector data = as<NumericVector>(dtm.slot("x"));
    int nnz = data.size();
    List dimnames = as<List>(dtm.slot("Dimnames"));
    CharacterVector terms = as<CharacterVector>(dimnames.at(1));

    NumericMatrix resp(k, rows);
    NumericMatrix mean(cols, k);
    NumericMatrix cov(cols, k);
    NumericVector class_weight(k);
    gmmInit(mean.begin(), cov.begin(), class_weight.begin(),
            data.begin(), index.begin(), row_ptr.begin(),
            rows, cols, nnz,
            k, seed, beta);
    std::vector<double> append_likehood = gmm(resp.begin(), mean.begin(), cov.begin(), class_weight.begin(),
                                              data.begin(), index.begin(), row_ptr.begin(),
                                              rows, cols, data.size(), k, max_itr, seed, alpha, beta);
    CharacterMatrix topicMatix(topics, k);
    vector<int> order(cols);
    for (int i = 0; i < cols; ++i) order[i] = i;
    struct Comp {
        const NumericMatrix::Column& mean;
        const NumericMatrix::Column& cov;

        Comp(const NumericMatrix::Column &mean, const NumericMatrix::Column &cov) : mean(mean), cov(cov) {}

        bool operator()(int a, int b) const {
//            if (cov[a] != cov[b]) return cov[a] < cov[b];
            return mean[a] > mean[b];
        }
    };
    for (int i = 0; i < k; ++i) {
        std::sort(order.begin(), order.end(), Comp(mean.column(i), cov.column(i)));
        for (int j = 0; j < topics; ++j) {
            topicMatix.at(j, i) = terms[order[j]];
        }
    }
    IntegerVector belong(rows);
    for (int i = 0; i < rows; ++i) belong[i] = which_max(resp.column(i));
    IntegerVector capacity(k);
    for (int i = 0; i < k; ++i) capacity[i] = 0;
    for (int i = 0; i < rows; ++i) capacity[belong[i]]++;
    return Rcpp::List::create(
            Rcpp::Named("mean") = mean,
            Rcpp::Named("cov") = cov,
            Rcpp::Named("class_weight") = class_weight,
            Rcpp::Named("capacity") = capacity,
            Rcpp::Named("resp") = resp,
            Rcpp::Named("belong") = belong,
            Rcpp::Named("likelihood") = append_likehood,
            Rcpp::Named("topics") = topicMatix);
}

RcppExport SEXP gmm(SEXP dtmSEXP, SEXP kSEXP, SEXP max_itrSEXP, SEXP seedSEXP, SEXP alphaSEXP, SEXP betaSEXP, SEXP topicsSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter<S4>::type dtm(dtmSEXP);
        Rcpp::traits::input_parameter<int>::type k(kSEXP);
        Rcpp::traits::input_parameter<int>::type max_itr(max_itrSEXP);
        Rcpp::traits::input_parameter<int>::type seed(seedSEXP);
        Rcpp::traits::input_parameter<double>::type alpha(alphaSEXP);
        Rcpp::traits::input_parameter<double>::type beta(betaSEXP);
        Rcpp::traits::input_parameter<int>::type topics(topicsSEXP);
        rcpp_result_gen = Rcpp::wrap(gmm(dtm, k, max_itr, seed, alpha, beta, topics));
        return rcpp_result_gen;
    END_RCPP
}

