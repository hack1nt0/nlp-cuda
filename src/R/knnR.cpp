//
// Created by DY on 17-9-17.
//

#include <R/Rutils.h>
#include <knn/VpTreeX.h>
#include <knn/knn.h>

typedef Knn<sm_type>               knn_t;
typedef knn_t::Neighbor            Neighbor;
typedef knn_t::NeighborList        NeighborList;
typedef knn_t::NeighborListList    NeighborListList;

List toNNListR(const NeighborListList& nll) {
    List r(nll.size());
    for (int i = 0; i < nll.size(); ++i) {
        const NeighborList& nl = nll[i];
        IntegerVector pointId(nl.size());
        NumericVector distance(nl.size());
        for (int j = 0; j < nl.size(); ++j) {
            pointId[j] = nl[j].second + 1;
            distance[j] = nl[j].first;
        }
        r[i] = DataFrame::create(Named("p.id") = pointId, Named("distance") = distance);
    }
    return r;
}

RcppExport
SEXP knn_dtm(SEXP trainSEXP, SEXP testSEXP, SEXP clSEXP,
             SEXP kSEXP, SEXP dtypeSEXP, SEXP lSEXP, SEXP probSEXP, SEXP useAllSEXP,
             SEXP methodSEXP, SEXP leafSizeSEXP, SEXP seedSEXP, SEXP verboseSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        sm_type train = toSparseMatrix(trainSEXP);
        sm_type test  = toSparseMatrix(testSEXP);
        int k = as<int>(kSEXP);
        int dist_t = 0;
        String dtype = CharacterVector(dtypeSEXP)[0];
        if (dtype == "euclidean") dist_t = 1;
        else if (dtype == "cosine") dist_t = 2;
        else if (dtype == "edit") dist_t = 3;
        int l = as<int>(lSEXP);
        bool prob = as<bool>(probSEXP);
        bool useAll = as<bool>(useAllSEXP);
        CharacterVector method(methodSEXP);
        int leafSize = as<int>(leafSizeSEXP);
        int seed = as<int>(seedSEXP);
        bool verbose = as<bool>(verboseSEXP);
        knn_t knn;
        List r;
        if (method[0] == "brute") {
            NeighborListList nll = knn.brute(train, test, k, dist_t, useAll, verbose);
            r["nn"] = toNNListR(nll);
        } else if (method[0] == "vptree") {
            IntegerVector visited(test.nrow());
            NeighborListList nll = knn.vptree(train, test, k, dist_t, useAll, leafSize, seed, visited.begin(), verbose);
            r["nn"] = toNNListR(nll);
            r["vn"] = visited;
        } else {
            Rcout << "Not impl..." << endl;
            return R_NilValue;
        }

        if (Rf_isNull(clSEXP)) {
            return r;
        } else {
            Rcout << "Not impl..." << endl;
            return R_NilValue;
        }
    END_RCPP
}

//typedef CDenseMatrix<double, int> dm_type;
//typedef VpTreeX<dm_type>         vptree_type;

//RcppExport
//SEXP vptree(SEXP trainSEXP, SEXP leafSizeSEXP, SEXP seedSEXP) {
//    NumericMatrix trainR(trainSEXP);
//    int leafSize = as<int>(leafSizeSEXP);
//    int seed = as<int>(seedSEXP);
//    dm_type train(trainR.nrow(), trainR.cols(), trainR.begin(), false);
//    XPtr<vptree_type> xp(new vptree_type(train, leafSize, seed), true);
//    return xp;
//}
//
//SEXP vptree_matrix(SEXP testSEXP, SEXP kSEXP, SEXP useAllSEXP, SEXP vptreeXpSEXP) {
//    NumericMatrix testR(testSEXP);
//    dm_type test(testR.nrow(), testR.ncol(), testR.begin(), false);
//    int k = as<int>(kSEXP);
//    bool useAll = as<bool>(useAllSEXP);
//    XPtr<vptree_type> vptreeXp(vptreeXpSEXP);
//    List r;
//    IntegerVector visited(testR.nrow());
//    NeighborListList nll = vptreeXp->knn(k, test, useAll, visited.begin());
//    r["nn"] = toNNListR(nll);
//    r["vn"] = visited;
//    return r;
//}
