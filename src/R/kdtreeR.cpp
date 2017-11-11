//
// Created by DY on 17-8-28.
//

#include <Rcpp.h>
#include "../knn/knn.h"
using namespace Rcpp;

struct KDTreeSegmentsVisitor {
    const KDTree<double, NumericMatrix>& visited;
    NumericVector x;
    NumericVector y;
    NumericVector xend;
    NumericVector yend;
    const int advance[10] = {0, 0, 1, 0, 1, 1, 0, 1, 0, 0};

    KDTreeSegmentsVisitor(const KDTree<double, NumericMatrix>& kdTree) : visited(kdTree) {
        visited.traverse(*this);
    }

    void visit(const KDTree<double, NumericMatrix>::Rect& curNode) {
        if (curNode.isLeaf && curNode.lchd < 0 && curNode.rchd < 0) {
            for (int i = 0; i < visited.dim; ++i) cout << visited.bound[i] << ' ';
            cout << endl << "-----------" << endl;
            for (int i = 0; i < 4 * 2; i += 2) {
                x.push_back(advance[i] == 0 ? visited.bound[0].first : visited.bound[0].second);
                y.push_back(advance[i + 1] == 0 ? visited.bound[1].first : visited.bound[1].second);
                xend.push_back(advance[i + 2] == 0 ? visited.bound[0].first : visited.bound[0].second);
                yend.push_back(advance[i + 3] == 0 ? visited.bound[1].first : visited.bound[1].second);
            }
        }
    }
};

Rcpp::DataFrame kdtreeSegments(const NumericMatrix& points) {
    cout << points << endl;
    int dim = points.ncol();
    assert(dim == 2);
    KDTree<double, NumericMatrix> kdTree(points);
    KDTreeSegmentsVisitor visitor(kdTree);
    return Rcpp::DataFrame::create(
            Rcpp::Named("x") = visitor.x,
            Rcpp::Named("y") = visitor.y,
            Rcpp::Named("xend") = visitor.xend,
            Rcpp::Named("yend") = visitor.yend
    );
}



RcppExport SEXP kdtreeSegments(SEXP pointsSEXP) {
    BEGIN_RCPP
        Rcpp::RObject rcpp_result_gen;
        Rcpp::RNGScope rcpp_rngScope_gen;
        Rcpp::traits::input_parameter<NumericMatrix>::type points(pointsSEXP);
        rcpp_result_gen = Rcpp::wrap(kdtreeSegments(points));
        return rcpp_result_gen;
    END_RCPP
}

