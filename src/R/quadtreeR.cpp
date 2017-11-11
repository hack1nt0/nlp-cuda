//
// Created by DY on 17-8-28.
//

#include <Rcpp.h>
#include <common_headers.h>
#include <tsne/quadtree.h>

using namespace Rcpp;

typedef QuadTree<Rcpp::NumericMatrix> QuadTreeR;

struct QuadTreeRAllNodesVisitor {
    XPtr<QuadTreeR>& quadTree;
    bool onlyLeaf;
    NumericVector& xmin;
    NumericVector& ymin;
    NumericVector& xmax;
    NumericVector& ymax;
    int size = 0;

    QuadTreeRAllNodesVisitor(XPtr<QuadTreeR>& quadTree, bool onlyLeaf,
                             NumericVector& xmin,
                             NumericVector& xmax,
                             NumericVector& ymin,
                             NumericVector& ymax) : quadTree(quadTree), onlyLeaf(onlyLeaf),
                                                    xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax) {}

    void visit() { quadTree->traverse(*this); }

    bool visit(const QuadTreeR::Node& curNode) {
        if (!(onlyLeaf && !curNode.isLeaf)) {
            xmin[size] = quadTree->bound[0].first;
            xmax[size] = quadTree->bound[0].second;
            ymin[size] = quadTree->bound[1].first;
            ymax[size] = quadTree->bound[1].second;
            size++;
        }
        return false;
    }
};

struct QuadTreeRNearbyNodesVisitor {
    const NumericVector& point;
    XPtr<QuadTreeR>& quadTree;
    double theta = 0.5;
    NumericVector xmin;
    NumericVector xmax;
    NumericVector ymin;
    NumericVector ymax;
    NumericVector xcenter;
    NumericVector ycenter;
    NumericVector capacity;

    QuadTreeRNearbyNodesVisitor(const NumericVector& point, XPtr<QuadTreeR>& quadTree, double theta) : point(point),
                                                                                                 quadTree(quadTree),
                                                                                                 theta(theta) {}

    void visit() { quadTree->traverse(*this); }

    bool visit(const QuadTreeR::Node& curNode) {
        if (curNode.isLeaf || (quadTree->bound[0].second - quadTree->bound[0].first) / curNode.dist(point) < theta) {
            xmin.push_back(quadTree->bound[0].first);
            xmax.push_back(quadTree->bound[0].second);
            ymin.push_back(quadTree->bound[1].first);
            ymax.push_back(quadTree->bound[1].second);
            xcenter.push_back(curNode.center[0]);
            ycenter.push_back(curNode.center[1]);
            capacity.push_back(curNode.capacity);
            return true;
        }
        return false;
    }
};

RcppExport SEXP QuadTree__new(SEXP pointsSEXP) {
    NumericMatrix points(pointsSEXP);
    XPtr<QuadTreeR> ptr( new QuadTreeR(points), true);
    return ptr;
}

RcppExport SEXP QuadTree__allNodes(SEXP xp, SEXP onlyLeafSEXP) {
    Rcpp::XPtr<QuadTreeR> ptr(xp);
    bool onlyLeaf = as<bool>(onlyLeafSEXP);
    NumericVector xmin(ptr->nodes);
    NumericVector xmax(ptr->nodes);
    NumericVector ymin(ptr->nodes);
    NumericVector ymax(ptr->nodes);
    QuadTreeRAllNodesVisitor visitor(ptr, onlyLeaf, xmin, xmax, ymin, ymax);
    visitor.visit();
    return DataFrame::create(
            Named("xmin") = visitor.xmin,
            Named("xmax") = visitor.xmax,
            Named("ymin") = visitor.ymin,
            Named("ymax") = visitor.ymax
    );
}

RcppExport SEXP QuadTree__nearbyNodes(SEXP xp, SEXP pointSEXP, SEXP thetaSEXP) {
    Rcpp::XPtr<QuadTreeR> ptr(xp);
    NumericVector point(pointSEXP);
    double theta = as<double>(thetaSEXP);
    QuadTreeRNearbyNodesVisitor visitor(point, ptr, theta);
    visitor.visit();
    return DataFrame::create(
            Named("xmin") = visitor.xmin,
            Named("xmax") = visitor.xmax,
            Named("ymin") = visitor.ymin,
            Named("ymax") = visitor.ymax,
            Named("xcenter") = visitor.xcenter,
            Named("ycenter") = visitor.ycenter,
            Named("capacity") = visitor.capacity
    );
}


