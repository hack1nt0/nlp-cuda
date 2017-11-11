//
// Created by DY on 17-9-17.
//

#ifndef NLP_CUDA_STATS_H
#define NLP_CUDA_STATS_H

#include <common_headers.h>
#include "matrix/DenseMatrix.h"
#include "VpTree.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <matrix/dist/DistMatrix.h>

using namespace std;
using namespace tbb;


template <typename T>
struct KNNTask {
    int* nn;
    T* nd;
    const DistMatrix<T> &D;
    int k;
    int n;

    KNNTask(int* nn, T* nd, const DistMatrix<T> &D, int k) : nn(nn), nd(nd), D(D), k(k), n(D.rows) {}

    void operator()(const blocked_range<int> &r) const {
        for (int from = r.begin(); from != r.end(); ++from) {
            int nnn = 0;
            int* nnq = nn + from * k;
            T* ndq = nd + from * k;
            for (int to = 0; to < n; ++to) {
                if (to == from) continue;
                T dist = D.at(from, to);
                if (nnn == k && dist < ndq[0]) {
                    swap(nnq[0], nnq[nnn - 1]);
                    swap(ndq[0], ndq[nnn - 1]);
                    nnn--;
                    int i = 0;
                    while (i < nnn) {
                        int j = i;
                        int lc = (i << 1) + 1;
                        if (lc < nnn && ndq[lc] > ndq[j]) j = lc;
                        int rc = (i << 1) + 2;
                        if (rc < nnn && ndq[rc] > ndq[j]) j = rc;
                        if (j == i) break;
                        swap(nnq[i], nnq[j]);
                        swap(ndq[i], ndq[j]);
                        i = j;
                    }
                }
                if (nnn < k) {
                    nnq[nnn] = to;
                    ndq[nnn] = dist;
                    nnn++;
                    int i = nnn - 1;
                    while (i > 0) {
                        int fa = (i - 1) >> 1;
                        if (ndq[i] <= ndq[fa]) break;
                        swap(nnq[i], nnq[fa]);
                        swap(ndq[i], ndq[fa]);
                        i = fa;
                    }
                }
            }
//            printf("%d\n", from);
        }
    }
};

template <typename T>
void knnCollect2(SparseMatrix<T>& nn, const DistMatrix<T>& dist, int k, bool verbose = false) {
    assert(0 < k && k <= dist.nrow());
    for (int i = 0; i <= nn.nrow(); ++i) nn.row_ptr[i] = k * i;
    CpuTimer timer;
    timer.start();
    if (verbose) printf("knn...");
    parallel_for(blocked_range<int>(0, nn.nrow()),
                 KNNTask<T>(nn.index, nn.data, dist, k));
    if (verbose) printf("done. %e ms\n", timer.elapsed());
}

template <typename T>
SparseMatrix<T> knnCollect2(const DistMatrix<T>& dist, int k, bool verbose = false) {
    assert(0 < k && k <= dist.nrow());
    SparseMatrix<T> nn(dist.nrow(), dist.nrow(), k * dist.nrow());
    knnCollect2<T>(nn, dist, k, verbose);
    return nn;
}

template <typename T = double, class Train = SparseMatrix<T> >
void knnCollect(SparseMatrix<T>& nn, const Train& points, int k, int seed = 1, bool verbose = false) {
    assert(0 < k && k <= points.nrow());
    for (int i = 0; i <= nn.nrow(); ++i) nn.row_ptr[i] = k * i;
    CpuTimer timer;
    timer.start();
    if (verbose) printf("knn.VpTree...");
    VpTree<T, SparseMatrix<T> > vpTree(points, seed);
#pragma omp parallel for
    for (int i = 0; i < points.nrow(); ++i) {
        auto nnRow = nn.row(i);
        vpTree.knn(nnRow.index, nnRow.data, k, i);
        nnRow.sortByIndex();//todo
    }
    if (verbose) printf("done. %e ms\n", timer.elapsed());
}

template <typename T = double, class Train = SparseMatrix<T> >
SparseMatrix<T> knnCollect(const Train& points, int k, int seed = 1, bool verbose = false) {
    SparseMatrix<T> nn(points.nrow(), points.nrow(), points.nrow() * k);
    knnCollect<T, Train>(nn, points, k, seed, verbose);
    return nn;
}

template <typename T = double, class Train = SparseMatrix<T>, class Test = Train >
void knnClassifier(SparseMatrix<T>& nn, const Train& trainPoints, const Train& testPoints, int k, bool tieIncluded = false, int seed = 1, bool verbose = false) {
    assert(0 < k && k <= trainPoints.nrow());
    for (int i = 0; i <= nn.nrow(); ++i) nn.row_ptr[i] = k * i;
    CpuTimer timer;
    timer.start();
    if (verbose) printf("knn.VpTree...");
    VpTree<T, SparseMatrix<T> > vpTree(trainPoints, seed);
#pragma omp parallel for
    for (int i = 0; i < testPoints.nrow(); ++i) {
        auto nnRow = nn.row(i);
        vpTree.knn(nnRow.index, nnRow.data, k, i);
        nnRow.sortByIndex();
    }
    if (verbose) printf("done. %e ms\n", timer.elapsed());
}

template <typename T, class Train = SparseMatrix<T>, class Test = Train >
SparseMatrix<T> knnClassifier(const Train& trainPoints, const Test& testPoints, int k, bool tieIncluded = false, int seed = 1, bool verbose = false) {
    SparseMatrix<T> nn(testPoints.nrow(), testPoints.nrow(), testPoints.nrow() * k);
    knnClassifier<T, Train, Test>(nn, trainPoints, testPoints, k, tieIncluded, seed, verbose);
    return nn;
}
#endif //NLP_CUDA_STATS_H
