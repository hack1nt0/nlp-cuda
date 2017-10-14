//
// Created by DY on 17-9-17.
//

#ifndef NLP_CUDA_STATS_H
#define NLP_CUDA_STATS_H

#include <common_headers.h>
#include "matrix/DenseMatrix.h"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <SymmetricDistance.h>

using namespace std;
using namespace tbb;

template <typename T>
struct Knn {
    SymmetricDistance<T> dist;
    bool verbose;

    Knn(const SparseMatrix<T>& sm, bool verbose=true) : dist(sm, verbose), verbose(verbose) {}

    SparseMatrix<T> knn(int k) {
        assert(0 < k && k <= dist.rows);
        SparseMatrix<T> nn(dist.rows, dist.rows, k * dist.rows);
        for (int i = 0; i <= nn.nrow(); ++i) nn.row_ptr[i] = k * i;
        CpuTimer timer;
        timer.start();
        if (verbose) printf("knn...");
        parallel_for(blocked_range<int>(0, nn.nrow()),
                     KNNTask(nn.index, nn.data, this->dist, k));
//    KNNTask<T> knnTask(nn, nd, D, k);
//    knnTask(blocked_range<int>(n - 10, n));
        if (verbose) printf("done. %e ms\n", timer.elapsed());
        return nn;
    }

    struct KNNTask {
        int* nn;
        T* nd;
        const SymmetricDistance<T> &D;
        int k;
        int n;

        KNNTask(int* nn, T* nd, const SymmetricDistance<T> &D, int k) : nn(nn), nd(nd), D(D), k(k), n(D.rows) {}

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


};

#endif //NLP_CUDA_STATS_H
