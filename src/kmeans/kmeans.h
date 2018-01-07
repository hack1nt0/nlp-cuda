//
// Created by DY on 17-9-27.
//

#ifndef NLP_CUDA_KMEANS_H
#define NLP_CUDA_KMEANS_H
                                  
#include "../dist/dist.h"


template <class Mat>
struct KMeans {
    typedef typename Mat::index_t index_t;
    typedef typename Mat::value_t value_t;
    typedef CDenseMatrix<value_t, index_t> DMatrix;
    typedef CDenseMatrix<index_t, index_t> IMatrix;
    typedef DistanceUtils<value_t, index_t> DU;

    struct Model {
        DMatrix         center;
        IMatrix         cluster;
        DMatrix         ss;
        IMatrix         size;
        vector<value_t> loss;
        vector<index_t> changed;
        index_t         itr;

        Model(index_t rows, index_t cols, index_t k) : center(k, cols), ss(rows, 1), cluster(rows, 1), size(k, 1), itr(0) {}

        Model(index_t rows, index_t cols, index_t k, value_t* center, index_t* cluster, value_t* ss, index_t* size) :
            center(k, cols, center), ss(rows, 1, ss), cluster(rows, 1, cluster), size(k, 1, size), itr(0) {}

        Model& operator=(const Model& o) {
            center = o.center;
            cluster = o.cluster;
            loss = o.loss;
            changed = o.changed;
            itr = o.itr;
            return *this;
        }
    };

    static void init(DMatrix& center, IMatrix& cluster, DMatrix& ss, int k, const Mat& train, int seed, bool verbose = false) {
        srand(seed);
        index_t n = train.nrow();
        ss = -1.0;
        index_t c = rand() % n;
        center.row(0) = train.row(c);
        ss[c] = 0;
        cluster[c] = 0;
        BlasWrapper blas;
        ProgressBar bar(k, verbose);

        if (verbose) {
            cout << "Kmeans++..." << endl;
            bar.increase();
        }
        for (index_t i = 1; i < k; ++i) {
#pragma omp parallel for
            for (index_t j = 0; j < n; ++j) {
                if (ss[j] == 0) continue; //centroid
                value_t dcj = DU::squaredEuclidean(train.row(c), train.row(j));
                if (ss[j] < 0 || dcj < ss[j]) {
                    ss[j] = dcj;
                    cluster[j] = i - 1;
                }
            }
            index_t nc = c;
            for (index_t j = 0; j < n; ++j) if (ss[j] > ss[nc]) nc = j;
            if (nc == c) throw invalid_argument("The k maybe too large, or too much duplicated Obs..");
            center.row(i) = train.row(nc);
            ss[nc] = 0;
            cluster[nc] = i;
            c = nc;
            if (i == k - 1) {
#pragma omp parallel for
                for (index_t j = 0; j < n; ++j) {
                    if (ss[j] == 0) continue; //centroid
                    value_t dcj = DU::squaredEuclidean(train.row(c), train.row(j));
                    if (dcj < ss[j]) {
                        ss[j] = dcj;
                        cluster[j] = k - 1;
                    }
                }
            }
            if (verbose) bar.increase();
            if (bar.interrupted()) throw interrupt_exception("SIGINT");
        }
    }

    static void train(Model& model, const Mat& train, index_t k, int maxItr = 10, int seed = 1, value_t tol = 1e-6, bool verbose = true) {
        DMatrix&         center = model.center;
        vector<value_t>& loss = model.loss;
        DMatrix&         ss = model.ss;
        IMatrix&         cluster = model.cluster;
        IMatrix&         size = model.size;
        index_t&         itr = model.itr;
        index_t n = train.nrow();
        index_t m = train.ncol();
        IMatrix changed(n, 1); changed = 0;
        DMatrix centerss(k, 1);
        BlasWrapper blas;

        ProgressBar bar(maxItr, verbose, 0.3);
        for (itr = 0; itr < maxItr; ++itr) {
            if (itr == 0) {
                init(center, cluster, ss, k, train, seed, verbose);
            } else {
#pragma omp parallel for
                for (index_t i = 0; i < k; ++i) {
                    centerss[i] = sum(center.row(i) ^ 2.);
                }
#pragma omp parallel for
                for (index_t i = 0; i < n; ++i) {
                    value_t minDist = -1;
                    index_t old = cluster[i];
                    value_t trainss = sum(train.row(i) ^ 2.);
                    for (index_t j = 0; j < k; ++j) {
                        value_t curDist = centerss[j] - blas.dot(center.row(j), train.row(i)) * 2 + trainss;
                        //              = dist(center.row(j), train.row(i), 0);
                        if (minDist < 0 || curDist < minDist) {
                            minDist = curDist;
                            cluster[i] = j;
                        }
                    }
                    ss[i] = minDist;
                    changed[i] = cluster[i] != old;
                }
            }
            loss.push_back(sum(ss));
            model.changed.push_back(sum(changed));
            if (verbose) {
                bar.increase(false);
                cout << "LOS: " << loss[itr] << "\tREA: " << model.changed[itr];
                cout.flush();
            }
            if (bar.interrupted()) throw interrupt_exception("SIGINT");
            if (itr > 0 && abs(loss[itr] - loss[itr - 1]) <= tol) break;
            center = 0;
            size   = 0;
            for (index_t j = 0; j < n; ++j) {
                center.row(cluster[j]) += train.row(j);
                ++size[cluster[j]];
            }
            for (index_t j = 0; j < k; ++j) if (size[j] > 0) center.row(j) /= size[j];
            if (bar.interrupted()) throw interrupt_exception("SIGINT");
        }
    }
};
#endif //NLP_CUDA_KMEANS_H
