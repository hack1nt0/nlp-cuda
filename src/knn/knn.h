//
// Created by DY on 17/12/6.
//

#ifndef NLP_CUDA_KNN_H
#define NLP_CUDA_KNN_H

#include <utils.h>
#include <dist/dist.h>
/**
 *
 * @tparam Mat
 */

template <class Mat>
struct Knn {
    typedef typename Mat::value_t     value_t;
    typedef typename Mat::index_t     index_t;
    typedef pair<value_t, index_t>    Neighbor;
    typedef vector<Neighbor>          NeighborList;
    typedef vector<NeighborList>      NeighborListList;
    typedef VpTreeX<Mat>              vptree_t;

    struct Comparator {
        inline bool operator()(const Neighbor& lhs, const Neighbor& rhs) { return lhs.first < rhs.first; }
        inline int compare(const Neighbor& lhs, const Neighbor& rhs) { return lhs.first == rhs.first ? 0 : (lhs.first > rhs.first ? +1 : -1); }
    };

    struct Heap : priority_queue<Neighbor, NeighborList, Comparator> {
        typedef priority_queue<Neighbor, NeighborList, Comparator> super_t;
        int bound;
        bool useAll;
        Comparator comp;

        Heap(int bound, bool useAll = false) : bound(bound), useAll(useAll) { assert(bound > 0); }

        inline void visit(Neighbor&& v) {
            if (super_t::size() < bound || useAll && comp.compare(v, super_t::top()) == 0) {
                super_t::emplace(v);
            } else if (comp.compare(v, super_t::top()) < 0) {
                do { super_t::pop(); } while (super_t::size() >= bound && comp.compare(v, super_t::top()) < 0);
                super_t::emplace(v);
            }
        }

        inline NeighborList getData() const { return move(super_t::c); }

    };

    NeighborListList brute(const Mat& train, const Mat& test, index_t k, int dist_t, bool useAll, bool verbose = false ) {
        NeighborListList r(test.nrow());
        ProgressBar pbar(test.nrow());
#pragma omp parallel for
        for (index_t te = 0; te < test.nrow(); ++te) {
            if (verbose) if (pbar.interrupted()) continue;
            auto terow = test.row(te);
            Heap heap(k, useAll);
            for (index_t tr = 0; tr < train.nrow(); ++tr) {
                heap.visit({dist(terow, train.row(tr), dist_t), tr});
            }
            r[te] = move(heap.getData());
            sort(r[te].begin(), r[te].end());
            if (verbose) pbar.increase();
        }
        if (pbar.interrupted()) throw std::runtime_error("[SIGINT]");
        return r;
    }

    NeighborListList vptree(const Mat& train, const Mat& test, index_t k, int dist_t, bool useAll, int leafSize, int seed, index_t* visited = nullptr, bool verbose = false) {
        index_t n = test.nrow();
        NeighborListList r(n);
        vptree_t vpt(train, dist_t, leafSize, seed);
        ProgressBar pbar(n);
#pragma omp parallel for
        for (index_t i = 0; i < n; ++i) {
            if (verbose) if (pbar.interrupted()) continue;
            typename vptree_t::NeighborHeap visitor(k, useAll);
            vpt.traverse(0, 0, test.row(i), visitor);
            if (visited != nullptr) visited[i] = visitor.visited;
            r[i] = move(visitor.data());
            sort(r[i].begin(), r[i].end());
            if (verbose) pbar.increase();
        }
        if (pbar.interrupted()) throw std::runtime_error("[SIGINT]");
        return r;
    }
};

#endif //NLP_CUDA_KNN_H
