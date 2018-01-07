//
// Created by DY on 17-11-6.
//

#ifndef NLP_CUDA_VPTREEX_H
#define NLP_CUDA_VPTREEX_H

#include <matrix/SparseMatrix.h>
#include <ds/SizeFixedHeap.h>
#include <ostream>
#include <dist/dist.h>

template <class Matrix>
struct VpTreeX {
    typedef typename Matrix::value_t  value_t;
    typedef typename Matrix::index_t  index_t;
    typedef typename Matrix::Row      Row;
    typedef DistanceUtils<value_t, index_t> DU;
    vector<index_t> vp;
    vector<value_t> radius1;
    vector<value_t> radius2;
    vector<index_t> lc;
    vector<index_t> rc;
    vector<index_t> from;
    vector<index_t> to;
    index_t size;
    const Matrix& points;
    typedef pair<value_t, index_t> Neighbor;
    typedef vector<Neighbor>       NeighborList;
    typedef vector<NeighborList>   NeighborListList;
    vector<Neighbor> order;
    index_t leafSize;
    int dist_t;


    inline bool isLeaf(const index_t& i) const { return to[i] - from[i] <= leafSize || lc[i] == rc[i]; }

    VpTreeX(const Matrix& points, int dist_t = 1, index_t leafSize = 2, int seed = 1) :
        points(points), dist_t(dist_t), order(points.nrow()), size(0), leafSize(leafSize),
        vp(points.nrow()),
        radius1(points.nrow()),
        radius2(points.nrow()),
        lc(points.nrow()),
        rc(points.nrow()),
        from(points.nrow()),
        to(points.nrow()) {
        assert(points.nrow() > 0);
        srand(seed);
        for (index_t i = 0; i < points.nrow(); ++i) order[i] = {0, i};
        buildTree(0, 0, points.nrow());
    }

    void buildTree(index_t cur, index_t from, index_t to) {
        ++size;
        this->from[cur] = from;
        this->to[cur] = to;
        if (to - from <= leafSize) return;
        value_t maxVar = -1;
        index_t vpi = from + rand() % (to - from);
        vp[cur] = order[vpi].second;
        swap(order[from], order[vpi]);
        // order[from] always be the vantage point, so ignore it.
        order[from].first = 0;
        ++from;
        Row vpRow = points.row(vp[cur]);
        for (index_t i = from; i < to; ++i) {
            order[i].first = DU::dist(vpRow, points.row(order[i].second), dist_t);
        }
        sort(order.begin() + from, order.begin() + to); //todo
        index_t mid = from + (to - from >> 1);
        radius1[cur] = order[mid - 1].first;
        radius2[cur] = order[mid].first;
//        radius3[cur] = order[to - 1].first;
        if (order[to - 1].first == 0) {//degenerated to one point//todo
            return;
        }
        index_t lchd = size;
        lc[cur] = lchd;
        buildTree(lchd, from, mid);
        index_t rchd = size;
        rc[cur] = rchd;
        buildTree(rchd, mid, to);
    }

    struct NeighborHeap : public priority_queue<Neighbor> {
        typedef priority_queue<Neighbor> super_t;
        index_t k;
        bool useAll;
        index_t visited;

        NeighborHeap(index_t k, bool useAll = false) :
            k(k), useAll(useAll), visited(0) {
            assert(k > 0);
        }

        void visit(Neighbor&& candidate) {
            if (super_t::size() < k || useAll && candidate.first == super_t::top().first) {
                super_t::emplace(candidate);
            } else if (candidate.first < super_t::top().first) {
                do { super_t::pop(); } while (super_t::size() >= k && candidate.first < super_t::top().first);
                super_t::emplace(candidate);
            }
            ++visited;
        }

        inline vector<Neighbor> data() const {
            return super_t::c;
        }

        inline bool useful(value_t lb) {
            return super_t::size() < k || lb < super_t::top().first || (useAll && lb == super_t::top().first);
        }
    };


    void traverse(index_t cur, value_t lb, const Row& p, NeighborHeap& neighborHeap) {
//        cerr << depth << ' ' << from[cur] << ' ' << to[cur] << ' ' << to[cur] - from[cur] << ' ' << lc[cur] << ' ' << rc[cur] << endl;
        if (isLeaf(cur)) {
            for (int i = from[cur]; i < to[cur]; ++i) {
                neighborHeap.visit({DU::dist(p, points.row(order[i].second), dist_t), order[i].second});
            }
            return;
        }
        value_t d = DU::dist(p, points.row(vp[cur]), dist_t);
        neighborHeap.visit({d, vp[cur]});
        value_t llb = max(lb, d - radius1[cur]);
        value_t rlb = max(lb, radius2[cur] - d);
        if (llb < rlb) {
            if (neighborHeap.useful(llb)) traverse(lc[cur], llb, p, neighborHeap);
            if (neighborHeap.useful(rlb)) traverse(rc[cur], rlb, p, neighborHeap);
        } else {
            if (neighborHeap.useful(rlb)) traverse(rc[cur], rlb, p, neighborHeap);
            if (neighborHeap.useful(llb)) traverse(lc[cur], llb, p, neighborHeap);
        }
    }


    vector<Neighbor> knn(index_t k, const Row& p, bool useAll, index_t& visited) {
        NeighborHeap visitor(k, useAll);
        traverse(0, 0, p, visitor);
        visited = visitor.visited;
        vector<Neighbor> r = move(visitor.data());
        sort(r.begin(), r.end());
        return move(r); //todo
    }
};
#endif //NLP_CUDA_VPTREE_H
