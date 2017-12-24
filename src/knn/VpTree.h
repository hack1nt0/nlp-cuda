//
// Created by DY on 17-11-6.
//

#ifndef NLP_CUDA_VPTREE_H
#define NLP_CUDA_VPTREE_H

#include <matrix/SparseMatrix.h>
#include <ds/SizeFixedHeap.h>
#include <ostream>

template <typename V = double, typename I = unsigned int, class Matrix = SparseMatrix<V, I> >
struct VpTree {
    typedef V value_t;
    typedef I index_t;
    struct Node {
        index_t vp = -1;
        value_t radius1 = 0;
        value_t radius2 = 0;
        Node* lchd = NULL;
        Node* rchd = NULL;

        ~Node() {
            delete lchd;
            delete rchd;
        }

        inline bool isLeaf() const {
            return lchd == NULL && rchd == NULL;
        }
    };
    const Matrix& points;
    Node root;
    struct Neighbor {
        index_t pointId;
        value_t dist;

        Neighbor() {}
        Neighbor(const index_t& pi, const value_t& dist) : pointId(pi), dist(dist) {}

        bool operator<(const Neighbor& o) const {
            return dist != o.dist ? dist < o.dist : pointId < o.pointId;
        }

        friend ostream &operator<<(ostream &os, const Neighbor &neighbor) {
            os << "pointId: " << neighbor.pointId << " dist: " << neighbor.dist;
            return os;
        }
    };
    vector<Neighbor> order;

    VpTree(const Matrix& points, int seed = 1) : points(points), order(points.nrow()) {
        srand(seed);
        for (index_t i = 0; i < points.nrow(); ++i) order[i] = {i, 0};
        buildTree(root, 0, points.nrow());
    }

    void buildTree(Node& curNode, index_t from, index_t to) {
        index_t size = to - from;
        if (size <= 2) {
            return;
        }
        // order[from] always be the vantage point, so ignore it.
        index_t vpi = from + rand() % size;
        curNode.vp = order[vpi].pointId;
        swap(order[from], order[vpi]);
        auto vpRow = points.row(curNode.vp);
        ++from;
        for (index_t i = from; i < to; ++i) {
            order[i].dist = vpRow.dist2(points.row(order[i].pointId));
        }
        sort(order.begin() + from, order.begin() + to); //todo
        if (order[to - 1].dist == 0) {//degenerated to one point//todo
            return;
        }
        index_t mid = from + (to - from >> 1);
        curNode.radius1 = order[mid - 1].dist;
        curNode.radius2 = order[mid].dist;
        curNode.lchd = new Node();
        buildTree(*curNode.lchd, from, mid);
        curNode.rchd = new Node();
        buildTree(*curNode.rchd, mid, to);
    }

    //actually a size-controlled max-heap
    struct KnnVisitor {
        typedef vector<Neighbor> Container;
//        Container* nearestPtr;
        Container nearest;
        index_t k;
        index_t startId;
        bool useAll;
        index_t visited;

        KnnVisitor(index_t k, index_t startId = -1, bool useAll = false) :
            k(k), startId(startId), useAll(useAll), visited(0) {}

        void visit(Neighbor&& candidate) {
            ++visited;
            if (candidate.pointId == startId) return;
            if (size() < k) {
                push(move(candidate));
            } else if (candidate.dist < top().dist) {
                do { pop(); } while (size() > 0 && candidate.dist < top().dist);
                push(move(candidate));
            } else if (useAll && candidate.dist == top().dist) {
                push(move(candidate));
            }
        }

        inline bool useful(value_t lb) {
            return size() < k || top().dist > lb || (useAll && top().dist == lb);
        }

        inline const Neighbor& top() const {
            return nearest[0];
        }

        inline index_t size() const { return nearest.size(); }

        void push(Neighbor&& candidate) {
            nearest.emplace_back(candidate);
            index_t ptr = size() - 1;
            while (ptr > 0) {
                index_t fa = ptr - 1 >> 1;
                if (nearest[fa].dist < nearest[ptr].dist) swap(nearest[fa], nearest[ptr]);
                else break;
                ptr = fa;
            }
        }

        void pop() {
            swap(nearest[0], nearest[size() - 1]);
            nearest.pop_back();
            index_t ptr = 0;
            while (ptr < size()) {
                index_t lc = (ptr << 1) + 1;
                index_t rc = lc + 1;
                if (lc >= size()) break;
                index_t ma = lc;
                if (rc < size() && nearest[lc].dist < nearest[rc].dist) ma = rc;
                if (nearest[ptr].dist < nearest[ma].dist) swap(nearest[ma], nearest[ptr]);
                else break;
                ptr = ma;
            }
        }

        inline index_t getVisited() { return visited; }
    };

    template <class Visitor>
    void knnTraverse(const Node& curNode, index_t from, index_t to, const typename Matrix::Row& p, Visitor& visitor) {
        if (curNode.isLeaf()) {
            for (index_t i = from; i < to; ++i) {
                visitor.visit({order[i].pointId, p.dist2(points.row(order[i].pointId))});
            }
        } else {
            value_t d = p.dist2(points.row(curNode.vp));
            visitor.visit({curNode.vp, d});
            ++from;
            index_t mid = from + (to - from >> 1);
            value_t d1 = abs(curNode.radius1 - d);
            value_t d2 = abs(curNode.radius2 - d);
            bool leftFirst = d1 <= d2;
            if (leftFirst) {
                knnTraverse(*curNode.lchd, from, mid, p, visitor);
                if (visitor.useful(d2))
                    knnTraverse(*curNode.rchd, mid, to, p, visitor);
            } else {
                knnTraverse(*curNode.rchd, mid, to, p, visitor);
                if (visitor.useful(d1))
                    knnTraverse(*curNode.lchd, from, mid, p, visitor);
            }
        }
    }

    vector<Neighbor> knn(index_t k, const typename Matrix::Row& p, bool useAll = false) {
        assert(0 < k && k < points.nrow());
        KnnVisitor visitor(k, -1, useAll);
        knnTraverse(root, 0, points.nrow(), p, visitor);
        vector<Neighbor> r = move(visitor.nearest);
        sort(r.begin(), r.end());
        return move(r);
    }

    vector<Neighbor> knn(index_t k, index_t pi, bool useAll = false) {
        assert(0 < k && k < points.nrow());
        assert(0 <= pi && pi < points.nrow());
        KnnVisitor visitor(k, pi, useAll);
        knnTraverse(root, 0, points.nrow(), points.row(pi), visitor);
        vector<Neighbor> r = move(visitor.nearest);
        sort(r.begin(), r.end());
        cout << "visited = " << visitor.getVisited() << " of " << points.nrow() << endl;
        cout << r.size() << endl;
        return move(r);
    }

};
#endif //NLP_CUDA_VPTREE_H
