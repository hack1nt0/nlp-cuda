//
// Created by DY on 17-11-6.
//

#ifndef NLP_CUDA_VPTREE_H
#define NLP_CUDA_VPTREE_H

#include <matrix/SparseMatrix.h>
#include <ds/SizeFixedHeap.h>

template <typename T = double, class Matrix = SparseMatrix<T> >
struct VpTree {
    struct Node {
        int vp = -1;
        T radius1 = 0;
        T radius2 = 0;
        Node* lchd = NULL;
        Node* rchd = NULL;

        ~Node() {
            delete lchd;
            delete rchd;
        }

        inline bool isLeaf() {
            return lchd == NULL && rchd == NULL;
        }
    };
    const Matrix& points;
    Node root;
    struct Pair {
        int pi;
        T dist;

        Pair() {}
        Pair(const int& pi, const T& dist) : pi(pi), dist(dist) {}
        Pair(const Pair& o) : pi(o.pi), dist(o.dist) {}

        bool operator<(const Pair &rhs) const {
            return dist < rhs.dist;
        }
    };
    vector<Pair> order;

    VpTree(const Matrix& points, int seed = 1) : points(points), order(points.nrow()) {
        srand(seed);
        for (int i = 0; i < points.nrow(); ++i) order[i] = Pair(i, T(0));
        buildTree(root, 0, points.nrow());
    }

    void buildTree(Node& curNode, int from, int to) {
        int size = to - from;
        if (size <= 2) {
            return;
        }
        // order[from] always be the vantage point, so ignore it.
        int vpi = from + rand() % size;
        curNode.vp = order[vpi].pi;
        swap(order[from], order[vpi]);
        auto vpRow = points.row(curNode.vp);
        ++from;
        for (int i = from; i < to; ++i) order[i].dist = vpRow.dist2(points.row(order[i].pi));
        sort(order.begin() + from, order.begin() + to); //todo
        if (order[to - 1].dist == 0) {//degenerated to one point//todo
            return;
        }
        int mid = from + (to - from >> 1);
        curNode.radius1 = order[mid - 1].dist;
        curNode.radius2 = order[mid].dist;
        curNode.lchd = new Node();
        buildTree(*curNode.lchd, from, mid);
        curNode.rchd = new Node();
        buildTree(*curNode.rchd, mid, to);
    }

    //actually the MaxHeap
    struct KnnVisitor {
        int* pis;
        T* dists;
        int k;
        int size;
        int start;

        KnnVisitor(int* index, T* data, int k, int start = -1) :
            k(k), size(0), pis(index), dists(data), start(start) {}

        void push(int pi, T dist) {
            if (pi == start) return;
            if (size == k && dist < dists[0]) {
                dists[0] = dist;
                pis[0] = pi;
                int cur = 0;
                while (true) {
                    int lchd = (cur << 1) + 1;
                    if (lchd >= k) break;
                    int rchd = lchd + 1;
                    int chd = (rchd < k && dists[lchd] < dists[rchd]) ? rchd : lchd;
                    if (dists[chd] <= dists[cur]) break;
                    swap(dists[cur], dists[chd]);
                    swap(pis[cur], pis[chd]);
                    cur = chd;
                }
                return;
            }
            if (size < k) {
                dists[size] = dist;
                pis[size] = pi;
                size++;
                int cur = size - 1;
                while (cur > 0) {
                    int fa = cur - 1 >> 1;
                    if (dists[cur] <= dists[fa]) break;
                    swap(dists[fa], dists[cur]);
                    swap(pis[fa], pis[cur]);
                    cur = fa;
                }
            }
        }

        inline T top() {
            return dists[0];
        }

        bool full() {
            return size == k;
        }
    };

//    struct KnnClassificationVisitor {
//        int* countOfLabel;
//        vector<int> countOfTie;
//        int nLabel;
//        int* label;
//        int k;
//        int size;
//        vector<Pair> heap;
//        int start;
//        bool tieIncluded;
//
//        virtual ~KnnClassificationVisitor() { //todo ?? valid or not
//            for (int i = 0; i < nLabel; ++i) countOfLabel[i] += countOfTie[i];
//        }
//
//        KnnClassificationVisitor(int k, int nLabel, int* label, int* data, int start = -1, bool tieIncluded = false) :
//            k(k), size(0), heap(k), nLabel(nLabel), label(label), countOfLabel(data), countOfTie(nLabel),
//            start(start), tieIncluded(tieIncluded) {
//            memset(countOfLabel, 0, sizeof(int) * nLabel);
//            memset(countOfTie.data(), 0, sizeof(int) * nLabel);
//        }
//
//        void push(int pi, T dist) {
//            if (pi == start) return;
//            if (size >= k) {
//                if (dist > dists[0]) return;
//                if (dist < dists[0]) {
//                    dists[0] = dist;
//                    int cur = 0;
//                    while (true) {
//                        int lchd = (cur << 1) + 1;
//                        if (lchd >= k) break;
//                        int rchd = lchd + 1;
//                        int chd = (rchd < k && dists[lchd] < dists[rchd]) ? rchd : lchd;
//                        if (dists[chd] <= dists[cur]) break;
//                        swap(dists[cur], dists[chd]);
//                        cur = chd;
//                    }
//                    memset(countOfTie.data(), 0, sizeof(int) * nLabel);
//                    countOfTie[label[]]++;
//                    return;
//                }
//            }
//            if (size < k) {
//                dists[size] = dist;
//                size++;
//                int cur = size - 1;
//                while (cur > 0) {
//                    int fa = (cur - 1) >> 1;
//                    if (dists[cur] <= dists[fa]) break;
//                    swap(dists[fa], dists[cur]);
//                    cur = fa;
//                }
//                countOfLabel[label[pi]]++;
//            }
//        }
//
//        T top() {
//            return dists[0];
//        }
//    };


    template <class Visitor>
    void knnTraverse(Node& curNode, int from, int to, const typename Matrix::Row& p, Visitor& visitor) {
        T d = p.dist2(points.row(curNode.vp));
        if (curNode.isLeaf()) {
            for (int i = from; i < to; ++i) visitor.push(order[i].pi, d);
        } else {
            ++from;
            int mid = from + (to - from >> 1);
            visitor.push(curNode.vp, d);
            if (d <= curNode.radius1) {
                knnTraverse(*curNode.lchd, from, mid, p, visitor);
                if (!visitor.full() || visitor.top() <= curNode.radius2 - d)
                    knnTraverse(*curNode.rchd, mid, to, p, visitor);
            } else if (d >= curNode.radius2) {
                    knnTraverse(*curNode.rchd, mid, to, p, visitor);
                if (!visitor.full() || visitor.top() <= d - curNode.radius1)
                    knnTraverse(*curNode.lchd, from, mid, p, visitor);
            }
        }
    }

    void knn(int* index, T* data, int k, const typename Matrix::Row& p) {
        assert(0 < k && k < points.nrow());
        KnnVisitor visitor(index, data, k, -1);
        knnTraverse(root, 0, points.nrow(), p, visitor);
    }

    void knn(int* index, T* data, int k, int pi) {
        assert(0 < k && k < points.nrow());
        assert(0 <= pi && pi < points.nrow());
        KnnVisitor visitor(index, data, k, pi);
        knnTraverse(root, 0, points.nrow(), points.row(pi), visitor);
    }
};
#endif //NLP_CUDA_VPTREE_H
