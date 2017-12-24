//
// Created by DY on 17-10-5.
//

#ifndef NLP_CUDA_QUADTREE_H
#define NLP_CUDA_QUADTREE_H

#include <utils.h>

template <class PointMatrix>
struct QuadTree {
    const PointMatrix& points;
    struct Node {
        int capacity = 0;
        vector<int> members;
        bool isLeaf = false;
        double center[2] = {0., 0.};
        Node* child[4] = {null, null, null, null};

        virtual ~Node() {
            for (int i = 0; i < 4; ++i) if (child[i] != null) delete child[i];
        }

        template <class Point>
        inline double dist(const Point& point) const {
            return sqrt((point[0] - center[0]) * (point[0] - center[0]) +
                        (point[1] - center[1]) * (point[1] - center[1]));
        }
    };
    Node root;
    int nodes = 0, leaves = 0;
    pair<double, double> bound[2];

    QuadTree(const PointMatrix& pointMatrix) : points(pointMatrix) {
        int n = points.nrow();
        assert(n > 0);
        double maxSpan = 0;
        for (int i = 0; i < 2; ++i) {
            bound[i].first = bound[i].second = points.at(0, i);
            for (int j = 1; j < n; ++j) {
                bound[i].first = min(bound[i].first, points.at(j, i));
                bound[i].second = max(bound[i].second, points.at(j, i));
            }
            maxSpan = max(maxSpan, bound[i].second - bound[i].first);
        }
        for (int i = 0; i < 2; ++i) {
            bound[i].first -= (maxSpan - (bound[i].second - bound[i].first)) / 2;
            bound[i].second += (maxSpan - (bound[i].second - bound[i].first)) / 2;
        }
        for (int i = 0; i < n; ++i)
            insert(i, root);
        NormCenterVisitor normCenterVisitor;
        traverse(normCenterVisitor);
    }

    void insert(int point, Node& curNode) {
        curNode.capacity++;
        for (int i = 0; i < 2; ++i) curNode.center[i] += points.at(point, i);
        if (curNode.capacity == 1) {
            curNode.members.push_back(point);
            curNode.isLeaf = true;
            leaves++;
            nodes++;
            return;
        }
        if (curNode.capacity > 1 && curNode.isLeaf) {
            curNode.members.push_back(point);
            double split[2];
            for (int i = 0; i < 2; ++i) split[i] = bound[i].first + (bound[i].second - bound[i].first) / 2;
            for (auto p : curNode.members) {
                int i = (points.at(p, 0) > split[0]) | ((points.at(p, 1) > split[1]) << 1);
                if (curNode.child[i] == null) {
                    curNode.child[i] = new Node();
                    curNode.child[i]->isLeaf = true;
                    leaves++;
                    nodes++;
                }
                Node& child = *curNode.child[i];
                child.capacity++;
                child.members.push_back(p);
                for (int i = 0; i < 2; ++i) child.center[i] += points.at(p, i);
            }
            curNode.members.clear(); //todo
            curNode.isLeaf = false;
            leaves--;
            return;
        }
        pair<double, double> tmp[2];
        memcpy(tmp, bound, sizeof(pair<double, double>) * 2);
        int grid = 0;
        for (int i = 0; i < 2; ++i) {
            double split = bound[i].first + (bound[i].second - bound[i].first) / 2;
            if (points.at(point, i) <= split) {
                bound[i].second = split;
                grid |= 0 << i;
            } else {
                bound[i].first = split;
                grid |= 1 << i;
            }
        }
        if (curNode.child[grid] == null) {
            curNode.child[grid] = new Node();
        }
        insert(point, *curNode.child[grid]);
        memcpy(bound, tmp, sizeof(pair<double, double>) * 2);
    }

    template <class Point>
    void nearbyNodes(vector<Node&> ns, const Point& point) {
        NearbyNodesVisitor<Point> visitor(point, ns, bound);
        traverse(visitor);
    }

    template <class Visitor>
    void traverse(Visitor& visitor) {
        traverse(visitor, root);
    }

    template <class Visitor>
    void traverse(Visitor& visitor, Node& curNode) {
        if (visitor.visit(curNode) || curNode.isLeaf) return;
        pair<double, double> tmp[2];
        memcpy(tmp, bound, sizeof(pair<double, double>) * 2);
        double split[2];
        for (int i = 0; i < 2; ++i) split[i] = bound[i].first + (bound[i].second - bound[i].first) / 2;
        for (int i = 0; i < 4; ++i) {
            if (curNode.child[i] == null) continue;
            for (int j = 0; j < 2; ++j) {
                if ((i >> j & 1) == 0) bound[j].second = split[j];
                else bound[j].first = split[j];
            }
            traverse(visitor, *curNode.child[i]);
            memcpy(bound, tmp, sizeof(pair<double, double>) * 2);
        }
    }

    struct NormCenterVisitor {
        bool visit(Node& curNode) {
            for (int i = 0; i < 2; ++i) curNode.center[i] /= curNode.capacity;
            return false;
        }
    };

    template <class Point>
    struct NearbyNodesVisitor {
        const Point& point;
        double theta = 0.5;
        const pair<double, double>* bound;
        vector<Node&>& ns;

        NearbyNodesVisitor(vector<Node &> &ns, const Point& point,
                           const pair<double, double>* bound) : ns(ns), point(point), bound(bound) {}

        bool visit(Node& curNode) {
            if (curNode.isLeaf || (bound[0].second - bound[0].first) / curNode.dist(point) < theta) {
                ns.push_back(curNode);
                return true;
            }
            return false;
        }
    };

};

#endif //NLP_CUDA_QUADTREE_H
