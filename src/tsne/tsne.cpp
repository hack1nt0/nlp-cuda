//
// Created by DY on 17-10-9.
//


#include "tsne.h"
#include <common_headers.h>
#include <quadtree.h>
#include <CpuTimer.h>
#include "matrix/DenseMatrix.h"

template <typename T>
DenseMatrix<T> Tsne<T>::tsne(int dim, int perplexity, T theta, int max_itr, int seed) {
    assert(dim == 2 || dim == 3);
    srand(seed);
    DenseMatrix<T> Y = rnorm<T>(this->X.nrow(), dim);
    DenseMatrix<T> dY(X.nrow(), dim);

    typedef QuadTree<DenseMatrix<T> > QuadTreeT;
    struct RepulsiveVisitor {
        QuadTree<DenseMatrix<T> >* visited;
        T* point;
        T* FrepZ;
        T Z;
        T theta;
        int dim;

        virtual ~RepulsiveVisitor() {
            delete[] FrepZ;
        }

        bool visit(const typename QuadTreeT::Node& curNode) {
            if (curNode.isLeaf || (visited->bound[0].second - visited->bound[0].first) / curNode.dist(point) < theta) {
                T QijZ = QZ(curNode.center, point);
                for (int i = 0; i < dim; ++i) FrepZ[i] += QijZ * QijZ * (point[i] - curNode.center[i]) * curNode.capacity;
                Z += QijZ * curNode.capacity;
                return true;
            }
            return false;
        }

        inline T QZ(T* Yi, T* Yj) {
            T QijZ = 0;
            for (int k = 0; k < dim; ++k) {
                QijZ += (Yi[k] - Yj[k]) * (Yi[k] - Yj[k]);
            }
            return T(1) / (T(1) + QijZ);
        };

    };

    vector<RepulsiveVisitor*> repVisitor(X.nrow());
    for (int i = 0; i < X.nrow(); ++i) {
        repVisitor[i] = new RepulsiveVisitor();
        repVisitor[i]->point = Y.data[i * dim];
        repVisitor[i]->FrepZ = new T[dim];
        memset(repVisitor[i]->FrepZ, 0, sizeof(T) * dim);
        repVisitor[i]->Z = 0;
        repVisitor[i]->theat = theta;
    }

    auto QZ = [&](int i, int j) -> T {
        T QijZ = 0;
        for (int k = 0; k < dim; ++k) {
            QijZ += (Y.at(i, k) - Y.at(j, k)) * (Y.at(i, k) - Y.at(j, k));
        }
        return T(1) / (T(1) + QijZ);
    };

    T lr = 0.5;

    CpuTimer timer;
    for (int itr = 0; itr < max_itr; ++itr) {

        dY = 0;
        QuadTreeT quadTree(Y);
        for (int i = 0; i < X.nrow(); ++i) {
            repVisitor[i]->visited = &quadTree;
            quadTree.traverse(*repVisitor[i]);
            for (int k = 0; k < dim; ++k) dY.at(i, k) -= repVisitor[i]->FrepZ[k] / repVisitor[i]->Z;
        }
        delete(&quadTree);
        for (int i = 0; i < X.nrow(); ++i) {
            for (int jj = P.row_ptr[i]; jj < P.row_ptr[i + 1]; ++jj) {
                int j = P.index[jj];
                double Pij = P.data[jj];
                double QijZ = QZ(i, j);
                for (int k = 0; k < dim; ++k) dY.at(i, k) += Pij * QijZ * (Y.at(i, k) - Y.at(j, k));
            }
        }
        if (itr % 10 == 0) {
            if (itr == 0) {
                printf("Itr\tKL-divergence\tcost(ms)\n");
                timer.start();
            }
            T KL = 0;
            T Z = 0;
            for (int i = 0; i < X.nrow(); ++i) Z += repVisitor[i]->Z;
            for (int i = 0; i < X.nrow(); ++i) {
                for (int jj = P.row_ptr[i]; jj < P.row_ptr[i + 1]; ++jj) {
                    int j = P.index[jj];
                    double Pij = P.data[jj];
                    KL += Pij * (log(Pij) - log(QZ(i, j) / Z));
                }
            }
            printf("%d\t%e\t%e\n", itr, KL, timer.elapsed());
            timer.start();
        }

        dY /= T(4);
        Y -= dY * lr;
    }
}
