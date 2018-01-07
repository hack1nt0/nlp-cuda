#ifndef NLP_CUDA_TSNE_H
#define NLP_CUDA_TSNE_H

#include <knn/knn.h>
#include <cfloat>
#include <optimizer/Adam.h>
#include "quadtree.h"

template <typename T = double, class InputType = SparseMatrix<T> >
struct TsneObjective {
    const InputType& X;
    SparseMatrix<T> P;
    vector<T> variance;
    vector<T> nns;// Actual Nearest Neighbors
    int size;
    int dim;
    CDenseMatrix<T>& Y;
    CDenseMatrix<T> grad;


    typedef QuadTree<CDenseMatrix<T> > QuadTreeT;
    struct RepulsiveVisitor {
        QuadTreeT* visited;
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

        inline T QZ(const double Yi[], T* Yj) {
            T QijZ = 0;
            for (int k = 0; k < dim; ++k) {
                QijZ += (Yi[k] - Yj[k]) * (Yi[k] - Yj[k]);
            }
            return T(1) / (T(1) + QijZ);
        }

    };
    vector<RepulsiveVisitor> repVisitor;

    TsneObjective(CDenseMatrix<T>& Y, const InputType& X, int dim, int perplexity, T theta, int seed, bool verbose) :
        X(X),
        P(X.nrow(), X.nrow(), X.nrow() * perplexity * 3),
        size(X.nrow() * dim),
        dim(dim),
        Y(Y),
        grad(X.nrow(), dim), repVisitor(X.nrow()) {

        srand(seed);
        Y = rnorm<T>(X.nrow(), dim);
        for (int i = 0; i < X.nrow(); ++i) {
            repVisitor[i].point = Y.data + i * dim;
            repVisitor[i].FrepZ = new T[dim];
            repVisitor[i].theta = theta;
        }
//        DistMatrix<T> distMatrix(X, verbose);
//        SparseMatrix<T> nn = knn(distMatrix, perplexity * 3, verbose);
//        SparseMatrix<T> nn; nn.read("knn.90.double.in");
        SparseMatrix<T> nn = knnCollect(X, perplexity * 3, seed, verbose);
        if (verbose) {
            nn.println();
            cout << "min(nn) = " << min(nn) << ", max(nn) = " << max(nn) << endl;
        }
        P = nn;

        for (int i = 0; i < P.nrow(); ++i) {
            auto PRow = P.row(i);
            auto nnRow = nn.row(i);
            T mi = T(1e-18);
            T ma = T(1e+18);
            int maxItr = 100;
            T mid;
            T curPerplexity;
            for (int itr = 0; itr < maxItr && mi < ma; ++itr) {
                mid = mi + (ma - mi) / 2;

//                PRow = exp(-(nnRow ^ T(2)) / mid / T(2));
//                PRow = maximum(PRow, DBL_MIN);
//                PRow /= sum(PRow);
//                curPerplexity = exp(-sum(PRow * log(PRow)));

                PRow = -(nnRow ^ T(2)) / mid / T(2);
                T s = sum(exp(PRow)) + DBL_MIN;
                curPerplexity = exp(-sum(exp(PRow) * PRow) / s + log(s));
                PRow = exp(PRow) / s;

                if (abs(curPerplexity - perplexity) < 1e-6 ) break;
                if (curPerplexity != curPerplexity) {

                    nnRow.println();
                    PRow = exp(-(nnRow ^ T(2)) / mid / T(2));
                    PRow.println();
                    T s = sum(PRow);
                    cerr << s << endl;
                    PRow /= sum(PRow);
                    PRow.println();

                    curPerplexity = exp(-sum(PRow * log(PRow)));
                    s = -sum(PRow * log(PRow));
                    cerr << s << ' ' << mid << ' ' << sum(PRow) << endl;
                    cerr << "Found NaN" << endl;
                    exit(1);
                }
                if (curPerplexity < perplexity) mi = mid;
                else ma = mid;
//                cerr << curPerplexity << endl;
            }
            if (verbose) {
                if (variance.size() == 0) {
                    variance.resize(X.nrow());
                    nns.resize(X.nrow());
                }
                variance.at(i) = mid;
                nns.at(i) = curPerplexity;
                if (abs((int) curPerplexity - perplexity) > 1) {
                    printf("%d %.3e %.3e %.3e %.3e %.3e\n", i, mi, ma, mid, curPerplexity, DBL_MIN);
                }
            }
        }
        if (verbose) {
            cout << "Gaussian Kernels:" << endl;
            summary(variance);
            summary(nns);
        }
        P = P + ~P;
        P /= sum(P);
    }

    inline T QZ(int i, int j) {
      T QijZ = 0;
      for (int k = 0; k < dim; ++k) {
          QijZ += (Y.at(i, k) - Y.at(j, k)) * (Y.at(i, k) - Y.at(j, k));
      }
      return T(1) / (T(1) + QijZ);
    }

    T operator()(const CDenseMatrix<T>& Y) {
        double KL = 0;
        double Z = 0;
        for (int i = 0; i < X.nrow(); ++i) Z += repVisitor[i].Z;
        for (int i = 0; i < X.nrow(); ++i) {
            for (int jj = P.row_ptr[i]; jj < P.row_ptr[i + 1]; ++jj) {
                int j = P.index[jj];
                double Pij = P.data[jj];
                double Qij = QZ(i, j) / Z;
//                if (Pij == 0 || Qij == 0) {
//                    cerr << Pij << ' ' << QZ(i, j) << ' ' << Z << endl;
//                }
                KL += Pij * (log(max(Pij, DBL_MIN) / max(Qij, DBL_MIN)));
                if (KL != KL) {
                    cerr << KL << ' ' << Pij << ' ' << QZ(i, j) << ' ' << Z << endl;
                    exit(1);
                }
            }
        }
        return T(KL);
    }

    const CDenseMatrix<T>& caculateGrad(const CDenseMatrix<T>& Y) {
        QuadTreeT quadTree(Y);
        grad = 0.;

#pragma omp parallel for
        for (int i = 0; i < X.nrow(); ++i) {
            repVisitor[i].visited = &quadTree;
            repVisitor[i].Z = 0;
            memset(repVisitor[i].FrepZ, 0, sizeof(T) * dim);
            quadTree.traverse(repVisitor[i]);
            for (int k = 0; k < dim; ++k) grad.at(i, k) -= repVisitor[i].FrepZ[k] / repVisitor[i].Z;
        }

#pragma omp parallel for
        for (int i = 0; i < P.nrow(); ++i) {
            for (int jj = P.row_ptr[i]; jj < P.row_ptr[i + 1]; ++jj) {
                int j = P.index[jj];
                double Pij = P.data[jj];
                double QijZ = QZ(i, j);
                for (int k = 0; k < dim; ++k) grad.at(i, k) += Pij * QijZ * (Y.at(i, k) - Y.at(j, k));
            }
        }
        return grad;
    }


    CDenseMatrix<T>& getParam() {
        return Y;
    }

    CDenseMatrix<T>& getGrad() {
        return grad;
    }

    int getSize() {
        return size;
    }
};

template <typename T = double, class InputType = SparseMatrix<T> >
void tsne(CDenseMatrix<T>& Y, const InputType& X, int dim = 2, int maxItr = 10, int perplexity = 30, T theta = 0.3, int seed = 1, bool verbose = true) {
    TsneObjective<T, InputType> f(Y, X, dim, perplexity, theta, seed, verbose);
    Adam<T, TsneObjective<T, InputType> > optimizer(f);
    optimizer.solve(maxItr, verbose);
};

template <typename T = double, class InputType = SparseMatrix<T> >
CDenseMatrix<T> tsne(const InputType& X, int dim = 2, int maxItr = 10, int perplexity = 30, T theta = 0.3, int seed = 1, bool verbose = true) {
    CDenseMatrix<T> Y(X.nrow(), dim);
    tsne(Y, X, dim, maxItr, perplexity, theta, seed, verbose);
    return Y;
};
#endif
        
    

