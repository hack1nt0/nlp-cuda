#ifndef NLP_CUDA_TSNE_H
#define NLP_CUDA_TSNE_H

#include <matrix/DenseMatrix.h>
#include <quadtree.h>
#include <knn.h>

void tsne(double* Y, int* landmarks, int newRows, int newCols,
          const double* data, const int* index, const int* row_ptr,
          int rows, int cols, int nnz,
          const int* belong, int classes,
          int perplexity, int max_itr, unsigned int seed);

template <typename T>
struct Tsne {
    const SparseMatrix<T>& X;
    SparseMatrix<T> P;

    DenseMatrix<T> variance;

    Tsne(const SparseMatrix<T> &X, T perplexity, bool verbose=true) : X(X),
                                                                      P(X.nrow(), X.nrow(), X.nrow() * perplexity),
                                                                      variance(1, X.nrow()) {
        SparseMatrix<T> nn(X.nrow(), X.nrow(), perplexity * X.nrow());
        Knn<T> knn(X, verbose);
        nn = knn.knn(perplexity);
        for (int i = 0; i < P.nrow(); ++i) {
            auto PRow = P.row(i);
            auto nnRow = nn.row(i);
            T minVar = T(1e-9);
            T maxVar = T(1e+9);
            int maxItr = 300;
            for (int itr = 0; itr < maxItr && minVar < maxItr; ++itr) {
                T midVar = minVar + (maxVar - minVar) / 2;
                PRow = exp(-nnRow ^ T(2) / midVar / T(2));
                PRow /= sum(PRow);
                T curPerplexity = exp(-sum(PRow * log(PRow)));
                if (curPerplexity < perplexity) minVar = midVar;
                else maxVar = midVar;
            }
            printf("%d %e %e\n", i, minVar, maxVar);
            variance.at(i) = minVar;
        }
        P = P + ~P;
        P /= T(2);
        P /= sum(P);
    }

    DenseMatrix<T> tsne(int dim, int perplexity, T theta, int max_itr, int seed);


};
#endif
        
    

