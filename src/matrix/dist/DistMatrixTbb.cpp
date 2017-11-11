//
// Created by DY on 17-10-9.
//

#include <common_headers.h>
#include "matrix/SparseMatrix.h"
#include "DistMatrix.h"
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

using namespace tbb;

template <typename T>
struct DistTask { //todo bug
    T* D;
    const int* row_ptr;
    const int* index;
    const T* data;
    int rows;

    DistTask(double *D, const int *row_ptr, const int *index, const double *data, int rows) : D(D), row_ptr(row_ptr),
                                                                                              index(index), data(data),
                                                                                              rows(rows) {}

    void operator() ( const blocked_range<size_t>& r ) const {
        for (size_t ti = r.begin(); ti != r.end(); ++ti) {
            int i, j;
            DistMatrix<T>::unzip(i, j, ti, rows);
            double d = 0;
            int p = row_ptr[i];
            int q = row_ptr[j];
            while (p < row_ptr[i + 1] && q < row_ptr[j + 1]) {
                int cmp = max(-1, min(index[p] - index[q], 1));
                switch (cmp) {
                    case 0:
                        d += (data[p] - data[q]) * (data[p] - data[q]);
                        p++;
                        q++;
                        break;
                    case 1:
                        d += data[q] * data[q];
                        q++;
                        break;
                    case -1:
                        d += data[p] * data[p];
                        p++;
                }
            }
            while (p < row_ptr[i + 1]) {
                d += data[p] * data[p];
                p++;
            }
            while (q < row_ptr[j + 1]) {
                d += data[q] * data[q];
                q++;
            }
            D[ti] = d;
        }
    }
};

template <typename T>
void DistMatrix<T>::init(const SparseMatrix<T> &dtm, bool verbose) {
    CpuTimer timer;
    if (verbose) {
        printf("dist.tbb...");
        timer.start();
    }
    this->rows = dtm.nrow();
    this->cols = dtm.nrow();
    this->nnz = dtm.nrow() * (dtm.nrow() - 1) / 2;
    this->data = new T[nnz]; //todo enable external storage
    this->needFree = false;
    parallel_for(blocked_range<size_t>(0, dtm.rows * (dtm.rows - 1) / 2),
                 DistTask<T>(this->data, dtm.row_ptr, dtm.index, dtm.data, dtm.rows));
    if (verbose) {
        printf("done. %e ms\n", timer.elapsed());
    }
}

template class DistMatrix<double>;
