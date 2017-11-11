//
// Created by DY on 17-10-9.
//

#include <common_headers.h>
#include "DistMatrix.h"

template <typename T>
void DistMatrix<T>::init(const SparseMatrix<T> &dtm, bool verbose) {
    CpuTimer timer;
    if (verbose) {
        printf("dist.omp...");
        fflush(stdout);
        timer.start();
    }

#pragma omp parallel for
    for (int ti = 0; ti < this->nnz; ++ti) {
        int i, j;
        DistMatrix<T>::unzip(i, j, ti, this->rows);
        this->data[ti] = dtm.row(i).dist2(dtm.row(j));
    }
    if (verbose) {
        printf("done. %e ms\n", timer.elapsed());
        fflush(stdout);
    }
}

template class DistMatrix<double>;

