//
// Created by DY on 17-10-9.
//

#include <utils.h>
#include "dist.h"

template <typename T>
void dist<T>::init(const SparseMatrix<T> &dtm, bool verbose) {
    CpuTimer timer;
    if (verbose) {
        printf("dist.omp...");
        fflush(stdout);
        timer.start();
    }

#pragma omp parallel for
    for (int ti = 0; ti < this->nnz; ++ti) {
        int i, j;
        dist<T>::unzip(i, j, ti, this->rows);
        this->data[ti] = dtm.row(i).dist2(dtm.row(j));
    }
    if (verbose) {
        printf("done. %e ms\n", timer.elapsed());
        fflush(stdout);
    }
}

template class dist<double>;

