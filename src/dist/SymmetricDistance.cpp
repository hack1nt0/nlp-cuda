//
// Created by DY on 17-10-9.
//

#include <CpuTimer.h>
#include "SymmetricDistance.h"
#include <common_headers.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

using namespace tbb;

struct DistTask { //todo bug
    double* D;
    const int* row_ptr;
    const int* index;
    const double* data;
    int rows;

    DistTask(double *D, const int *row_ptr, const int *index, const double *data, int rows) : D(D), row_ptr(row_ptr),
                                                                                              index(index), data(data),
                                                                                              rows(rows) {}

    void operator() ( const blocked_range<size_t>& r ) const {
        for (size_t ti = r.begin(); ti != r.end(); ++ti) {
            /*
               s = ti + 1
               r = rows-1
               (r+r-i+1)*i/2 >= s
               (2*r-i+1)*i >= 2*s
               (2*r+1)i-i^2 >= 2*s
               i^2-(2*r+1)*i+2*s <= 0
               a = 1, b = -2*r-1, c = 2*s
               (-b-sqrt(b^2-4*a*c))/2/a <= i
               */
            int s = ti + 1;
            int r = rows - 1;
            float a = 1;
            float b = -2.f * r - 1;
            float c = 2.f * s;
            int i = (int)ceil((-b - sqrt(b * b - 4 * a * c)) / 2 / a);
            //avoid float error
            i = max(1, min(i, rows));
            if ((2 * r - (i - 1) + 1) * (i - 1) / 2 >= s) i--;
            if ((2 * r - i + 1) * i / 2 < s) i++;
            int rect = (2 * r - (i - 1) + 1) * (i - 1) / 2;
            int j = i + 1 + s - rect - 1;
            i--; j--;

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
void SymmetricDistance<T>::SymmetricDistance(const SparseMatrix<T> &dtm, bool verbose) {
    CpuTimer timer;
    if (verbose) {
        printf("dist...");
        timer.start();
    }
    parallel_for(blocked_range<size_t>(0, dtm.rows() * (dtm.rows() - 1) / 2),
                 DistTask(this->data, dtm.row_ptr(), dtm.index(), dtm.data(), dtm.rows()));
    if (verbose) {
        printf("done. %e ms\n", timer.elapsed());
    }
}