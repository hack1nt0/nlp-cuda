//
// Created by DY on 17-9-27.
//

#ifndef NLP_CUDA_KMEANS_H
#define NLP_CUDA_KMEANS_H

#include <common_headers.h>

template <typename T>
void initKmeans(T* h_mean, unsigned int k,
                const T* data, const int* index, const int* row_ptr,
                unsigned int rows, unsigned int cols, unsigned int nnz, unsigned int seed) {
    srand(seed);
//    for (int i = 0; i < k * cols; ++i) h_mean[i] = (double)rand() / RAND_MAX;
    T* minDist = new T[rows];
    for (int i = 0; i < rows; ++i) minDist[i] = 3e300;
    int c = rand() % rows;
    minDist[c] = 0;
    for (int i = row_ptr[c]; i < row_ptr[c + 1]; ++i) h_mean[index[i]] = data[i];
    for (int i = 1; i < k; ++i) {
        int nc = c;
        for (int j = 0; j < rows; ++j) {
            if (minDist[j] == 0) continue; //centroid
            T dist = 0;
            int p = row_ptr[c];
            int q = row_ptr[j];
            while (p < row_ptr[c + 1] && q < row_ptr[j + 1]) {
                if (index[p] == index[q]) {
                    dist += (data[p] - data[q]) * (data[p] - data[q]);
                    p++; q++;
                } else if (index[p] < index[q]) {
                    dist += data[p] * data[p];
                    p++;
                } else if (index[p] > index[q]) {
                    dist += data[q] * data[q];
                    q++;
                }
            }
            while (p < row_ptr[c + 1]) {
                dist += data[p] * data[p];
                p++;
            }
            while (q < row_ptr[j + 1]) {
                dist += data[q] * data[q];
                q++;
            }
            minDist[j] = min(minDist[j], dist);
            if (minDist[j] > minDist[nc]) nc = j;
        }
        assert(minDist[nc] > 0);
        c = nc;
        for (int j = row_ptr[c]; j < row_ptr[c + 1]; ++j) h_mean[i * cols + index[j]] = data[j];
        minDist[nc] = 0;
    }
    delete[] minDist;
}

template <typename T>
vector<T> kmeans(T* h_mean, int* h_class,
                 const T* data, const int* index, const int* row_ptr,
                 unsigned int rows, unsigned int cols, unsigned int nnz,
                 unsigned int k, unsigned int max_itr, unsigned int seed);
#endif //NLP_CUDA_KMEANS_H
