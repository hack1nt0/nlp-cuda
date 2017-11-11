//
// Created by DY on 17-10-22.
//

#ifndef NLP_CUDA_SIZE_FIXED_HEAP_H
#define NLP_CUDA_SIZE_FIXED_HEAP_H

#include <common_headers.h>
#include <matrix/functors.h>

template <typename T = double>
struct SizeFixedMaxHeap {
    int capacity;
    int size;
    int* index;
    T* value;
    SizeFixedMaxHeap(int capacity, int* index, T* value) :
        capacity(capacity), size(0), index(index), value(value) {}

    void push(int i, T v) {
        if (size >= capacity && v < value[0]) {
            value[0] = v;
            index[0] = i;
            int cur = 0;
            while (true) {
                int lchd = (cur << 1) + 1;
                if (lchd >= capacity) break;
                int rchd = lchd + 1;
                int chd = (rchd < capacity && value[lchd] < value[rchd]) ? rchd : lchd;
                if (value[chd] <= value[cur]) break;
                swap(value[cur], value[chd]);
                swap(index[cur], index[chd]);
                cur = chd;
            }
            return;
        }
        if (size < capacity) {
            value[size] = v;
            index[size] = i;
            size++;
            int cur = size - 1;
            while (cur > 0) {
                int fa = (cur - 1) >> 1;
                if (value[cur] <= value[fa]) break;
                swap(value[fa], value[cur]);
                swap(index[fa], index[cur]);
                cur = fa;
            }
        }
    }

    inline T top() { return value[0]; }
};
#endif
