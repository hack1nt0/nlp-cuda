//
// Created by DY on 17-10-9.
//

#ifndef NLP_CUDA_SYMMETRICDISTANCE_H
#define NLP_CUDA_SYMMETRICDISTANCE_H

#include <common_headers.h>
#include <ds/DocumentTermMatrix.h>

template <typename T>
struct SymmetricDistance {
    T* data;
    int rows;
    int size;
    bool needFree;

    virtual ~SymmetricDistance() {
        delete[] this->data;
    }

    SymmetricDistance(const SparseMatrix<T>& dtm, bool verbose);


    inline
    T at(int s, int t) const {
        assert(0 <= s && s < this->rows && 0 <= t && t < this->rows);
        if (s == t) return T(0);
        if (s > t) swap(s, t);
        s++; t++;
        //rows - 1 + ... + rows - s + 1 + (t - (s + 1) + 1)
        //(2*rows - s) * (s - 1) / 2 + t - s
        int i = (2 * this->rows - s) * (s - 1) / 2 + t - s - 1;
        assert(0 <= i && i < size);
        return this->data[i];
    }

    friend ostream &operator<<(ostream &os, const SymmetricDistance<T> &matrix) {
        os << "HostSymmetricMatrix [rows, cols] = [" << matrix.rows << ", " << matrix.rows << "]" << endl;
        for (int i = 0; i < min(10, matrix.rows); ++i) {
            for (int j = 0; j < min(10, matrix.rows); ++j) {
                printf("%e\t", matrix.at(i, j));
            }
            os << endl;
        }
        return os;
    }

};



#endif //NLP_CUDA_SYMMETRICDISTANCE_H
