//
// Created by DY on 17-10-9.
//

#ifndef NLP_CUDA_DIST_MATRIX_H
#define NLP_CUDA_DIST_MATRIX_H

#include "../matrix/matrix.h"

template <class LHS, class RHS>
typename LHS::value_t dist(const LHS& lhs, const RHS& rhs, int k = 1) {
    switch (k) {
        case 0: return lhs.squaredEuclideanDist(rhs);
        case 1: return lhs.euclideanDist(rhs);
        default: throw runtime_error("Not implemented dist kind.");
    }
};


//template <typename V, typename I>
//struct DistMatrix : DenseMatrix<V, I> {
//    typedef DenseMatrix<V, I> super_t;
//    typedef DistMatrix<V, I>  self_t;
//    typedef V                 value_t;
//    typedef I                 index_t;
//
//    DistMatrix() {}
//
//    DistMatrix(T* data, const SparseMatrix<T>& dtm, bool verbose) {
//        this->data = data;
//        this->rows = dtm.nrow();
//        this->cols = dtm.nrow();
//        this->nnz = dtm.nrow() * (dtm.nrow() - 1) / 2;
//        this->needFree = false;
//        init(dtm, verbose);
//    }
//
//    DistMatrix(const SparseMatrix<T>& dtm, bool verbose) {
//        this->rows = dtm.nrow();
//        this->cols = dtm.nrow();
//        this->nnz = dtm.nrow() * (dtm.nrow() - 1) / 2;
//        this->data = new T[this->nnz];
//        this->needFree = true;
//        init(dtm, verbose);
//    }
//
//    void init(const SparseMatrix<T>& dtm, bool verbose);
//    inline value_t at(int s, int t) const {
//        assert(0 <= s && s < this->rows && 0 <= t && t < this->rows);
//        if (s == t) return T(0);
//        if (s > t) swap(s, t);
//        s++; t++;
//        //rows - 1 + ... + rows - s + 1 + (t - (s + 1) + 1)
//        //(2*rows - s) * (s - 1) / 2 + t - s
//        int i = (2 * this->rows - s) * (s - 1) / 2 + t - s - 1;
//        assert(0 <= i && i < this->nnz);
//        return this->data[i];
//    }
//
//    static inline void unzip(int& i, int& j, int ii, int rows) {
//        /*
//           s = ii + 1
//           r = rows-1
//           (r+r-i+1)*i/2 >= s
//           (2*r-i+1)*i >= 2*s
//           (2*r+1)i-i^2 >= 2*s
//           i^2-(2*r+1)*i+2*s <= 0
//           a = 1, b = -2*r-1, c = 2*s
//           (-b-sqrt(b^2-4*a*c))/2/a <= i
//         */
//
//        int s = ii + 1;
//        int r = rows - 1;
//        double a = 1;
//        double b = -2. * r - 1;
//        double c = 2. * s;
//        i = (int)ceil((-b - sqrt(b * b - 4 * a * c)) / 2 / a);
//        //avoid float error
//        i = max(1, min(i, rows));
//        if ((2 * r - (i - 1) + 1) * (i - 1) / 2 >= s) i--;
//        if ((2 * r - i + 1) * i / 2 < s) i++;
//        int rect = (2 * r - (i - 1) + 1) * (i - 1) / 2;
//        j = i + 1 + s - rect - 1;
//        i--; j--;
//    }
//
//    friend ostream &operator<<(ostream &os, const dist<T> &matrix) {
//        os << "DistMatrix [rows, cols] = [" << matrix.rows << ", " << matrix.rows << "]" << endl;
//        for (int i = 0; i < min(10, matrix.rows); ++i) {
//            for (int j = 0; j < min(10, matrix.rows); ++j) {
//                os << setiosflags(ios::scientific) << matrix.at(i, j) << '\t';
//            }
//            os << endl;
//        }
//        return os;
//    }
//
//    vector<pair<T, double> > summary(bool exact, T mi=0, T ma=-1) {
//        vector<pair<T, double> > split(5);
//        if (exact) {
//            T* data2 = new T[this->nnz];
//            memcpy(data2, this->data, sizeof(T) * this->nnz);
//            sort(data2, data2 + this->nnz);
//            int perSize = this->nnz / 5;
//            for (int i = 0; i < 5; ++i) split[i].first = split[i].second = data2[i * perSize];
//            split[4].first = split[4].second = data2[this->nnz - 1];
//            delete[](data2);
//            return split;
//        }
//        if (ma <= mi) {
//            mi = ma = this->data[0];
//            for (int i = 1; i < this->nnz; ++i) {
//                mi = min(mi, this->data[i]);
//                ma = max(ma, this->data[i]);
//            }
//        }
//        double perSize = (ma - mi) / 4.;
//        for (int i = 0; i < 5; ++i) {
//            split[i].first = mi + perSize * i;
//            split[i].second = 0;
//        }
//        split[4].first = ma;
//        for (int i = 1; i < this->nnz; ++i) {
//            for (int j = 0; j < 4; ++j) if (this->data[i] < split[j + 1].first) {
//                    split[j].second += 1;
//                    break;
//                }
//        }
//        cout << "hi"  << endl;
//        return split;
//    }
//
//};



#endif //NLP_CUDA_SYMMETRICDISTANCE_H
