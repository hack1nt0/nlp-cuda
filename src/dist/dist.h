//
// Created by DY on 17-10-9.
//

#ifndef NLP_CUDA_DIST_MATRIX_H
#define NLP_CUDA_DIST_MATRIX_H

#include "../matrix/matrix.h"
#include "../utils/utils.h"

template <typename V, typename I>
struct DistanceUtils {
    typedef V                             value_t;
    typedef I                             index_t;
    typedef DenseVector<V, I>             dv_t;
    typedef SparseVector<V, I>            sv_t;

    template <class LHS, class RHS>
    static value_t dist(const LHS& lhs, const RHS& rhs, int kind) {
        switch (kind) {
            case 0: return squaredEuclidean(lhs, rhs);
            case 1: return euclidean(lhs, rhs);
            default: throw runtime_error("Not implemented dist kind.");
        }
    };

    static value_t squaredEuclidean(const dv_t& a, const dv_t& b) {
        assert(a.getNnz() == b.getNnz());
        value_t r = 0;
        for (index_t i = 0; i < a.getNnz(); ++i) r += (a.at(i) - b.at(i)) * (a.at(i) - b.at(i));
        return r;
    }
    static inline value_t euclidean(const dv_t& a, const dv_t& b) { return std::sqrt(squaredEuclidean(a, b)); }

    static value_t squaredEuclidean(const sv_t& a, const sv_t& b) {
        value_t r = 0;
        index_t i = 0;
        index_t oi = 0;
        while (i < a.nnz && oi < b.nnz) {
            if (a.index[i] == b.index[oi]) {r += (a.at(i) - b.at(oi)) * (a.at(i) - b.at(oi)); ++i; ++oi; }
            else if (a.index[i] < b.index[oi]) {r += (a.at(i)) * (a.at(i)); ++i; }
            else {r += (b.at(oi)) * (b.at(oi)); ++oi; }
        }
        while (i < a.nnz) {r += (a.at(i)) * (a.at(i)); ++i;}
        while (oi < b.nnz) {r += (b.at(oi)) * (b.at(oi)); ++oi;}
        return r;
    }
    static inline value_t euclidean(const sv_t& a, const sv_t& b) { return std::sqrt(squaredEuclidean(a, b)); }

    static value_t squaredEuclidean(const dv_t& a, const sv_t& b) {
        value_t r = 0;
        index_t i = 0;
        index_t j = 0;
        while (i < a.getNnz()) {
            value_t vi = a[i];
            value_t vj = j < b.getNnz() && i == b.index[j] ? b[j++] : 0;
            r += (vi - vj) * (vi - vj);
            ++i;
        }
        return r;
    }
    static inline value_t squaredEuclidean(const sv_t& a, const dv_t& b) { return squaredEuclidean(b, a); }
    static inline value_t euclidean(const dv_t& a, const sv_t& b) { return std::sqrt(squaredEuclidean(a, b)); }
    static inline value_t euclidean(const sv_t& a, const dv_t& b) { return std::sqrt(squaredEuclidean(a, b)); }
};

template <typename index_t>
inline index_t zip(index_t s, index_t t, unsigned long long rows) {
    s++; t++;
    //rows - 1 + ... + rows - s + 1 + (t - (s + 1) + 1)
    //(2*rows - s) * (s - 1) / 2 + t - s
    return (rows * 2 - s) * (s - 1) / 2 + t - s - 1;
}

template <typename index_t>
inline pair<index_t, index_t> unzip(index_t ii, index_t rows) {
    /*
       s = ii + 1
       r = rows-1
       (r+r-i+1)*i/2 >= s
       (2*r-i+1)*i >= 2*s
       (2*r+1)i-i^2 >= 2*s
       i^2-(2*r+1)*i+2*s <= 0
       a = 1, b = -2*r-1, c = 2*s
       (-b-sqrt(b^2-4*a*c))/2/a <= i
     */

    index_t s = ii + 1;
    index_t r = rows - 1;
    double a = 1;
    double b = -2. * r - 1;
    double c = 2. * s;
    index_t i = (index_t)ceil((-b - sqrt(b * b - 4 * a * c)) / 2 / a);
    //avoid float error
    i = max(1, min(i, rows));
    if ((2 * r - (i - 1) + 1) * (i - 1) / 2 >= s) i--;
    if ((2 * r - i + 1) * i / 2 < s) i++;
    index_t rect = (2 * r - (i - 1) + 1) * (i - 1) / 2;
    index_t j = i + 1 + s - rect - 1;
    i--; j--;
    return {i, j};
}

template <class Dist, class Mat>
void dist(Dist& D, const Mat& x, int kind, bool verbose) {
    typedef typename Mat::value_t value_t;
    typedef typename Mat::index_t index_t;
    typedef DistanceUtils<value_t, index_t> DU;
    index_t rows = x.nrow();
    index_t tot = rows * (rows - 1) / 2;
    ProgressBar bar(tot);
//#pragma omp parallel for
//    for (index_t i = 0; i < tot; ++i) {
//        pair<index_t, index_t> p = unzip(i, n);
//        D[i] = dist(x.row(p.first), x.row(p.second), kind);
//        if (verbose) bar.increase();
//        if (bar.interrupted()) throw new interrupt_exception("[SIGINT]");
//    }

#pragma omp parallel for
    for (index_t i = 0; i < rows; ++i) {
        auto ri = x.row(i);
        for (index_t j = i + 1; j < rows; ++j) {
            index_t ii = zip(i, j, rows);
            if (ii < 0) throw new invalid_argument("<0");
            D[ii] = DU::dist(ri, x.row(j), kind);
            if (verbose) bar.increase();
            if (bar.interrupted()) throw new interrupt_exception("[SIGINT]");
        }
    }
};

template <typename V = double, typename I = int>
struct DistMatrix : CDenseMatrix<V, I> {
    typedef V                             value_t;
    typedef I                             index_t;
    typedef CDenseMatrix<value_t, index_t> super_t;
    typedef DistMatrix<value_t, index_t>  self_t;

    DistMatrix() {}

    DistMatrix(value_t* value, index_t rows) {
        this->value = value;
        this->rows = rows;
        this->cols = rows;
        this->nnz = rows * (rows - 1) / 2;
        this->needFree = false;
    }

    DistMatrix(index_t n) {
        this->rows = n;
        this->cols = n;
        this->nnz = n * (n - 1) / 2;
        this->value = new value_t[this->nnz];
        this->needFree = true;
    }

    inline value_t at(int s, int t) const {
        assert(0 <= s && s < this->rows && 0 <= t && t < this->rows);
        if (s == t) return V(0);
        if (s > t) swap(s, t);
        s++; t++;
        //rows - 1 + ... + rows - s + 1 + (t - (s + 1) + 1)
        //(2*rows - s) * (s - 1) / 2 + t - s
        int i = (2 * this->rows - s) * (s - 1) / 2 + t - s - 1;
        assert(0 <= i && i < this->nnz);
        return this->data[i];
    }

    friend ostream &operator<<(ostream &os, const DistMatrix<V, I> &matrix) {
        os << "DistMatrix [rows, cols] = [" << matrix.rows << ", " << matrix.rows << "]" << endl;
        for (int i = 0; i < min(10, matrix.rows); ++i) {
            for (int j = 0; j < min(10, matrix.rows); ++j) {
                os << setiosflags(ios::scientific) << matrix.at(i, j) << '\t';
            }
            os << endl;
        }
        return os;
    }

    vector<pair<V, double> > summary(bool exact, V mi=0, V ma=-1) {
        vector<pair<V, double> > split(5);
        if (exact) {
            V* data2 = new value_t[this->nnz];
            memcpy(data2, this->data, sizeof(value_t) * this->nnz);
            sort(data2, data2 + this->nnz);
            int perSize = this->nnz / 5;
            for (int i = 0; i < 5; ++i) split[i].first = split[i].second = data2[i * perSize];
            split[4].first = split[4].second = data2[this->nnz - 1];
            delete[](data2);
            return split;
        }
        if (ma <= mi) {
            mi = ma = this->data[0];
            for (int i = 1; i < this->nnz; ++i) {
                mi = min(mi, this->data[i]);
                ma = max(ma, this->data[i]);
            }
        }
        double perSize = (ma - mi) / 4.;
        for (int i = 0; i < 5; ++i) {
            split[i].first = mi + perSize * i;
            split[i].second = 0;
        }
        split[4].first = ma;
        for (int i = 1; i < this->nnz; ++i) {
            for (int j = 0; j < 4; ++j) if (this->data[i] < split[j + 1].first) {
                    split[j].second += 1;
                    break;
                }
        }
        cout << "hi"  << endl;
        return split;
    }

};



#endif //NLP_CUDA_SYMMETRICDISTANCE_H
