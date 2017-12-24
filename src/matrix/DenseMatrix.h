//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_DENSE_MATRIX_H
#define NLP_CUDA_DENSE_MATRIX_H

#include "DenseExpr.h"
#include "SparseMatrix.h"

/**
 * Shallow copy, Col Major matrix
 */

template <class V = double, typename I = int>
class DenseMatrix : public DenseExpr<V, I, DenseMatrix<V, I> > {
  public:
    typedef DenseMatrix<V, I> self_t;
    typedef V                 value_t;
    typedef I                 index_t;
    index_t rows = 0;
    index_t cols = 0;
    value_t* value = nullptr;
    bool needFree = false;

    virtual ~DenseMatrix() {
        if (needFree) {
            delete[] value;
        }
    }

    DenseMatrix() {}

    DenseMatrix(const self_t& o) {
        this->rows = o.rows;
        this->cols = o.cols;
        this->value = o.value;
        this->needFree = false;
    }

    DenseMatrix(self_t&& o) {
        this->~DenseMatrix();
        this->rows = o.rows;
        this->cols = o.cols;
        this->value = o.value;
        this->needFree = o.needFree;
        o.value = nullptr;
    }

    DenseMatrix(index_t rows, index_t cols, value_t* value, bool needFree = false) {
        this->rows = rows;
        this->cols = cols;
        this->value = value;
        this->needFree = needFree;
    }

    DenseMatrix(index_t rows, index_t cols) : rows(rows), cols(cols), value(new value_t[rows * cols]), needFree(true) {}

    self_t& operator=(const self_t& o) {
        for (index_t i = 0; i < rows; ++i)
            for (index_t j = 0; j < cols; ++j) at(i, j) = o.at(i, j);
        return *this;
    }

    inline const value_t& at(index_t r, index_t c) const {
        return value[c * rows + r];
    }

    inline value_t& at(index_t r, index_t c) {
        return value[c * rows + r];
    }

    inline const value_t& at(index_t i) const {
        return value[i];
    }

    inline value_t& at(index_t i) {
        return value[i];
    }

    inline const value_t& operator[](index_t i) const {
        return value[i];
    }

    inline value_t& operator[](index_t i) {
        return value[i];
    }

    inline index_t nrow() const {return rows;}

    inline index_t ncol() const {return cols;}

    inline index_t getNnz() const {return rows * cols;}

    /** Utils methods **/

    template <class Etype>
    self_t& operator=(const DenseExpr<V, I, Etype>& e) {
        auto ee = e.self();
#pragma omp parallel for
        for (index_t c = 0; c < cols; ++c) {
            for (index_t r = 0; r < rows; ++r) {
                at(r, c) = ee.at(r, c);
            }
        }
        return *this;
    }

    self_t& operator=(value_t v) {
        //if (v == 0 || v == -1) memset(value, v, sizeof(value_t) * getNnz());
        *this = ConstDenseExpr<V, I>(v);
        return *this;
    }

    template <class Etype>
    self_t& operator+=(const DenseExpr<V, I, Etype>& e) {
        *this = *this + e;
        return *this;
    }
    template <class Etype>
    self_t& operator-=(const DenseExpr<V, I, Etype>& e) {
        *this = *this - e;
        return *this;
    }

    template <class Etype>
    self_t& operator*=(const DenseExpr<V, I, Etype>& e) {
        *this = *this * e;
        return *this;
    }

    template <class Etype>
    self_t& operator/=(const DenseExpr<V, I, Etype>& e) {
        *this = *this / e;
        return *this;
    }

    void print(bool head = true) const {
        if (head) std::printf("DenseMat %d x %d\n", rows, cols);
        for (index_t i = 0; i < rows; ++i) row(i).print(false);
    }

    void save(ofstream& s) {
        s.write((char*)(&rows), sizeof(rows));
        s.write((char*)(&cols), sizeof(cols));
        s.write((char*)value,   sizeof(value_t));
    }

    void read(ifstream& s) {
        s.read((char*)(&rows), sizeof(rows));
        s.read((char*)(&cols), sizeof(cols));
        s.read((char*)value,   sizeof(value_t));
    }

    static self_t rnorm(index_t rows, index_t cols, value_t mean=0., value_t std=1.) {
        DenseMatrix<value_t> res(rows, cols);
        index_t size = rows * cols;
        for (index_t i = 0; i < size; i += 2) {
            double x, y, radius;
            do {
                x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
                y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
                radius = (x * x) + (y * y);
            } while ((radius >= 1.0) || (radius == 0.0));
            radius = sqrt(-2 * log(radius) / radius);
            x *= radius;
            y *= radius;
            res.at(i) = x * std + mean;
            if (i + 1 < size) res.at(i + 1) = y * std + mean;
        }
        return res;
    }

    static self_t runif(index_t rows, index_t cols, value_t from=0, value_t to=1) {
        DenseMatrix<value_t> res(rows, cols);
        index_t size = rows * cols;
        for (index_t i = 0; i < size; ++i) res.at(i) = runif(from, to);
        return res;
    }

    struct Vector : DenseExpr<V, I, Vector> {
        typedef self_t         belong_t;
        typedef self_t::Vector self_t;
        typedef V              value_t;
        typedef I              index_t;
        index_t stride;
        index_t nnz;
        value_t* value;

        Vector(index_t stride, index_t nnz, value_t *value) : nnz(nnz), stride(stride), value(value) {}

        inline const value_t& at(index_t i) const {
            assert(i < nnz);
            return value[i * stride];
        }

        inline value_t& at(index_t i) {
            assert(i < nnz);
            return value[i*stride];
        }

        template <class EType>
        self_t& operator=(const DenseExpr<V, I, EType> &expr) {
            const EType& e = expr.self();
            for (index_t i = 0; i < nnz; ++i) at(i) = e.at(i);
            return *this;
        }

        self_t& operator=(V v) {
            *this = ConstDenseExpr<V, I>(v);
            return *this;
        }

        self_t& operator=(const typename SparseMatrix<V, I>::Vector& sv) {
            //memset(value, 0, sizeof(value_t) * nnz);
            for (index_t i = 0; i < nnz; ++i) at(i) = 0;
            for (index_t i = 0; i < sv.nnz; ++i) at(sv.index[i]) = sv.at(i);
            return *this;
        }

        template <class Etype>
        self_t& operator+=(const DenseExpr<V, I, Etype>& e) {
            *this = *this + e;
            return *this;
        }

        self_t& operator+=(const typename SparseMatrix<V, I>::Vector& sv) {
            for (index_t i = 0; i < sv.nnz; ++i) {
                at(sv.index[i]) += sv.at(i);
            }
            return *this;
        }

        template <class Etype>
        self_t& operator-=(const DenseExpr<V, I, Etype>& e) {
            *this = *this - e;
            return *this;
        }

        template <class Etype>
        self_t& operator*=(const DenseExpr<V, I, Etype>& e) {
            *this = *this * e;
            return *this;
        }

        template <class Etype>
        self_t& operator/=(const DenseExpr<V, I, Etype>& e) {
            *this = *this / e;
            return *this;
        }

        self_t& operator/=(V v) {
            *this = *this / ConstDenseExpr<V, I>(v);
            return *this;
        }

        void print(bool head = true) const {
            if (head) std::printf("DenseMat::Vec 1 x %d\n", nnz);
            for (int i = 0; i < nnz; ++i) cout << at(i) << '\t';
            cout << endl;
        }

        value_t squaredEuclideanDist(const Vector& o) const {
            value_t r = 0;
            for (index_t i = 0; i < nnz; ++i) r += (o.at(i) - at(i)) * (o.at(i) - at(i));
            return r;
        }
        inline value_t euclideanDist(const Vector& o) const { return std::sqrt(squaredEuclideanDist(o)); }

        value_t squaredEuclideanDist(const typename SparseMatrix<V, I>::Vector& o) const {
            value_t r = 0;
            index_t i = 0;
            index_t j = 0;
            while (i < nnz) {
                value_t vi = at(i);
                value_t vj = j < o.nnz && i == o.index[j] ? o.at(j++) : 0;
                r += (vi - vj) * (vi - vj);
                ++i;
            }
            return r;
        }
        inline value_t euclideanDist(const typename SparseMatrix<V, I>::Vector& o) const { return std::sqrt(squaredEuclideanDist(o)); }
    };

    typedef Vector Row;
    typedef Vector Col;

    Row row(index_t i) const {
        return Row(rows, cols, value + i);
    }

    Col col(index_t i) const {
        return Col(1, rows, value + i * rows);
    }
};
#endif


