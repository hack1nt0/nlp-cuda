//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_ROW_DENSE_MATRIX_H
#define NLP_CUDA_ROW_DENSE_MATRIX_H

#include "DenseExpr.h"
#include "DenseVector.h"

/**
 * Shallow copy, Col Major matrix
 */

template <class V, typename I>
struct RDenseMatrix : public DenseExpr<V, I, RDenseMatrix<V, I> > {
    typedef RDenseMatrix<V, I> self_t;
    typedef V value_t;
    typedef I index_t;
    index_t rows = 0;
    index_t cols = 0;
    index_t nnz = 0;
    value_t *value = nullptr;
    bool needFree = false;

    virtual ~RDenseMatrix() {
        if (needFree) {
            delete[] value;
        }
    }

    RDenseMatrix() {}

    RDenseMatrix(const self_t &o) {
        this->rows = o.rows;
        this->cols = o.cols;
        this->value = o.value;
        this->needFree = false;
    }

    RDenseMatrix(self_t &&o) {
        this->rows = o.rows;
        this->cols = o.cols;
        this->value = o.value;
        this->needFree = o.needFree;
        o.value = nullptr;
    }

    RDenseMatrix(index_t rows, index_t cols, value_t *value, bool needFree = false) {
        this->rows = rows;
        this->cols = cols;
        this->value = value;
        this->needFree = needFree;
    }

    RDenseMatrix(index_t rows, index_t cols) : rows(rows), cols(cols), needFree(true) {
        value = new value_t[rows*cols];
    }

    self_t &operator=(const self_t &o) {
#pragma omp parallel for
        for (index_t r = 0; r < rows; ++r)
            for (index_t c = 0; c < cols; ++c) at(r, c) = o.at(r, c);
        return *this;
    }

    virtual inline const value_t &at(index_t r, index_t c) const {
        return value[r * cols + c];
    }

    virtual inline value_t &at(index_t r, index_t c) {
        return value[r * cols + c];
    }

    virtual inline const value_t &at(index_t i) const {
        return value[i];
    }

    inline value_t &at(index_t i) {
        return value[i];
    }

    inline const value_t &operator[](index_t i) const {
        return value[i];
    }

    inline value_t &operator[](index_t i) {
        return value[i];
    }

    inline index_t nrow() const { return rows; }

    inline index_t ncol() const { return cols; }

    inline index_t getNnz() const { return rows*cols; }

    inline value_t *getValue() const { return value; }

    /** Utils methods **/

    template<class Etype>
    self_t &operator=(const DenseExpr<V, I, Etype> &e) {
        auto ee = e.self();
#pragma omp parallel for
        for (index_t c = 0; c < cols; ++c) {
            for (index_t r = 0; r < rows; ++r) {
                at(r, c) = ee.at(r, c);
            }
        }
        return *this;
    }

    self_t &operator=(value_t v) {
        *this = ConstDenseExpr<V, I>(v);
        return *this;
    }

    void print(bool head = true) const {
        if (head) std::printf("RowMajorDenseMat %d x %d\n", rows, cols);
        for (index_t i = 0; i < rows; ++i) (this->row(i)).print(false);
    }

    void save(ofstream &s) {
        s.write((char *) (&rows), sizeof(rows));
        s.write((char *) (&cols), sizeof(cols));
        s.write((char *) value, sizeof(value_t));
    }

    void read(ifstream &s) {
        s.read((char *) (&rows), sizeof(rows));
        s.read((char *) (&cols), sizeof(cols));
        s.read((char *) value, sizeof(value_t));
    }

    typedef DenseVector<V, I> Vector;
    typedef DenseVector<V, I> Row;
    typedef DenseVector<V, I> Col;

    virtual Row row(index_t i) const {
        return Row(1, cols, value + i * cols);
    }

    virtual Col col(index_t i) const {
        return Col(cols, rows, value + i);
    }

    static self_t rnorm(index_t rows, index_t cols, value_t mean=0., value_t std=1.) {
        self_t res(rows, cols);
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
        self_t res(rows, cols);
        index_t size = rows * cols;
        for (index_t i = 0; i < size; ++i) res.at(i) = from + (value_t)rand() / RAND_MAX * (to - from);
        return res;
    }

    static inline ConstDenseExpr<V, I> Const(value_t v) { return ConstDenseExpr<V, I>(v); }
};
#endif


