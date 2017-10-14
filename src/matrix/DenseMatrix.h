//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_MATRIX_H
#define NLP_CUDA_MATRIX_H

#include "DenseExpr.h"
//#include "cuda_utils.cuh"

using namespace std;

//namespace cutils {
/**
 *
 * Shallow copy, Row Major matrix class
 *
 */

template <class T>
class DenseMatrix : public DenseExpr<T, DenseMatrix<T> > {
public:
    T* data;
    int rows, cols;
    bool needFree;

    virtual ~DenseMatrix() {
        if (needFree) {
            delete[] data;
        }
    }

    DenseMatrix(const DenseMatrix<T>& that) {
        this->rows = that.rows;
        this->cols = that.cols;
        this->data = that.data;
        this->needFree = false;
    }

    DenseMatrix(T* data, int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        this->data = data;
        this->needFree = false;
    }

    DenseMatrix(int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        this->data = new T[rows * cols];
    }

    DenseMatrix() {}

    template <class DeviceDenseMatrix>
    DenseMatrix& operator=(const DeviceDenseMatrix& d_matrix) {
        assert(this->rows == d_matrix.rows && this->cols == d_matrix.cols);
        this->needFree = true;
        cudaMemcpy(this->data, d_matrix.data, sizeof(T) * (rows * cols), cudaMemcpyDeviceToHost);
        return *this;
    }

    template <class EType>
    DenseMatrix<T>& operator=(const DenseExpr<T, EType>& expr) {
        const EType& e = expr.self();
        for (int i = 0; i < rows * cols; ++i) data[i] = e.at(i);
    }

    template <class EType>
    DenseMatrix<T>& operator-=(const DenseExpr<T, EType>& expr) {
        *this = *this - expr.self();
        return *this;
    }

    void t() {
        T* oldData = data;
        data = new T[rows * cols];
        for (int i = 0; i < cols; ++i) {
            for (int j = 0; j < rows; ++j) {
                data[i * rows + j] = oldData[j * cols + i];
            }
        }
        delete[] oldData;
        swap(rows, cols);
    }

    inline T& at(int r, int c) const {
        return data[r * cols + c];
    }

    inline T& at(int i) const {
        return data[i];
    }

    inline int nrow() const {return rows;}

    inline int ncol() const {return cols;}

    friend ostream &operator<<(ostream &os, const DenseMatrix &matrix) {
        os << "HostDenseMatrix [rows, cols] = [" << matrix.rows << ", " << matrix.cols << "]" << endl;
        for (int i = 0; i < min(10, matrix.rows); ++i) {
            for (int j = 0; j < min(10, matrix.cols); ++j) {
                printf("%e\t", matrix.at(i, j));
            }
            os << endl;
        }
        return os;
    }
};


template <typename T>
T sum(const SparseMatrix<T>& sm) {
    T s = 0;
    for (int i = 0; i < sm.nnz; ++i) s += sm.data[i];
    return s;
}

template <typename T>
inline T rnorm(T mean=0., T std=1.) {
    double x, y, radius;
    do {
        x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
        radius = (x * x) + (y * y);
    } while((radius >= 1.0) || (radius == 0.0));
    radius = sqrt(-2 * log(radius) / radius);
    x *= radius;
    y *= radius;
    return T(x);
}

template <typename T>
DenseMatrix<T> rnorm(int rows, int cols, T mean=0., T std=1.) {
    DenseMatrix<T> res(rows, cols);
    int size = rows * cols;
    for (int i = 0; i < size; i += 2) {
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

template <typename T>
inline T runif(T from=0, T to=1) {
    double x = (double)rand() / RAND_MAX;
    return from + (to - from) * x;
}

template <typename T>
DenseMatrix<T> runif(int rows, int cols, T from=0, T to=1) {
    DenseMatrix<T> res(rows, cols);
    int size = rows * cols;
    for (int i = 0; i < size; ++i) res.at(i) = runif(from, to);
    return res;
}

#endif


