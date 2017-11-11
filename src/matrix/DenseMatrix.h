//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_DENSE_MATRIX_H
#define NLP_CUDA_DENSE_MATRIX_H

#include "DenseExpr.h"

/**
 * Shallow copy, Row Major matrix class
 */

template <class T>
class DenseMatrix : public DenseExpr<T, DenseMatrix<T> > {
  public:
    int rows = 0;
    int cols = 0;
    int nnz = 0;

    T* data = NULL;
    int offset = 0;
    int stride = 0;

    bool needFree = false;

    static const T MIN_VALUE;

    static const DenseMatrix<T> NULL_VALUE;

    virtual ~DenseMatrix() {
        if (needFree) {
            delete[] data;
        }
    }

    DenseMatrix() {}

    DenseMatrix(const DenseMatrix<T>& o) {
        this->rows = o.rows;
        this->cols = o.cols;
        this->nnz = o.nnz;
        this->data = o.data;
        this->offset = o.offset;
        this->stride = o.stride;
        this->needFree = false;
    }

    DenseMatrix(DenseMatrix<T>&& o) {
        cout << "hi && " << endl;
        this->rows = o.rows;
        this->cols = o.cols;
        this->nnz = o.nnz;
        this->data = o.data;
        o.data = NULL;
        this->offset = o.offset;
        this->stride = o.stride;
        this->needFree = o.needFree;
    }

    DenseMatrix(T* data, int rows, int cols, int offset = 0, int stride = 0) {
        this->rows = rows;
        this->cols = cols;
        this->nnz = rows * cols;
        this->data = data;
        this->offset = offset;
        this->stride = stride == 0 ? cols : stride;
        this->needFree = false;
    }

    DenseMatrix(int rows, int cols) :
        rows(rows), cols(cols), nnz(rows * cols),
        offset(0), stride(cols),
        needFree(true) {
        data = new T[rows * cols];
    }

    DenseMatrix<T>& operator=(const DenseMatrix<T>& o) {
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < cols; ++j) at(i, j) = o.at(i, j);
        return *this;
    }

    template <class EType>
    DenseMatrix<T>& operator=(const DenseExpr<T, EType>& expr) {
        const EType& e = expr.self();
        for (int i = 0; i < rows * cols; ++i) {
            at(i) = e.at(i);
        }
        return *this;
    }

    DenseMatrix<T>& operator=(T value) {
        *this = ConstDenseExpr<T>(value);
        return *this;
    }

    template <class EType>
    DenseMatrix<T>& operator-=(const DenseExpr<T, EType>& expr) {
        const EType& e = expr.self();
        *this = *this - e;
        return *this;
    }

    template <class EType>
    DenseMatrix<T>& operator+=(const DenseExpr<T, EType>& expr) {
        const EType& e = expr.self();
        *this = *this + e;
        return *this;
    }

    DenseMatrix<T>& operator/=(T value) {
        *this = *this / value;
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

    T sum() {
        T s = 0;
        for (int i = 0; i < rows * cols; ++i) s += data[i];
        return s;
    }

    inline T at(int r, int c) const {
        return data[r * stride + c + offset];
    }

    inline T at(int i) const {
        return at(i / cols, i % cols);
    }

    inline T& at(int r, int c) {
        return data[r * stride + c + offset];
    }

    inline T& at(int i) {
        return at(i / cols, i % cols);
    }

    inline int nrow() const {return rows;}

    inline int ncol() const {return cols;}

    inline int getNnz() const {return nnz;}

    DenseMatrix<T> at(int fromRow, int fromCol, int toRow, int toCol) {
        int offset = (fromRow * cols + fromCol);
        return DenseMatrix<T>(this->data, toRow - fromRow, toCol - fromCol, offset, cols);
    }

    void println(int rows=10, int cols=10) const {
        printf("DenseMat[rows=%d,cols=%d]\n", nrow(), ncol());
        for (int i = 0; i < min(rows, nrow()); ++i) {
            for (int j = 0; j < min(cols, ncol()); ++j) {
                T v = this->at(i, j);
                printf("%f\t", v);
            }
            printf("\n");
        }
    }

    void save(const string& path) {
        FILE* f = fopen(path.c_str(), "w");
        fprintf(f, "%d\t%d\t%d\n", rows, cols, nnz);
        fwrite(data, sizeof(T), nnz, f);
        fclose(f);
    }

    void read(const string& path) {
        FILE* f = fopen(path.c_str(), "r");
        fscanf(f, "%d\t%d\t%d\n", &rows, &cols, &nnz);
        data = new T[nnz];
        fread(data, sizeof(T), nnz, f);
        fclose(f);
    }
};

template <typename T>
const DenseMatrix<T> DenseMatrix<T>::NULL_VALUE = *(new DenseMatrix<T>());

template <typename T>
const T DenseMatrix<T>::MIN_VALUE = FLT_MIN;

template <typename T = double>
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
    return T(x * std + mean);
}

template <typename T = double>
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

template <typename T = double>
inline T runif(T from=0, T to=1) {
    double x = (double)rand() / RAND_MAX;
    return from + T((to - from) * x);
}

template <typename T = double>
DenseMatrix<T> runif(int rows, int cols, T from=0, T to=1) {
    DenseMatrix<T> res(rows, cols);
    int size = rows * cols;
    for (int i = 0; i < size; ++i) res.at(i) = runif(from, to);
    return res;
}

#endif


