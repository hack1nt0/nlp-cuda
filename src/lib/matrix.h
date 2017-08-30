//
// Created by DY on 17-8-14.
//

#ifndef NLP_CUDA_MATRIX_H
#define NLP_CUDA_MATRIX_H

#include <iostream>
#include <vector>
#include <host_defines.h>
#include <CudaUtils.cu>
#include <expression_templates.h>

using namespace std;

//namespace cutils {
template <typename Matrix>
class ScatterMatrix {
public:
    const Matrix &matrix;
    int rows, cols;

    __device__ __host__
    inline float operator()(int r, int c) const {
        return matrix(r % rows, c % cols);
    }
};

template <class T>
class Matrix {
public:
    vector<T> data;
    int rows, cols;

    Matrix() {}

    static Matrix generateRandomMatrix(int rows, int cols) {
        Matrix matrix;
        matrix.rows = rows;
        matrix.cols = cols;
        for (int i = 0; i < rows * cols; ++i)
            matrix.data.push_back((T)(rand() % 100));
        return matrix;
    }

    Matrix(const vector<T> &vec, int rows, int cols) {
        this->rows = rows;
        this->cols = cols;
        this->data = vec;
    }

    inline T operator()(int r, int c) const {
        return data[r * cols + c];
    }

    friend ostream &operator<<(ostream &os, const Matrix &matrix) {
        os << "HostDenseMatrix [rows, cols] = [" << matrix.rows << ", " << matrix.cols << "]" << endl;
        for (int i = 0; i < matrix.rows; ++i) {
            for (int j = 0; j < matrix.cols; ++j) {
                printf("%10.3f\t", matrix.data[i * matrix.cols + j]);
            }
            os << endl;
        }
        return os;
    }
};

template <class T>
class SparseMatrix : public BaseExpr<T, SparseMatrix<T> > {
public:
    T *data = 0;
    int *index = 0;
    int *row_ptr = 0;
    int rows = 0;
    int cols = 0;
    int nnz = 0;
    bool is_root = true;

    SparseMatrix() {}

    virtual ~SparseMatrix() {
        if (is_root) {
            if (data != 0) {
                delete[] data;
                delete[] index;
                delete[] row_ptr;
            }
        }
    }

    SparseMatrix(int rows, int cols, float density) {
        this->rows = rows;
        this->cols = cols;
        this->row_ptr = new int[rows + 1];
        this->row_ptr[0] = 0;
        vector<T> data;
        vector<int> index;
        srand(time(0));
        this->nnz = 0;
        for (int i = 0; i < rows; ++i) {
            int count = 0;
            for (int j = 0; j < cols; ++j) {
                float p = (float) rand() / RAND_MAX;
                if (p < density) {
                    data.push_back((T) p);
                    index.push_back(j);
                    this->nnz++;
                    count++;
                }
            }
            this->row_ptr[i + 1] = (count + this->row_ptr[i]);
        }
        this->data = new T[this->nnz];
        this->index = new int[this->nnz];
        for (int i = 0; i < this->nnz; ++i) {
            this->data[i] = data[i];
            this->index[i] = index[i];
        }
        this->is_root = true;
    }

    SparseMatrix(const SparseMatrix<T> &that) {
        this->data = that.data;
        this->index = that.index;
        this->row_ptr = that.row_ptr;
        this->rows = that.rows;
        this->cols = that.cols;
        this->nnz = that.nnz;
        this->is_root = false;
    }

    template <class DeviceSparseMatrix>
    SparseMatrix &operator=(const DeviceSparseMatrix &d_matrix) {
        if (this->nnz != d_matrix.nnz) {
            this->nnz = d_matrix.nnz;
            delete[] this->data;
            this->data = new T[this->nnz];
            delete[] this->index;
            this->index = new int[this->nnz];
        }
        if (this->rows != d_matrix.rows) {
            this->rows = d_matrix.rows;
            delete[] this->row_ptr;
            this->row_ptr = new int[this->rows + 1];
        }
        this->cols = d_matrix.cols;
        this->is_root = true;
        checkCudaErrors(cudaMemcpy(this->data, d_matrix.data, sizeof(T) * this->nnz, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(this->index, d_matrix.index, sizeof(int) * this->nnz, cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaMemcpy(this->row_ptr, d_matrix.row_ptr, sizeof(int) * (this->rows + 1), cudaMemcpyDeviceToHost));
        return *this;
    }

    SparseMatrix &operator=(const TransExpr<T, SparseMatrix<T> > &transExpr) {
        const SparseMatrix<T> &that = transExpr.lhs;
        if (this->data == that.data) { //todo
            this->row_ptr = new int[this->cols + 1];
            this->data = new T[this->nnz];
            this->index = new int[this->nnz];
            for (int i = 0; i < this->cols + 1; ++i) this->row_ptr[i] = 0;
            for (int i = 0; i < that.nnz; ++i) {
                this->row_ptr[that.index[i] + 1]++;
            }
            for (int i = 2; i < this->cols + 1; ++i) this->row_ptr[i] += this->row_ptr[i - 1];
            for (int r = 0; r < that.rows; ++r) {
                int from = that.row_ptr[r];
                int to = that.row_ptr[r + 1];
                for (int i = from; i < to; ++i) {
                    int c = that.index[i];
                    this->index[this->row_ptr[c]] = r;
                    this->data[this->row_ptr[c]] = that.data[i];
                    this->row_ptr[c]++;
                }
            }
            for (int i = this->rows - 1; i > 0; --i) this->row_ptr[i] = this->row_ptr[i - 1];
            this->row_ptr[0] = 0;
            swap(this->cols, this->rows);
            delete[] that.data;
            delete[] that.index;
            delete[] that.row_ptr;
        } else {
            if (this->rows != that.cols) {
                delete[] this->row_ptr;
                this->row_ptr = new int[that.cols + 1];
            }
            fill(this->row_ptr, this->row_ptr + that.cols + 1, 0);
            for (int i = 0; i < that.nnz; ++i) {
                this->row_ptr[that.index[i] + 1]++;
            }
            for (int i = 2; i < this->rows + 1; ++i) this->row_ptr[i] += this->row_ptr[i - 1];
            if (this->nnz != that.nnz) {
                delete[] this->data;
                delete[] this->index;
                this->data = new T[that.nnz];
                this->index = new int[that.nnz];
            }
            for (int r = 0; r < that.rows; ++r) {
                int from = that.row_ptr[r];
                int to = that.row_ptr[r + 1];
                for (int i = from; i < to; ++i) {
                    int c = that.index[i];
                    this->index[this->row_ptr[c]] = r;
                    this->data[this->row_ptr[c]] = that.data[i];
                    this->row_ptr[c]++;
                }
            }
            for (int i = this->rows - 1; i > 0; --i) this->row_ptr[i] = this->row_ptr[i - 1];
            this->row_ptr[0] = 0;
            this->cols = that.rows;
            this->rows = that.cols;
            this->nnz = that.nnz;
        }
        return *this;
    }

    inline T at(int i) const {
        return data[i];
    }

    friend ostream &operator<<(ostream &os, const SparseMatrix &matrix) {
        os << "SparseMatrix [rows, cols, nnz] = [" << matrix.rows << ", " << matrix.cols << ", " << matrix.nnz
           << "]" << endl;
        for (int i = 0; i < matrix.rows; ++i) {
            int from = matrix.row_ptr[i], to = matrix.row_ptr[i + 1];
            for (int j = 0; j < matrix.cols; ++j) {
                if (from < to && j == matrix.index[from]) {
                    printf("%10.3f\t", matrix.data[from++]);
                } else {
                    printf("%10s\t", ".");
                }
            }
            os << endl;
        }
        return os;
    }
};
//}

#endif


