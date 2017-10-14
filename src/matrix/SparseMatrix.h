//
// Created by DY on 17-10-14.
//

#ifndef NLP_CUDA_SPARSEMATRIX_H
#define NLP_CUDA_SPARSEMATRIX_H

/**
 *
 * Shallow copy, Compressed Sparse Row(CSR) matrix class
 *
 */
    template <typename T>
    class SparseMatrix : public SparExpr<T, SparseMatrix<T> > {
    public:
        T *data;
        int *index;
        int *row_ptr;
        int rows;
        int cols;
        int nnz;
        bool needFree;

        virtual ~SparseMatrix() {
            if (needFree) {
                delete[] data;
                delete[] index;
                delete[] row_ptr;
            }
        }

        SparseMatrix() {
            this->data = 0;
            this->index = 0;
            this->row_ptr = 0;
            this->rows = this->cols = this->nnz = 0;
            this->needFree = false;
        }

        SparseMatrix(const SparseMatrix<T> &that) {
            this->data = that.data;
            this->index = that.index;
            this->row_ptr = that.row_ptr;
            this->rows = that.rows;
            this->cols = that.cols;
            this->nnz = that.nnz;
            this->needFree = false;
        }

        SparseMatrix(T* data, int* index, int* row_ptr,
                     int rows, int cols, int nnz) : data(data), index(index), row_ptr(row_ptr),
                                                    rows(rows), cols(cols), nnz(nnz) {
            this->needFree = false;
        }

        SparseMatrix(int rows, int cols, int nnz) {
            this->rows = rows;
            this->cols = cols;
            this->nnz = nnz;
            this->row_ptr = new int[rows + 1];
            memset(this->row_ptr, 0, sizeof(int) * (rows + 1));
            this->index = new int[nnz];
            this->data = new T[nnz];
            memset(this->data, 0, sizeof(T) * nnz);
            this->needFree = true;
        }

        SparseMatrix(int rows, int cols, float density, unsigned int seed) {
            this->rows = rows;
            this->cols = cols;
            this->row_ptr = new int[rows + 1];
            this->row_ptr[0] = 0;
            vector<T> data;
            vector<int> index;
            srand(seed);
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
            this->needFree = true;
        }

        template <class EType>
        SparseMatrix& operator=(const SparExpr<T, EType> &expr) {
            const EType& e = expr.self();
            for (int i = 0; i < nnz; ++i) this->data[i] = e.at(i);
            return *this;
        }

        SparseMatrix<T> operator=(SparseMatrix<T>&& that) {
            delete(this); //todo
            this->data = that.data;
            this->index = that.index;
            this->row_ptr = that.row_ptr;
            this->rows = that.rows;
            this->cols = that.cols;
            this->nnz = that.nnz;
            this->needFree = true;
            that.data = NULL;
            that.index = NULL;
            that.row_ptr = NULL;
            return *this;
        }

//    template <class DeviceSparseMatrix>
//    SparseMatrix &operator=(const DeviceSparseMatrix &d_matrix) {
//        if (this->nnz != d_matrix.nnz) {
//            if (this->data != 0) {
//                delete[] this->data;
//                delete[] this->index;
//            }
//            this->nnz = d_matrix.nnz;
//            this->data = new T[this->nnz];
//            this->index = new int[this->nnz];
//        }
//        if (this->rows != d_matrix.rows) {
//            if (this->row_ptr != 0) {
//                delete[] this->row_ptr;
//            }
//            this->rows = d_matrix.rows;
//            this->row_ptr = new int[this->rows + 1];
//        }
//        this->cols = d_matrix.cols;
//        this->needFree = true;
//        cudaMemcpy(this->data, d_matrix.data, sizeof(T) * this->nnz, cudaMemcpyDeviceToHost);
//        cudaMemcpy(this->index, d_matrix.index, sizeof(int) * this->nnz, cudaMemcpyDeviceToHost);
//        cudaMemcpy(this->row_ptr, d_matrix.row_ptr, sizeof(int) * (this->rows + 1), cudaMemcpyDeviceToHost);
//        return *this;
//    }

        template <class EType>
        SparseMatrix& operator/=(const SparExpr<T, EType> &expr) {
            *this = *this / expr.self();
            return *this;
        }

        SparseMatrix& operator/=(T value) {
            *this = *this / value;
            return *this;
        }

        template <class EType>
        SparseMatrix<T> &operator=(const TransSparExpr<T, EType> &transExpr) {
            const EType& e = transExpr.lhs;
            int* oldIndex = index;
            int* oldRow_ptr = row_ptr;
            index = new int[nnz];
            row_ptr = new int[cols + 1];
            memset(row_ptr, 0, sizeof(int) * (cols + 1)); //!
            for (int i = 0; i < rows; ++i) {
                int from = oldRow_ptr[i];
                int to   = oldRow_ptr[i + 1];
                for (int j = from; j < to; ++j) row_ptr[oldIndex[j] + 1]++;
            }
            for (int i = 1; i <= cols; ++i) row_ptr[i] += row_ptr[i - 1];
            for (int i = 0; i < rows; ++i) {
                int from = oldRow_ptr[i];
                int to   = oldRow_ptr[i + 1];
                for (int j = from; j < to; ++j) {
                    index[row_ptr[oldIndex[j]]] = i;
                    data[row_ptr[oldIndex[j]]] = e.at(j);
                    row_ptr[oldIndex[j]]++;
                }
            }
            for (int i = cols; i > 0; --i) row_ptr[i] = row_ptr[i - 1];
            row_ptr[0] = 0;
            swap(rows, cols);
            delete[] oldIndex;
            delete[] oldRow_ptr;
            return *this;
        }


        SparseMatrix<T> operator~() {
            const SparseMatrix<T>& a = *this;
            int rows = a.cols;
            int cols = a.rows;
            int nnz = a.nnz;
            T* data = new T[nnz];
            int* index = new int[nnz];
            int* row_ptr = new int[rows + 1];
            memset(row_ptr, 0, sizeof(int) * (rows + 1)); //!
            for (int i = 0; i < a.rows; ++i) {
                int from = a.row_ptr[i];
                int to   = a.row_ptr[i + 1];
                for (int j = from; j < to; ++j) row_ptr[a.index[j] + 1]++;
            }
            for (int i = 1; i <= rows; ++i) row_ptr[i] += row_ptr[i - 1];
            for (int i = 0; i < a.rows; ++i) {
                int from = a.row_ptr[i];
                int to   = a.row_ptr[i + 1];
                for (int j = from; j < to; ++j) {
                    assert(row_ptr[a.index[j]] < nnz);
                    index[row_ptr[a.index[j]]] = i;
                    data[row_ptr[a.index[j]]] = a.data[j];
                    row_ptr[a.index[j]]++;
                }
            }
            for (int i = rows; i > 0; --i) row_ptr[i] = row_ptr[i - 1];
            row_ptr[0] = 0;
            SparseMatrix<T> t(data, index, row_ptr, rows, cols, nnz);
            t.needFree = true;
            return t;
        }

        SparseMatrix<T> operator+(const SparseMatrix<T>& b) {
            const SparseMatrix& a = *this;
            assert(b.rows == a.rows && b.cols == a.cols);
            int* row_ptr = new int[a.rows + 1];
            memset(row_ptr, 0, sizeof(a.rows + 1));
            for (int i = 0; i < a.rows; ++i) {
                int aj = a.row_ptr[i];
                int bj = b.row_ptr[i];
                while (aj < a.row_ptr[i + 1] && bj < b.row_ptr[i + 1]) {
                    if (a.index[aj] == b.index[bj]) aj++, bj++;
                    else if (a.index[aj] < b.index[bj]) aj++;
                    else if (a.index[aj] > b.index[bj]) bj++;
                    row_ptr[i + 1]++;
                }
                row_ptr[i + 1] += row_ptr[i + 1] - aj + b.row_ptr[i + 1] - bj;
            }
            for (int i = 1; i <= rows; ++i) row_ptr[i] += row_ptr[i - 1];
            int nnz = row_ptr[a.rows];
            int* index = new int[nnz];
            T* data = new T[nnz];
            for (int i = 0; i < a.rows; ++i) {
                int aj = a.row_ptr[i];
                int bj = b.row_ptr[i];
                while (aj < a.row_ptr[i + 1] && bj < b.row_ptr[i + 1]) {
                    if (a.index[aj] == b.index[bj]) {
                        index[row_ptr[i]] = a.index[aj];
                        data[row_ptr[i]] = a.data[aj] + b.data[bj];
                        row_ptr[i]++, aj++, bj++;
                    }
                    else if (a.index[aj] < b.index[bj]) {
                        index[row_ptr[i]] = a.index[aj];
                        data[row_ptr[i]] = a.data[aj];
                        row_ptr[i]++, aj++;
                    }
                    else if (a.index[aj] > b.index[bj]) {
                        index[row_ptr[i]] = b.index[bj];
                        data[row_ptr[i]] = b.data[bj];
                        row_ptr[i]++, bj++;
                    }
                }
                while (aj < a.row_ptr[i + 1]) {
                    index[row_ptr[i]] = a.index[aj];
                    data[row_ptr[i]] = a.data[aj];
                    row_ptr[i]++, aj++;
                }
                while (bj < b.row_ptr[i + 1]) {
                    index[row_ptr[i]] = b.index[bj];
                    data[row_ptr[i]] = b.data[bj];
                    row_ptr[i]++, bj++;
                }
            }
            for (int i = rows; i > 0; --i) row_ptr[i] = row_ptr[i - 1];
            row_ptr[0] = 0;
            SparseMatrix<T> c(data, index, row_ptr, a.rows, a.cols, nnz);
            c.needFree = true;
            return c;
        }

        inline T& at(int i) const {
            return data[i];
        }

        friend ostream &operator<<(ostream &os, const SparseMatrix &matrix) {
            os << "HostSparseMatrix [rows, cols, nnz] = [" << matrix.rows << ", " << matrix.cols << ", " << matrix.nnz
               << "]" << endl;
            for (int i = 0; i < min(10, matrix.rows); ++i) {
                int from = matrix.row_ptr[i], to = matrix.row_ptr[i + 1];
                for (int j = 0; j < min(10, matrix.cols); ++j) {
                    if (from < to && j == matrix.index[from]) {
                        printf("%e\t", matrix.data[from++]);
                    } else {
                        printf("%10s\t", ".");
                    }
                }
                os << endl;
            }
            return os;
        }

        inline
        int nrow() const {
            return rows;
        }

        inline
        int ncol() const {
            return cols;
        }

        inline SparseMatrix<T> row(int i) {
            return SparseMatrix<T>(data + row_ptr[i], index + row_ptr[i], row_ptr + i, 1, cols, row_ptr[i + 1] - row_ptr[i]);
        }
    };

};


#endif //NLP_CUDA_SPARSEMATRIX_H
