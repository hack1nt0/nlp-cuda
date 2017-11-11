//
// Created by DY on 17-10-14.
//

#ifndef NLP_CUDA_SPARSEMATRIX_H
#define NLP_CUDA_SPARSEMATRIX_H

#include "SparseExpr.h"
#include <common_headers.h>
#include <matrix/dist/MetricType.h>


/**
 *
 * Shallow copy, Compressed Sparse Row(CSR) matrix class
 *
 */
template <typename T>
struct SparseMatrix : SparExpr<T, SparseMatrix<T> > {
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

    SparseMatrix(int rows, int cols, int nnz, int* row_ptr, int* index, T* data) : data(data), index(index), row_ptr(row_ptr),
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
        for (int i = 0; i < nnz; ++i) this->at(i) = e.at(i);
        return *this;
    }

    SparseMatrix<T>& operator=(const SparseMatrix<T>& o) {
        cerr << "hi &" << endl;
        if (this == &o) {
            return *this;
        }
        assert(nnz == o.nnz && rows == o.rows && needFree);
        rows = o.rows;
        cols = o.cols;
        memcpy(row_ptr, o.row_ptr, sizeof(int) * (rows + 1));
        memcpy(index, o.index, sizeof(int) * nnz);
        memcpy(data, o.data, sizeof(T) * nnz);
        return *this;
    }

    SparseMatrix<T>& operator=(SparseMatrix<T>&& o) {
        cerr << "hi &&" << endl;
        if (this == &o) {
            return *this;
        }
        delete[](this->row_ptr);
        delete[](this->index);
        delete[](this->data);
        this->data = o.data;
        this->index = o.index;
        this->row_ptr = o.row_ptr;
        this->rows = o.rows;
        this->cols = o.cols;
        this->nnz = o.nnz;
        this->needFree = true;
        o.data = NULL;
        o.index = NULL;
        o.row_ptr = NULL;
        return *this;
    }

//    template <class CuSparseMatrix>
//    SparseMatrix &operator=(const CuSparseMatrix &d_matrix) {
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

//    template <class EType>
//    SparseMatrix<T> &operator=(const TransSparExpr<T, EType> &transExpr) {
//        const EType& e = transExpr.lhs;
//        int* oldIndex = index;
//        int* oldRow_ptr = row_ptr;
//        index = new int[nnz];
//        row_ptr = new int[cols + 1];
//        memset(row_ptr, 0, sizeof(int) * (cols + 1)); //!
//        for (int i = 0; i < rows; ++i) {
//            int from = oldRow_ptr[i];
//            int to   = oldRow_ptr[i + 1];
//            for (int j = from; j < to; ++j) row_ptr[oldIndex[j] + 1]++;
//        }
//        for (int i = 1; i <= cols; ++i) row_ptr[i] += row_ptr[i - 1];
//        for (int i = 0; i < rows; ++i) {
//            int from = oldRow_ptr[i];
//            int to   = oldRow_ptr[i + 1];
//            for (int j = from; j < to; ++j) {
//                index[row_ptr[oldIndex[j]]] = i;
//                data[row_ptr[oldIndex[j]]] = e.at(j);
//                row_ptr[oldIndex[j]]++;
//            }
//        }
//        for (int i = cols; i > 0; --i) row_ptr[i] = row_ptr[i - 1];
//        row_ptr[0] = 0;
//        swap(rows, cols);
//        delete[] oldIndex;
//        delete[] oldRow_ptr;
//        return *this;
//    }


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
            for (int j = from; j < to; ++j) {
                if (a.index[j] + 1 >= rows + 1) {
                    cout << j << ' ' << from << ' ' << to << ' ' << a.index[j] + 1 << endl;
                    for (int k = from; k < to; ++k) cout << a.index[k] << endl;
                }
                row_ptr[a.index[j] + 1]++;
            }
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
        SparseMatrix<T> t(rows, cols, nnz, row_ptr, index, data);
        t.needFree = true;
        return t;
    }

    SparseMatrix<T> operator+(const SparseMatrix<T>& b) {
        const SparseMatrix<T>& a = *this;
        assert(b.rows == a.rows && a.cols == b.cols);
        int* row_ptr = new int[a.rows + 1];
        memset(row_ptr, 0, sizeof(int) * (a.rows + 1));
        for (int i = 0; i < a.rows; ++i) {
            int aj = a.row_ptr[i];
            int bj = b.row_ptr[i];
            while (aj < a.row_ptr[i + 1] && bj < b.row_ptr[i + 1]) {
                if (a.index[aj] == b.index[bj]) aj++, bj++;
                else if (a.index[aj] < b.index[bj]) aj++;
                else if (a.index[aj] > b.index[bj]) bj++;
                row_ptr[i + 1]++;
            }
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
        SparseMatrix<T> c(rows, cols, nnz, row_ptr, index, data);
        c.needFree = true;
        return c;
    }

    inline T at(int i) const {
        return data[i];
    }

    inline T& at(int i) {
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

    inline
    int getNnz() const {
        return nnz;
    }

    struct Row : SparExpr<T, SparseMatrix<T>::Row > {
        const int cols;
        const int nnz;
        int* index;
        T* data;

        Row(const int cols, const int nnz, int *index, T *data)
            : cols(cols), nnz(nnz), index(index), data(data) {}

        inline T& at(int i) const {
            return data[i];
        }

        void println(int nnz=10) {
            printf("SparRow[nnz=%d,cols=%d]\n", this->nnz, cols);
            for (int i = 0; i < min(nnz, this->nnz); ++i) printf("(%d,%f)\t", index[i], data[i]);
            printf("\n");
        }

        template <class EType>
        Row& operator=(const SparExpr<T, EType> &expr) {
            const EType& e = expr.self();
            for (int i = 0; i < nnz; ++i) this->data[i] = e.at(i);
            return *this;
        }

        Row& operator/=(T value) {
            *this = *this / value;
            return *this;
        }

        T dist2(const Row& b, MetricType metricType = EUCLIDEAN) const {
            const Row& a = *this;
            T d = 0;
            switch (metricType) {
                case EUCLIDEAN: {
                    int ai = 0;
                    int bi = 0;
                    while (ai < a.nnz && bi < b.nnz) {
                        if (a.index[ai] == b.index[bi]) {
                            d += (a.data[ai] - b.data[bi]) * (a.data[ai] - b.data[bi]);
                            ai++;
                            bi++;
                        } else if (a.index[ai] > b.index[bi]) {
                            d += b.data[bi] * b.data[bi];
                            bi++;
                        } else {
                            d += a.data[ai] * a.data[ai];
                            ai++;
                        }
                    }
                    while (ai < a.nnz) {
                        d += a.data[ai] * a.data[ai];
                        ai++;
                    }
                    while (bi < b.nnz) {
                        d += b.data[bi] * b.data[bi];
                        bi++;
                    }
                    return d;
                }
                default:
                    fprintf(stderr, "other metric dist method not implmented yet...\n");
                    exit(1);
            }
        }

        template <class Comparator = std::less<T> >
        void sortByIndex() {
            vector<pair<int, T> > order(nnz);
            for (int i = 0; i < nnz; ++i) order[i] = make_pair(index[i], data[i]);
            Comparator compare;
            sort(order.begin(), order.end(), [&](const pair<int, T>& a, const pair<int, T>& b) -> bool {return compare(a.first, b.first);});
            for (int i = 0; i < nnz; ++i) index[i] = order[i].first, data[i] = order[i].second;
        }

        template <class Comparator = std::less<T> >
        void sortByData() {
            vector<pair<int, T> > order(nnz);
            for (int i = 0; i < nnz; ++i) order[i] = make_pair(index[i], data[i]);
            Comparator compare;
            sort(order.begin(), order.end(), [&](const pair<int, T>& a, const pair<int, T>& b) -> bool {return compare(a.second, b.second);});
            for (int i = 0; i < nnz; ++i) index[i] = order[i].first, data[i] = order[i].second;
        }

        inline int nrow() const { return 1; }

        inline int ncol() const { return cols; }

        inline int getNnz() const { return nnz; }
    };


    inline SparseMatrix<T>::Row row(int i) const {
        return SparseMatrix<T>::Row(cols, row_ptr[i + 1] - row_ptr[i], index + row_ptr[i], data + row_ptr[i]);
    }

    void println(int rows=10, int nnz=10) {
        printf("SparMat[rows=%d,cols=%d,nnz=%d]\n", nrow(), ncol(), getNnz());
        for (int i = 0; i < min(rows, nrow()); ++i) {
            int from = row_ptr[i], to = row_ptr[i + 1];
            int nnz2 = nnz;
            while (nnz2 > 0 && from < to) {
                printf("(%d,%f)\t", index[from], data[from]);
                --nnz2; ++from;
            }
            printf("\n");
        }
    }

    void save(const string& path) {
        FILE* f = fopen(path.c_str(), "w");
        fprintf(f, "%d\t%d\t%d\n", rows, cols, nnz);
        fwrite(row_ptr, sizeof(int), (rows + 1), f);
        fwrite(index, sizeof(int), nnz, f);
        fwrite(data, sizeof(T), nnz, f);
        fclose(f);
    }

    void read(const string& path) {
        FILE* f = fopen(path.c_str(), "r");
        fscanf(f, "%d\t%d\t%d\n", &rows, &cols, &nnz);
        row_ptr = new int[rows + 1];
        index = new int[nnz];
        data = new T[nnz];
        fread(row_ptr, sizeof(int), (rows + 1), f);
        fread(index, sizeof(int), nnz, f);
        fread(data, sizeof(T), nnz, f);
        needFree = true;
        fclose(f);
    }
};


#endif //NLP_CUDA_SPARSEMATRIX_H
