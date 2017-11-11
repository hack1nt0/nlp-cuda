//
// Created by DY on 2017/9/8.
//

#ifndef NLP_CUDA_DOCUMENT_TERM_MATRIX_H
#define NLP_CUDA_DOCUMENT_TERM_MATRIX_H

#include <common_headers.h>
#include <matrix/SparseMatrix.h>

template <typename T>
struct DocumentTermMatrix : SparseMatrix<T> {
    vector<string> dict;
//    string* dict;
    T* idf;
    bool normalized;
    typedef SparseMatrix<T> super;

    virtual ~DocumentTermMatrix() {
        delete[] idf;
    }

    void save(const string& path) {
        ofstream out(path.c_str());
        out << this->rows << '\t' << this->cols << '\t' << this->nnz;
        for (int i = 0; i <= this->rows; ++i) out << this->row_ptr[i] << '\n';
        for (int i = 0; i < this->nnz; ++i) out << this->index[i] << '\n';
        for (int i = 0; i <= this->nnz; ++i) out << this->data[i] << '\n';
        for (int i = 0; i < this->cols; ++i) out << dict[i] << '\n';
        for (int i = 0; i < this->cols; ++i) out << idf[i] << '\n';
        out.close();
    }

    void read(const string& path) {
        ifstream in(path.c_str());
        int rows, cols, nnz;
        in >> rows >> cols >> nnz;
        this->rows = rows;
        this->cols = cols;
        this->nnz = nnz;
        this->row_ptr = new int[rows + 1];
        this->index = new int[nnz];
        this->data = new T[nnz];
        for (int i = 0; i <= rows; ++i) in >> this->row_ptr[i];
        for (int i = 0; i < nnz; ++i) in >> this->index[i];
        for (int i = 0; i < nnz; ++i) in >> this->data[i];
        this->needFree = true;
        dict.resize(cols);
        idf = new T[cols];
        string word;
        getline(in, word);
        for (int i = 0; i < cols; ++i) {
            getline(in, dict[i]);
        }
        for (int i = 0; i < cols; ++i) {
            in >> idf[i];
        }
        normalized = false;
        in.close();
    }

    void normalize() {
        if (!normalized) {
            normalized = true;
            for (int i = 0; i < this->rows; ++i) {
                int from = this->row_ptr[i];
                int to   = this->row_ptr[i + 1];
                T norm2 = 0;
                for (int j = from; j < to; ++j) norm2 += this->data[j] * this->data[j];
                norm2 = sqrt(norm2);
                assert(from == to || from < to && norm2 > 0);
                for (int j = from; j < to; ++j) this->data[j] /= norm2;
            }
        }
    }


    struct GreaterCmp {
        const T* data;
        GreaterCmp(const T* data) : data(data) {}
        bool operator()(int i, int j) { return data[i] > data[j]; }
    };

    struct LessCmp {
        const T* data;
        LessCmp(const T* data) : data(data) {}
        bool operator()(int i, int j) { return data[i] < data[j]; }
    };

    vector<int> commonWords(int k) {
        vector<int> order(this->cols);
        for (int i = 0; i < this->cols; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + k, order.end(), LessCmp(this->idf));
        return vector<int>(order.begin(), order.begin() + k);
    }

    vector<int> rareWords(int k) {
        vector<int> order(this->cols);
        for (int i = 0; i < this->cols; ++i) order[i] = i;
        std::partial_sort(order.begin(), order.begin() + k, order.end(), GreaterCmp(this->idf));
        return vector<int>(order.begin(), order.begin() + k);
    }
};
#endif //NLP_CUDA_DOCUMENT_TERM_MATRIX_H
