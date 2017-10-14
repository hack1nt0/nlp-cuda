//
// Created by DY on 2017/9/8.
//

#ifndef NLP_CUDA_DOCUMENT_TERM_MATRIX_H
#define NLP_CUDA_DOCUMENT_TERM_MATRIX_H

#include "matrix/DenseMatrix.h"
#include <common_headers.h>

template <typename T>
class DocumentTermMatrix {
public:
    string* dict;
	T* idf;
    SparseMatrix<T>* csr;
    bool normalized = false;

    virtual ~DocumentTermMatrix() {
        delete[] dict;
        delete[] idf;
        delete csr;
    }

	template <class IStream>
    DocumentTermMatrix(IStream& in) {
        printf("Dtm...");
        int rows, cols, nnz;
        in >> rows;
        in >> cols;
        in >> nnz;
        cout << rows << '\t' << cols << '\t' << nnz << endl;
        csr = new SparseMatrix<T>(rows, cols, nnz);
        dict = new string[cols];
		idf = new T[cols];
        string word;
		getline(in, word);
        for (int i = 0; i < cols; ++i) {
			getline(in, word);
//            dict->push_back(word);
//            if (i + 10 > cols) cout << dict->at(i)  << endl;
        }
		T a;
		int b;
        for (int i = 0; i < cols; ++i) {
            in >> a;
			idf[i] = a;
        }
        for (int i = 0; i <= rows; ++i) {
            in >> b;
            csr->row_ptr[i] = b;
        }
		assert(csr->row_ptr[0] == 0 && csr->row_ptr[rows] == nnz);
        for (int i = 0; i < nnz; ++i) {
            in >> b;
            csr->index[i] = b;
        }
        for (int i = 0; i < nnz; ++i) {
            in >> a;
            csr->data[i] = a;
        }
        this->normalized = false;
    }

    DocumentTermMatrix(int* row_ptr, int* index, T* data, int rows, int cols, int nnz, T* idf=null, string* dict=null) {

    }

    void normalize() {
        if (!normalized) {
            normalized = true;
            T* data = csr->data;
            int* row_ptr = csr->row_ptr;
            int rows = csr->rows;
            for (int i = 0; i < rows; ++i) {
                int from = row_ptr[i];
                int to   = row_ptr[i + 1];
                T norm2 = 0;
                for (int j = from; j < to; ++j) norm2 += data[j] * data[j];
                norm2 = sqrt(norm2);
                assert(from == to || from < to && norm2 > 0);
                for (int j = from; j < to; ++j) data[j] /= norm2;
            }
        }
    }
    
    inline 
    int rows() const { return csr->rows; }
    
    inline
    int cols() const { return csr->cols; }
    
    inline
    int nnz() const { return csr->nnz; }
    
    inline
    int size() const { return csr->nnz; }
    
    inline
    T* data() const { return csr->data; }
    
    inline
    int* index() const { return csr->index; }
    
    inline
    int* row_ptr() const { return csr->row_ptr; }

};
#endif //NLP_CUDA_DOCUMENT_TERM_MATRIX_H
