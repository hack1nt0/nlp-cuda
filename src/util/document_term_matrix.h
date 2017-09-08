//
// Created by DY on 2017/9/8.
//

#ifndef NLP_CUDA_DOCUMENT_TERM_MATRIX_H
#define NLP_CUDA_DOCUMENT_TERM_MATRIX_H

#include "matrix.h"
#include <string>
#include <iostream>
using namespace std;

class DocumentTermMatrix {
public:
    vector<string>* dict;
	double* idf;
    SparseMatrix<double>* csr;

	template <class IStream>
    DocumentTermMatrix(IStream& in) {
        int rows, cols, nnz;
        in >> rows;
        in >> cols;
        in >> nnz;
        csr = new SparseMatrix<double>(rows, cols, nnz);
        dict = new vector<string>(cols);
		idf = new double[cols];
        string word;
		getline(in, word);
        for (int i = 0; i < cols; ++i) {
			getline(in, word);
            dict->push_back(word);
        }
		double a;
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
    }

    virtual ~DocumentTermMatrix() {
        delete dict;
		delete idf;
        delete csr;
    }
};
#endif //NLP_CUDA_DOCUMENT_TERM_MATRIX_H
