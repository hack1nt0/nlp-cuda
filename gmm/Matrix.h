//
// Created by DY on 17-6-18.
//

#ifndef NLP_CUDA_MATRIX_H
#define NLP_CUDA_MATRIX_H

template <class T>
class Matrix {
public:
    T* array;
    int rows, cols;
    Matrix(int rows, int cols) {

        this->rows = rows;
        this->cols = cols;
        array = new T[rows * cols];
    }

    T& operator() (int r, int c) {
        return array[r * cols + c];
    }

    T* operator() (int r) {
        return &array[r * cols];
    }

    virtual ~Matrix() {
        delete[] array;
    }
};

#endif //NLP_CUDA_MATRIX_H
