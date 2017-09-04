//
// Created by DY on 17-6-28.
//

#ifndef NLP_CUDA_ARRAY_H
#define NLP_CUDA_ARRAY_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
using namespace std;



template <class T>
class Array {
public:
    T* data;
    int size;
    bool inHost;

//    virtual Array<T>& operator = (const Array<T>& array) {
//        cerr << "operator = of Array not implemented." << endl;
//        exit(1);
//    }

    virtual ~Array() {}

    T& operator [] (int i) {
        return data[i];
    }

    void print() {
        print("null", 100);
    }

    void print(string name, int len) {
        printf("HostArray %s of obs %d\n", name.c_str(), size);
        for (int i = 0; i < min(size, len); ++i) cout << data[i] << ' ';
        if (len < size) cout << "...";
        cout << endl;
    }
};

template <class T>
class HostArray : public Array<T> {
public:

    HostArray() {
        this->size = 0;
        this->data = NULL;
        this->inHost = true;
    }

    HostArray(int size) {
        this->size = size;
        this->data = new T[size];
        this->inHost = true;
    }

    HostArray<T>& operator = (const Array<T>& array) {
        if (this == &array) return *this;
        if (this->size != array.size) {
            delete[] this->data;
            this->size = array.size;
            this->data = new T[array.size];
        }
        if (!array.inHost) {
            cudaMemcpy(this->data, array.data, sizeof(T) * array.size, cudaMemcpyDeviceToHost);
        } else {
            memcpy(this->data, array.data, sizeof(T) * array.size);
        }
        return *this;
    }

    virtual ~HostArray(){
        delete [] this->data;
    }
};

template <class T>
class DeviceArray : public Array<T> {
public:
    DeviceArray(int size) {
        this->size = size;
        this->inHost = false;
        cudaMalloc((void**)&this->data, sizeof(T) * size);
    }

    DeviceArray(T* begin, T* end) {
        int size = end - begin;
        this->size = size;
        this->inHost = false;
        cudaMalloc((void**)&this->data, sizeof(T) * size);
        cudaMemcpy((void*)this->data, (void*)begin, sizeof(T) * size, cudaMemcpyHostToDevice);
    }

    DeviceArray<T>& operator = (const vector<T>& array) {
        if (this->size != array.size()) {
            cudaFree(this->data);
            this->size = array.size();
            cudaMalloc((void**)this->data, sizeof(T) * array.size());
        }
        cudaMemcpy(this->data, array.data(), sizeof(T) * array.size(), cudaMemcpyHostToDevice); // todo (void*)
        return *this;
    }


    DeviceArray<T>& operator = (const Array<T>& array) {
        if (this == &array) return *this;
        if (this->size != array.size) {
            cudaFree(this->data);
            this->size = array.size;
            cudaMalloc((void**)this->data, sizeof(T) * array.size);
        }
        if (array.inHost) {
            cudaMemcpy(this->data, array.data, sizeof(T) * array.size, cudaMemcpyHostToDevice);
        } else {
            cudaMemcpy(this->data, array.data, sizeof(T) * array.size, cudaMemcpyDeviceToDevice);
        }
        return *this;
    }

    virtual ~DeviceArray() {
        cudaFree(this->data);
    }

    void resize(int newLen) {
        this->size = newLen;
        cudaFree(this->data);
        cudaMalloc((void**)&this->data, sizeof(T) * newLen);
    }

    friend ostream &operator<<(ostream &os, const DeviceArray<T> &device) {
        T* tmp = new T[device.size];
        cudaMemcpy(tmp, device.data, sizeof(T) * device.size, cudaMemcpyDeviceToDevice);
        os << device.size << '\t' << '[';
        if (device.size == 0) {
            os << ']';
        } else {
            for (int i = 0; i < device.size; ++i) {
                os << tmp[i];
                os << (i == device.size - 1 ? "]" : ", ");
            }
        }
        delete[] tmp;
        return os;
    }
};

template <class T>
class MatrixHost {
public:
    T* data;
    int rows, cols;
    MatrixHost(int rows, int cols) {

        this->rows = rows;
        this->cols = cols;
        data = new T[rows * cols];
    }

    T& operator() (int r, int c) {
        return data[r * cols + c];
    }

    T* operator() (int r) {
        return &data[r * cols];
    }

    virtual ~MatrixHost() {
        delete[] data;
    }

    void print() {
        print("null", 10, 10);
    }

    void print(string name, int rows, int cols) {
        printf("MatrixHost %s of dim (%d, %d)\n", name.c_str(), this->rows, this->cols);
        for (int i = 0; i < min(rows, this->rows); ++i) {
            for (int j = 0; j < min(cols, this->cols); ++j) cout << data[i] << ' ';
            if (cols < this->cols) cout << "...";
            cout << endl;
        }
        if (rows < this->rows) cout << "..." << endl;
    }
};
#endif //NLP_CUDA_ARRAY_H
