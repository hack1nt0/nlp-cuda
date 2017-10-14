//
// Created by DY on 17-8-5.
//

#ifndef NLP_CUDA_DEVICE_MATRIX_H
#define NLP_CUDA_DEVICE_MATRIX_H

//namespace cutils {

    class CudaSparseContext {
    public:
        cusparseHandle_t handle;

        CudaSparseContext() {
            cusparseCreate(&handle);
            cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
        }

        virtual ~CudaSparseContext() {
            cusparseDestroy(handle);
        }
    };

    const CudaSparseContext cudaSparseContext;

template <typename T>
    class DeviceSparseMatrix : public SparExpr<T, DeviceSparseMatrix<T> > {
    public:
        T *data = 0;
        int *index = 0;
        int *row_ptr = 0;
        bool needFree;
        int rows = 0;
        int cols = 0;
        int nnz = 0;
        cusparseMatDescr_t descr; //todo

        virtual ~DeviceSparseMatrix() {
            if (needFree) {
                cudaFree(data);
                cudaFree(index);
                cudaFree(row_ptr);
                checkCusparseErrors(cusparseDestroyMatDescr(descr));
            }
        }

        DeviceSparseMatrix(int rows, int cols, int nnz) {
            checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * nnz));
            checkCudaErrors(cudaMalloc((void **) &this->index, sizeof(int) * nnz));
            checkCudaErrors(cudaMalloc((void **) &this->row_ptr, sizeof(int) * (rows + 1)));
            this->rows = rows;
            this->cols = cols;
            this->nnz = nnz;
            this->needFree = true;
            checkCusparseErrors(cusparseCreateMatDescr(&descr));
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        }

        DeviceSparseMatrix(const T* data, const int* index, const int* row_ptr, int rows,
                           int cols, int nnz) {
            checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * nnz));
            checkCudaErrors(cudaMemcpy(this->data, data, sizeof(T) * nnz, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMalloc((void **) &this->index, sizeof(int) * nnz));
            checkCudaErrors(cudaMemcpy(this->index, index, sizeof(int) * nnz, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMalloc((void **) &this->row_ptr, sizeof(int) * (rows + 1)));
            checkCudaErrors(cudaMemcpy(this->row_ptr, row_ptr, sizeof(int) * (rows + 1), cudaMemcpyHostToDevice));
            this->rows = rows;
            this->cols = cols;
            this->nnz = nnz;
            this->needFree = true;
            checkCusparseErrors(cusparseCreateMatDescr(&descr));
            cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
            cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
        }

        DeviceSparseMatrix(const DeviceSparseMatrix &that) {
            this->data = that.data;
            this->index = that.index;
            this->row_ptr = that.row_ptr;
            this->rows = that.rows;
            this->cols = that.cols;
            this->nnz = that.nnz;
            this->descr = that.descr;
            this->needFree = false;
        }

        DeviceSparseMatrix& operator=(const DeviceSparseMatrix &that) {
            if (this->nnz != that.nnz) {
                if (this->data != 0) {
                    checkCudaErrors(cudaFree(this->data));
                    checkCudaErrors(cudaFree(this->index));
                }
                checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * that.nnz));
                checkCudaErrors(cudaMalloc((void **) &this->index, sizeof(int) * that.nnz));
            }
            if (this->rows != that.rows) {
                if (this->data != 0) checkCudaErrors(cudaFree(this->row_ptr));
                checkCudaErrors(cudaMalloc((void **) &this->row_ptr, sizeof(int) * (that.rows + 1)));
            }
            checkCudaErrors(cudaMemcpy(this->data, that.data, sizeof(T) * that.nnz, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(this->index, that.index, sizeof(int) * that.nnz, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaMemcpy(this->row_ptr, that.row_ptr, sizeof(int) * (that.rows + 1), cudaMemcpyDeviceToDevice));
            this->rows = that.rows;
            this->cols = that.cols;
            this->nnz = that.nnz;
            this->needFree = true;
            return *this;
        }


        friend ostream &operator<<(ostream &os, const DeviceSparseMatrix &d_matrix) {
            SparseMatrix<T> matrix;
            matrix = d_matrix;
            os << "DeviceSparseMatrix [rows, cols, nnz] = [" << matrix.rows << ", " << matrix.cols << ", " << matrix.nnz << "]" << endl;
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

        __device__
        T at(size_t i) const {
            return data[i];
        }

        template <class ETYPE>
        DeviceSparseMatrix& operator=(const SparExpr<T, ETYPE> &expr) {
            foreachDevice(data, expr, nnz);
            checkCudaErrors(cudaDeviceSynchronize());
            return *this;
        }

        DeviceSparseMatrix& operator=(const SparseMatrix<T>& h_matrix) {
            if (this->nnz != h_matrix.nnz) {
                if (this->data != 0) {
                    checkCudaErrors(cudaFree(this->data));
                    checkCudaErrors(cudaFree(this->index));
                }
                checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * h_matrix.nnz));
                checkCudaErrors(cudaMalloc((void **) &this->index, sizeof(int) * h_matrix.nnz));
            }
            if (this->rows != h_matrix.rows) {
                if (this->row_ptr != 0) checkCudaErrors(cudaFree(this->row_ptr));
                checkCudaErrors(cudaMalloc((void **) &this->row_ptr, sizeof(int) * (h_matrix.rows + 1)));
            }
            checkCudaErrors(cudaMemcpy(this->data, h_matrix.data, sizeof(T) * h_matrix.nnz, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(this->index, h_matrix.index, sizeof(int) * h_matrix.nnz, cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(this->row_ptr, h_matrix.row_ptr, sizeof(int) * (h_matrix.rows + 1), cudaMemcpyHostToDevice));
            this->rows = h_matrix.rows;
            this->cols = h_matrix.cols;
            this->nnz = h_matrix.nnz;
            this->needFree = true;
            return *this;
        }

        template <class EType>
        DeviceSparseMatrix& operator=(const TransSparExpr<T, EType> &expr) {
            SparseMatrix<T> matrix;
            matrix = expr.lhs;
            matrix.t();
            *this = matrix;
            return *this;
        }

        void t() {
            *this = ~(*this);
        }

        int size() {
            return nnz;
        }

        __device__ inline
        T distRow(int i, int j) {
            T d = 0;
            int p = row_ptr[i];
            int q = row_ptr[j];
            while (p < row_ptr[i + 1] && q < row_ptr[j + 1]) {
                if (index[p] == index[q]) {
                    d += (data[p] - data[q]) * (data[p] - data[q]);
                    p++;
                    q++;
                } else if (index[p] < index[q]) {
                    d += data[q] * data[q];
                    q++;
                } else {
                    d += data[p] * data[p];
                    p++;
                }
            }
            while (p < row_ptr[i + 1]) {
                d += data[p] * data[p];
                p++;
            }
            while (q < row_ptr[j + 1]) {
                d += data[q] * data[q];
                q++;
            }
            return sqrt(d);
        }
    };




//}

#endif


