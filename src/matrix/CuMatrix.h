//
// Created by DY on 17-10-14.
//

#ifndef NLP_CUDA_CUMATRIX_H
#define NLP_CUDA_CUMATRIX_H


template <typename T>
class CuMatrix : public DenseExpr<T, DeviceDenseMatrix<T> > {
    class CudaBlasContext {
    public:
        cublasHandle_t handle;

        CudaBlasContext() {
            cublasCreate(&handle);
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        }

        virtual ~CudaBlasContext() {
            cublasDestroy(handle);
        }
    };

    const CudaBlasContext cudaBlasContext;

    /*
     * Row major, Shadow copy
     * when factory method return new instance, the \copyDepth MUST BE -1.
     */
        T *data;
        bool needFree;
        int rows, cols;

        virtual ~DeviceDenseMatrix() {
            if (needFree) {
                checkCudaErrors(cudaFree(data));
            }
        }

        void initData(const T *data, int rows, int cols) {
            checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * (rows * cols)));
            checkCudaErrors(cudaMemcpy(this->data, data, sizeof(T) * (rows * cols), cudaMemcpyHostToDevice));
        }

        DeviceDenseMatrix(const T* data, int rows, int cols) {
            initData(data, rows, cols);
            this->rows = rows;
            this->cols = cols;
            this->needFree = true;
        }

        DeviceDenseMatrix(int rows, int cols) {
            checkCudaErrors(cudaMalloc((void **) &this->data, sizeof(T) * (rows * cols)));
            this->rows = rows;
            this->cols = cols;
            this->needFree = true;
        }

        DeviceDenseMatrix(int rows, int cols, unsigned int seed) {
            srand(seed);
            vector<T> data(rows * cols);
            for (int i = 0; i < rows * cols; ++i) data[i] = (T) rand() % 100 / 10;
            initData(data.data(), rows, cols);
            this->rows = rows;
            this->cols = cols;
            this->needFree = true;
        }

        DeviceDenseMatrix(const DeviceDenseMatrix &that) {
            this->data = that.data;
            this->rows = that.rows;
            this->cols = that.cols;
            this->needFree = false;
        }

        DeviceDenseMatrix& operator=(const DeviceDenseMatrix &that) {
            if (this != &that) {
                if (this->rows * this->cols != that.rows * this->cols) {
                    this->~DeviceDenseMatrix();
                }
                this->data = that.data;
                this->rows = that.rows;
                this->cols = that.cols;
                this->needFree = true;
            }
            return *this;
        }

        void toHost(T* h_data) {
            checkCudaErrors(cudaMemcpy(h_data, data, sizeof(T) * (rows * cols), cudaMemcpyDeviceToHost));
        }

//        template <typename T>
//        friend T* operator=(T* h_data, const DeviceDenseMatrix& d_data) {
//            checkCudaErrors(cudaMemcpy(h_data, d_data.data, sizeof(T) * (d_data.rows * d_data.cols), cudaMemcpyDeviceToHost));
//        }


        /*
         * C = alpha * A * B + beta * C
         * */
        static void cudaSparseMultiplyDense(DeviceDenseMatrix<double> &C, double beta,
                                            double alpha, const DeviceSparseMatrix<double> &A, bool transposeA,
                                            const DeviceDenseMatrix<double> &B, bool transposeB) {
            cusparseOperation_t transA = transposeA ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
            cusparseOperation_t transB = transposeB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
            checkCusparseErrors(
                    cusparseDcsrmm2(cudaSparseContext.handle, //todo
                                    transA,
                                    transB,
                                    A.rows,
                                    B.cols,
                                    A.cols,
                                    A.nnz,
                                    &alpha,
                                    A.descr,
                                    A.data,
                                    A.row_ptr,
                                    A.index,
                                    B.data,
                                    B.rows,
                                    &beta,
                                    C.data,
                                    C.rows)
            );
        }

        void reshape(int rows, int cols) {
            assert(rows * cols == this->rows * this->cols && rows > 0 && cols > 0);
            this->rows = rows;
            this->cols = cols;
        }

        friend ostream &operator<<(ostream &os, const DeviceDenseMatrix &matrix) {
            int size = matrix.rows * matrix.cols;
            T *data = new T[size];
            checkCudaErrors(cudaMemcpy(data, matrix.data, sizeof(T) * size, cudaMemcpyDeviceToHost));
            os << "DeviceDenseMatrix [rows, cols] = [" << matrix.rows << ", " << matrix.cols << "]" << endl;
            for (int i = 0; i < min(matrix.rows, 10); ++i) {
                for (int j = 0; j < min(matrix.cols, 10); ++j) {
//                    os << data[i * matrix.cols + j] <<"\t";
                    printf("%e\t", data[i * matrix.cols + j]);
                }
                os << endl;
            }
            delete[] data;
            return os;
        }

        __device__ inline
                T& at(int r, int c) const {
            r %= rows; // broad-casting
            c %= cols;
            return data[r * cols + c];
        }

        __device__ inline
                T& at(int i) const {
            return data[i];
        }

        template <class ETYPE>
        DeviceDenseMatrix& operator=(const DenseExpr<T, ETYPE> &expr) {
            foreachDevice(data, expr, rows, cols);
            checkCudaErrors(cudaDeviceSynchronize());
            return *this;
        }

        void t() {
            if (this->rows == this->cols) {
                transposeDevice(*this);
                checkCudaErrors(cudaDeviceSynchronize());
            } else {
                DenseMatrix<T> h_matrix(this->rows, this->cols);
                h_matrix = *this;
                h_matrix.t();
                *this = h_matrix;
            }
        }

        DeviceDenseMatrix& operator=(const DenseMatrix<T>& h_matrix) {
            assert(this->rows * this->cols == h_matrix.rows * h_matrix.cols);
            this->rows = h_matrix.rows;
            this->cols = h_matrix.cols;
            checkCudaErrors(cudaMemcpy(this->data, h_matrix.data, sizeof(T) * (this->rows * this->cols), cudaMemcpyHostToDevice));
            return *this;
        }

        DeviceDenseMatrix& operator=(T value) {
            foreachDevice(data, ConstViewer<T>(value), rows, cols);
            checkCudaErrors(cudaDeviceSynchronize());
            return *this;
        }



        template <class E>
        DeviceDenseMatrix& operator+=(E expr) {
            return *this = *this + expr;
        }

        template <class E>
        DeviceDenseMatrix& operator-=(E expr) {
            return *this = *this - expr;
        }

        template <class E>
        DeviceDenseMatrix& operator*=(E expr) {
            return *this = *this * expr;
        }

        template <class E>
        DeviceDenseMatrix& operator/=(E expr) {
            return *this = *this / expr;
        }

        template <class OP, class LHS>
        DeviceDenseMatrix& operator=(const ZipExpr<OP, LHS, T> &expr) {
            foreachDevice(data, expr, rows * cols);
            checkCudaErrors(cudaDeviceSynchronize());
            return *this;
        }

        __device__ inline
        void set(int row, int col, T value) {
            //todo bound checking
            data[row * cols + col] = value;
        }

        __device__ inline
        void set(int i, T value) {
            data[i] = value;
        }

        int size()const {
            return rows * cols;
        }
    };

template <typename T>
typedef CuMatrix<T> CuDenseMatrix<T>;

template <typename T>
T sum(const DeviceDenseMatrix<T>& m) {
    thrust::device_ptr<T> devicePtr(m.data);
    return thrust::reduce(devicePtr, devicePtr + m.size());
};


#endif //NLP_CUDA_CUMATRIX_H
