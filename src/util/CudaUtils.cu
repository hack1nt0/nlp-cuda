#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cassert>
#include <iomanip>
#include <Array.h>
#include <cusparse.h>
#include <typeinfo>
#include <algorithm>
//#include <cxxabi.h>

#ifndef NLP_CUDA_CUDAUTILS_CU
#define NLP_CUDA_CUDAUTILS_CU

__device__
void __syncthreads();

//namespace cutils {


const int SHARED_MEM_PER_BLOCK = 49152; //bytes
const int MAX_THREADS_PER_BLOCK = 1024;

    class GpuTimer {
        cudaEvent_t starT;
        cudaEvent_t stopT;
    public:
        GpuTimer() {
            cudaEventCreate(&starT);
            cudaEventCreate(&stopT);
        }

        virtual ~GpuTimer() {
            cudaEventDestroy(starT);
            cudaEventDestroy(stopT);
        }

        void start() {
            cudaEventRecord(starT, 0);
        }

        void stop() {
            cudaEventRecord(stopT, 0);
            printf("gpu consumed %f ms\n", elapsed());
        }

        float elapsed() {
            cudaEventRecord(stopT, 0);
            float elapsed;
            cudaEventSynchronize(stopT);
            cudaEventElapsedTime(&elapsed, starT, stopT);
            return elapsed;
        }
    };

#define checkCudaErrors(val) check( (val), #val, __FILE__, __LINE__)

#define checkCusparseErrors(val) checkCusparse( (val), #val, __FILE__, __LINE__)

    template<typename T>
    void check(T err, const char *const func, const char *const file, const int line) {
        if (err != cudaSuccess) {
            std::cerr << "CUDA error at: " << file << ":" << line << std::endl;
            std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
            exit(1);
        }
    }

    template<typename T>
    void checkCusparse(T err, const char *const func, const char *const file, const int line) {
        if (err != CUSPARSE_STATUS_SUCCESS) {
            std::cerr << "CUDA Sparse error at: " << file << ":" << line << std::endl;
            if (err == CUSPARSE_STATUS_ALLOC_FAILED)
                std::cerr << "CUSPARSE_STATUS_ALLOC_FAILED " << func << std::endl;
            else if (err == CUSPARSE_STATUS_ARCH_MISMATCH)
                std::cerr << "CUSPARSE_STATUS_ARCH_MISMATCH " << func << std::endl;
            else if (err == CUSPARSE_STATUS_EXECUTION_FAILED)
                std::cerr << "CUSPARSE_STATUS_EXECUTION_FAILED " << func << std::endl;
            else if (err == CUSPARSE_STATUS_INTERNAL_ERROR)
                std::cerr << "CUSPARSE_STATUS_INTERNAL_ERROR " << func << std::endl;
            else if (err == CUSPARSE_STATUS_INVALID_VALUE)
                std::cerr << "CUSPARSE_STATUS_INVALID_VALUE " << func << std::endl;
            else if (err == CUSPARSE_STATUS_MAPPING_ERROR)
                std::cerr << "CUSPARSE_STATUS_MAPPING_ERROR " << func << std::endl;
            else if (err == CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
                std::cerr << "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED " << func << std::endl;
            else if (err == CUSPARSE_STATUS_NOT_INITIALIZED)
                std::cerr << "CUSPARSE_STATUS_NOT_INITIALIZED " << func << std::endl;
            else if (err == CUSPARSE_STATUS_ZERO_PIVOT)
                std::cerr << "CUSPARSE_STATUS_ZERO_PIVOT " << func << std::endl;
            else
                std::cerr << "Unspecified error " << func << std::endl;
            exit(1);
        }
    }

    template<typename T>
    void checkResultsExact(const T *const gpu, const T *const ref, size_t numElem) {
        //check that the GPU result matches the CPU result
        for (size_t i = 0; i < numElem; ++i) {
            if (abs(ref[i] - gpu[i]) > 0.01f) {
                std::cerr << "Difference at pos " << i << std::endl;
                //the + is magic to convert char to int without messing
                //with other types
                std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
                          "\nGPU      : " << +gpu[i] << std::endl;
                exit(1);
            }
        }
    }

    template<typename T>
    void checkResultsEps(const T *const gpu, const T *const ref, size_t numElem, double eps1, double eps2) {
        assert(eps1 >= 0 && eps2 >= 0);
        unsigned long long totalDiff = 0;
        unsigned numSmallDifferences = 0;
        for (size_t i = 0; i < numElem; ++i) {
            //subtract smaller from larger in case of unsigned types
            T smaller = std::min(ref[i], gpu[i]);
            T larger = std::max(ref[i], gpu[i]);
            T diff = larger - smaller;
            if (diff > 0 && diff <= eps1) {
                numSmallDifferences++;
            } else if (diff > eps1) {
                std::cerr << "Difference at pos " << +i << " exceeds tolerance of " << eps1 << std::endl;
                std::cerr << "Reference: " << std::setprecision(17) << +ref[i] <<
                          "\nGPU      : " << +gpu[i] << std::endl;
                exit(1);
            }
            totalDiff += diff * diff;
        }
        double percentSmallDifferences = (double) numSmallDifferences / (double) numElem;
        if (percentSmallDifferences > eps2) {
            std::cerr << "Total percentage of non-zero pixel difference between the two images exceeds " << 100.0 * eps2
                      << "%" << std::endl;
            std::cerr << "Percentage of non-zero pixel differences: " << 100.0 * percentSmallDifferences << "%"
                      << std::endl;
            exit(1);
        }
    }

//Uses the autodesk method of image comparison
//Note the the tolerance here is in PIXELS not a percentage of input pixels
    template<typename T>
    void
    checkResultsAutodesk(const T *const gpu, const T *const ref, size_t numElem, double variance, size_t tolerance) {

        size_t numBadPixels = 0;
        for (size_t i = 0; i < numElem; ++i) {
            T smaller = std::min(ref[i], gpu[i]);
            T larger = std::max(ref[i], gpu[i]);
            T diff = larger - smaller;
            if (diff > variance)
                ++numBadPixels;
        }

        if (numBadPixels > tolerance) {
            std::cerr << "Too many bad pixels in the image." << numBadPixels << "/" << tolerance << std::endl;
            exit(1);
        }
    }

    template<class T, class BINARY_FUNCTION>
    __global__
    void reduceKernel(const T *arrayDevice, int from, int to, BINARY_FUNCTION op, T *maxDevice) {
        int dOffset = blockIdx.x * blockDim.x;
        int dIdx = dOffset + threadIdx.x;
        if (dIdx >= to - from)
            return;
        int tIdx = threadIdx.x;
        int leaves = blockDim.x;
        int minIdx = (leaves - 1 - 1) >> 1;
        int maxIdx = (leaves + leaves - 1 - 1 - 1) >> 1;
        int nodes = leaves >> 1;
        if (minIdx <= tIdx && tIdx <= maxIdx) {
            int lchd = (tIdx << 1) + 1 - (leaves - 1) + dOffset;
            int rchd = lchd + 1;
            maxDevice[dIdx] = op(arrayDevice[lchd], arrayDevice[rchd]);
        }
        nodes >>= 1;
        minIdx = (minIdx - 1) >> 1;
        maxIdx = (maxIdx - 1) >> 1;
        __syncthreads();
        while (nodes >= 1) {
            if (minIdx <= tIdx && tIdx <= maxIdx) {
                int lchd = (tIdx << 1) + 1 + dOffset;
                int rchd = lchd + 1;
                maxDevice[dIdx] = op(maxDevice[lchd], maxDevice[rchd]);
            } else if (maxIdx < tIdx) {
                return;
            }
            nodes >>= 1;
            minIdx = (minIdx - 1) >> 1;
            maxIdx = (maxIdx - 1) >> 1;
            __syncthreads();
        }
    }

    template<class T, class BINARY_FUNCTION>
    __global__
    void reduceKernelX(const T *arrayDevice, int from, int to, T init, BINARY_FUNCTION op, T *maxDevice,
                       int WORK_PER_THREAD) {
        extern __shared__ float maxShared[];
        int tIdx = threadIdx.x;
        int arrayOffsetBlock = blockDim.x * blockIdx.x * WORK_PER_THREAD;
        int arrayIdx = arrayOffsetBlock + threadIdx.x;
        if (arrayIdx >= to)
            return;
        if (arrayOffsetBlock + WORK_PER_THREAD * blockDim.x >= to) {
            WORK_PER_THREAD = (to - arrayOffsetBlock) / blockDim.x;
        }
        if (WORK_PER_THREAD % 2 != 0) {
            maxShared[tIdx] = arrayDevice[arrayIdx];
            arrayIdx += blockDim.x;
        } else {
            maxShared[tIdx] = init;
        }
        int stride = blockDim.x * (WORK_PER_THREAD / 2);
        for (int wi = 0; wi < WORK_PER_THREAD / 2; ++wi) {
            maxShared[tIdx] = op(maxShared[tIdx], op(arrayDevice[arrayIdx], arrayDevice[arrayIdx + stride]));
//        printf("%g %g\n", maxShared[tIdx], op(arrayDevice[arrayIdx], arrayDevice[arrayIdx + stride]));
            arrayIdx += blockDim.x;
        }
//    if (tIdx == 0) {
//        printf("%d, %d, %f \n", blockIdx.x, WORK_PER_THREAD, maxShared[tIdx]);
//    }
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 32; stride >>= 1) {
            if (tIdx < stride) {
                maxShared[tIdx] = op(maxShared[tIdx], maxShared[tIdx + stride]);
            } else {
                return;
            }
            __syncthreads();
        }
        if (tIdx < 32) {
            maxShared[tIdx] = op(maxShared[tIdx], maxShared[tIdx + 32]);
            maxShared[tIdx] = op(maxShared[tIdx], maxShared[tIdx + 16]);
            maxShared[tIdx] = op(maxShared[tIdx], maxShared[tIdx + 8]);
            maxShared[tIdx] = op(maxShared[tIdx], maxShared[tIdx + 4]);
            maxShared[tIdx] = op(maxShared[tIdx], maxShared[tIdx + 2]);
            maxShared[tIdx] = op(maxShared[tIdx], maxShared[tIdx + 1]);
        }
        if (tIdx == 0) {
            maxDevice[blockIdx.x] = maxShared[0];
//        printf("%d\t%f\n", blockIdx.x, maxShared[0]);
        }
    }

    template<class T, class BINARY_FUNCTION>
    T reduceDevice(const T *arrayDevice, int from, int to, const T &init, BINARY_FUNCTION op,
                   T *tmpDevice, int THREADS_PER_BLOCK, int WORK_PER_THREAD) {
        assert(0 <= from && from < to);
        if (from == to)
            return init;
        int arraySize = to - from;
        assert(arraySize > THREADS_PER_BLOCK && arraySize % THREADS_PER_BLOCK == 0);
        dim3 threads(THREADS_PER_BLOCK);
        int WORK_PER_BLOCK = WORK_PER_THREAD * THREADS_PER_BLOCK;
        dim3 blocks((arraySize + WORK_PER_BLOCK - 1) / WORK_PER_BLOCK);
        T ans = init;
        reduceKernelX <<< blocks, threads, sizeof(float) * THREADS_PER_BLOCK >>>
                                            (arrayDevice, from, to, init, op, tmpDevice, WORK_PER_THREAD);
        checkCudaErrors(cudaDeviceSynchronize());
        T *tmpHost = new T[arraySize];
        cudaMemcpy(tmpHost, tmpDevice, sizeof(T) * blocks.x, cudaMemcpyDeviceToHost);
        for (int i = 0; i < blocks.x; ++i) {
            //std::cout << tmpHost[i] << std::endl;
//        printf("%f \n", tmpHost[i]);
            ans = op(ans, tmpHost[i]);
        }
        delete[] tmpHost;
//    T vi;
//    for (int i = 0; i < arraySize; i += threads.x) {
//        cudaMemcpy((void *) &vi, tmpDevice + i, sizeof(T), cudaMemcpyDeviceToHost);
////        printf("%f \n", vi);
//        ans = op(ans, vi);
//    }
        return ans;
    }

    template<class T, class BINARY_FUNCTION>
    T reduceHost(const T *arrayHost, int from, int to, const T &init, BINARY_FUNCTION op, T *tmpDevice,
                 int THREADS_PER_BLOCK, int WORK_PER_THREAD) {
        assert(0 <= from && from < to);
        if (from == to)
            return init;
        int arraySize = to - from;
        T ans = init;
        for (int i = 0; i < arraySize % THREADS_PER_BLOCK; ++i) {
            ans = op(ans, arrayHost[to - 1 - i]);
        }
        arraySize -= arraySize % THREADS_PER_BLOCK;
        T *arrayDevice;
        cudaMalloc((void **) &arrayDevice, sizeof(T) * arraySize);
        cudaMemcpy(arrayDevice, arrayHost, sizeof(T) * arraySize, cudaMemcpyHostToDevice);
        ans = op(ans, reduceDevice(arrayDevice, from, from + arraySize, init, op, tmpDevice, THREADS_PER_BLOCK,
                                   WORK_PER_THREAD));
        cudaFree(arrayDevice);
        return ans;
    }


    template<class T, class BINARY_FUNCTION>
    __global__
    void scanKernel(const T *array, int from, int to, const T init, BINARY_FUNCTION op, T *prefix, T *blockSum) {
        extern __shared__ T shared[];
        int arrayId = (blockDim.x * blockIdx.x << 1) + (threadIdx.x << 1);
        int tId = threadIdx.x;
        shared[tId + blockDim.x - 1] = op(array[arrayId], array[arrayId + 1]);
//    shared[tId + blockDim.x - 1] = array[arrayId];
        __syncthreads();

        int leaves = blockDim.x;
        for (int sharedOffset = blockDim.x / 2 - 1, parents = leaves >> 1; parents > 0;) {
            if (tId < parents) {
                int pi = tId + sharedOffset;
                int li = (pi << 1) + 1;
                int ri = (pi << 1) + 2;
                shared[pi] = op(shared[li], shared[ri]);
            }
            parents >>= 1;
            sharedOffset -= parents;
            __syncthreads();
        }

        if (tId == 0) {
            blockSum[blockIdx.x] = shared[0];
            shared[0] = init;
            //printf("%g\n", shared[0]);
        }
//    __syncthreads();

        for (int sharedOffset = 0, parents = 1; parents < leaves;) {
            if (tId < parents) {
                int pi = tId + sharedOffset;
                int li = (pi << 1) + 1;
                int ri = (pi << 1) + 2;
//            if (parents == leaves >> 1) {
//                shared[ri] = op(op(shared[pi], shared[li]), shared[ri]);
//                shared[li] = op(shared[pi], shared[li]);
//            } else {
                shared[ri] = op(shared[pi], shared[li]);
                shared[li] = shared[pi];
            }
            sharedOffset += parents;
            parents <<= 1;
            __syncthreads();
        }

        prefix[arrayId] = op(shared[tId + blockDim.x - 1], array[arrayId]);
        prefix[arrayId + 1] = op(prefix[arrayId], array[arrayId + 1]);
//    prefix[arrayId] = shared[tId + blockDim.x - 1];

    }

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

    template<class T, class BINARY_FUNCTION>
    __global__
    void scanKernelX(const T *array, int from, int to, const T init, BINARY_FUNCTION op, T *prefix, T *blockSum) {
        extern __shared__ T shared[];

        int tId = threadIdx.x;
        int offset = 1;
        int arrayId = (blockDim.x * blockIdx.x << 1) + (threadIdx.x << 1);
        int leaves = blockDim.x * 2;

        shared[tId * 2] = array[arrayId];
        shared[tId * 2 + 1] = array[arrayId + 1];

//    int ai = tId;
//    int bi = tId + (leaves/2);
//    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
//    int bankOffsetB = CONFLICT_FREE_OFFSET(ai);
//    shared[ai + bankOffsetA] = array[ai + blockDim.x * blockIdx.x * 2];
//    shared[bi + bankOffsetB] = array[bi + blockDim.x * blockIdx.x * 2];

        for (int d = leaves >> 1; d > 0; d >>= 1) {
            __syncthreads();
            if (tId < d) {

                int ai = offset * (tId * 2 + 1) - 1;
                int bi = offset * (tId * 2 + 2) - 1;

//            int ai = offset*(2*tId+1)-1;
//            int bi = offset*(2*tId+2)-1;
//            ai += CONFLICT_FREE_OFFSET(ai);
//            bi += CONFLICT_FREE_OFFSET(bi);

                shared[bi] = op(shared[bi], shared[ai]);
            }
            offset <<= 1;
        }

        if (tId == 0) {

            blockSum[blockIdx.x] = shared[leaves - 1];
            shared[leaves - 1] = init;

//        blockSum[blockIdx.x] = shared[leaves - 1 + CONFLICT_FREE_OFFSET(leaves - 1)];
//        shared[leaves - 1 + CONFLICT_FREE_OFFSET(leaves - 1)] = init;
        }

        for (int d = 1; d < leaves; d <<= 1) {
            offset >>= 1;
            __syncthreads();
            if (tId < d) {

                int ai = offset * (tId * 2 + 1) - 1;
                int bi = offset * (tId * 2 + 2) - 1;

//            int ai = offset*(2*tId+1)-1;
//            int bi = offset*(2*tId+2)-1;
//            ai += CONFLICT_FREE_OFFSET(ai);
//            bi += CONFLICT_FREE_OFFSET(bi);

                T t = shared[ai];
                shared[ai] = shared[bi];
                shared[bi] = op(shared[bi], t);
            }
        }
        __syncthreads();

        prefix[arrayId] = op(shared[tId * 2], array[arrayId]);
        prefix[arrayId + 1] = op(shared[tId * 2 + 1], array[arrayId + 1]);
//    prefix[ai + blockDim.x * blockIdx.x * 2] = shared[ai + bankOffsetA];
//    prefix[bi + blockDim.x * blockIdx.x * 2] = shared[bi + bankOffsetB];
    }


    template<class T, class BINARY_FUNCTION>
    __global__
    void scanMapKernel(T *prefix, BINARY_FUNCTION op, T *blockPrefix) {
        if (blockIdx.x == 0)
            return;
        T delta = blockPrefix[blockIdx.x - 1];
        int arrayId = (blockDim.x * blockIdx.x << 1) + (threadIdx.x << 1);
        prefix[arrayId] = op(prefix[arrayId], delta);
        prefix[arrayId + 1] = op(prefix[arrayId + 1], delta);
//    int arrayId = (blockDim.x * blockIdx.x) + (threadIdx.x);
//    prefix[arrayId] = op(prefix[arrayId], delta);
    }

    template<class T, class BINARY_FUNCTION>
    void
    scanInclusivelyDevice(const T *arrayDevice, int from, int to, const T &init, BINARY_FUNCTION op, T *prefixDevice,
                          int THREADS_PER_BLOCK) {
        int threads = THREADS_PER_BLOCK;
        int blocks = (to - from) / THREADS_PER_BLOCK / 2;
        DeviceArray<T> blockSum(blocks);
        scanKernel<<<blocks, threads, sizeof(T) * THREADS_PER_BLOCK * 2>>>
                                         (arrayDevice, from, to, init, op, prefixDevice, blockSum.data);	
        cudaDeviceSynchronize();

        HostArray<T> blockPrefix;
        blockPrefix = blockSum;
        for (int i = 1; i < blockPrefix.size; ++i) blockPrefix[i] = op(blockPrefix[i], blockPrefix[i - 1]);
//    blockPrefix.print();

        blockSum = blockPrefix;
        scanMapKernel<<< blocks, threads >>>(prefixDevice, op, blockSum.data);
        cudaDeviceSynchronize();
    }

    template<class T, class BINARY_FUNCTION>
    HostArray<T>
    scanInclusivelyHost(const T *arrayHost, int from, int to, const T &init, BINARY_FUNCTION op, T *prefixDevice,
                        int THREADS_PER_BLOCK) {
        int arraySize = to - from;
        int remainder = arraySize % (THREADS_PER_BLOCK * 2);
        arraySize -= remainder;
        DeviceArray<T> arrayDevice(arraySize);
        checkCudaErrors(cudaMemcpy(arrayDevice.data, arrayHost + from, sizeof(T) * arraySize, cudaMemcpyHostToDevice));

        scanInclusivelyDevice(arrayDevice.data, 0, arraySize, init, op, prefixDevice, THREADS_PER_BLOCK);

        HostArray<T> prefixHost(to - from);
        checkCudaErrors(cudaMemcpy(prefixHost.data, prefixDevice, sizeof(T) * arraySize, cudaMemcpyDeviceToHost));
        for (int i = arraySize; i < to; ++i) {
            prefixHost[i] = op(arrayHost[i], prefixHost[i - 1]);
        }
        return prefixHost;
    }

/*
 * cxxabi header file not found on Windows...
 */
//    template <class T>
//    string type(const T &variable) {
//        const char *abiName = typeid(variable).name();
//        int status;
//        char *typeNameChars = abi::__cxa_demangle(abiName, 0, 0, &status);
//        string typeName(typeNameChars);
//        free(typeNameChars);
//        return typeName;
//    }

    template<typename T, typename ETYPE>
    __global__
    void fillKernel(T *target, ETYPE expression, int rows, int cols) {
        int col = threadIdx.x + blockIdx.x * blockDim.x;
        int row = threadIdx.y + blockIdx.y * blockDim.y;
        if (col >= cols || row >= rows) return;
        int idx = row * cols + col;
        target[idx] = expression.at(row, col);
    };

template<typename T, typename ETYPE>
__global__
void fillKernel(T *target, ETYPE expression, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    target[idx] = expression.at(idx);
};

    template<typename T, typename ETYPE>
    void fillDevice(T *target, const ETYPE &expression, int rows, int cols) {
        dim3 threads(min(16, cols), min(16, rows));
        dim3 blocks((cols + threads.x - 1) / threads.x, (rows + threads.y - 1) / threads.y);
        fillKernel<<<blocks, threads>>>(target, expression, rows, cols);
    };


template<typename T, typename ETYPE>
    void fillDevice(T *target, const ETYPE &expression, int size) {
        int threads = 16 * 16;
        int blocks = (size + threads - 1) / threads;
        fillKernel<<<blocks, threads>>>(target, expression, size);
    };

template<typename T, class F>
__global__
void mapKernel(T *target, F f, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;
    target[idx] = f(idx, target[idx]);
};

template<typename T, class F>
void mapDevice(T *target, F f, int size) {
    int threads = 16 * 16;
    int blocks = (size + threads - 1) / threads;
    mapKernel<<<blocks, threads>>>(target, f, size);
};


template <class MatrixA, class MatrixB, class T>
__global__
void dotKernel(const T *C, MatrixA A, MatrixB B) {
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    if (row >= A.rows || col >= B.cols) return;
    T c = 0;
    for (int i = 0; i < A.cols; ++i) c += A.at(row, i) * B.at(i, col);
    C[row * B.cols + col] = c;
};

template <class MatrixA, class MatrixB, class MatrixC>
void dotDevice(const MatrixA &A, const MatrixB &B, MatrixC &C) {
    assert(A.cols == B.rows && C.rows == A.rows && C.cols == B.cols);
};

const int MAX_ROW_LENGTH_OF_SPARSE_MATRIX = 5500; // bytes

template <class MatrixA, class MatrixB, class T>
__global__
void spDotKernel(T *C, MatrixA A, MatrixB B, int cPadding) {
    __shared__ float data[MAX_ROW_LENGTH_OF_SPARSE_MATRIX];
    __shared__ float index[MAX_ROW_LENGTH_OF_SPARSE_MATRIX];
    __shared__ int nnz;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int row = idx / cPadding;
    int col = idx % cPadding;
    if (col >= B.cols) return;
    if (threadIdx.x == 0) {
        int from = A.row_ptr[row];
        int to = A.row_ptr[row + 1];
        nnz = to - from;
        printf("nnz = %d\n", nnz);
        for (int i = from; i < to; ++i) {
            data[i - from] = A.data[i];
            index[i - from] = A.index[i];
        }
    }
    __syncthreads();
    printf("here.\n");
    T c = 0;
    for (int i = 0; i < nnz; ++i) {
//        if (row >= 44) {
            printf("index=%d\n", index[i]);
            printf("index=%d,B(index, col)=%f\n", index[i], B.at(index[i], col));
//        }
        c += data[i] * B.at(index[i], col);
    }
    printf("r=%d,c=%d,v=%f\n", row, col, c);
    C[row * B.cols + col] = c;
};

template <class MatrixA, class MatrixB, class MatrixC>
void spDotDevice(const MatrixA &A, const MatrixB &B, MatrixC &C) {
    assert(A.cols == B.rows && C.rows == A.rows && C.cols == B.cols);
    int rowSplits = (C.cols + MAX_THREADS_PER_BLOCK - 1) / MAX_THREADS_PER_BLOCK;
    int threads = (C.cols + rowSplits - 1) / rowSplits;
    int blocks = rowSplits * C.rows;
    int cPadding = rowSplits * threads;
    spDotKernel<<<blocks, threads>>>(C.data, A, B, cPadding);
    checkCudaErrors(cudaDeviceSynchronize());
};

template <typename T>
__global__
void transposeKernel(T *data, int rows, int cols) {
    int r = threadIdx.y + blockDim.y * blockIdx.y;
    int c = threadIdx.x + blockDim.x * blockIdx.x;
    if (r >= rows || c >= cols || r >= c) return;
    T from = data[r * cols + c];
    T to = data[c * cols + r];
    data[r * cols + c] = to;
    data[c * cols + r] = from;
}

template <typename Matrix>
void transposeDevice(Matrix &matrix) {
    dim3 threads(min(matrix.cols, 16), min(matrix.rows, 16));
    dim3 blocks((matrix.cols + threads.x - 1) / threads.x, (matrix.rows + threads.y - 1) / threads.y);
    transposeKernel<<<blocks, threads>>>(matrix.data, matrix.rows, matrix.cols);
}

//}


#endif
