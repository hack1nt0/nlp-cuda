//
// Created by DY on 17-6-26.
//

#ifndef NLP_CUDA_CUDAUTILS_H
#define NLP_CUDA_CUDAUTILS_H

class CpuTimer {
    clock_t time;
    float duration;
public:
    void start() {
        time = clock();
    }

    void stop() {
        duration = (float)(clock() - time)/ CLOCKS_PER_SEC * 1000;
        printf("cpu consumed %f ms \n", duration);
    }

    float elapsed() {
        return duration;
    }
};

#endif //NLP_CUDA_CUDAUTILS_H
