//
// Created by DY on 17-6-18.
//

#ifndef NLP_CUDA_NORMALPDF_H
#define NLP_CUDA_NORMALPDF_H

class DiagNormalPDF {
public:
    const float* mean;
    const float* variance;
    int n;

    DiagNormalPDF(const float* mean, const float* variance, int n) {
        this->mean = mean;
        this->variance = variance;
        this->n = n;
    }

    static void probability(const float* mean, const float* variance, const float* obs, int n) {

    }
};

#endif //NLP_CUDA_NORMALPDF_H
