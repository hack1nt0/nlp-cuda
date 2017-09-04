//
// Created by DY on 17-6-22.
//

#ifndef NLP_CUDA_GMM_H
#define NLP_CUDA_GMM_H

//template<class TIterator, class IndexIterator, typename T>
//void gmm(TIterator &data, IndexIterator &index, IndexIterator &row_ptr, int rows, int cols, int k, int max_itr,
//         unsigned int seed);

class GmmModel {
public:
    unsigned int rows, cols, k, max_itr, valid_itr, seed;
    double* resp;
    double* mean;
    double* conv;
    double* class_weight;
    double* likelihood;

    GmmModel(unsigned int rows, unsigned int cols, unsigned int k, unsigned int max_itr, unsigned int valid_itr,
                       unsigned int seed) : rows(rows), cols(cols), k(k), max_itr(max_itr), valid_itr(valid_itr),
                                            seed(seed) {
        resp = new double[rows * k];
        mean = new double[k * cols];
        conv = new double[k * cols];
        class_weight = new double[k];
        likelihood = new double[max_itr];
    }

    virtual ~GmmModel() {
        delete[] resp;
        delete[] mean;
        delete[] conv;
        delete[] class_weight;
        delete[] likelihood;
    }
};

void gmmInit(double* h_mean, double* h_conv, double* h_class_weight, unsigned int k, unsigned int cols);

/*
 * return valid_itr
 */
unsigned int gmm(double* h_resp, double* h_mean, double* h_conv, double* h_class_weight, double* h_likelihood,
                 const double* data, const int* index, const int* row_ptr,
                 unsigned int rows, unsigned int cols, unsigned int nnz,
                 unsigned int k, unsigned int max_itr, unsigned int seed,
                 double alpha, double beta);

#endif //NLP_CUDA_GMM_H
