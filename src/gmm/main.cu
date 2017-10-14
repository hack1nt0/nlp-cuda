#include <iostream>
#include <ds/DocumentTermMatrix.h>
#include "gmm.h"

template <typename T>
class GmmModel {
public:
    unsigned int rows, cols, k, valid_itr, seed;
    T alpha, beta;
    T* resp;
    T* mean;
    T* conv;
    T* class_weight;
    T* likelihood;

    GmmModel(unsigned int rows, unsigned int cols, unsigned int k,
             unsigned int valid_itr,
             unsigned int seed, T alpha, T beta) : rows(rows), cols(cols), k(k), valid_itr(valid_itr),
                                                   seed(seed), alpha(alpha), beta(beta) {
        resp = new T[rows * k];
        mean = new T[k * cols];
        conv = new T[k * cols];
        class_weight = new T[k];
        likelihood = new T[valid_itr];
    }

    virtual ~GmmModel() {
        delete[] resp;
        delete[] mean;
        delete[] conv;
        delete[] class_weight;
        delete[] likelihood;
    }
};

template <typename T>
void gmm(GmmModel<T>& model, DocumentTermMatrix<T>& dtm, int max_itr) {
    gmmInit(model.mean, model.conv, model.class_weight,
            dtm.data(), dtm.index(), dtm.row_ptr(),
            dtm.rows(), dtm.cols(), dtm.nnz(),
            model.k, model.seed, model.beta);
    vector<double> likelihood = gmm(model.resp, model.mean, model.conv, model.class_weight,
                                    dtm.csr->data, dtm.csr->index, dtm.csr->row_ptr,
                                    dtm.csr->rows, dtm.csr->cols, dtm.csr->nnz,
                                    model.k, max_itr, model.seed, model.alpha, model.beta);
    double* old_likelihood = model.likelihood;
    model.likelihood = new double[model.valid_itr + likelihood.size()];
    memcpy(model.likelihood, old_likelihood, sizeof(double) * model.valid_itr);
    memcpy(model.likelihood + model.valid_itr, likelihood.data(), sizeof(double) * likelihood.size());
    model.valid_itr += likelihood.size();
    delete[] old_likelihood;
}

int main(int argc, char* argv[]) {
    int k = atoi(argv[1]);
    int max_itr = atoi(argv[2]);
    int seed = atoi(argv[3]);

    DocumentTermMatrix<double> dtm(std::cin);
    cout << "DTM done." << endl;
    GmmModel<double> model(dtm.csr->rows, dtm.csr->cols, k, 0, seed, 1e-5, 1e-5);
    gmm(model, dtm, max_itr);

    return 0;
}
