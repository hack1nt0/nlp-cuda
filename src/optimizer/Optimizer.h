//
// Created by DY on 17-10-31.
//

#ifndef NLP_CUDA_OPTIMIZER_H
#define NLP_CUDA_OPTIMIZER_H

#include <matrix/CDenseMatrix.h>

template <typename T = double, typename Matrix = CDenseMatrix<T> >
struct Optimizer {

    virtual void update(Matrix& original, const Matrix& grad) = 0;
};

template <typename T = double, class Matrix = CDenseMatrix<T> >
struct Objective {

    virtual T operator()(const Matrix& X) = 0;

    virtual void caculateGrad(const Matrix& X) const = 0;

    virtual Matrix& getGrad() const = 0;

    virtual Matrix& getParam() const = 0;
};

typedef CDenseMatrix<double> Matrix;

#endif //NLP_CUDA_OPTIMIZER_H
