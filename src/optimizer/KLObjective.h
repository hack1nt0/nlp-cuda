//
// Created by DY on 17-11-2.
//

#ifndef NLP_CUDA_KLOBJECTIVE_H
#define NLP_CUDA_KLOBJECTIVE_H

#include <matrix/DenseMatrix.h>

template <typename T = double, class Matrix = DenseMatrix<T> >
struct KLObjective {
    const Matrix& P;
    int size;
    T alpha; // Penalty sum(Q) = 1
    T beta; // Penalty Q >= 0
    T gamma; // Penalty L2
    Matrix Q;
    Matrix grad;

    KLObjective(const Matrix& P, T alpha = 100, T beta = 10, T gamma = 2) :
        P(P),
        size(P.getNnz()),
        alpha(alpha), beta(beta), gamma(gamma),
        Q(size, 1), grad(size, 1) {
        Q = P;
    }

    T operator()(const Matrix& Q) const {
        T cost = sum(P * (log(maximum(P, Matrix::MIN_VALUE)) - log(maximum(Q, Matrix::MIN_VALUE))))
            + alpha * pow(sum(Q) - 1, 2.)
            + beta * sum(Q < 0.)
            + gamma * sum(Q ^ 2.);
        return cost;
    }

    void caculateGrad(const Matrix& Q) {
        grad = -P / maximum(Q, Matrix::MIN_VALUE)
            + alpha * 2. * (sum(Q) - 1.)
            + (Q < 0.) * beta
            + Q * 2. * gamma;
    }

    Matrix& getGrad() {
        return grad;
    }

    Matrix& getParam() {
        return Q;
    }

    int getSize() const { return size; }
};
/**
 * KL-divergence
 * Objective    : min_Q sum(P * log(P / Q))
 * Constraint   : sum(Q) = 1
 * Constraint   : Q >= 0
 * KKT  : -P / Q + lambda * 1 - Beta_Matrix = 0
 * ==> Q = P
 * TODO sparse matrix
 */
template <typename T = double, class Matrix = DenseMatrix<T> >
struct KLObjectiveLagrangian {
    const Matrix& P;
    int size;
    Matrix param;
    Matrix Q;
    Matrix lambda;
    Matrix beta;

    Matrix grad;
    Matrix QGrad;
    Matrix lambdaGrad;
    Matrix betaGrad;


    KLObjectiveLagrangian(const Matrix& P) :
        P(P),
        size(P.getNnz() * 2 + 1),
        param(size, 1),
        Q(param.at(0, 0, P.getNnz(), 1)),
        lambda(param.at(P.getNnz(), 0, P.getNnz() + 1, 1)),
        beta(param.at(P.getNnz() + 1, 0, size, 1)),
        grad(size, 1),
        QGrad(grad.at(0, 0, P.getNnz(), 1)),
        lambdaGrad(grad.at(P.getNnz(), 0, P.getNnz() + 1, 1)),
        betaGrad(grad.at(P.getNnz() + 1, 0, size, 1)) {

//        Q = runif(P.getNnz(), 1);
        Q = P;
        lambda = 1;
        beta = runif(P.getNnz(), 1);
    }

    T operator()(const Matrix& Q) const {
        return sum(P * (log(maximum(P, Matrix::MIN_VALUE)) - log(maximum(Q, Matrix::MIN_VALUE)))) + sum(lambda * (sum(Q) - 1.)) - sum(beta * Q);
    }

    void caculateGrad(const Matrix& Q) {
        QGrad = -P / maximum(Q, Matrix::MIN_VALUE) + lambda.at(0) - beta; //todo
        lambdaGrad = sum(Q) - 1.;
        betaGrad = -Q;
    }

    Matrix& getGrad() {
        return grad;
    }

    Matrix& getParam() {
        return param;
    }

    int getSize() const { return size; }
};
#endif //NLP_CUDA_KLOBJECTIVE_H
