//
// Created by DY on 17-10-31.
//

#ifndef NLP_CUDA_ADAM_H
#define NLP_CUDA_ADAM_H

#include <matrix/CDenseMatrix.h>

template <typename T, class Objective>
struct Adam {
    Objective& f;
    T beta1 = 0.9;
    T beta2 = 0.999;
    T beta1PowT = 1;
    T beta2PowT = 1;
    T epsilon = 1e-8;
    T lr = 0.01;
    CDenseMatrix<T> momentum1;
    CDenseMatrix<T> momentum2;

    Adam(Objective& f, T lr = 0.01, T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8) :
        f(f),
        beta1(beta1), beta2(beta2), epsilon(epsilon), lr(lr),
        beta1PowT(1), beta2PowT(1),
        momentum1(f.getSize(), 1),
        momentum2(f.getSize(), 1) {
        momentum1 = T(0);
        momentum2 = T(0);
    }

    void update(CDenseMatrix<T>& original, const CDenseMatrix<T>& grad) {
        momentum1 = momentum1 * beta1 + grad * (1. - beta1);
        momentum2 = momentum2 * beta2 + grad * grad * (1. - beta2);
        beta1PowT *= beta1;
        beta2PowT *= beta2;
        original -= (momentum1 / (1 - beta1PowT)) * lr / (sqrt(momentum2 / (1 - beta2PowT)) + epsilon);
        if (sum(original != original) > 0) {
            printf("Error found NaN\n");
            for (int i = 0; i < original.nrow(); ++i) {
                for (int j = 0; j < original.ncol(); ++j) {
                    if (original.at(i, j)!=original.at(i, j)) {
                        printf("%e\n",
                               (momentum1.at(i, j)/(1 - beta1PowT))*lr
                                   /(sqrt(momentum2.at(i, j)/(1 - beta2PowT)) + epsilon));
                        printf("%e %e %e %e %e %e \n",
                               momentum1.at(i, j),
                               (1 - beta1PowT),
                               lr,
                               momentum2.at(i, j),
                               (1 - beta2PowT),
                               epsilon);
                        printf("%e\n", original.at(i, j));
                    }
                }
            }
        }
    }

    void solve(int maxItr, bool verbose = false) {
        for (int itr = 0; itr < maxItr; ++itr) {
            CDenseMatrix<T>& grad = f.getGrad();
            CDenseMatrix<T>& param = f.getParam();
            update(param, f.caculateGrad(param));
            if (verbose) {
                if (itr == 0) printf("itr\tf(x)\n");
                printf("%d\t%f\n", itr, f(param));
                printf("Param:\n");
                param.println();
//                printf("Grad:\n");
//                grad.println();
                printf("-----------------------\n");
            }
        }
    }
};


#endif //NLP_CUDA_ADAM_H
