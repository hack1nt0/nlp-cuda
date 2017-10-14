//
// Created by DY on 17-10-11.
//
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <avxintrin.h>

using Eigen::MatrixXd;

int main()
{
#if defined(__AVX__)
    std::cout << "hi" << std::endl;
#endif
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    int n = 10;
    Eigen::SparseMatrix<double> sm(n, n);
    sm.setIdentity();
    sm = sm + sm + sm * sm;
    sm *= 3.;
    std::cout << sm << std::endl;
}
