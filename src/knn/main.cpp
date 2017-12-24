//
// Created by DY on 17-9-29.
//

#include <utils.h>
#include <matrix/DocumentTermMatrix.h>
#include <ostream>
#include <matrix/DenseMatrix.h>
#include "VpTree.h"
#include "VpTreeX.h"
#include "knn.h"

int main(int argc, char* args[]) {
    int n = atoi(args[1]);
//    typedef DenseMatrix<> dm_t;
    typedef SparseMatrix<> sm_t;
    typedef Knn<sm_t>        knn_t;
//    typedef VpTreeX<sm_t>  vpt_t;
    sm_t sm = sm_t::rnorm(n, n, 0.5);
////    dm_t dm = dm_t::rnorm(2, 2);
//    sm.print();
    knn_t knn;
    try {
        knn.brute(sm, sm, 10, 1, true, true);
    } catch (...) {
        cout << "HI" << endl;
    }
//    int n = 100;
//    ProgressBar pbar(n);
//#pragma omp parallel for
//    for (int i = 0; i < n; ++i) {
//        double x = rand();
//        for (int j = 0; j < 100000000; ++j) x = sqrt((sqrt(sqrt(x))));
//        pbar.checkInterrupt([](){cout << "DONE." << endl; });
//        pbar.increase();
//    }

//    while (true) {
//        if (pbar.aborted) {
//            cout << "interrupted.." << endl;
//            pbar.aborted = false;
//        }
//    }
    return 0;
}
