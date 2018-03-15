//
// Created by DY on 17-11-10.
//

#include <utils.h>
#include "matrix.h"

int main(int argc, char* args[]) {
//    typedef vector<string> doc_type;
//    typedef DocumentTermMatrix<double, int, doc_type> dtm_t;
//    int n = 4;
//    vector<doc_type> documents(n);
//    documents[0] = {"A", "B"};
//    documents[1] = {"B", "C"};
//    documents[2] = {"E", "F"};
//    documents[3] = {"A", "B"};
//    dtm_t dtm(n, documents.data(), false);
//    dtm.print();

    typedef CDenseMatrix<double, int> dm_t;
    typedef SparseMatrix<double, int> sm_t;
    typedef MatrixUtils<double, int>  mu;


//    int n = atoi(args[1]);
//    int k = atoi(args[2]);
//    int m = atoi(args[3]);

    int n = 2;
    int k = 3;
    int m = 4;
    sm_t A = mu::rnormSPM(n, k, 0.5);
    A.print();
    cout << "-------------" << endl;
    dm_t B = mu::rnormCDM(k, m);
    B.print();
    cout << "-------------" << endl;
    dm_t C(n, m);
    BlasWrapper blas;
    CpuTimer timer;
    timer.start();
    blas.to1BaseIndexing(A);
    blas.mm(C, 1, A, B, 0);
    blas.to0BaseIndexing(A);
    cout << timer.elapsed() << " ms" << endl;
    cout << "----------------------" << endl;
    timer.start();
    blas.mm(C, 1, A, B, 0);
    C.print();
    cout << timer.elapsed() << " ms" << endl;
    cout << "----------------------" << endl;

    timer.start();
    mu::mm(C, 1, A, B, 0);
    C.print();
    cout << timer.elapsed() << " ms" << endl;
    cout << "----------------------" << endl;


    return 0;
}
