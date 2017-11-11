//
// Created by DY on 17-10-31.
//

#include "Adam.h"
#include "KLObjective.h"

int main() {
    int n = 3;
    Matrix P(n, 1);
    P = 1. / n;
//
    KLObjective<> f(P);
    Adam<KLObjective<> > optimizer(f);

    optimizer.solve(10, true);

//    Matrix P1(n, n);
//    P1 = 1.;
//    P1.println();
//    cout << sum(P1 < 1.1) << endl;
    return 0;
}
