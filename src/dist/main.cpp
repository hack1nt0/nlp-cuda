//
// Created by DY on 17-10-18.
//

#include <matrix/DocumentTermMatrix.h>
#include "dist.h"

int main() {

    typedef DocumentTermMatrix<> dtm_t;
    typedef DistMatrix<>         dist_t;
    dtm_t dtm;
    dtm.read("../data/spamsms.dtm");
    rnormalize(dtm, 2);
    dist_t D(dtm.nrow());
    dist(D, dtm, 1, true);
    return 0;

}
