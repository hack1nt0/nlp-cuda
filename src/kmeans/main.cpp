//
// Created by DY on 17-9-15.
//

#include <matrix/DocumentTermMatrix.h>
#include "kmeans.h"

int main(int argc, char* args[]) {
    typedef SparseMatrix<> sm_t;
    typedef KMeans<sm_t>   km_t;
    typedef DocumentTermMatrix<> dtm_t;

    dtm_t dtm;
    dtm.read("../data/spamsms.dtm");
    sm_t sm = dtm.clone();
    rnormalize(sm, 2);
    sm.updateCsc();

//    int k = atoi(args[1]);
//    km_t::Model model(dtm.nrow(), dtm.ncol(), k);
//    km_t::go(model, dtm, k);
//    model.size.print();
    return 0;
}
