//
// Created by DY on 17/11/22.
//

#include  <utils.h>
#include "Indexer.h"
#include "ConcurrentIndexer.h"

int main(int argc, char* args[]) {
    typedef unsigned long long index_t;
    size_t k = atoi(args[1]);
    size_t n = atoi(args[2]);
    vector<index_t> v(n);
    for (index_t i = 0; i < n; ++i) v[i] = rand();
    ConcurrentIndexer<index_t, index_t> a;
    Indexer<index_t, index_t> b;
    if (k == 1) {
#pragma omp parallel for
        for (index_t i = 0; i < n; ++i) a.insert(v[i]);
    }
    if (k == 2) {
        for (index_t i = 0; i < n; ++i) b.insert(v[i]);
    }
    if (k == 3) {
        assert(a.size() == b.size());
        vector<index_t> &at = a.id2term;
        vector<index_t> &bt = b.id2term;
        sort(at.begin(), at.end());
        sort(bt.begin(), bt.end());
        assert(at == bt);
    }
    return 0;
}
