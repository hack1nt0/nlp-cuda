//
// Created by DY on 17-6-18.
//

#ifndef NLP_CUDA_DOCUMENTTERMMATRIX_H
#define NLP_CUDA_DOCUMENTTERMMATRIX_H

#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include "Indexer.h"
using namespace std;

class DocumentTermMatrix {
public:
    int dn = 0, tn = 0;
    float* value;
    int* index;
    int* rowPtr;

    Indexer<string> indexer;

    DocumentTermMatrix(const vector<vector<string>>& docs, vector<string>& ids) {
        assert(docs.size() == ids.size());
        this->dn = docs.size();
        this->rowPtr = new int[docs.size()];
        for (auto sent = docs.begin(); sent != docs.end(); ++sent) {
            this->tn += sent->size();
        }
        this->value = new float[this->tn];
        this->index = new int[this->tn];

        for (auto sent = docs.begin(); sent != docs.end(); ++sent) {
            for (auto w = sent->begin(); w != sent->end(); ++w) {
                int id = indexer.addAndGet((*w));

            }

        }

    }

};


#endif //NLP_CUDA_DOCUMENTTERMMATRIX_H
