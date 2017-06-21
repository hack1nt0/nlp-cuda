//
// Created by DY on 17-6-18.
//

#ifndef NLP_CUDA_DOCUMENTTERMMATRIX_H
#define NLP_CUDA_DOCUMENTTERMMATRIX_H

#include <iostream>
#include <vector>
#include <string>
#include "Indexer.h"
using namespace std;

class DocumentTermMatrix {
public:
    int dn, tn;
    vector<pair<vector<int>, vector<float> > > matrix;
    Indexer<string> indexer;

    DocumentTermMatrix(string filePath) {

    }

};


#endif //NLP_CUDA_DOCUMENTTERMMATRIX_H
