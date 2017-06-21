//
// Created by DY on 17-6-18.
//

#ifndef NLP_CUDA_INDEXER_H
#define NLP_CUDA_INDEXER_H

#include <hash_map.h>
#include <string>
using namespace std;

template <class T>
class Indexer {
public:
    hash_map<string, int> term2id;
    hash_map<int, string> id2term;

    string& getTerm(int id) {
        return id2term[id];
    }

    int getId(string& term) {
        return term2id[term];
    }
};


#endif //NLP_CUDA_INDEXER_H
