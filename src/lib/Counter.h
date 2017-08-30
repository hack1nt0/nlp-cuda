//
// Created by DY on 17-7-7.
//

#include <hash_map.h>
#ifndef NLP_CUDA_COUNTER_H
#define NLP_CUDA_COUNTER_H

template <class T>
class Counter {
    hash_map<T, int> map;

public:

    void add(const T& t) {
        map[t]++;
    }

    int get(const T& t){
        return map[t];
    }

};
#endif //NLP_CUDA_COUNTER_H
