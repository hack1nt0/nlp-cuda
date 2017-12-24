//
// Created by DY on 17-6-18.
//

#ifndef NLP_CUDA_CONCURRENT_INDEXER_H
#define NLP_CUDA_CONCURRENT_INDEXER_H

#include "utils.h"

template <class T = string, class I = unsigned int>
struct ConcurrentIndexer {
    typedef T value_t;
    typedef T index_t;
    unordered_map<T, I> term2id;
    vector<T>        id2term;
    std::mutex lock;

    index_t insert(const T& term) {
        if (!exist(term)) {
            lock.lock();
            if (!exist(term)) {
                index_t id = term2id.size();
                term2id[term] = id;
                id2term.push_back(term);
            }
            lock.unlock();
        }
        return term2id[term];
    }

    inline bool exist(const T& term) const {
        return term2id.find(term) != term2id.end();
    }

    inline index_t size() const {
        return term2id.size();
    }

    inline const value_t& getTerm(index_t id) const {
        return id2term[id];
    }

    inline const index_t& getId(value_t & term) const {
        return term2id[term];
    }
};

#endif
