//
// Created by DY on 2017/9/8.
//

#ifndef NLP_CUDA_DOCUMENT_TERM_MATRIX_H
#define NLP_CUDA_DOCUMENT_TERM_MATRIX_H

#include "matrix.h"
#include <utils.h>

using namespace tbb;

template <typename V = double, typename I = int, class Document = std::vector<std::string> >
struct DocumentTermMatrix : SparseMatrix<V, I> {
    typedef DocumentTermMatrix<V, I, Document> self_t;
    typedef SparseMatrix<V, I> base_t;
    typedef I                  index_t;
    typedef V                  value_t;
    typedef vector<pair<string, index_t> > dict_t;
    typedef unordered_map<string, index_t> word2Id_t;
    shared_ptr<dict_t> dict;
    shared_ptr<word2Id_t> word2Id;
    vector<index_t> nterm;
    bool normalized = false;

    virtual ~DocumentTermMatrix() {}

    DocumentTermMatrix() {}

    DocumentTermMatrix(index_t docNum, const Document* documents, index_t par = 0, const self_t* train = NULL) : nterm(docNum) {
        assert(par >= 1);
        bool needFree = 1;
        index_t rows = docNum;
        index_t cols;
        index_t nnz  = 0;
        index_t* csrPtr = new index_t[rows + 1]; csrPtr[0] = 0;
        index_t* csrInd;
        value_t* csrVal;
        index_t* cscPtr;
        index_t* cscInd;
        value_t* cscVal;
        if (train == NULL) {
            word2Id = make_shared<word2Id_t>();
            if (par==0) {
                vector<map<index_t, index_t> > tf(rows); // count of term
                unordered_map<index_t, index_t> cd; //count of document occurrence
                //csr
                for (index_t i = 0; i < rows; ++i) {
                    for (const string &word : documents[i]) {
                        index_t wordId =
                            word2Id->find(word)==word2Id->end() ? (word2Id->operator[](word) = word2Id->size()) : word2Id->operator[](word);
                        if (tf[i].find(wordId)==tf[i].end()) ++cd[wordId];
                        ++tf[i][wordId];
                    }
                    csrPtr[i + 1] = tf[i].size() + csrPtr[i];
                    nnz += tf[i].size();
                    nterm[i] = documents[i].size();
                }
                assert(csrPtr[rows]==nnz);
                cols = cd.size();
                csrInd = new index_t[nnz];
                csrVal = new value_t[nnz];
                cscPtr = new index_t[cols + 1]; cscPtr[0] = 0;
                for (auto p : cd) cscPtr[p.first + 1] = p.second;
                for (index_t i = 0; i < cols; ++i) cscPtr[i + 1] += cscPtr[i];
                assert(cscPtr[cols]==nnz);
                cscInd = new index_t[nnz];
                cscVal = new value_t[nnz];
                for (index_t i = 0; i < rows; ++i) {
                    index_t rowPtr = csrPtr[i];
                    for (auto p : tf[i]) {
                        const index_t &wordId = p.first;
                        const index_t &wordCnt = p.second;
                        value_t val = tfidf(wordCnt, documents[i].size(), cd[wordId], docNum);
                        index_t &colPtr = cscPtr[wordId];
                        csrInd[rowPtr] = wordId;
                        csrVal[rowPtr] = val;
                        cscInd[colPtr] = i;
                        cscVal[colPtr] = val;
                        ++rowPtr;
                        ++colPtr;
                    }
                }
                for (index_t i = cols; i > 0; --i) cscPtr[i] = cscPtr[i - 1];
                cscPtr[0] = 0;
                //dict
                dict = make_shared<dict_t>(cols);
                for (auto p : *word2Id) {
                    const index_t &wordId = p.second;
                    dict->operator[](wordId).first = p.first;
                    dict->operator[](wordId).second = cd[wordId];
                }
            } else {
                typedef concurrent_hash_map<string, set<index_t> > ConcurrentHashMap;
                typedef typename ConcurrentHashMap::accessor WriteVisitor;
                typedef typename ConcurrentHashMap::const_accessor ReadVisitor;
                vector<map<string, index_t> > tf(rows); // count of term
                ConcurrentHashMap cd; //count of document occurrence
                parallel_for(blocked_range<index_t>(0, docNum, max(1, docNum/par)),
                             [&](const blocked_range<index_t> &range) {
                               WriteVisitor visitor;
                               for (auto p = range.begin(); p!=range.end(); ++p) {
                                   for (const string &word : documents[p]) {
                                       ++tf[p][word];
                                       cd.insert(visitor, word);
                                       visitor->second.insert(p);
                                       visitor.release();
                                   }
                                   nterm[p] = documents[p].size();
                               }
                             }
                );
                //dict and csc
                cols = cd.size();
                cscPtr = new index_t[cols + 1];
                cscPtr[0] = 0;
                dict = make_shared<dict_t>(cols);
                index_t wordId = 0;
                for (const auto &p : cd) {
                    dict->operator[](wordId).first = p.first;
                    dict->operator[](wordId).second = p.second.size();
                    nnz += p.second.size();
                    ++wordId;
                }
                parallel_sort(dict->begin(), dict->end());
                for (index_t i = 0; i < cols; ++i) {
                    (*word2Id)[(*dict)[i].first] = i;
                    cscPtr[i + 1] = (*dict)[i].second + cscPtr[i];
                }
                assert(cscPtr[cols]==nnz);
                cscInd = new index_t[nnz];
                cscVal = new value_t[nnz];
                parallel_for(blocked_range<index_t>(0, cols, max(1, cols/par)),
                             [&](const blocked_range<index_t> &range) {
                               ReadVisitor visitor;
                               for (auto wordId = range.begin(); wordId!=range.end(); ++wordId) {
                                   index_t ptr = cscPtr[wordId];
                                   const string &word = (*dict)[wordId].first; //todo
                                   cd.find(visitor, word);
                                   for (const index_t &docId : visitor->second) {
                                       const index_t &tc = tf[docId][word];
                                       const index_t &dc = (*dict)[wordId].second;
                                       const index_t &ws = documents[docId].size();
                                       cscInd[ptr] = docId;
                                       cscVal[ptr] = tfidf(tc, ws, dc, docNum);
                                       ++ptr;
                                   }
                                   visitor.release();
                               }
                             }
                );
                //csr
                for (index_t i = 0; i < rows; ++i) csrPtr[i + 1] = tf[i].size() + csrPtr[i];
                assert(csrPtr[rows]==nnz);
                csrInd = new index_t[nnz];
                csrVal = new value_t[nnz];
#pragma omp parallel for
                for (index_t i = 0; i < rows; ++i) {
                    index_t ptr = csrPtr[i];
                    for (auto p : tf[i]) {
                        const string &word = p.first;
                        const index_t &wordId = (*word2Id)[word];
                        const index_t &tc = p.second;
                        const index_t &dc = (*dict)[wordId].second;
                        const index_t &ws = documents[i].size();
                        csrInd[ptr] = wordId;
                        csrVal[ptr] = tfidf(tc, ws, dc, docNum);
                        ++ptr;
                    }
                }
            }
        } else {
            vector<map<index_t, index_t> > tf(rows); // count of term
            unordered_map<index_t, index_t> cd; //count of document occurrence
            //csr
            for (index_t i = 0; i < rows; ++i) {
                for (const string &word : documents[i]) {
                    auto wordAndId = train->word2Id->find(word); //?? threaded safe
                    if (wordAndId != train->word2Id->end()) {
                        const index_t& wordId = wordAndId->second;
                        if (tf[i].find(wordId) == tf[i].end()) ++cd[wordId];
                        ++tf[i][wordId];
                    }
                }
                csrPtr[i + 1] = tf[i].size() + csrPtr[i];
                nnz += tf[i].size();
                nterm[i] = documents[i].size();
            }
            assert(csrPtr[rows]==nnz);
            cols = train->ncol();
            csrInd = new index_t[nnz];
            csrVal = new value_t[nnz];
            cscPtr = new index_t[cols + 1]; cscPtr[0] = 0;
            for (auto p : cd) cscPtr[p.first + 1] = p.second;
            for (index_t i = 0; i < cols; ++i) cscPtr[i + 1] += cscPtr[i];
            assert(cscPtr[cols]==nnz);
            cscInd = new index_t[nnz];
            cscVal = new value_t[nnz];
            for (index_t i = 0; i < rows; ++i) {
                index_t rowPtr = csrPtr[i];
                for (auto p : tf[i]) {
                    const index_t &wordId = p.first;
                    const index_t &wordCnt = p.second;
                    value_t val = tfidf(wordCnt, documents[i].size(), train->dict->operator[](wordId).second, docNum);
                    index_t &colPtr = cscPtr[wordId];
                    csrInd[rowPtr] = wordId;
                    csrVal[rowPtr] = val;
                    cscInd[colPtr] = i;
                    cscVal[colPtr] = val;
                    ++rowPtr;
                    ++colPtr;
                }
            }
            for (index_t i = cols; i > 0; --i) cscPtr[i] = cscPtr[i - 1];
            cscPtr[0] = 0;
            //dict
            dict = train->dict;
            word2Id = train->word2Id;
        }
        base_t::rows = rows;
        base_t::cols = cols;
        base_t::nnz  = nnz;
        base_t::csrPtr = csrPtr;
        base_t::csrInd = csrInd;
        base_t::csrVal = csrVal;
        base_t::cscPtr = cscPtr;
        base_t::cscInd = cscInd;
        base_t::cscVal = cscVal;
        base_t::needFree = needFree;
    }

    inline value_t tfidf(index_t ct, index_t words, index_t cd, index_t docs) {
        return log((1.0 + docs) / cd) * ct / words;
    }

    void save(ofstream& s) {
        base_t::save(s);
        for (auto p : *dict) s << p.first << '\n';
        for (auto p : *dict) s.write((char*)(&p.second), sizeof(p.second));
    }

    void save(const string& path) {
        ofstream s(path);
        save(s);
        s.flush();
        s.close();
    }

    void read(ifstream& s) {
        base_t::read(s);
        dict = make_shared<dict_t>(base_t::cols);
        word2Id = make_shared<word2Id_t>();
        for (index_t i = 0; i < base_t::cols; ++i) {
            s >> (*dict)[i].first;
            (*word2Id)[(*dict)[i].first] = i;
            s.get();
        }
        for (index_t i = 0; i < base_t::cols; ++i) s.read((char*)(&(*dict)[i].second), sizeof((*dict)[i].second));
    }

    void read(const string& path) {
        ifstream s(path);
        read(s);
        s.close();
    }

    vector<index_t> orderOfDict(index_t topk = 10) {
        auto comp = [&](index_t i, index_t j) -> bool {
          if ((*dict)[i].second != (*dict)[j].second) {
              return topk >= 0 ? (*dict)[i].second < (*dict)[j].second : (*dict)[i].second > (*dict)[j].second;
          } else {
              return (*dict)[i].first < (*dict)[j].first;
          }
        };
        vector<index_t> order(base_t::cols);
        for (index_t i = 0; i < base_t::cols; ++i) order[i] = i;
        if (topk == 0 || abs(topk) > log2(base_t::cols)) {
            tbb::parallel_sort(order.begin(), order.end(), comp);
        } else {
            std::partial_sort(order.begin(), order.begin() + std::abs(topk), order.end(), comp);
        }
        return order;
    }
};


#endif //NLP_CUDA_DOCUMENT_TERM_MATRIX_H
