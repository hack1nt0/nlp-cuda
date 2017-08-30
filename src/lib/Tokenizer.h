//
// Created by DY on 17-7-7.
//

#include <cppjieba/Jieba.hpp>
using namespace std;

#ifndef NLP_CUDA_TOKENIZER_H
#define NLP_CUDA_TOKENIZER_H

class Tokenizer {
    const char* const DICT_PATH = "../src/cppjieba/dict/jieba.dict.utf8";
    const char* const HMM_PATH = "../src/cppjieba/dict/hmm_model.utf8";
    const char* const USER_DICT_PATH = "../src/cppjieba/dict/user.dict.utf8";
    const char* const IDF_PATH = "../src/cppjieba/dict/idf.utf8";
    const char* const STOP_WORD_PATH = "../src/cppjieba/dict/stop_words.utf8";
    cppjieba::Jieba* tokenizer;

public:

    Tokenizer() {
        tokenizer = new cppjieba::Jieba(DICT_PATH,
                              HMM_PATH,
                              USER_DICT_PATH,
                              IDF_PATH,
                              STOP_WORD_PATH);
    }

    ~Tokenizer() {
        delete tokenizer;
    }

    void splitFilterSpace(vector<string>& words, const string& sentence) {
        vector<string> cand;
        tokenizer->Cut(sentence, cand);
        for (auto w = cand.begin(); w != cand.end(); ++w) if ((*w)[0] != ' ') words.insert(words.end(), *w);
    }

    void split(vector<string>& words, const string& sentence) {
        tokenizer->Cut(sentence, words);
    }

};
#endif //NLP_CUDA_TOKENIZER_H
