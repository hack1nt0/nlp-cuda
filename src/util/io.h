//
// Created by DY on 2017/9/8.
//

#ifndef NLP_CUDA_IO_H
#define NLP_CUDA_IO_H

#include <iostream>
#include <vector>
#include <cstring>

using namespace std;
/*
 * UTF-8 char reader
 */
class InputReader {
public:
    istream& stream;
    unsigned long long count = 0;
    char c = 0;
    vector<char>* buf;

    InputReader(istream &stream) : stream(stream) {
        buf = new vector<char>(1024);
    }

    virtual ~InputReader() {
        delete buf;
    }

    void read() {
        if (stream.eof()) c = -1;
        else {
            c = stream.get();
            count++;
        }
    }

    bool isExausted() {
        if (c == -1) return true;
        while (isSpace(stream.peek())) stream.get();
        return stream.peek() == -1;
    }

    bool isSpace(int c) {
        return c == ' ' || c == '\t' || c == '\r' || c == '\n';
    }

    InputReader& operator>>(string& str) {
        if (isExausted()) throw std::runtime_error("End of istream...");
        buf->clear();
        read();
        while (c != -1 && !isSpace(c)) {
            buf->push_back(c);
            if ((c >> 7) == 0) {
                ;
            }
            else if ((c >> 5) == 6) {
                read();
                if ((c >> 6) == 2) {
                    buf->push_back(c);
                }
                else throw std::runtime_error("Utf-8 recognize error");
            }
            else if ((c >> 4) == 14) {
                read();
                char c1 = c;
                read();
                char c2 = c;
                if ((c1 >> 6) == 2 && (c2 >> 6) == 2) {
                    buf->push_back(c1);
                    buf->push_back(c2);
                }
                else throw std::runtime_error("Utf-8 recognize error");
            }
            else if ((c >> 3) == 30) {
                read();
                char c1 = c;
                read();
                char c2 = c;
                read();
                char c3 = c;
                if ((c1 >> 6) == 2 && (c2 >> 6) == 2 && (c3 >> 6) == 2) {
                    buf->push_back(c1);
                    buf->push_back(c2);
                    buf->push_back(c3);
                }
                else throw std::runtime_error("Utf-8 recognize error");
            }
            read();
        }

        str.resize(buf->size());
        for (int i = 0; i < buf->size(); ++i) str[i] = buf->at(i);
        return *this;
    }

    InputReader& operator>>(int& d) {
        if (!isExausted()) stream >> d;
        else throw std::runtime_error("End of istream...");
		return *this;
    }

    InputReader& operator>>(double& d) {
        if (!isExausted()) stream >> d;
        else throw std::runtime_error("End of istream...");
		return *this;
    }

};
#endif //NLP_CUDA_IO_H
