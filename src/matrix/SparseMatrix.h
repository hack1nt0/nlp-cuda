//
// Created by DY on 17-10-14.
//

#ifndef NLP_CUDA_SPARSE_MAVRIX_H
#define NLP_CUDA_SPARSE_MAVRIX_H

#include "SparseExpr.h"
#include "SparseVector.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <map>

using namespace std;

template <typename V, typename I>
struct SparseMatrix : SparExpr<V, I, SparseMatrix<V, I> > {
  public:
    typedef SparseMatrix<V, I>     self_t;
    typedef I                      index_t;
    typedef V                      value_t;
    index_t rows = 0;
    index_t cols = 0;
    index_t nnz = 0;
  protected:
    index_t* csrPtr = nullptr;
    index_t* csrInd = nullptr;
    value_t* csrVal = nullptr;
    index_t* cscPtr = nullptr;
    index_t* cscInd = nullptr;
    value_t* cscVal = nullptr;
    const int CSR = 1;
    const int CSC = 2;
    int needFree = 0; //0: dont'free, 3: free all, 1: free only csr data, 2: free only csc data;

  public:
    self_t clone() const {
        self_t o(rows, cols, nnz);
        memcpy(o.csrPtr, csrPtr, sizeof(index_t) * (rows + 1));
        memcpy(o.csrInd, csrInd, sizeof(index_t) * nnz);
        memcpy(o.csrVal, csrVal, sizeof(value_t) * nnz);
        memcpy(o.cscPtr, cscPtr, sizeof(index_t) * (cols + 1));
        memcpy(o.cscInd, cscInd, sizeof(index_t) * nnz);
        memcpy(o.cscVal, cscVal, sizeof(value_t) * nnz);
        return o;
    }

    virtual ~SparseMatrix() {
        if (needFree & CSR) {
            delete[] csrPtr;
            delete[] csrInd;
            delete[] csrVal;
        }
        if (needFree & CSC) {
            delete[] cscPtr;
            delete[] cscInd;
            delete[] cscVal;
        }
    }

    template <class EVype>
    self_t& operator=(const SparExpr<V, I, EVype>& expr) {
        const EVype& e = expr.self();
        for (index_t i = 0; i < nnz; ++i) this->at(i) = e.at(i);
        updateCsr();
        return *this;
    }

    self_t& operator=(const self_t& o) {
        if (this == &o) {
            return *this;
        }
        assert(nnz == o.nnz && rows == o.rows && cols == o.cols);
        memcpy(csrPtr, o.csrPtr, sizeof(index_t) * (rows + 1));
        memcpy(csrInd, o.csrInd, sizeof(index_t) * nnz);
        memcpy(csrVal, o.csrVal, sizeof(value_t) * nnz);
        memcpy(cscPtr, o.cscPtr, sizeof(index_t) * (cols + 1));
        memcpy(cscInd, o.cscInd, sizeof(index_t) * nnz);
        memcpy(cscVal, o.cscVal, sizeof(value_t) * nnz);
        return *this;
    }

    self_t& operator=(self_t&& o) {
        cerr << "hi &&" << endl;
        if (this == &o) {
            return *this;
        }
        self_t::~SparseMatrix();
        csrPtr = o.csrPtr;
        csrInd = o.csrInd;
        csrVal = o.csrVal;
        cscPtr = o.cscPtr;
        cscInd = o.cscInd;
        cscVal = o.cscVal;
        needFree = 1;
        o.csrPtr = nullptr;
        o.csrInd = nullptr;
        o.csrVal = nullptr;
        o.cscPtr = nullptr;
        o.cscInd = nullptr;
        o.cscVal = nullptr;
        return *this;
    }

    SparseMatrix(index_t rows,
                 index_t cols,
                 index_t nnz,
                 index_t *csrPtr,
                 index_t *csrInd,
                 value_t *csrVal,
                 index_t *cscPtr,
                 index_t *cscInd, value_t *cscVal, bool needFree = false)
        : rows(rows),
          cols(cols),
          nnz(nnz),
          csrPtr(csrPtr),
          csrInd(csrInd),
          csrVal(csrVal),
          cscPtr(cscPtr),
          cscInd(cscInd),
          cscVal(cscVal), needFree(needFree ? (CSR | CSC) : 0) {}

    SparseMatrix() {}

    SparseMatrix(const SparseMatrix<V, I> &o) : rows(o.rows), cols(o.cols), nnz(o.nnz), csrPtr(o.csrPtr), csrInd(o.csrInd), csrVal(o.csrVal), cscPtr(o.cscPtr), cscInd(o.cscInd), cscVal(o.cscVal), needFree(0) {
    }

    SparseMatrix(index_t rows, index_t cols, index_t nnz) :
        rows(rows), cols(cols), nnz(nnz),
        needFree(CSR | CSC) {
        csrPtr = new index_t[rows + 1]();
        csrInd = new index_t[nnz];
        csrVal = new value_t[nnz];
        cscPtr = new index_t[cols + 1]();
        cscInd = new index_t[nnz];
        cscVal = new value_t[nnz];
    }

    SparseMatrix(index_t rows, index_t cols, index_t nnz,
                     index_t* ptr, index_t* ind, value_t* val, bool csr = true, bool needFree = false) : rows(rows), cols(cols), nnz(nnz) {
        if (csr) {
            csrPtr = ptr;
            csrInd = ind;
            csrVal = val;
            if (needFree) this->needFree |= CSR;
            updateCsc();
        } else {
            cscPtr = ptr;
            cscInd = ind;
            cscVal = val;
            if (needFree) this->needFree |= CSC;
            updateCsr();
        }
    }

    void updateCsr() {
        if (csrPtr == nullptr) {
            csrPtr = new index_t[rows + 1]();
            csrInd = new index_t[nnz];
            csrVal = new value_t[nnz];
            needFree |= CSR;
            for (index_t i = 0; i < cols; ++i) {
                for (index_t j = cscPtr[i]; j < cscPtr[i + 1]; ++j) {
                    ++csrPtr[cscInd[j] + 1];
                }
            }
            for (index_t i = 1; i < rows + 1; ++i) csrPtr[i] += csrPtr[i - 1];
            for (index_t i = 0; i < cols; ++i) {
                for (index_t j = cscPtr[i]; j < cscPtr[i + 1]; ++j) {
                    index_t &ptr = csrPtr[cscInd[j]];
                    csrInd[ptr] = i;
                    csrVal[ptr] = cscVal[j];
                    ++ptr;
                }
            }
            for (index_t i = rows; i > 0; --i) csrPtr[i] = csrPtr[i - 1]; csrPtr[0] = 0;
        } else {
            for (index_t i = 0; i < cols; ++i) {
                for (index_t j = cscPtr[i]; j < cscPtr[i + 1]; ++j) {
                    index_t &ptr = csrPtr[cscInd[j]];
                    csrVal[ptr] = cscVal[j];
                    ++ptr;
                }
            }
            for (index_t i = rows; i > 0; --i) csrPtr[i] = csrPtr[i - 1]; csrPtr[0] = 0;
        }
    }

    void updateCsc() {
        if (cscPtr == nullptr) {
            cscPtr = new index_t[cols + 1]();
            cscInd = new index_t[nnz];
            cscVal = new value_t[nnz];
            needFree |= CSC;
            for (index_t i = 0; i < rows; ++i) {
                for (index_t j = csrPtr[i]; j < csrPtr[i + 1]; ++j) {
                    ++cscPtr[csrInd[j] + 1];
                }
            }
            for (index_t i = 1; i < cols + 1; ++i) cscPtr[i] += cscPtr[i - 1];
            for (index_t i = 0; i < rows; ++i) {
                for (index_t j = csrPtr[i]; j < csrPtr[i + 1]; ++j) {
                    index_t &ptr = cscPtr[csrInd[j]];
                    cscInd[ptr] = i;
                    cscVal[ptr] = csrVal[j];
                    ++ptr;
                }
            }
            for (index_t i = cols; i > 0; --i) cscPtr[i] = cscPtr[i - 1]; cscPtr[0] = 0;
        } else {
            for (index_t i = 0; i < rows; ++i) {
                for (index_t j = csrPtr[i]; j < csrPtr[i + 1]; ++j) {
                    index_t &ptr = cscPtr[csrInd[j]];
                    cscVal[ptr] = csrVal[j];
                    ++ptr;
                }
            }
            for (index_t i = cols; i > 0; --i) cscPtr[i] = cscPtr[i - 1]; cscPtr[0] = 0;
        }
    }

    inline const value_t& at(index_t i) const {
        return cscVal[i];
    }

    inline value_t& at(index_t i) {
        return cscVal[i];
    }

    inline index_t nrow() const {
        return rows;
    }

    inline index_t ncol() const {
        return cols;
    }

    inline index_t getNnz() const {
        return nnz;
    }
    inline I *getCsrPtr() const {
        assert(csrPtr != nullptr);
        return csrPtr;
    }
    inline I *getCsrInd() const {
        assert(csrInd != nullptr);
        return csrInd;
    }
    inline V *getCsrVal() const {
        assert(csrVal != nullptr);
        return csrVal;
    }
    inline I *getCscPtr() const {
        assert(cscPtr != nullptr);
        return cscPtr;
    }
    inline I *getCscInd() const {
        assert(cscInd != nullptr);
        return cscInd;
    }
    inline V *getCscVal() const {
        assert(cscVal != nullptr);
        return cscVal;
    }
    void print(bool head = true) const {
        if (head) std::printf("SparMat %d x %d = %d\n", rows, cols, nnz);
        for (int i = 0; i < rows; ++i) row(i).print(false);
    }

    virtual void save(ofstream& s) {
        s.write((char*)(&rows), sizeof(index_t));
        s.write((char*)(&cols), sizeof(index_t));
        s.write((char*)(&nnz), sizeof(index_t));
        s.write((char*)csrPtr, sizeof(index_t) * (rows + 1));
        s.write((char*)csrInd, sizeof(index_t) * (nnz));
        s.write((char*)csrVal, sizeof(value_t) * (nnz));
        s.write((char*)cscPtr, sizeof(index_t) * (cols + 1));
        s.write((char*)cscInd, sizeof(index_t) * (nnz));
        s.write((char*)cscVal, sizeof(value_t) * (nnz));
    }

    virtual void read(ifstream& s) {
        self_t::~SparseMatrix();
        s.read((char*)(&rows), sizeof(index_t));
        s.read((char*)(&cols), sizeof(index_t));
        s.read((char*)(&nnz), sizeof(index_t));
        csrPtr = new index_t[rows + 1];
        csrInd = new index_t[nnz];
        csrVal = new value_t[nnz];
        cscPtr = new index_t[cols + 1];
        cscInd = new index_t[nnz];
        cscVal = new value_t[nnz];
        s.read((char*)csrPtr, sizeof(index_t) * (rows + 1));
        s.read((char*)csrInd, sizeof(index_t) * (nnz));
        s.read((char*)csrVal, sizeof(value_t) * (nnz));
        s.read((char*)cscPtr, sizeof(index_t) * (cols + 1));
        s.read((char*)cscInd, sizeof(index_t) * (nnz));
        s.read((char*)cscVal, sizeof(value_t) * (nnz));
        needFree = true;
    }

    typedef SparseVector<V, I> Vector;
    typedef SparseVector<V, I> Row;
    typedef SparseVector<V, I> Col;

    inline Row row(int i) const {
        return Row(cols, csrPtr[i + 1] - csrPtr[i], csrInd + csrPtr[i], csrVal + csrPtr[i]);
    }

    inline Col col(int i) const {
        return Col(rows, cscPtr[i + 1] - cscPtr[i], cscInd + cscPtr[i], cscVal + cscPtr[i]);
    }

    /** Utils methods **/

    void t() {
        swap(rows, cols);
        swap(csrPtr, cscPtr);
        swap(csrInd, cscInd);
        swap(csrVal, cscVal);
    }

    inline self_t operator~() const { return self_t(cols, rows, nnz, cscPtr, cscInd, cscVal, csrPtr, csrInd, csrVal); }

    self_t operator+(const SparseMatrix<V, I>& o) {
        assert(rows == o.rows && cols == o.cols);
        vector<map<index_t, value_t>> rmap(rows);
#pragma omp parallel for
        for (index_t i = 0; i < rows; ++i) {
            Vector arow = row(i);
            Vector brow = o.row(i);
            index_t ai = 0;
            index_t bi = 0;
            while (ai < arow.nnz && bi < brow.nnz) {
                if (arow.index[ai] < brow.index[bi]) {
                    rmap[i][arow.index[ai]] = arow.at(ai);
                    ++ai;
                } else if (arow.index[ai] > brow.index[bi]) {
                    rmap[i][brow.index[bi]] = brow.at(bi);
                    ++bi;
                } else {
                    rmap[i][brow.index[bi]] = brow.at(bi) + arow.at(ai);
                    ++ai;
                    ++bi;
                }
            }
            while (ai < arow.nnz) { rmap[i][arow.index[ai]] = arow.at(ai); ++ai; }
            while (bi < brow.nnz) { rmap[i][brow.index[bi]] = brow.at(bi); ++bi; }
        }
        index_t* csrPtr = new index_t[rows + 1];
        index_t nnz = 0;
        for (index_t i = 0; i < rows; ++i) {
            csrPtr[i + 1] = rmap[i].size() + csrPtr[i];
            nnz += rmap[i].size();
        }
        index_t* csrInd = new index_t[nnz];
        value_t* csrVal = new value_t[nnz];
#pragma omp parallel for
        for (index_t i = 0; i < rows; ++i) {
            index_t ptr = csrPtr[i];
            for (auto p : rmap[i]) {
                csrInd[ptr] = p.first;
                csrVal[ptr] = p.second;
                ++ptr;
            }
        }
        return self_t(rows, cols, nnz, csrPtr, csrInd, csrVal, true, true);
    }

    self_t& operator^=(value_t v) {
        for (index_t i = 0; i < nnz; ++i) {
            csrVal[i] = std::pow(csrVal[i], v);
            cscVal[i] = std::pow(cscVal[i], v);
        }
        return *this;
    }

    inline bool operator==(const self_t& o) const {
        return equals(o, false);
    }

    bool equals(const self_t& o, bool structural = false) const {
        if (rows != o.rows || cols != o.cols || nnz != o.nnz) return false;
        for (index_t i = 0; i < rows + 1; ++i) if (csrPtr[i] != o.csrPtr[i]) return false;
        for (index_t i = 0; i < nnz; ++i) if (csrInd[i] != o.csrInd[i]) return false;
        if (structural) return true;
        for (index_t i = 0; i < nnz; ++i) if (csrVal[i] != o.csrVal[i]) return false;
        return true;
    }

    static self_t rnorm(index_t rows, index_t cols, double density) {
        index_t nnz = 0; //actual nnz may not equal to given nnz.
        index_t* csrPtr = new index_t[rows + 1]; csrPtr[0] = 0;
        vector<value_t> vs;
        vector<index_t> is;
        for (index_t i = 0; i < rows; ++i) {
            index_t count = 0;
            for (index_t j = 0; j < cols; ++j) {
                float p = (float) rand() / RAND_MAX;
                if (p < density) {
                    vs.push_back((value_t) p);
                    is.push_back(j);
                    nnz++;
                    count++;
                }
            }
            csrPtr[i + 1] = count + csrPtr[i];
        }
        value_t* csrVal = new value_t[nnz];
        index_t* csrInd = new index_t[nnz];
        for (index_t i = 0; i < nnz; ++i) {
            csrVal[i] = vs[i];
            csrInd[i] = is[i];
        }
        return self_t(rows, cols, nnz, csrPtr, csrInd, csrVal, true, true);
    }
};

#endif //NLP_CUDA_SPARSE_MAVRIX_H
