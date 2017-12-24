//
// Created by DY on 17-10-14.
//

#ifndef NLP_CUDA_SPARSE_MAVRIX_H
#define NLP_CUDA_SPARSE_MAVRIX_H

#include "SparseExpr.h"                 
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>
#include <map>

using namespace std;

template <typename V = double, typename I = int>
struct SparseMatrix : SparExpr<V, I, SparseMatrix<V, I> > {
  public:
    typedef SparseMatrix<V, I>     self_t;
    typedef I                      index_t;
    typedef V                      value_t;
  protected:
    index_t rows = 0;
    index_t cols = 0;
    index_t nnz = 0;
    index_t* csrPtr = nullptr;
    index_t* csrInd = nullptr;
    value_t* csrVal = nullptr;
    index_t* cscPtr = nullptr;
    index_t* cscInd = nullptr;
    value_t* cscVal = nullptr;
    int needFree = 0; //0: dont'free, 1: free all, 2: free only csr data, 3: free only csc data;

  public:

    virtual ~SparseMatrix() {
        switch (needFree) {
            case 0: break;
            case 1:
                delete[] csrPtr;
                delete[] csrInd;
                delete[] csrVal;
                delete[] cscPtr;
                delete[] cscInd;
                delete[] cscVal;
                break;
            case 2:
                delete[] csrPtr;
                delete[] csrInd;
                delete[] csrVal;
                break;
            case 3:
                delete[] cscPtr;
                delete[] cscInd;
                delete[] cscVal;
                break;
        }
    }

    template <class EVype>
    self_t& operator=(const SparExpr<V, I, EVype>& expr) {
        updateCscValue();
        const EVype& e = expr.self();
        for (index_t i = 0; i < nnz; ++i) this->at(i) = e.at(i);
        updateCsrValue();
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

    template <class EVype>
    SparseMatrix& operator/=(const SparExpr<V, I, EVype>& expr) {
        *this = *this / expr.self();
        return *this;
    }

    SparseMatrix& operator/=(V csrVal) {
        *this = *this / csrVal;
        return *this;
    }

    SparseMatrix(index_t rows,
                 index_t cols,
                 index_t nnz,
                 index_t *csrPtr,
                 index_t *csrInd,
                 value_t *csrVal,
                 index_t *cscPtr,
                 index_t *cscInd,
                 value_t *cscVal,
                 bool _needFree = false)
        : rows(rows),
          cols(cols),
          nnz(nnz),
          csrPtr(csrPtr),
          csrInd(csrInd),
          csrVal(csrVal),
          cscPtr(cscPtr),
          cscInd(cscInd),
          cscVal(cscVal),
          needFree(_needFree ? 1 : 0) {}

    SparseMatrix() {
    }

    SparseMatrix(const SparseMatrix<V, I> &o) : rows(o.rows), cols(o.cols), nnz(o.nnz), csrPtr(o.csrPtr), csrInd(o.csrInd), csrVal(o.csrVal), cscPtr(o.cscPtr), cscInd(o.cscInd), cscVal(o.cscVal), needFree(0) {
    }

    SparseMatrix(index_t rows, index_t cols, index_t nnz) :
        rows(rows), cols(cols), nnz(nnz),
        csrPtr(new index_t[rows + 1]), csrInd(new index_t[nnz]), csrVal(new value_t[nnz]),
        cscPtr(new index_t[rows + 1]), cscInd(new index_t[nnz]), cscVal(new value_t[nnz]),
        needFree(1) {
    }

    SparseMatrix(index_t rows, index_t cols, index_t nnz,
                     index_t* ptr, index_t* ind, value_t* val, bool csr = false, bool _needFree = false) : rows(rows), cols(cols), nnz(nnz) {
        if (csr) {
            csrPtr = ptr;
            csrInd = ind;
            csrVal = val;
            cscPtr = new index_t[cols + 1];
            cscInd = new index_t[nnz];
            cscVal = new value_t[nnz];
            transpose(cscPtr, cscInd, cscVal, rows, cols, nnz, csrPtr, csrInd, csrVal, true);
            needFree = _needFree ? 1 : 3;
        } else {
            cscPtr = ptr;
            cscInd = ind;
            cscVal = val;
            csrPtr = new index_t[rows + 1];
            csrInd = new index_t[nnz];
            csrVal = new value_t[nnz];
            transpose(csrPtr, csrInd, csrVal, rows, cols, nnz, cscPtr, cscInd, cscVal, false);
            needFree = _needFree ? 1 : 2;
        }
    }

    static void transpose(index_t*& newPtr, index_t*& newInd, value_t*& newVal, index_t rows, index_t cols, index_t nnz, const index_t* oldPtr, const index_t* oldInd, const value_t* oldVal, bool csr = false) {
        if (csr) {
            memset(newPtr, 0, sizeof(index_t) * (cols + 1));
            for (index_t i = 0; i < rows; ++i) {
                for (index_t j = oldPtr[i]; j < oldPtr[i + 1]; ++j) {
                    ++newPtr[oldInd[j] + 1];
                }
            }
            for (index_t i = 1; i <= cols; ++i) newPtr[i] += newPtr[i - 1];
            for (index_t i = 0; i < rows; ++i) {
                for (index_t j = oldPtr[i]; j < oldPtr[i + 1]; ++j) {
                    index_t &ptr = newPtr[oldInd[j]];
                    newInd[ptr] = i;
                    newVal[ptr] = oldVal[j];
                    ++ptr;
                }
            }
            for (index_t i = cols; i > 0; --i) newPtr[i] = newPtr[i - 1]; newPtr[0] = 0;
        } else {
            memset(newPtr, 0, sizeof(index_t) * (rows + 1));
            for (index_t i = 0; i < cols; ++i) {
                for (index_t j = oldPtr[i]; j < oldPtr[i + 1]; ++j) {
                    ++newPtr[oldInd[j] + 1];
                }
            }
            for (index_t i = 1; i <= rows; ++i) newPtr[i] += newPtr[i - 1];
            for (index_t i = 0; i < cols; ++i) {
                for (index_t j = oldPtr[i]; j < oldPtr[i + 1]; ++j) {
                    index_t &ptr = newPtr[oldInd[j]];
                    newInd[ptr] = i;
                    newVal[ptr] = oldVal[j];
                    ++ptr;
                }
            }
            for (index_t i = rows; i > 0; --i) newPtr[i] = newPtr[i - 1]; newPtr[0] = 0;
        }
    }

    void updateCsrValue() {
        for (index_t i = 0; i < cols; ++i) {
            for (index_t j = cscPtr[i]; j < cscPtr[i + 1]; ++j) {
                index_t &ptr = csrPtr[cscInd[j]];
                csrVal[ptr] = cscVal[j];
                ++ptr;
            }
        }
        for (index_t i = rows; i > 0; --i) csrPtr[i] = csrPtr[i - 1]; csrPtr[0] = 0;
    }

    void updateCscValue() {
        for (index_t i = 0; i < rows; ++i) {
            for (index_t j = csrPtr[i]; j < csrPtr[i + 1]; ++j) {
                index_t &ptr = cscPtr[csrInd[j]];
                cscVal[ptr] = csrVal[j];
                ++ptr;
            }
        }
        for (index_t i = cols; i > 0; --i) cscPtr[i] = cscPtr[i - 1]; cscPtr[0] = 0;
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
    I *getCsrPtr() const {
        return csrPtr;
    }
    I *getCsrInd() const {
        return csrInd;
    }
    V *getCsrVal() const {
        return csrVal;
    }
    I *getCscPtr() const {
        return cscPtr;
    }
    I *getCscInd() const {
        return cscInd;
    }
    V *getCscVal() const {
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

    static inline ConstSparExpr<V, I> C(V v) {
        return ConstSparExpr<V, I>(v);
    };

    struct Vector: SparExpr<V, I, Vector> {
        typedef self_t           belong_t;
        typedef belong_t::Vector self_t;
        typedef I                index_t;
        typedef V                value_t;
        const index_t rows = 0;
        const index_t cols = 0;
        const index_t nnz = 0;
        const index_t* index = nullptr;
        value_t* value = nullptr;

        Vector() {}

        Vector(const I rows, const I cols, const I nnz, const I *index, V *value)
            : rows(rows), cols(cols), nnz(nnz), index(index), value(value) {}

        inline value_t& at(index_t i) {
            assert(i < nnz);
            return value[i];
        }

        inline const value_t& at(index_t i) const {
            assert(i < nnz);
            return value[i];
        }

        template <class EType>
        self_t& operator=(const SparExpr<V, I, EType> &expr) {
            const EType& e = expr.self();
            for (index_t i = 0; i < nnz; ++i) this->at(i) = e.at(i);
            return *this;
        }

        template <class Etype>
        self_t& operator+=(const SparExpr<V, I, Etype>& e) {
            *this = *this + e;
            return *this;
        }

        template <class Etype>
        self_t& operator-=(const SparExpr<V, I, Etype>& e) {
            *this = *this - e;
            return *this;
        }

        template <class Etype>
        self_t& operator*=(const SparExpr<V, I, Etype>& e) {
            *this = *this * e;
            return *this;
        }

        template <class Etype>
        self_t& operator/=(const SparExpr<V, I, Etype>& e) {
            *this = *this / e;
            return *this;
        }

        self_t& operator/=(V v) {
            *this = *this / belong_t::C(v);
            return *this;
        }

        inline index_t nrow() const { return rows; }

        inline index_t ncol() const { return cols; }

        inline index_t getNnz() const { return nnz; }

        void print(bool head = true) const {
            if (head) std::printf("SparMat::Vec %d x %d = %d\n", rows, cols, nnz);
            if (nnz == 0) {
                for (index_t i = 0; i < max(rows, cols); ++i) std::printf(".\t");
                return;
            }
            for (index_t i = 0; i < index[0]; ++i) std::printf(".\t");
            for (index_t i = 0; i < nnz; ++i) {
                std::printf("%e\t", value[i]);
                for (int j = index[i] + 1; j < (i == nnz - 1 ? cols : index[i + 1]); ++j) std::printf(".\t");
            }
            std::printf("\n");
        }

        value_t squaredEuclideanDist(const Vector& o) const {
            value_t r = 0;
            index_t i = 0;
            index_t oi = 0;
            while (i < nnz && oi < o.nnz) {
                if (index[i] == o.index[oi]) {r += (at(i) - o.at(oi)) * (at(i) - o.at(oi)); ++i; ++oi; }
                else if (index[i] < o.index[oi]) {r += (at(i)) * (at(i)); ++i; }
                else {r += (o.at(oi)) * (o.at(oi)); ++oi; }
            }
            while (i < nnz) {r += (at(i)) * (at(i)); ++i;}
            while (oi < o.nnz) {r += (o.at(oi)) * (o.at(oi)); ++oi;}
            return r;
        }
        inline value_t euclideanDist(const Vector& o) const { return std::sqrt(squaredEuclideanDist(o)); }
    };

    typedef Vector Row;
    typedef Vector Col;

    inline Row row(int i) const {
        return Row(1, cols, csrPtr[i + 1] - csrPtr[i], csrInd + csrPtr[i], csrVal + csrPtr[i]);
    }

    inline Col col(int i) const {
        return Col(rows, 1, cscPtr[i + 1] - cscPtr[i], cscInd + cscPtr[i], cscVal + cscPtr[i]);
    }

    /** Utils methods **/

    static self_t rnorm(index_t rows, index_t cols, double density, int seed) {
        srand(seed);
        self_t r;
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

    SparseMatrix<V, I> operator~() {
        return SparseMatrix<V, I>(cols, rows, nnz, cscPtr, cscInd, cscVal, csrPtr, csrInd, csrVal);
    }

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

};

#endif //NLP_CUDA_SPARSE_MAVRIX_H
