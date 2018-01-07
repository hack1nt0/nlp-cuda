//
// Created by DY on 17-9-27.
//

#ifndef NLP_CUDA_GMM_H
#define NLP_CUDA_GMM_H

#include "../utils/utils.h"
#include "../matrix/matrix.h"

template <class Input>
struct MixtureModel {
    typedef typename Input::value_t        value_t;
    typedef typename Input::index_t        index_t;
    typedef RDenseMatrix<value_t, index_t> dm_t;
    typedef SparseMatrix<value_t, index_t> spm_t;

    struct GmmModel {
        dm_t prior;
        dm_t mean;
        dm_t var;
        dm_t resp;

        GmmModel(index_t rows, index_t cols, index_t k) : prior(1, k), mean(cols, k), var(cols, k), resp(rows, k) {
            prior = dm_t::Const(1. / k);
            mean = dm_t::rnorm(cols, k);
            var  = dm_t::runif(cols, k, 1., 5.);
        }

        GmmModel(index_t rows, index_t cols, index_t k, value_t* prior, value_t* mean, value_t* var, value_t* resp) :
            prior(1, k, prior), mean(cols, k, mean), var(cols, k, var), resp(rows, k, resp) {}
    };

    static void dnormLog(dm_t& resp, const spm_t& x, const dm_t& prior, const dm_t& mean, const dm_t& var, vector<value_t>& normNormLog, vector<value_t>& normConst, vector<value_t>& probLog) {
        index_t k = resp.ncol();
        index_t rows = x.nrow();
        index_t cols = x.ncol();
#pragma omp parallel for
        for (index_t i = 0; i < k; ++i) {
            value_t r = log(3.14*2)*cols;
            auto vari = var.col(i);
            for (index_t j = 0; j < cols; ++j) {
                r += log(vari[j]);
            }
            normNormLog[i] = r/2;
            value_t r3 = 0;
            auto meani = mean.col(i);
            for (index_t i = 0; i < meani.getNnz(); ++i) r3 += meani[i]*meani[i]/vari[i];
            normConst[i] = r3;
        }
#pragma omp parallel for
        for (index_t i = 0; i < rows; ++i) {
            auto xi = x.row(i);
            value_t respiNormLog = 0;
            value_t maxRespiLog = 0;
            for (index_t j = 0; j < k; ++j) {
                auto meanj = mean.col(j);
                auto varj  = var.col(j);
                value_t respij = 0;
                for (index_t m = 0; m < xi.getNnz(); ++m) respij += xi[m] * (xi[m] - meanj[xi.index[m]] * 2.) / varj[xi.index[m]];
                respij = -(respij + normConst[j]) / 2. - (normNormLog[j]);
                respij += log(prior[j]);
                if (j==0) maxRespiLog = respij;
                else maxRespiLog = max(maxRespiLog, respij);
                resp.at(i, j) = respij;
            }
            for (index_t j = 0; j < k; ++j) respiNormLog += exp(resp.at(i, j) - maxRespiLog);
            respiNormLog = maxRespiLog + log(respiNormLog);
            for (index_t j = 0; j < k; ++j) resp.at(i, j) = exp(resp.at(i, j) - respiNormLog);
            probLog[i] = respiNormLog;
        }
    }

//    static inline value_t dnorm(const dv_t& x, const dv_t& mean, const dv_t& var, value_t norm) {
//        value_t r = 0;
//        for (index_t i = 0; i < x.getNnz(); ++i) r += (x[i] - mean[i]) / var[i] * (x[i] - mean[i]);
//        return exp(-r / 2.) / norm;
//    }

    static void gmm(GmmModel& model, Input& x, index_t k, index_t maxItr, value_t tol, value_t lbPrior, value_t lbVar, vector<value_t>& logProb, bool verbose) {
        dm_t& prior = model.prior;
        dm_t& mean = model.mean;
        dm_t& var  = model.var;
        dm_t& resp = model.resp;
        index_t rows = x.nrow();
        index_t cols = x.ncol();
        vector<value_t> normNormLog(k);
        vector<value_t> normConst(k);
        vector<value_t> probLog(rows);
        Input xp2 = x.clone();
        xp2 ^= 2;
        dm_t respk(1, k);
        BlasWrapper blas;

        ProgressBar bar(maxItr );
        for (index_t itr = 0; itr < maxItr; ++itr) {

            if (bar.interrupted()) throw interrupt_exception("SIGINT when E");

            dnormLog(resp, x, prior, mean, var, normNormLog, normConst, probLog);

            value_t curLogProb = 0;
            for (index_t i = 0; i < rows; ++i) curLogProb += probLog[i];
            curLogProb /= rows;
            logProb.push_back(curLogProb);
            if (verbose) {
                bar.increase();
                cout <<  "LogP: " << std::setw(20) << curLogProb;
                cout.flush();
            }
            if (itr > 0 && abs(curLogProb - logProb[itr - 1]) <= tol) return;

            if (bar.interrupted()) throw interrupt_exception("SIGINT when M");

            csum(respk, resp);
            respk = max(respk, dm_t::Const(1e-5)); // todo

            //var[d, k] = sum_i ( (x[d, i] - mean[k, d])^2 * resp[i, k] )
            //var[d, k] = sum_i ( x[d, i]^2 * resp[i, k] ) - 2 * sum_i ( x[d, i] * resp[i, k] ) * mean[d, k] + sum_i (mean[d, k]^2 * resp[i, k])

            blas.mm(var, -2., ~x, resp, 0.);
            var = var * mean;
            blas.mm(var, 1., ~xp2, resp, 1.);
            var = var + mean * mean * respk;

            /** DEBUG **/
//            dm_t var2(cols, k);
//#pragma omp parallel for
//            for (index_t i = 0; i < k; ++i) {
//                for (index_t d = 0; d < cols; ++d) {
//                    var2.at(d, i) = mean.at(d, i)*mean.at(d, i)*respk[i];
//                }
//            }
//            for (index_t i = 0; i < rows; ++i) {
//                auto xi = x.row(i);
//                for (index_t j = 0; j < xi.getNnz(); ++j) {
//                    index_t d = xi.index[j];
//                    index_t v = xi.value[j];
//                    for (index_t kk = 0; kk < k; ++kk) {
//                        var2.at(d, kk) += v * v * resp.at(i, kk) - 2. * v * resp.at(i, kk) * mean.at(d, kk);
//                    }
//                }
//            }
//            if (!diff(var, var2)) throw "";
            /** END OF DEBUG **/

            var = var / respk;
            var = max(var, dm_t::Const(lbVar));

            //mean[d, k] = sum_i (x[d, i] * resp[i, k]) / resp[k];
            blas.mm(mean, 1., ~x, resp, 0.);
            mean = mean / respk;
            prior = respk / dm_t::Const(sum(respk));
            prior = max(prior, dm_t::Const(lbPrior));
            prior = prior / dm_t::Const(sum(prior)); //todo
//            if (verbose) bar.increase();
        }
    }

    static bool diff(const dm_t& A, const dm_t& B) {
        bool r = true;
        for (index_t i = 0; i < A.nrow(); ++i)
            for (index_t j = 0; j < A.ncol(); ++j)
                if (A.at(i, j) != B.at(i, j)) {
                    cerr << i << ' ' << j << ' ' << A.at(i, j) << "!=" << B.at(i, j) << endl;
                    r = false;
                }
        return r;
    }
};


#endif //NLP_CUDA_GMM_H
