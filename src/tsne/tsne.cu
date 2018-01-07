#include "tsne.h"
#include "matrix/CuSparseMatrix.cu"
#include "../knn/knn.h"

using namespace std;

typedef DeviceDenseMatrix<double> Mat;

__global__
void QKernel(Mat Q, Mat Y);

__global__
void GKernel(Mat G, Mat P, Mat Q, double QNorm, Mat Y);

void tsne(double* Y, int* landmarks, int newRows, int newCols,
        const double* data, const int* index, const int* row_ptr, 
        int rows, int cols, int nnz,
        const int* belong, int classes,
        int perplexity, int max_itr, unsigned int seed) {
    printf("tsne...");
    assert(newCols < cols && newRows >= classes);
    srand(seed);
    if (classes > 0) {
        vector<vector<int> *> staff(classes);
        for (int i = 0; i < classes; ++i) staff[i] = new vector<int>();
        printf("here");
        for (int i = 0; i < rows; ++i) staff[belong[i]]->push_back(i);
        printf("hi");
        // make sure every class has at least one member in sample
        for (int i = 0; i < classes; ++i)
            if (staff[i]->size() > 0) {
                int reserved = rand() % staff[i]->size();
                landmarks[i] = staff[i]->at(reserved);
                swap(staff[i]->at(reserved), *(staff[i]->end() - 1));
                staff[i]->pop_back();
            }
        printf("hi");
        int i = 0;
        int j = -1;
        int k = classes;
        while (k < rows) {
            //advance
            j++;
            while (j >= staff[i]->size()) {
                i++;
                j = 0;
            }
            //Reservoir Sampling
            if (k < newRows) {
                landmarks[k] = staff[i]->at(j);
            } else {
                if (double(rand()) / RAND_MAX < double(newRows - classes) / (k - classes + 1)) {
                    landmarks[classes + rand() % (newRows - classes)] = k;
                }
            }
            k++;
        }
    } else {
        int i = 0;
        while (i < rows) {
            //advance
            //Reservoir Sampling
            if (i < newRows) {
                landmarks[i] = i;
            } else {
                if (double(rand()) / RAND_MAX < double(newRows) / (i + 1 - classes + 1)) {
                    landmarks[rand() % newRows] = i;
                }
            }
            i++;
        }
    }
    cout << "landmarks:" << endl;
    for (int i = 0; i < newRows; ++i) {
        cout << landmarks[i] << '\t';
    }
    cout << endl;
    /*
       Loss Function:
       C = KL(P|Q) = \Sum_i\Sum_j Pij*log(Pij/Qij)
       P[N,M] : joint prob (Gussian) distribution of original data
       Q[N,K] : joint prob (Student t-) distribution of mapped data(K << M)
       Pij = Pji = (Pj|i + Pi|j)/2/N
       Pj|i = e^(-|Xi-Xj|^2/2/var_i^2) / Norm
       Qij = Qji = (1 + |Yi-Yj|^2)^-1 / Norm
       Pii = Qii = 0
       
       Gradient Desending:
       Yt = Yt-1 + lr * grad + momentum * (Yt-1 - Yt-2)
       grad[C,Yi] = 4 * \Sum_j (Pij-Qij) * (Yi-Yj) * (1+|Yi-Yj|^2)^-1
     */
    vector<double> var(newRows);
    DistMatrix<double> D(newRows);
//    knn(D.data, data, index, row_ptr, rows, cols, nnz, landmarks, newRows);
    cout << "D" << endl;
    cout << D << endl;
    double eps = pow(2., -52.);
    double logPerplexity = log(perplexity);
    double tol = 1e-6;
    CDenseMatrix<double> P(newRows, newRows);
    double INF = 3e+300;
    vector<double> perp(newRows);
    for (int i = 0; i < newRows; ++i) {
        double l = -INF;
        double r = INF; //todo
        double mid = 1;
        for (int j = 0; j < 50; ++j) {
//            printf("var_i %e\n", var_i);
            double Norm = 0;
            perp[i] = 0;
            for (int j = 0; j < newRows; ++j) {
                if (j == i) continue;
                P.at(i, j) = exp(-D.at(i, j) * D.at(i, j) * mid); //todo
                Norm += P.at(i, j);
                perp[i] += D.at(i, j) * P.at(i, j);
            }
            perp[i] = log(Norm) + mid * perp[i] / Norm;
//            if (perp[i] != perp[i]) {
//                cout << "hi " << Norm << '\t' <<  mid << '\t' << perp[i] << endl;
//                for (int j = 0; j < newRows; ++j) {
//                    cout << D.at(i, j) << endl;
//                }
//                return;
//            }
            for (int j = 0; j < newRows; ++j) P.at(i, j) /= Norm;
//            printf("%d perp %e mid %e l %e r %e\n", i, exp(perp), mid, l, r);
            double diff = perp[i] - logPerplexity;
            if (abs(diff) < tol) break;
            if (diff < 0 || perp[i] != perp[i] /* Norm ~= 0 */ ) {
                r = mid;
                mid = l == -INF ? r / 2 : (l + r) / 2;
            } else {
                l = mid;
                mid = r == INF ? l * 2 : (l + r) / 2;
            }
        }
        printf("perp_%d  %e precision %e var %e\n", i, exp(perp[i]), mid, sqrt(1. / 2 / mid));
    }
    double sumP = 0.;
    for (int i = 0; i < newRows; ++i) {
        P.at(i, i) = 0.;
        for (int j = i + 1; j < newRows; ++j) {
            P.at(i, j) = P.at(j, i) = (P.at(i, j) + P.at(j, i)) * 0.5;
            sumP += P.at(i, j) * 2;
        }
    }
    for (int i = 0; i < newRows; ++i) {
        for (int j = 0; j < newRows; ++j) P.at(i, j) = max(P.at(i, j) /  sumP, eps);
    }
    cout << "P" << endl;
    cout << P << endl;
    Mat d_P(newRows, newRows);
    d_P = P;
    d_P *= 12.;
    double init_mean = 0.;
    double init_var = 1e-4;
    //Box-Muller Transform
    for (int i = 0; i < newRows * newCols; i += 2) {
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        Y[i] = sqrt(-2. * log(u1)) * cos(2. * 3.14 * u2) * init_var + init_mean;
        if (i + 1 < newRows * newCols)
            Y[i + 1] = sqrt(-2. * log(u1)) * sin(2. * 3.14 * u2) * init_var + init_mean;
    }
    CDenseMatrix<double> h_Y(Y, newRows, newCols);
    Mat* d_Y[3];
    for (int i = 0; i < 3; ++i) {
        d_Y[i] = new Mat(newRows, newCols);
        *d_Y[i] = h_Y;
    }
    Mat d_Q(newRows, newRows);
    Mat d_G(newRows, newCols);
    double lr = 0.1;
    double momentum = 0.5;
    int t = 2;

    Mat d_gains(newRows, newCols);
    d_gains = 1.;
    Mat d_incs(newRows, newCols);
    d_incs = 0.;
    double min_gain = 0.01;
    double epsilon = 200;

    printf("Iteration\tCost(KL-devergence)\n");
    for (int itr = 0; itr < max_itr; ++itr) {

        int threads1 = 16 * 16;
        int blocks1 = (newRows + threads1 - 1) / threads1;
        QKernel<<<blocks1, threads1>>>(d_Q, *d_Y[t]);
        checkCudaErrors(cudaDeviceSynchronize());

        double QNorm = sum(d_Q);
        d_Q /= QNorm;
        d_Q = maximum(d_Q, eps);

        int threads2 = 16 * 16;
        int blocks2 = (newRows + threads2 - 1) / threads2;
        GKernel<<<blocks2, threads2>>>(d_G, d_P, d_Q, QNorm, *d_Y[t]);
        checkCudaErrors(cudaDeviceSynchronize());

//        *d_Y[t] = *d_Y[(t - 1 + 3) % 3] - d_G * lr
//                  + (*d_Y[(t - 1 + 3) % 3] - *d_Y[(t - 2 + 3) % 3]) * momentum;
//        if (itr < max_itr - 1) t = (t - 2 + 3) % 3;
        d_gains = (d_gains + 0.2) * (sign(d_G) != sign(d_incs)) +
                  d_gains * 0.8 * (sign(d_G) == sign(d_incs));
        d_gains = maximum(d_gains, min_gain);
        d_incs = d_incs * momentum - d_gains * d_G * epsilon;
        *d_Y[t] += d_incs;

        if (itr % 10 == 0) {
            d_Q = d_P * (log(d_P) - log(d_Q));
            double cost = sum(d_Q);
            printf("%d\t%.30f\n", itr, cost);
        }
        if (itr > 250) {
            momentum = 0.8;
        }
        if (itr == 100) {
            d_P /= 12.;
        }
    }
    (*d_Y[t]).toHost(Y);
}

__global__
void QKernel(Mat Q, Mat Y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid;
    for (int j = 0; j < Q.cols; ++j) {
        double d = 0;
        for (int k = 0; k < Y.cols; ++k) d += (Y.at(i, k) -Y.at(j, k)) * (Y.at(i, k) -Y.at(j, k));
        Q.at(i, j) = 1. / (1. + d);
    }
    Q.at(i, i) = 0.;
}

//grad[C,Yi] = 4 * \Sum_j (Pij-Qij) * (Yi-Yj) * (1+|Yi-Yj|^2)^-1
__global__
void GKernel(Mat G, Mat P, Mat Q, double QNorm, Mat Y) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int i = tid;
    for (int k = 0; k < G.cols; ++k) {
        double g = 0;
        for (int j = 0; j < G.rows; ++j)
            g += (P.at(i, j) - Q.at(i, j)) * (Y.at(i, k) - Y.at(j, k)) * Q.at(i, j) * QNorm;
        G.at(i, k) = g * 4.;
    }
}

