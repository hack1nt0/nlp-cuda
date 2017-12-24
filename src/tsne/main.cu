#include "tsne.h"
#include <iostream>
#include <matrix/DenseMatrix.h>
#include <matrix/DocumentTermMatrix.h>
#include <fstream>

using namespace std;

int main() {
    int newRows;
    int newCols;
    int perplexity;
    int max_itr;
    int seed;
    cin >> newRows >> newCols >> perplexity >> max_itr >> seed;
    cout << newRows << '\t' << newCols << '\t' << perplexity << '\t' << max_itr << '\t' << seed << endl;
    DenseMatrix<double> Y(newRows, newCols);
    vector<int> landmarks(newRows);
    ifstream fin("/Users/dy/TextUtils/data/train/spamsms.dtm");
    DocumentTermMatrix dtm(fin);
    fin.close();
    dtm.normalize();

    printf("hi\n");
    tsne(Y.data, landmarks.data(), newRows, newCols, dtm.data(), dtm.index(), dtm.row_ptr(),
         dtm.rows(), dtm.cols(), dtm.nnz(), NULL, 0, perplexity, max_itr, seed);

    cout << Y << endl;
    return 0;
}