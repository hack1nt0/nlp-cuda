//
// Created by DY on 17-10-18.
//

#include <matrix/DocumentTermMatrix.h>
#include <dist/dist.h>

int main() {
    bool verbose = true;
    DocumentTermMatrix<double> dtm;
    ifstream in("/Users/dy/TextUtils/data/train/spamsms.dtm");
    dtm.read(in);
    dtm.normalize();

    dist<double> distMatrix(dtm, verbose);

    CpuTimer timer;
    timer.start();
    distMatrix.save("dist.double.in");
    printf("w cost %f ms", timer.elapsed());

    timer.start();
    dist<double> distMatrix2; distMatrix2.read("dist.double.in");
    printf("r cost %f ms", timer.elapsed());

    distMatrix.println(4, 4);
    distMatrix2.println(4, 4);

    double s = sum(distMatrix != distMatrix2);
    if (s != 0) {
        cout << s << endl;
        exit(1);
    }
//    cout << distMatrix << endl;
//    for (int i = 0; i < 10; ++i) {
//        for (int j = 0; j < 10; ++j) {
//            double d = distMatrix.at(i, j);
//            cout << setiosflags(ios::scientific) << d << endl;
//        }
//    }
    return 0;

}
