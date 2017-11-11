#include "jni_cuda_CuMatrix.h"
#include "CuDenseExprJ.cu"
#include "CuDenseMatrixJ.cu"

using namespace JNI;

typedef CuDenseExprJ<double> Expr;
typedef CuDenseMatrixJ<double> Mat;

JNIEXPORT void JNICALL Java_jni_cuda_CuMatrix_destruct
  (JNIEnv *, jobject, jlong pointer) {
    Mat* mp = reinterpret_cast<Mat*>(pointer);
    delete(mp);
}

JNIEXPORT jlong JNICALL Java_jni_cuda_CuMatrix_init
  (JNIEnv *, jobject, jint rows, jint cols) {
    Mat* mat = new Mat(rows, cols);
    return reinterpret_cast<jlong>(mat);
}

JNIEXPORT jlong JNICALL Java_jni_cuda_CuMatrix_pow
  (JNIEnv *, jobject, jlong, jdouble) {;}

JNIEXPORT jlong JNICALL Java_jni_cuda_CuMatrix_sqrt
  (JNIEnv *, jobject, jlong pointer) {
    Expr* ep = reinterpret_cast<Expr*>(pointer);
    return reinterpret_cast<jlong>(sqrt(ep));
}

JNIEXPORT jdouble JNICALL Java_jni_cuda_CuMatrix_sum
  (JNIEnv *, jobject, jlong pointer) {
    Expr* ep = reinterpret_cast<Expr*>(pointer);
    return sum(ep);
}

JNIEXPORT jlong JNICALL Java_jni_cuda_CuMatrix_add
        (JNIEnv *, jobject, jlong pointerA, jlong pointerB) {
    Expr* ap = reinterpret_cast<Expr*>(pointerA);
    Expr* bp = reinterpret_cast<Expr*>(pointerB);
    return reinterpret_cast<jlong>(add(ap, bp));
}

JNIEXPORT jlong JNICALL Java_jni_cuda_CuMatrix_addDouble
        (JNIEnv *, jobject, jlong pointerA, jdouble valueB) {
    Expr *ap = reinterpret_cast<Expr *>(pointerA);
    return reinterpret_cast<jlong>(add(ap, valueB));
}
