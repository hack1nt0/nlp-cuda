#include <iostream>
#include <Array.h>
#include <cuda_utils.h>
#include <CpuTimer.h>
#include <math.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

int main() {
    int n = 262144;
//    int n = 4 * 1024 * 1024 + 1e5;
    HostArray<float> array(n);
    for (int i = 0; i < n; ++i) array[i] = drand48();
    array.print();
    printf("--------------------------------\n");

    CpuTimer timer;
    timer.start();
    HostArray<float> ansHost(n);
    Multiply<float> binaryFunction;
    float init = 1.0f;
    float acc = init;
    for (int i = 0; i < array.size; ++i) {
        ansHost[i] = binaryFunction(acc, array[i]);
        acc = binaryFunction(acc, array[i]);
    }
    printf("cpu output: \n");
    ansHost.print();
    timer.stop();
    printf("--------------------------------\n");

    GpuTimer gpuTimer;
    gpuTimer.start();
    DeviceArray<float> bufferDevice(n);
    HostArray<float> ansDevice = scanInclusivelyHost(array.data, 0, array.size, init, binaryFunction, bufferDevice.data, 256);
    printf("gpu output: \n");
    ansDevice.print();
    gpuTimer.stop();

    printf("--------------------------------\n");
    checkResultsExact(ansDevice.data, ansHost.data, n);


//    // generate random data serially
    thrust::host_vector<float> h_vec(n);
    for (int i = 0; i < n; ++i) h_vec[i] = array[i];
//    std::generate(h_vec.begin(), h_vec.end(), rand);
    thrust::device_vector<float> d_prefix(n);
    gpuTimer.start();
    thrust::device_vector<float> d_vec = h_vec;

    // transfer to device and compute sum
    thrust::inclusive_scan(d_vec.begin(), d_vec.end(), d_prefix.begin(), thrust::multiplies<float>());

    thrust::host_vector<float> h_prefix = d_prefix;
    printf("thrust output: \n");
    for (int i = 0; i < 100; ++i) printf("%g ", h_prefix[i]);
    printf("\n");
    gpuTimer.stop();

    return 0;
}
