#include <iostream>
#include <Array.h>
#include <cuda_utils.h>
#include <CpuTimer.h>
#include <thrust/reduce.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <math.h>
using namespace std;

int main() {
    int n = 4 * 1024 * 1024 + 1e5;
    HostArray<float> array(n);
    for (int i = 0; i < n; ++i) array[i] = drand48();
    array.print();
    printf("%d %g\n", n, array[n - 1]);

    CpuTimer timer;
    timer.start();
    float ansHost = 0;
    Max<float> binaryFunction;
    for (int i = 0; i < array.size; ++i) ansHost = binaryFunction(ansHost, array[i]);
    printf("cpu output %g \n", ansHost);
    timer.stop();

    GpuTimer gpuTimer;
    gpuTimer.start();
    DeviceArray<float> bufferDevice(n);
    float ansDevice = reduceHost(array.data, 0, array.size, 0.0f, binaryFunction, bufferDevice.data, 16 * 16, 1024);
    printf("gpu output %g \n", ansDevice);
    gpuTimer.stop();


    // generate random data serially
//    std::generate(h_vec.begin(), h_vec.end(), rand);
    gpuTimer.start();
    thrust::device_vector<float> d_vec(array.data, array.data + n);

    // transfer to device and compute sum
    float x = thrust::reduce(d_vec.begin(), d_vec.end(), 0.0f, binaryFunction);
    printf("thrust output %g \n", x);
    gpuTimer.stop();

    return 0;
}
