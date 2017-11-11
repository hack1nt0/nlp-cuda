#include <stdio.h>

struct B {
    int value;
    
    B(int value) : value(value) {}

    __device__ __host__ 
    void f() const { printf("B f() %d \n", value); }

    //B(const B& o) { this->value = o.value; }
};

struct A {
    B* b;
    int value;

    A(B* b, int value) : b(b), value(value) {}

    __device__ __host__ 
    void f() const { 
        printf("A f() %d \n", value);
        b->f();
    }

    A(const A& o) {
        this->value = o.value;
        cudaMalloc(&this->b, sizeof(B));
        cudaDeviceSynchronize();
        cudaMemcpy(this->b, new B(*o.b), sizeof(B), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
    }

};


__global__
void fKernel(A a) {
    a.f();
}


int main() {
    B b(102);
    A a(&b, 101);
    a.f();

    fKernel<<<1, 1>>>(a);
    cudaDeviceSynchronize();

    return 0;
} 
