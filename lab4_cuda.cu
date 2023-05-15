// 一维FFT
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define PI 3.14159

class Complex {
public:
    double real;
    double imag;

    Complex() {

    }

    // Wn 获取n次单位复根中的主单位根
    __device__ static Complex W(int n) {
        Complex res = Complex(cos(2.0 * PI / n), sin(2.0 * PI / n));
        return res;
    }

    // Wn^k 获取n次单位复根中的第k个
    __device__ static Complex W(int n, int k) {
        Complex res = Complex(cos(2.0 * PI * k / n), sin(2.0 * PI * k / n));
        return res;
    }
    
    // 实例化并返回一个复数（只能在Host调用）
    static Complex GetComplex(double real, double imag) {
        Complex r;
        r.real = real;
        r.imag = imag;
        return r;
    }

    // 随机返回一个复数
    static Complex GetRandomComplex() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = (double)rand() / rand();
        return r;
    }

    // 随即返回一个实数
    static Complex GetRandomReal() {
        Complex r;
        r.real = (double)rand() / rand();
        r.imag = 0;
        return r;
    }

    // 随即返回一个纯虚数
    static Complex GetRandomPureImag() {
        Complex r;
        r.real = 0;
        r.imag = (double)rand() / rand();
        return r;
    }

    // 构造函数（只能在Device上调用）
    __device__ Complex(double real, double imag) {
        this->real = real;
        this->imag = imag;
    }
    
    // 运算符重载
    __device__ Complex operator+(const Complex &other) {
        Complex res(this->real + other.real, this->imag + other.imag);
        return res;
    }

    __device__ Complex operator-(const Complex &other) {
        Complex res(this->real - other.real, this->imag - other.imag);
        return res;
    }

    __device__ Complex operator*(const Complex &other) {
        Complex res(this->real * other.real - this->imag * other.imag, this->imag * other.real + this->real * other.imag);
        return res;
    }
};

// 根据数列长度n获取二进制位数
int GetBits(int n) {
    int bits = 0;
    while (n >>= 1) {
        bits++;
    }
    return bits;
}

// 在二进制位数为bits的前提下求数值i的二进制逆转
__device__ int BinaryReverse(int i, int bits) {
    int r = 0;
    do {
        r += i % 2 << --bits;
    } while (i /= 2);
    return r;
}

// 蝴蝶操作, 输出结果直接覆盖原存储单元的数据, factor是旋转因子
__device__ void Bufferfly(Complex *a, Complex *b, Complex factor) {
    Complex a1 = (*a) + factor * (*b);
    Complex b1 = (*a) - factor * (*b);
    *a = a1;
    *b = b1;
}

__global__ void FFT(Complex nums[], Complex result[], int n, int bits) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= n) return;
    for (int i = 2; i < 2 * n; i *= 2) {
        if (tid % i == 0) {
            int k = i;
            if (n - tid < k) k = n - tid;
            for (int j = 0; j < k / 2; ++j) {
                Bufferfly(&nums[BinaryReverse(tid + j, bits)], &nums[BinaryReverse(tid + j + k / 2, bits)], Complex::W(k, j));
            }
        }
        __syncthreads();
    }
    result[tid] = nums[BinaryReverse(tid, bits)];
}

// 打印数列
void printSequence(Complex nums[], const int N) {
    printf("[");
    for (int i = 0; i < N; ++i) {
        double real = nums[i].real, imag = nums[i].imag;
        if (imag == 0) printf("%.16f", real);
        else {
            if (imag > 0) printf("%.16f+%.16fi", real, imag);
            else printf("%.16f%.16fi", real, imag);
        }
        if (i != N - 1) printf(", ");
    }
    printf("]\n");
}

int main() {
    srand(time(0));  // 设置随机数种子
    const int TPB = 1024;  // 每个Block的线程数，即blockDim.x
    const int N = 1024 * 1024;  // 数列大小
    const int bits = GetBits(N);
    
    // 随机生成实数数列
    Complex *nums = (Complex*)malloc(sizeof(Complex) * N), *dNums, *dResult;
    for (int i = 0; i < N; ++i) {
        nums[i] = Complex::GetRandomReal();
    }
    printf("Length of Sequence: %d\n", N);
    // printf("Before FFT: \n");
    // printSequence(nums, N);
    
    // 保存开始时间
    clock_t s = clock();
    
    // 分配device内存，拷贝数据到device
    cudaMalloc((void**)&dNums, sizeof(Complex) * N);
    cudaMalloc((void**)&dResult, sizeof(Complex) * N);
    cudaMemcpy(dNums, nums, sizeof(Complex) * N, cudaMemcpyHostToDevice);
    
    // 调用kernel
    dim3 threadPerBlock = dim3(TPB);
    dim3 blockNum = dim3((N + threadPerBlock.x - 1) / threadPerBlock.x);
    FFT<<<blockNum, threadPerBlock>>>(dNums, dResult, N, bits);

    // 拷贝回结果
    cudaMemcpy(nums, dResult, sizeof(Complex) * N, cudaMemcpyDeviceToHost);
    
    // 计算用时
    clock_t cost = clock() - s;
    // printf("After FFT: \n");
    // printSequence(nums, N);
    printf("Time of Transfromation: %f\n", (double)(cost)/(double)(CLOCKS_PER_SEC));
  
    // 释放内存
    free(nums);
    cudaFree(dNums);
    cudaFree(dResult);
}
