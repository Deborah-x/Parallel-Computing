#include <stdlib.h>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

// using namespace std;

#define N (1024*256)    // 每次处理的数据量
#define SIZE (N*20)     // 数据总量

// 向量加法核函数
__global__ void add(int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x = threadIdx.x;
    if(i < N)
        c[i] = a[i] + b[i];
}

int main()
{
    cudaDeviceProp mprop;
    cudaGetDeviceProperties(&mprop, 0);
    if(!mprop.deviceOverlap)
    {
        printf("Device not support overlaps, so stream is invalid!\n");
        return 0;
    }

    // 创建计时事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float elapsedTime;

    // 创建流
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    // 开辟主机页锁定内存，并随机初始化数据
    int *host_a, *host_b, *host_c;
    cudaHostAlloc((void**)&host_a, SIZE*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_b, SIZE*sizeof(int), cudaHostAllocDefault);
    cudaHostAlloc((void**)&host_c, SIZE*sizeof(int), cudaHostAllocDefault);
    for(size_t i = 0; i < SIZE; i++) 
    {
        host_a[i] = rand();
        host_b[i] = rand();
    }

    // 声明并开辟相关变量内存
    int *dev_a0, *dev_b0, *dev_c0;  // 用于流0的数据
    int *dev_a1, *dev_b1, *dev_c1;  // 用于流1的数据
    cudaMalloc((void**)&dev_a0, N*sizeof(int));
    cudaMalloc((void**)&dev_b0, N*sizeof(int));
    cudaMalloc((void**)&dev_c0, N*sizeof(int));
    cudaMalloc((void**)&dev_a1, N*sizeof(int));
    cudaMalloc((void**)&dev_b1, N*sizeof(int));
    cudaMalloc((void**)&dev_c1, N*sizeof(int));

    cudaEventRecord(start, 0);
    for(size_t i = 0; i < SIZE; i += 2*N)
    {
        // 复制流0数据a
        cudaMemcpyAsync(dev_a0, host_a+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
        // 复制流1数据a
        cudaMemcpyAsync(dev_a1, host_a+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
        // 复制流0数据b
        cudaMemcpyAsync(dev_b0, host_b+i, N*sizeof(int), cudaMemcpyHostToDevice, stream0);
        // 复制流1数据b
        cudaMemcpyAsync(dev_b0, host_b+i+N, N*sizeof(int), cudaMemcpyHostToDevice, stream1);
        // 执行流0核函数
        add<<<N/256,256,stream0>>>(dev_a0,dev_b0,dev_c0);
        // 执行流1核函数
        add<<<N/256,256,stream1>>>(dev_a1,dev_b1,dev_c1);
        // 复制流0数据c
        cudaMemcpyAsync(host_c+i*N, dev_c0, N*sizeof(int), cudaMemcpyDeviceToHost, stream0);
        // 复制流1数据c
        cudaMemcpyAsync(host_c+i*N+N, dev_c1, N*sizeof(int), cudaMemcpyDeviceToHost, stream1);
    }

    // 流同步
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);

    // 处理计时
    cudaEventSynchronize(stop);
    cudaEventRecord(stop, 0);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    // cout << "GPU Time: " << elapsedTime << "ms" << endl;

    // 销毁所有开辟的内存
    cudaFreeHost(host_a);
    cudaFreeHost(host_b);
    cudaFreeHost(host_c);
    cudaFree(dev_a0); cudaFree(dev_b0); cudaFree(dev_c0);
    cudaFree(dev_a1); cudaFree(dev_b1); cudaFree(dev_c1);

    // 销毁流以及计时事件
    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}