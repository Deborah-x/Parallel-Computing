#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>

#define N 10

__global__ void Dot(int *a, int *b, int *c) // 明 Kernel函数
{
    __shared__ int temp[N]; // 声明在共享内存中的变量
    temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];    // Kernel 函数中利用 threadIdx.x获得线程索引号
                                                            //threadIdx 是内建变量，它指定 block 内thread索引号
    __syncthreads();    // 线程同步，确保线程模块中每个线程都执行完前面的语句才会执行下一条语句
    if(0 == threadIdx.x)
    {
        int sum = 0;
        for(int i = 0; i < N; i++) 
            sum += temp[i];
        *c = sum;
        printf("sum Calculated on Device: %d\n", *c);
    }
}

void random_inits(int *a, int n) 
{
    for(int i = 0; i < n; i++)
        *(a +i) = rand() % 10;
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    int size = N *sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, sizeof(int));

    a = (int *)malloc(size); random_inits(a, N);
    b = (int *)malloc(size); random_inits(b, N);
    c = (int *)malloc(sizeof(int));

    printf("Array a[N]: \n");
    for(int i = 0; i < N; i++) 
        printf("%d ", a[i]);
    printf("\n");
    printf("Array b[N]: \n");
    for(int i = 0; i <N; i++)
        printf("%d ", b[i]);
    printf("\n\n");

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    Dot<<<1, N>>>(d_a, d_b, d_c);   // 单block多thread

    cudaMemcpy(c, d_c, sizeof(int), cudaMemcpyDeviceToHost);

    int sumHost = 0;
    for(int i = 0; i < N; i++)
        sumHost += a[i] *b[i];
    printf("sum Calculated on Host: %d\n", sumHost);
    printf("Device to Host: a * b = %d\n", *c);
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}

/* Command and Output: 
xiaoli@xiaoli-KLVC-WXX9:~/Project/Parallel-computing$ nvcc dot.cu -o dot
xiaoli@xiaoli-KLVC-WXX9:~/Project/Parallel-computing$ ./dot 
Array a[N]: 
3 6 7 5 3 5 6 2 9 1 
Array b[N]: 
2 7 0 9 3 6 0 6 2 6 

sum Calculated on Device: 168
sum Calculated on Host: 168
Device to Host: a * b = 168
xiaoli@xiaoli-KLVC-WXX9:~/Project/Parallel-computing$ nvprof ./dot
==10285== NVPROF is profiling process 10285, command: ./dot
Array a[N]: 
3 6 7 5 3 5 6 2 9 1 
Array b[N]: 
2 7 0 9 3 6 0 6 2 6 

sum Calculated on Device: 168
sum Calculated on Host: 168
Device to Host: a * b = 168
==10285== Profiling application: ./dot
==10285== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   94.03%  56.426us         1  56.426us  56.426us  56.426us  Dot(int*, int*, int*)
                    3.57%  2.1440us         2  1.0720us     928ns  1.2160us  [CUDA memcpy HtoD]
                    2.40%  1.4400us         1  1.4400us  1.4400us  1.4400us  [CUDA memcpy DtoH]
      API calls:   99.73%  119.18ms         3  39.726ms  2.4210us  119.17ms  cudaMalloc
                    0.09%  101.98us       101  1.0090us     122ns  42.859us  cuDeviceGetAttribute
                    0.08%  91.740us         3  30.580us  6.8740us  68.518us  cudaMemcpy
                    0.04%  53.615us         3  17.871us  2.8870us  45.161us  cudaFree
                    0.03%  36.391us         1  36.391us  36.391us  36.391us  cuDeviceGetName
                    0.03%  30.555us         1  30.555us  30.555us  30.555us  cudaLaunchKernel
                    0.01%  7.6460us         1  7.6460us  7.6460us  7.6460us  cuDeviceGetPCIBusId
                    0.00%  1.8500us         3     616ns     170ns  1.2020us  cuDeviceGetCount
                    0.00%     902ns         2     451ns     148ns     754ns  cuDeviceGet
                    0.00%     441ns         1     441ns     441ns     441ns  cuDeviceTotalMem
                    0.00%     292ns         1     292ns     292ns     292ns  cuDeviceGetUuid
*/