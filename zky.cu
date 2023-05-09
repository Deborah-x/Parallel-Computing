#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define DATA_SIZE 10000000

int data[DATA_SIZE];

// 产生大量0-9之间的随机数
void GenerateNumbers(int *number, int size)
{
    for(int i = 0; i < size; i++)
        number[i] = rand()%10;
}

// CUDA 初始化
bool InitCUDA()
{
    int count;

    // 取得支持 CUDA 的装置的数目
    cudaGetDeviceCount(&count);
    if(count == 0) {
        fprintf(stderr, "There is no device.\n");
        return false;
    }
    int i;
    for(i = 0; i < count; i++) {
        cudaDeviceProp prop;
        if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if(prop.major >= 1) {
                break;
            }
        }
    }
    if(i == count) {
        fprintf(stderr, "There is no device supporting CUDA.\n");
        return false;
    }    
    cudaSetDevice(i);
    return true;
}

// __global__ 函数（GPU上执行）计算立方和
__global__ static void sumOfcubes(int *num, int *result) 
{
    int sum = 0;
    int i;
    // clock_t start = clock();
    
    for(i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i] * num[i];
    }
    *result = sum;
    // *time = clock() - start;
   
}

int main()
{
    // CUDA 初始化
    if(!InitCUDA()) {
        return 0;
    }
    // 生成随机数
    GenerateNumbers(data, DATA_SIZE);
    // for(int i = 0; i < DATA_SIZE; i++) {
    //     printf("%d ", data[i]);
    // }
    // printf("\n");
    int *gpudata, *result;
    float time_used1, time_used2;
    cudaMalloc((void**)&gpudata, sizeof(int)*DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int));

    // cudaMemcpy将产生的随机数复制到显卡内存中
    cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);
    
    struct timespec time1 = {0, 0};
    struct timespec time2 = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time1);
    sumOfcubes<<<1 ,1, 0>>>(gpudata, result);
    clock_gettime(CLOCK_REALTIME, &time2);
    time_used1 = (time2.tv_sec - time1.tv_sec)*1000000 +(time2.tv_nsec - time1.tv_nsec)/1000;   // 现在的单位为 us 微秒
    // printf("%ld\n", time2.tv_sec - time1.tv_sec);
    // printf("%ld\n", time2.tv_nsec - time1.tv_nsec);

    long long int sum = 0;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    printf("GPUsum: %lld  Time: %lf\n", sum, time_used1);
    sum = 0;

    struct timespec time3 = {0, 0};
    struct timespec time4 = {0, 0};
    clock_gettime(CLOCK_REALTIME, &time3);
    for(int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i] * data[i];
    }
    clock_gettime(CLOCK_REALTIME, &time4);
    time_used2 = (time4.tv_sec - time3.tv_sec)*1000000 +(time4.tv_nsec - time3.tv_nsec)/1000;
    printf("CPUsum: %lld Time: %lf\n", sum, time_used2);
    return 0;
}   