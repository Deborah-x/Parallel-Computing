#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define DATA_SIZE 1000
#define THREAD_NUM 256

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


__global__ static void sumOfSquares(int *num, int *result, clock_t *time) {
    const int tid = threadIdx.x;
    // 计算每个线程要完成的量
    const int size = DATA_SIZE/THREAD_NUM;
    int sum = 0;
    int i;
    
    clock_t start;
    // 只在 thread 0 进行记录
    if(tid == 0) start = clock();
    for(i = tid*size; i < (tid+1)*size; i++) {
        sum += num[i] * num[i] * num[i];
        result[tid] = sum;
        if(tid == 0) *time = clock() - start;
    }
}

int main() {
    // CUDA 初始化
    if(!InitCUDA()) {
        return 0;
    }
    // 生成随机数
    GenerateNumbers(data, DATA_SIZE);

    int *gpudata, *result;
    clock_t *time;
    clock_t time_used;
    cudaMalloc((void**)&gpudata, sizeof(int)*DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int)*THREAD_NUM);
    cudaMalloc((void**)&time, sizeof(clock_t));

    // cudaMemcpy将产生的随机数复制到显卡内存中
    cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);
    
    sumOfSquares<<<1 ,THREAD_NUM, 0>>>(gpudata, result, time);

    int sum[THREAD_NUM];
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);

    int final_sum = 0;
    for(int i = 0; i < THREAD_NUM; i++) {
        final_sum += sum[i];
    }
    printf("GPUsum: %d  Time: %ld\n", final_sum, time_used);
    final_sum = 0;
    for(int i = 0; i < DATA_SIZE; i++) {
        final_sum += data[i] * data[i] * data[i];
    }
    printf("CPUsum: %d \n", final_sum);
    return 0;
}