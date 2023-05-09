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
__global__ static void sumOfcubes(int *num, int *result, clock_t *time) 
{
    int sum = 0;
    int i;
    clock_t start = clock();
    for(i = 0; i < DATA_SIZE; i++) {
        sum += num[i] * num[i] * num[i];
    }
    *result = sum;
    *time = clock() - start;
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
    clock_t *time;
    clock_t time_used;
    cudaMalloc((void**)&gpudata, sizeof(int)*DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int));
    cudaMalloc((void**)&time, sizeof(clock_t));

    // cudaMemcpy将产生的随机数复制到显卡内存中
    cudaMemcpy(gpudata, data, sizeof(int)*DATA_SIZE, cudaMemcpyHostToDevice);
    
    sumOfcubes<<<1 ,1, 0>>>(gpudata, result, time);
    // 核函数只能在主机端调用，调用时必须申明执行参数
    // 形式: Kernel<<<Dg, Db, Ns, S>>>(param list);
    /*
    <<< >>> 内是核函数的执行参数，告诉编译器运行时如何启动核函数，用于说明内核函数中线程数量，以及线程是如何组织的
    1. 参数 Dg 用于定义整个 grid 的维度和尺寸，即一个 grid 有多少个 block，为 dim3 类型
    Dim3 Dg(Dg.x, Dg.y, Dg.z) 表示一个 grid 中每行有 Dg.x 个 block，每列有 Dg.y 个 block，第三维一般为 1（目前一个核函数对应一个 grid ）
    这样的整个 grid 中共有 Dg.x*Dg.y 个 block
    2. 参数 Db 用于定义一个 block 的维度和尺寸，即每一个 block 有多少个 thread，为 dim3 类型
    Dim3 Db(Db.x, Db.y, Db.z) 表示一个 block 中每行有 Db.x 个 thread，每列有 Db.y 个 thread，高度为 Db.z。Db.x 和 Db.y 最大值为 1024，Db.z 最大值为64
    这样的整个 brid 中共有 Db.x*Db.y*Db.z 个 thread
    3. 参数 Ns 是一个可选参数，用于设置每个 block 除了静态分配的 shared Memory 之外，最多能动态分配的 shared Memory大小，单位为 byte，不需要动态分配时该值为 0 或省略不写
    4. 参数 S 是一个 cudaStream_t 类型的可选参数，初始值为0，表示该核函数处在哪个流中
    */

    int sum = 0;
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&time_used, time, sizeof(clock_t), cudaMemcpyDeviceToHost);
    cudaFree(gpudata);
    cudaFree(result);
    cudaFree(time);
    printf("GPUsum: %d  Time: %ld\n", sum, time_used);
    sum = 0;
    clock_t s = clock();
    for(int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i] * data[i];
    }
    time_used = clock() - s;
    printf("CPUsum: %d Time: %ld\n", sum, time_used);
    return 0;
}   