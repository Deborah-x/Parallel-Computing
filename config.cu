#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

void printDeviceProp(const cudaDeviceProp &prop) 
{
    printf("Device Name: %s.\n", prop.name);
    printf("totalGlobalMem: %.0f MBytes ----- %ld Bytes. \n", (float)prop.totalGlobalMem/1024/1024, prop.totalGlobalMem);
    printf("shareMemPerBlock: %ld.\n", prop.sharedMemPerBlock);
    printf("warpSize: %d.\n", prop.warpSize);
    printf("memPitch: %ld.\n", prop.memPitch);
    printf("maxThreadsPer`Block: %d.\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0-2]: %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("maxGridSize[0-2]: %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("totalConstMem: %ld.\n", prop.totalConstMem);
    printf("major.minor: %d %d.\n", prop.major, prop.minor);
    printf("clockRate: %d.\n", prop.clockRate);     // GPU 时钟频率
    printf("textureAlignment: %ld.\n", prop.textureAlignment);
    printf("deviceOverlap: %d.\n", prop.deviceOverlap);
    printf("multiProcessorCount: %d.\n", prop.multiProcessorCount);
}

int main(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printDeviceProp(prop);

    return 0;
}

/*
xiaoli@xiaoli-KLVC-WXX9:~/Project/Parallel-computing$ nvcc config.cu -o config
xiaoli@xiaoli-KLVC-WXX9:~/Project/Parallel-computing$ ./config 
Device Name: NVIDIA GeForce MX350.
totalGlobalMem: 2001 MBytes ----- 2098331648 Bytes. 
shareMemPerBlock: 49152.
warpSize: 32.
memPitch: 2147483647.
maxThreadsPer`Block: 1024.
maxThreadsDim[0-2]: 1024 1024 64.
maxGridSize[0-2]: 2147483647 65535 65535.
totalConstMem: 65536.
major.minor: 6 1.
clockRate: 1468000.
textureAlignment: 512.
deviceOverlap: 1.
multiProcessorCount: 5.
*/