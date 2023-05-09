#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1000

__global__ void matrixMulGlobalKernel(int* pfMatrixA, int* pfMatrixB, int* pfMatrixC, int m, int n, int k)
{
    int nRow = blockIdx.y * blockDim.y + threadIdx.y;
    int nCol = blockIdx.x * blockDim.x + threadIdx.x;
    int fCVal = 0;

    // for(int i =0; i < k; i++)
    // {
    //     fCVal += pfMatrixA[nRow * k + i] * pfMatrixB[i * n + nCol];
    // }

    // pfMatrixC[nRow * n + nCol] = fCVal;
    if (nRow < m && nCol < n) {
        for (int i = 0; i < k; ++i) {
            fCVal += pfMatrixA[nRow * k + i] * pfMatrixB[i * n + nCol];
        }
        pfMatrixC[nRow * n + nCol] = fCVal;
    }

    // __syncthreads();
    // if((threadIdx.x == 0) && (threadIdx.y == 0) && (blockIdx.x == 0) && (blockIdx.y == 0)) {
    //     for(int i = 0; i <N*N; i++) {
    //         printf("%d ", pfMatrixC[i]);
    //         if(i % N == N-1) 
    //             printf("\n");
    //     }
    //     printf("\n");
    // }
    
}

void randomInit(int *a, int n)
{
    for(int i = 0; i < n*n; i++)
        *(a +i) = rand() % 10;
}

int main() {
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    clock_t time;
    clock_t time_used;

    int size = N*N*sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size); randomInit(a, N);
    b = (int *)malloc(size); randomInit(b, N);
    c = (int *)malloc(size); 
    // memset(c, 0, size);

    // printf("Array a: \n");
    // for(int i = 0; i < N*N; i++) {
    //     printf("%d ", a[i]);
    //     if(i % N == N-1) 
    //         printf("\n");
    // }
    // printf("Array b: \n");
    // for(int i = 0; i <N*N; i++) {
    //     printf("%d ", b[i]);
    //     if(i % N == N-1) 
    //         printf("\n");
    // }
    // printf("\n");

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 启动kernel函数计算矩阵乘法
    dim3 block(N, N);
    dim3 grid(1, 1);
    time = clock();
    matrixMulGlobalKernel<<<grid, block>>>(d_a, d_b, d_c, N, N, N);
    time_used = clock() - time;
    printf("GPU Time: %ld\n", time_used);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    int *mulHost; 
    mulHost = (int *)malloc(size); memset(mulHost, 0, size);
    // printf("Matrix Calculate on Host: \n");
    time = clock();
    for(int i = 0; i < N; i++) {
        for(int k = 0; k < N; k++) {
            for(int j = 0; j < N; j++) {
                mulHost[i*N+j] += a[i*N+k]*b[k*N+j];
            }
        }
    }
    time_used = clock() - time;
    printf("CPU Time: %ld\n", time_used);
    // for(int i = 0; i <N*N; i++) {
    //     printf("%d ", mulHost[i]);
    //     if(i % N == N-1) 
    //         printf("\n");
    // }
    // printf("\n");

    // printf("Matrix Calculate on Device: \n");
    // for(int i = 0; i <N*N; i++) {
    //     printf("%d ", c[i]);
    //     if(i % N == N-1) 
    //         printf("\n");
    // }
    // printf("\n");

    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);

    return 0;
}
