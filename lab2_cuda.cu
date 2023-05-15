#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

__global__ void swap(int *data, int i, int j){
    int temp;
    temp = data[i]; data[i] = data[j]; data[j] = temp;
    return;
}

__global__ void partition(int *data, int p, int r, int *lx, int *rx){
    int k = (p + r)/2, i = p - 1;
    swap<<<1,1>>>(data, k, r);
    for(int j = p; j < r; j++){
        if(data[j] < data[r]){
            i++;
            swap<<<1,1>>>(data, i, j);
        }
    }
    swap<<<1,1>>>(data, i+1, r);
    int m = i+1;
    *lx = (m-1)*sizeof(int);
    *rx = (m+1)*sizeof(int); 
}

__global__ void qsort(int *data, int l, int r) {
        int *lx = data+l, *rx = data+r;
        // Partition data around pivot value
        partition<<<1,1>>>(data, l , r, lx, rx);
        // Now the recursive launch part
        // Use streams this time
        cudaStream_t s1, s2;
        cudaStreamCreateWithFlags(&s1, 1);
        cudaStreamCreateWithFlags(&s2, 2);

        if(l < *lx)
                qsort<<<1,32,0,s1>>>(data, l, *lx);
        if(r > *rx)
                qsort<<<1,32,0,s2>>>(data, *rx, r);
}

void init(int *data) {
        for(int i = 0; i < N; i++) {
                data[i] = rand()%10;
        }

}

void show(int *data) {
        for(int i = 0; i < N; i++) {
                printf("%d ", data[i]);
        }
}

int main() {
        int *s;
        int *d_s;

        clock_t time;
        clock_t time_used;

        int size = N*sizeof(int);

        s = (int *) malloc(size); init(s);
        for (int i = 0; i < 10; i++)
        {
            printf("%d ", s[i]);
        }
        printf("\n");

        cudaMalloc((void**)&d_s, size);
        cudaMemcpy(d_s, s, size, cudaMemcpyHostToDevice);

        time = clock();
        qsort<<<1, 1>>>(d_s, 0, N-1);
        time_used = clock() - time;
        printf("GPU Time: %ld\n", time_used);
        for (int i = 0; i < 10; i++)
        {
            printf("%d ", s[i]);
        }
        printf("\n");

        free(s);
        cudaFree(d_s);
        return 0;
}

